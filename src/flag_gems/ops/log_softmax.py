import logging

import torch
import triton
import copy
import triton.language as tl

from ..utils import libentry, TOTAL_CLUSTER_NUM

MAX_C_MLU_LOG_SOFTMAX_FORWARD = 16384
MAX_C_MLU_LOG_SOFTMAX_BACKWARD = 8192
MLU_GRID_MAX = 65535

def heuristics_for_tile_k(m, k, max_tile_k, num_sms):
    tile_k = 1
    upper_bound = min(k, max_tile_k)
    while tile_k <= upper_bound:
        num_blocks = m * triton.cdiv(k, tile_k)
        num_waves = num_blocks / num_sms
        if (num_waves > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k

def heuristics_for_num_warps(tile_size):
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b

@triton.heuristics(
    values={
        "TILE_K": lambda args: heuristics_for_tile_k(
            args["M"],
            args["K"],
            MAX_C_MLU_LOG_SOFTMAX_FORWARD,
            TOTAL_CLUSTER_NUM,
        )
    }
)
@triton.heuristics(values={"TILE_N": lambda args: triton.cdiv(MAX_C_MLU_LOG_SOFTMAX_FORWARD, args["TILE_K"])})
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.heuristics(
    values={
        "num_warps": lambda args: heuristics_for_num_warps(
            args["TILE_N"] * args["TILE_K"]
        )
    },
)
@triton.jit
def log_softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    log2e = 1.442695
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        o = e / z
        out = tl.log2(o) / log2e
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            alpha = tl.exp(m - m_new)
            z = z * alpha + tl.exp(inp - m_new)
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced
        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            out = tl.log2(o) / log2e
            tl.store(output_ptr + offsets, out, mask=mask)


def config_prune(configs, named_args, **kwargs):
    N = named_args["N"]
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            config.num_warps,
            config.num_stages,
        )

        # When N is less than MAX_C_MLU_LOG_SOFTMAX_FORWARD, no reduction loops
        doopt = N < MAX_C_MLU_LOG_SOFTMAX_FORWARD
        if doopt:
            BLOCK_N = N
            num_stages = 1
        key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
        if key in configs_map:
            continue
        # change config
        if doopt:
            config = copy.deepcopy(config)
            config.kwargs["BLOCK_N"] = N
            config.num_stages = 1
        # keep config
        configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    return pruned_configs


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": m,
            "BLOCK_N": 2**n
        },
                      num_stages=s,
                      num_warps=1) for m in range(1, 30, 3)
        for n in range(6, 13, 1) for s in [1, 3]
    ],
    key=[
        "M",
        "N",
    ],
    prune_configs_by={'early_config_prune': config_prune},
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["N"] <= args["BLOCK_N"],
    }, )
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    log2e = 1.442695
    if ONE_TILE_PER_CTA:
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offset = tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask,
                      other=-float("inf")).to(tl.float32)
        row_minus_max = inp - tl.max(inp, axis=1)[:, None]
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        softmax_output = numerator / denominator
        output = tl.log2(softmax_output) / log2e
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, output, mask=mask)
    else:
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        tmp0 = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
        for col_offset in range(0, N, BLOCK_N):
            n_offset = col_offset + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M and n_offset[None, :] < N
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask,
                          other=-float("inf")).to(tl.float32)
            # get max for each block
            tmp1 = tl.maximum(inp, tmp0)
            tmp0 = tmp1

        tmp2 = tl.max(tmp0, 1)[:, None]
        tmp3 = tl.full([BLOCK_M, BLOCK_N], 0, tl.float32)
        for col_offset in range(0, N, BLOCK_N):
            n_offset = col_offset + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M and n_offset[None, :] < N
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask,
                          other=-float("inf")).to(tl.float32)
            # minus max value each line
            row_minus_max = inp - tmp2
            numerator = tl.exp(row_minus_max)
            tmp4 = tmp3 + numerator
            tmp3 = tmp4

        denominator = tl.sum(tmp3, axis=1)[:, None]
        for col_offset in range(0, N, BLOCK_N):
            n_offset = col_offset + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M and n_offset[None, :] < N
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask,
                          other=-float("inf")).to(tl.float32)
            row_minus_max = inp - tmp2
            numerator = tl.exp(row_minus_max)
            softmax_output = numerator / denominator
            output = tl.log2(softmax_output) / log2e
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, output, mask)


# ------------------------  backward -------------------------------
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_K": 2**k}, num_warps=1, num_stages = s)
        for k in range(3, 11, 1)
        for s in [1, 3]
    ],
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.heuristics(
    values={
        "TILE_K":
        lambda args: max(triton.cdiv(args["K"], MLU_GRID_MAX), args["TILE_K"])
    })
@triton.heuristics(
    values={
        "TILE_N": lambda args: max(1, MAX_C_MLU_LOG_SOFTMAX_BACKWARD // args["TILE_K"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def log_softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        
        scale = tl.sum(out_grad_tile, axis=0)
        in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[None, :]

        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[None, :]
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K

@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD // (2**k)}, num_warps=1, num_stages = s)
        for k in range(0, 6, 1)
        for s in [1, 3]
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={
        "TILE_M": lambda args: max(1, MAX_C_MLU_LOG_SOFTMAX_BACKWARD // args["TILE_N"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def log_softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_grad_tile, 1)
        in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[:, None]
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            ).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[:, None]
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug("GEMS LOG_SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N

        with torch.mlu.device(inp.device):
            if K > 1:
                logging.debug("GEMS LOGSOFTMAX USE NON INNER")
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                log_softmax_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                logging.debug("GEMS LOGSOFTMAX USE INNER")
                grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), 1, 1)
                log_softmax_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS LOG_SOFTMAX VJP")

        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]

        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N

        with torch.mlu.device(in_grad.device):
            if K > 1:
                logging.debug("GEMS LOG SOFTMAX VJP USE NON INNER")
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                log_softmax_backward_kernel_non_inner[grid](
                    out,
                    out_grad,
                    in_grad,
                    M,
                    N,
                    K,
                )
            else:
                logging.debug("GEMS LOG SOFTMAX VJP USE INNER")
                grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
                log_softmax_backward_kernel_inner[grid](
                    out,
                    out_grad,
                    in_grad,
                    M,
                    N,
                )
        return in_grad, None, None


def log_softmax(x, dim=-1, dtype=None):
    return LogSoftmax.apply(x, dim, dtype)
