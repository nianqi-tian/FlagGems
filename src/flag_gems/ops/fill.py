import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry, TOTAL_CORE_NUM


@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 65536}, num_stages=1, num_warps=1),
    ],
    key=['N'],
)
@triton.jit(do_not_specialize=["value_scalar"])
def fill_scalar_kernel(
    out_ptr,
    N,
    value_scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, N, step):
        offset = block_start_offset + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 65536}, num_stages=1, num_warps=1),
    ],
    key=['N'],
)
@triton.jit
def fill_tensor_kernel(
    out_ptr,
    N,
    value_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, N, step):
        offset = block_start_offset + tl.arange(0, BLOCK_SIZE)
        value_scalar = tl.load(value_ptr)  # load the value from the tensor.
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def fill_tensor(input, value):
    logging.debug("GEMS FILL TENSOR")
    out = torch.empty_like(input)
    N = out.numel()
    # grid = triton.cdiv(N, BLOCK_SIZE)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch.cuda.device(input.device):
        fill_tensor_kernel[grid_fn](out, N, value)
    return out


def fill_scalar(input, value):
    logging.debug("GEMS FILL SCALAR")
    out = torch.empty_like(input)
    N = out.numel()
    # grid = triton.cdiv(N, BLOCK_SIZE)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch.cuda.device(input.device):
        fill_scalar_kernel[grid_fn](out, N, value)
    return out
