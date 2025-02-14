import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry, offsetCalculator, restride_dim, TOTAL_CORE_NUM
from .. import runtime
from ..utils import triton_lang_extension as tle
from ..ops.copy import copy
from ..utils.shape_utils import MemOverlap, has_internal_overlapping

@libentry()
@triton.autotune(
    configs = [
        triton.Config({"BLOCK_SIZE": 2**n}, num_stages=s)
            for n in range(6, 16, 2)
            for s in [1, 3]
    ],
    key = ["src_elements"],
)
@triton.jit
def select_scatter_2d_kernel(
    inp_ptr,
    src_ptr,
    dim,
    index,
    src_elements,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    job_id = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    start = job_id * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    w_stride = 1 if dim == 0 else W
    h_stride = W if dim == 0 else 1
    for off in range(start, src_elements, step):
        src_offsets = off + tl.arange(0, BLOCK_SIZE)
        src_mask = src_offsets < src_elements
        src = tl.load(src_ptr + src_offsets, mask=src_mask)
        inp_offsets = (off + tl.arange(0, BLOCK_SIZE)) * w_stride + index * h_stride
        tl.store(inp_ptr + inp_offsets, src, mask=src_mask)


def select_scatter(inp, src, dim, index):
    logging.debug("GEMS SELECT_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    # if inp.ndim == 2:
    #     src_elements = src.numel()
    #     H = out.shape[0]
    #     W = out.shape[1]
    #     grid = lambda meta: (min(triton.cdiv(src_elements, meta["BLOCK_SIZE"]),
    #                              TOTAL_CORE_NUM), )
    #     select_scatter_2d_kernel[grid](out, src, dim, index, src_elements, H,
    #                                    W)
    #     return out

    copy(inp, out0=out)
    indices = [slice(None)] * inp.ndim
    indices[dim] = index
    copy(src, out0=out[indices])

    return out
