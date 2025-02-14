import logging

import torch
import triton

from flag_gems.ops.rand import rand_kernel
from ..utils import TOTAL_CORE_NUM
from flag_gems.utils.random_utils import philox_backend_seed_offset

from ..runtime import torch_device_fn

UNROLL = 4

def rand_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logging.debug("GEMS RAND_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),)
    philox_seed, philox_offset = philox_backend_seed_offset(N)
    with torch_device_fn.device(x.device):
        rand_kernel[grid_fn](out, N, philox_seed, philox_offset, num_stages=3, num_warps=1)
    return out
