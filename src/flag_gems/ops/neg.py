import logging

import triton

from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    logging.debug("GEMS NEG")
    return neg_func(A)
