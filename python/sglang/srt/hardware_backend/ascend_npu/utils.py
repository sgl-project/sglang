import functools
import logging
from typing import Callable

import torch

from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)


def _call_once(func: Callable):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if getattr(func, "_has_call_once", False):
            logger.debug("Function {} has already been called.".format(func.__name__))
            return
        else:
            func._has_call_once = True
            return func(*args, **kwargs)

    return wrapper


@_call_once
def init_ascend_npu_backend():
    assert (
        is_npu()
    ), "This function should only be called within Ascend NPU environment."

    import sgl_kernel_npu  # noqa: F401
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    # Re-mock torch.cuda.is_available cuz transfer_to_npu mocks it True
    torch.cuda.is_available = lambda: False

    torch_npu.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)
