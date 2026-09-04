import inspect
import sys

import pytest
from cutlass._mlir.dialects import nvvm

from sglang.kernels.ops.attention.flash_attn.cute.utils import (
    _NVVM_FMAX_REQUIRES_EXPLICIT_RESULT_TYPE,
    _nvvm_fmax_requires_explicit_result_type,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _old_fmax(res, a, b, *, c=None, loc=None, ip=None):
    return res, a, b, c, loc, ip


def _new_fmax(a, b, *, c=None, results=None, loc=None, ip=None):
    return a, b, c, results, loc, ip


class _OpaqueFmax:
    @property
    def __signature__(self):
        raise ValueError("signature unavailable")

    def __call__(self, *args, **kwargs):
        return args, kwargs


def test_detects_old_and_new_nvvm_fmax_bindings():
    assert _nvvm_fmax_requires_explicit_result_type(_old_fmax)
    assert not _nvvm_fmax_requires_explicit_result_type(_new_fmax)


def test_rejects_an_uninspectable_nvvm_fmax_binding():
    with pytest.raises(RuntimeError, match="Unable to inspect"):
        _nvvm_fmax_requires_explicit_result_type(_OpaqueFmax())


def test_selected_call_shape_binds_to_installed_nvvm_fmax():
    signature = inspect.signature(nvvm.fmax)
    old_args = (object(), object(), object())
    new_args = (object(), object())

    if _NVVM_FMAX_REQUIRES_EXPLICIT_RESULT_TYPE:
        signature.bind(*old_args, c=None)
        with pytest.raises(TypeError):
            signature.bind(*new_args, c=None)
    else:
        signature.bind(*new_args, c=None)
        with pytest.raises(TypeError):
            signature.bind(*old_args, c=None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
