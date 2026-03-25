import pytest

from sglang.jit_kernel.utils import _REGISTERED_DEPENDENCIES
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)


@pytest.mark.parametrize("name", _REGISTERED_DEPENDENCIES.keys())
def test_availability(name: str) -> None:
    # NOTE: the path resolution should not fail
    _REGISTERED_DEPENDENCIES[name]()
