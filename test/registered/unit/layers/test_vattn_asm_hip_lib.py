"""CPU tests for the vattn_asm HIP runtime binding.

ROCm 10 wheels map PyTorch to ``_rocm_sdk_core`` while ``LD_LIBRARY_PATH``
points at ``_rocm_sdk_devel``. Selecting the wrong copy makes ctypes launches
on a torch stream fail with HIP 709. These tests pin the selector only; they
do not need a GPU.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.kernels.ops.attention.vattn_asm_gfx950 import (  # noqa: E402
    _select_amdhip64_path,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_selects_torch_rocm_sdk_core_over_devel():
    core = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7"
    devel = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libamdhip64.so.7"
    assert _select_amdhip64_path([devel, core]) == core
    assert _select_amdhip64_path([core]) == core
    assert _select_amdhip64_path([devel]) == devel
    assert _select_amdhip64_path([]) == "libamdhip64.so"
