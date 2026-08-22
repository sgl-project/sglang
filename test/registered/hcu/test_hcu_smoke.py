import importlib
import sys
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import (
    REGISTER_MAPPING,
    HWBackend,
    register_hcu_ci,
    ut_parse_one_file,
)

register_hcu_ci(est_time=30, suite="stage-a-test-1-hcu")


def _torch():
    return importlib.import_module("torch")


def test_hcu_registration_api():
    assert REGISTER_MAPPING["register_hcu_ci"] is HWBackend.HCU
    assert register_hcu_ci(est_time=30, suite="stage-a-test-1-hcu") is None


def test_hcu_smoke_file_registration():
    registrations, has_main = ut_parse_one_file(str(Path(__file__)))

    assert has_main
    assert len(registrations) == 1
    assert registrations[0].backend is HWBackend.HCU
    assert registrations[0].effective_suite == "stage-a-test-1-hcu"


def test_hip_runtime_and_device_available():
    torch = _torch()

    assert torch.version.hip
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1


def test_hcu_tensor_copy():
    torch = _torch()
    source = torch.arange(16, dtype=torch.float32)

    assert torch.equal(source.cuda().cpu(), source)


def test_hcu_bfloat16_addition():
    torch = _torch()
    left = torch.ones(32, device="cuda", dtype=torch.bfloat16)
    right = torch.full((32,), 2, device="cuda", dtype=torch.bfloat16)

    assert torch.all(left + right == 3)


def test_hcu_bfloat16_matmul():
    torch = _torch()
    left = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    right = torch.eye(16, device="cuda", dtype=torch.bfloat16)

    torch.testing.assert_close(left @ right, left)


def test_hcu_device_synchronize():
    torch = _torch()
    value = torch.ones(1, device="cuda") + 1

    torch.cuda.synchronize()
    assert value.item() == 2


def test_sgl_kernel_common_ops_loaded():
    sgl_kernel = importlib.import_module("sgl_kernel")

    assert sgl_kernel.common_ops is not None


def test_sgl_kernel_silu_and_mul():
    torch = _torch()
    sgl_kernel = importlib.import_module("sgl_kernel")
    inputs = torch.randn((2, 64), device="cuda", dtype=torch.float16)
    expected = inputs[..., 32:] * torch.nn.functional.silu(inputs[..., :32])

    torch.testing.assert_close(
        sgl_kernel.silu_and_mul(inputs), expected, rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
