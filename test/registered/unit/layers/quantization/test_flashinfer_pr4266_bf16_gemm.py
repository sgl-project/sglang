import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import (
    _FLASHINFER_PR4266_TUNED_TACTICS,
    Bf16GemmBackend,
    should_enable_bf16_splitk_gemm,
    use_flashinfer_pr4266_bf16_gemm,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.mark.parametrize("m,n,k", _FLASHINFER_PR4266_TUNED_TACTICS)
def test_flashinfer_pr4266_selects_tuned_oakhaven_shape(m: int, n: int, k: int):
    assert use_flashinfer_pr4266_bf16_gemm(m, n, k)


@pytest.mark.parametrize("m", [0, 33, 64])
@pytest.mark.parametrize("n,k", [(256, 8192), (512, 8192), (2304, 8192), (2560, 8192)])
def test_flashinfer_pr4266_keeps_large_m_on_existing_path(m: int, n: int, k: int):
    assert not use_flashinfer_pr4266_bf16_gemm(m, n, k)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1024, 2048),
        (3, 256, 8192),
        (16, 8192, 4096),
        (32, 4096, 8192),
    ],
)
def test_flashinfer_pr4266_rejects_unmeasured_shapes(shape: tuple[int, int, int]):
    assert not use_flashinfer_pr4266_bf16_gemm(*shape)


def test_flashinfer_pr4266_backend_is_explicit():
    assert Bf16GemmBackend.FLASHINFER_PR4266.value == "flashinfer_pr4266"


def test_bf16_splitk_is_enabled_by_default():
    assert envs.SGLANG_ENABLE_BF16_SPLITK_GEMM.default is True
    with envs.SGLANG_ENABLE_BF16_SPLITK_GEMM.override(True):
        assert should_enable_bf16_splitk_gemm(Bf16GemmBackend.CUTEDSL)


def test_bf16_splitk_env_kill_switch():
    with envs.SGLANG_ENABLE_BF16_SPLITK_GEMM.override(False):
        assert not should_enable_bf16_splitk_gemm(Bf16GemmBackend.CUTEDSL)


def test_bf16_splitk_does_not_override_torch_backend():
    with envs.SGLANG_ENABLE_BF16_SPLITK_GEMM.override(True):
        assert not should_enable_bf16_splitk_gemm(Bf16GemmBackend.TORCH)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
