"""CPU unit tests for the NVFP4 MoE method's ``--moe-runner-backend auto``
resolution.

``auto`` must be resolved once, when the method is built, and the same answer
must drive the weight layout (``enable_flashinfer_trtllm_moe``) and the runner
``create_moe_runner`` picks. Resolving only in ``create_moe_runner`` left the
weights in the CUTLASS layout (no ``g1_scale_c``) while ``apply`` ran the
TRT-LLM runner, which failed the first forward of Kimi-K3-NVFP4 with
``FusedMoE has no attribute g1_scale_c``.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant as mq
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a-test-cpu")


@contextlib.contextmanager
def _device(capability, *, is_blackwell):
    with (
        mock.patch.object(mq, "is_cuda", return_value=True),
        mock.patch.object(mq, "get_device_capability", return_value=capability),
        mock.patch.object(
            mq, "get_platform", return_value=SimpleNamespace(is_blackwell=is_blackwell)
        ),
        mock.patch.object(
            mq,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_megamoe=lambda: False),
        ),
    ):
        yield


def _method(backend: MoeRunnerBackend):
    with mock.patch.object(mq, "get_moe_runner_backend", return_value=backend):
        return mq.ModelOptNvFp4FusedMoEMethod(quant_config=mock.Mock())


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0)])
def test_auto_resolves_to_trtllm_on_blackwell(capability):
    with _device(capability, is_blackwell=True):
        assert (
            mq.resolve_nvfp4_moe_runner_backend(MoeRunnerBackend.AUTO)
            == MoeRunnerBackend.FLASHINFER_TRTLLM
        )
        m = _method(MoeRunnerBackend.AUTO)
    assert m._moe_runner_backend == MoeRunnerBackend.FLASHINFER_TRTLLM
    # the weight layout the TRT-LLM runner reads (g1_scale_c, shuffled scales)
    assert m.enable_flashinfer_trtllm_moe is True
    assert m.use_flashinfer_trtllm_weight_layout is True


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0)])
def test_auto_resolves_to_marlin_before_blackwell(capability):
    with _device(capability, is_blackwell=False):
        assert (
            mq.resolve_nvfp4_moe_runner_backend(MoeRunnerBackend.AUTO)
            == MoeRunnerBackend.MARLIN
        )
        m = _method(MoeRunnerBackend.AUTO)
    assert m._moe_runner_backend == MoeRunnerBackend.MARLIN
    assert m.enable_flashinfer_trtllm_moe is False


def test_explicit_backend_is_kept():
    with _device((10, 0), is_blackwell=True):
        for backend in (
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            assert mq.resolve_nvfp4_moe_runner_backend(backend) == backend
            m = _method(backend)
            assert m._moe_runner_backend == backend
            assert m.enable_flashinfer_trtllm_moe is (
                backend == MoeRunnerBackend.FLASHINFER_TRTLLM
            )


def test_non_blackwell_without_marlin_fallback_is_rejected():
    with _device((9, 0), is_blackwell=False):
        with pytest.raises(ValueError, match="does not support NVFP4"):
            _method(MoeRunnerBackend.FLASHINFER_TRTLLM)
