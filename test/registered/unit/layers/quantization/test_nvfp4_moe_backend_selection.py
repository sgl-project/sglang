"""Which MoE runner backend "auto" resolves to for an NVFP4 checkpoint.

Resolution only -- no kernels are launched, so this runs on any device via
``override_platform``. The numerics for each backend live in
``test_nvfp4_moe_backends.py``, which needs real NVFP4 hardware.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.runtime_context import get_flags, override_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

E, H, I, TOPK = 8, 256, 256, 2


def _resolve(request, **platform_flags) -> ModelOptNvFp4FusedMoEMethod:
    """Build the NVFP4 MoE method under a platform and let "auto" resolve."""
    platform = override_platform(**platform_flags)
    platform.install()
    request.addfinalizer(platform.restore)

    quant_config = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
    with get_flags().moe.override(runner_backend=MoeRunnerBackend.AUTO):
        method = ModelOptNvFp4FusedMoEMethod(quant_config)
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=E,
                num_local_experts=E,
                hidden_size=H,
                intermediate_size_per_partition=I,
                top_k=TOPK,
                activation="silu",
                is_gated=True,
            ),
        )
    return method


def test_auto_resolves_to_cutlass_on_sm120(request):
    """Consumer Blackwell must not land on the SM100-only TRTLLM FP4 path.

    Regression test: "auto" used to select FLASHINFER_TRTLLM for every device
    with compute capability >= 10.0, and apply() then raised
    NotImplementedError on SM120/SM121 because the TRTLLM NVFP4 kernels only
    cover SM100.
    """
    method = _resolve(request, is_sm120=True, is_blackwell=True, is_sm100=False)
    assert method._moe_runner_backend is MoeRunnerBackend.FLASHINFER_CUTLASS


def test_cutlass_flag_follows_the_resolved_backend(request):
    """enable_flashinfer_cutlass_moe must reflect resolution, not the raw arg.

    It reads the global --moe-runner-backend, which stays "auto" here. If it
    answered from that, apply() would skip its CUTLASS branch and fall through
    to the NotImplementedError telling the user to pass the very backend that
    was already selected.
    """
    method = _resolve(request, is_sm120=True, is_blackwell=True, is_sm100=False)
    assert method.enable_flashinfer_cutlass_moe is True


def test_auto_still_resolves_to_trtllm_on_sm100(request):
    """Datacenter Blackwell keeps the TRTLLM default."""
    method = _resolve(request, is_sm120=False, is_blackwell=True, is_sm100=True)
    assert method._moe_runner_backend is MoeRunnerBackend.FLASHINFER_TRTLLM
    assert method.enable_flashinfer_cutlass_moe is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
