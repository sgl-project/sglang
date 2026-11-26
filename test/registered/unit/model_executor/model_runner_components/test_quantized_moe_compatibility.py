"""The post-load check must accept FP8 layouts validated during weight creation."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_components import (  # noqa: E402
    moe_ep_setup,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _check(*, architecture="BailingMoeV3ForCausalLM", quant_method="fp8", size=1280):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[architecture],
            model_type="bailing_hybrid",
            use_kda=True,
            quantization_config={
                "quant_method": quant_method,
                "weight_block_size": [128, 128],
                "is_checkpoint_fp8_serialized": True,
            },
        ),
        hf_text_config=SimpleNamespace(moe_intermediate_size=size),
    )
    with (
        patch.object(moe_ep_setup, "_use_aiter", False),
        patch.object(
            moe_ep_setup.envs.SGLANG_SHARED_EXPERT_TP1, "get", return_value=False
        ),
    ):
        moe_ep_setup.check_quantized_moe_compatibility(
            model_config=model_config, tp_size=8, moe_ep_size=1, moe_dp_size=1
        )


@pytest.mark.parametrize(
    "architecture",
    ["BailingMoeV3ForCausalLM", "BailingMoeForCausalLMNextN", "DeepseekV3ForCausalLM"],
)
def test_fp8_partial_blocks_defer_to_weight_creation(architecture):
    _check(architecture=architecture)


def test_non_fp8_partial_blocks_remain_rejected():
    with pytest.raises(ValueError, match="For quantized MoE models"):
        _check(quant_method="mxfp8")


def test_non_fp8_aligned_blocks_remain_supported():
    _check(quant_method="mxfp8", size=1024)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
