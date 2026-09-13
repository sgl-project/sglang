import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _load_common_utils():
    source = (
        Path(__file__).resolve().parents[5]
        / "benchmark/kernels/fused_moe_triton/common_utils.py"
    )
    spec = importlib.util.spec_from_file_location("fused_moe_common_utils", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_model_config_supports_kimi_vl():
    common_utils = _load_common_utils()
    text_config = SimpleNamespace(
        hidden_size=2048,
        n_routed_experts=64,
        num_experts_per_tok=6,
        moe_intermediate_size=1408,
        torch_dtype=torch.bfloat16,
    )
    model_config = SimpleNamespace(
        architectures=["KimiVLForConditionalGeneration"],
        text_config=text_config,
        get_text_config=lambda: text_config,
    )

    with patch.object(common_utils, "get_config", return_value=model_config):
        tuned_config = common_utils.get_model_config(
            "moonshotai/Kimi-VL-A3B-Instruct", tp_size=4, ep_size=4
        )

    assert tuned_config == {
        "num_experts": 16,
        "topk": 6,
        "hidden_size": 2048,
        "shard_intermediate_size": 2816,
        "dtype": torch.bfloat16,
        "block_shape": None,
        "architecture": "KimiVLForConditionalGeneration",
    }


@pytest.mark.parametrize(
    "tp_size,ep_size,experts,intermediate",
    [(4, 4, 72, 4096), (8, 8, 36, 4096), (8, 4, 72, 2048)],
)
def test_get_model_config_supports_glm5_next(tp_size, ep_size, experts, intermediate):
    common_utils = _load_common_utils()
    text_config = SimpleNamespace(
        hidden_size=4096,
        n_routed_experts=288,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        torch_dtype=torch.bfloat16,
    )
    model_config = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        quantization_config={"weight_block_size": [128, 128]},
        text_config=text_config,
        get_text_config=lambda: text_config,
    )
    with patch.object(common_utils, "get_config", return_value=model_config):
        tuned_config = common_utils.get_model_config(
            "zai-org/GLM-5.3-Flash",
            tp_size=tp_size,
            ep_size=ep_size,
            disable_shared_experts_fusion=True,
        )
    assert tuned_config == {
        "num_experts": experts,
        "topk": 8,
        "hidden_size": 4096,
        "shard_intermediate_size": intermediate,
        "dtype": torch.bfloat16,
        "block_shape": [128, 128],
        "architecture": "Glm5NextForConditionalGeneration",
    }


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
