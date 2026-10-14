import importlib.util
from pathlib import Path

import pytest
import torch

from sglang.srt.configs.beam import BeamConfig
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


@pytest.mark.parametrize(
    "tp_size,ep_size,num_experts,shard_size", [(4, 1, 256, 1280), (8, 8, 32, 5120)]
)
def test_get_model_config_supports_beam(
    tmp_path, tp_size, ep_size, num_experts, shard_size
):
    BeamConfig().save_pretrained(tmp_path)
    tuned_config = _load_common_utils().get_model_config(
        str(tmp_path), tp_size, ep_size
    )

    assert tuned_config == {
        "num_experts": num_experts,
        "topk": 8,
        "hidden_size": 5120,
        "shard_intermediate_size": shard_size,
        "dtype": torch.bfloat16,
        "block_shape": None,
        "architecture": "BeamForCausalLM",
        "num_layers": 52,
        # DeepSeek first_k_dense_replace metadata, absent on Beam; not Beam's dense
        # layer count, which only the unsupported Beam capture/replay would need.
        "dense_layers": 0,
    }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
