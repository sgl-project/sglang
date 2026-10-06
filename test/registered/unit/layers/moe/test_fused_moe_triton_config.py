import json
import sys
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe,
    fused_moe_triton_config,
)
from sglang.srt.runtime_context import get_context

BENCHMARK_DIR = Path(__file__).parents[5] / "benchmark" / "kernels" / "fused_moe_triton"
sys.path.insert(0, str(BENCHMARK_DIR))
import common_utils  # noqa: E402


@pytest.mark.parametrize("runtime_triton_version", ["3.6.0", "3.8.0"])
def test_down_moe_reuses_tuned_up_config_when_separate_config_is_absent(
    monkeypatch, tmp_path, runtime_triton_version
):
    config_root = tmp_path / "configs" / "triton_3_6_0"
    config_root.mkdir(parents=True)
    tuned_config = {
        "128": {"BLOCK_SIZE_M": 64, "USE_TMA": True},
        "256": {"BLOCK_SIZE_M": 128, "USE_TMA": False},
    }
    (config_root / "up.json").write_text(json.dumps(tuned_config))

    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        fused_moe_triton_config.triton, "__version__", runtime_triton_version
    )
    monkeypatch.setattr(
        fused_moe_triton_config,
        "get_config_file_name",
        lambda *args, down_moe=False, **kwargs: "down.json" if down_moe else "up.json",
    )
    fused_moe_triton_config.get_moe_configs.cache_clear()
    monkeypatch.setattr(fused_moe, "_moe_support_tma", lambda: True)
    monkeypatch.setattr(
        fused_moe, "moe_align_block_size", lambda *args: (None, None, None)
    )

    try:
        # get_moe_configs reads get_exec().deterministic.
        with get_context().override_server_args(enable_deterministic_inference=False):
            assert fused_moe_triton_config.get_moe_configs(
                32, 768, None, down_moe=True
            ) == {
                128: {"BLOCK_SIZE_M": 64},
                256: {"BLOCK_SIZE_M": 128},
            }
            _, _, down_tma, up_tma, *_ = fused_moe._prepare_fused_moe_run(
                torch.empty((128, 2048), dtype=torch.bfloat16, device="meta"),
                torch.empty((32, 1536, 2048), device="meta"),
                torch.empty((32, 2048, 768), device="meta"),
                torch.empty((128, 8), dtype=torch.int32, device="meta"),
                use_fp8_w8a8=True,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=[128, 128],
            )
            assert up_tma is True
            assert down_tma is False
            assert fused_moe_triton_config.get_moe_configs(
                32, 768, None, down_moe=False
            ) == {int(key): value for key, value in tuned_config.items()}
    finally:
        fused_moe_triton_config.get_moe_configs.cache_clear()


@pytest.mark.parametrize("use_tma", [False, True])
def test_separate_down_moe_config_preserves_tma_setting(monkeypatch, tmp_path, use_tma):
    config_root = tmp_path / "configs" / "triton_3_6_0"
    config_root.mkdir(parents=True)
    tuned_config = {"128": {"BLOCK_SIZE_M": 64, "USE_TMA": use_tma}}
    (config_root / "down.json").write_text(json.dumps(tuned_config))

    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(fused_moe_triton_config.triton, "__version__", "3.6.0")
    monkeypatch.setattr(
        fused_moe_triton_config,
        "get_config_file_name",
        lambda *args, down_moe=False, **kwargs: "down.json" if down_moe else "up.json",
    )
    fused_moe_triton_config.get_moe_configs.cache_clear()

    try:
        with get_context().override_server_args(enable_deterministic_inference=False):
            assert fused_moe_triton_config.get_moe_configs(
                32, 768, None, down_moe=True
            ) == {128: {"BLOCK_SIZE_M": 64, "USE_TMA": use_tma}}
    finally:
        fused_moe_triton_config.get_moe_configs.cache_clear()


def test_int4_tuner_filename_uses_runtime_down_projection_dimension(monkeypatch):
    monkeypatch.setattr(
        common_utils,
        "get_config_file_name",
        lambda E, N, *_args: f"E={E},N={N}.json",
    )

    filename = common_utils.get_config_filename(
        num_experts=256,
        shard_intermediate_size=512,
        hidden_size=1024,
        topk=8,
        dtype=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=[128, 128],
    )

    assert filename == "E=256,N=256.json"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
