import json
import sys
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config
from sglang.srt.runtime_context import get_context

BENCHMARK_DIR = Path(__file__).parents[5] / "benchmark" / "kernels" / "fused_moe_triton"
sys.path.insert(0, str(BENCHMARK_DIR))
import common_utils  # noqa: E402


def test_h200_bf16_config_is_available_for_current_triton_runtime():
    config_path = (
        Path(fused_moe_triton_config.__file__).parent
        / "configs"
        / "triton_3_6_0"
        / "E=128,N=768,device_name=NVIDIA_H200.json"
    )

    assert config_path.is_file()
    assert json.loads(config_path.read_text())["128"]["BLOCK_SIZE_M"] > 0


def test_h100_lingbot_video_configs_enable_tma_only_for_the_tuned_shape():
    config_root = (
        Path(fused_moe_triton_config.__file__).parent / "configs" / "triton_3_7_1"
    )

    for suffix in ("", "_down"):
        config_path = (
            config_root / f"E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3{suffix}.json"
        )
        configs = json.loads(config_path.read_text())

        assert configs["4096"]["USE_TMA"] is True
        assert all(
            "USE_TMA" not in config
            for num_tokens, config in configs.items()
            if num_tokens != "4096"
        )


def test_down_moe_reuses_tuned_up_config_when_separate_config_is_absent(
    monkeypatch, tmp_path
):
    config_root = tmp_path / "configs" / "triton_3_6_0"
    config_root.mkdir(parents=True)
    tuned_config = {"128": {"BLOCK_SIZE_M": 64}}
    (config_root / "up.json").write_text(json.dumps(tuned_config))

    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(fused_moe_triton_config.triton, "__version__", "3.6.0")
    monkeypatch.setattr(
        fused_moe_triton_config,
        "get_config_file_name",
        lambda *args, down_moe=False, **kwargs: "down.json" if down_moe else "up.json",
    )
    fused_moe_triton_config.get_moe_configs.cache_clear()

    try:
        # get_moe_configs reads get_exec().deterministic.
        with get_context().override_server_args(enable_deterministic_inference=False):
            assert fused_moe_triton_config.get_moe_configs(
                32, 768, None, down_moe=True
            ) == {128: {"BLOCK_SIZE_M": 64}}
    finally:
        fused_moe_triton_config.get_moe_configs.cache_clear()


def test_int4_w4a16_tuner_filename_matches_runtime_config_key(monkeypatch):
    """The tuner must build the config filename from the same N the runtime
    derives from w2.shape[2] (shard_intermediate_size // 2).

    An extra int4-only halving of N made tuning_fused_moe_triton.py write
    filenames the server never looked up for int4_w4a16 models (#35252).
    """
    # get_device_name() depends on the host's accelerator and returns None on
    # GPU-less runners; pin it so both sides below exercise the real filename
    # format on identical inputs.
    monkeypatch.setattr(fused_moe_triton_config, "get_device_name", lambda: "CI_DEVICE")

    filename = common_utils.get_config_filename(
        num_experts=256,
        shard_intermediate_size=512,
        hidden_size=1024,
        topk=8,
        dtype=torch.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=[0, 128],
    )

    # The runtime looks up configs with N = w2.shape[2]
    # (fused_moe.py: (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size)),
    # which is shard_intermediate_size // 2 = 256 for this shape.
    runtime_key = fused_moe_triton_config.get_config_file_name(
        256, 256, "int4_w4a16", [0, 128], False
    )

    assert filename == runtime_key
    assert filename == "E=256,N=256,device_name=CI_DEVICE,dtype=int4_w4a16.json"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
