import json
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config
from sglang.srt.runtime_context import get_context


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


def test_nemotron_h200_config_covers_serving_token_counts():
    config_path = (
        Path(fused_moe_triton_config.__file__).parent
        / "configs"
        / "triton_3_6_0"
        / "E=128,N=1856,device_name=NVIDIA_H200.json"
    )
    expected_token_counts = {
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        72,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        384,
        512,
        1024,
        1536,
        2048,
        2560,
        3072,
        5120,
        8192,
    }
    assert set(map(int, json.loads(config_path.read_text()))) == expected_token_counts
    assert set(map(int, json.loads(config_path.read_text()))) == expected_token_counts


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


def test_nearest_moe_config_lookup_is_cached(monkeypatch):
    calls = 0
    configs = {
        32: {"BLOCK_SIZE_M": 16},
        64: {"BLOCK_SIZE_M": 32},
    }

    def get_moe_configs(*args, **kwargs):
        nonlocal calls
        calls += 1
        return configs

    monkeypatch.setattr(fused_moe_triton_config, "get_moe_configs", get_moe_configs)
    fused_moe_triton_config._get_nearest_moe_config.cache_clear()

    try:
        expected = (configs[64], 32)
        args = (128, 1856, None, 0, 0, False, False, 60)
        assert fused_moe_triton_config._get_nearest_moe_config(*args) == expected
        assert fused_moe_triton_config._get_nearest_moe_config(*args) == expected
        assert calls == 1
    finally:
        fused_moe_triton_config._get_nearest_moe_config.cache_clear()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
