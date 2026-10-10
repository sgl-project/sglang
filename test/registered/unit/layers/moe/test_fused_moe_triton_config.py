import json
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config
from sglang.srt.runtime_context import get_context

BENCHMARK_DIR = Path(__file__).parents[5] / "benchmark" / "kernels" / "fused_moe_triton"
sys.path.insert(0, str(BENCHMARK_DIR))


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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
