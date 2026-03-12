import pytest

from sglang.srt.debug_utils.comparator.preset import PRESETS, expand_preset
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default", nightly=True)


class TestExpandPreset:
    """Test preset expansion logic."""

    def test_explicit_preset(self):
        """--preset sglang_megatron expands into its argv."""
        argv = [
            "--baseline-path",
            "/a",
            "--preset",
            "sglang_megatron",
            "--diff-threshold",
            "0.01",
        ]
        result = expand_preset(argv, presets=PRESETS)
        assert "--preset" not in result
        assert "--grouping-skip-keys" in result
        assert "concat_steps" in result
        assert "--baseline-path" in result
        assert "--diff-threshold" in result

    def test_default_preset_applied(self):
        """No --preset and no --grouping-skip-keys triggers default preset."""
        argv = ["--baseline-path", "/a"]
        result = expand_preset(argv, presets=PRESETS)
        assert "--grouping-skip-keys" in result

    def test_explicit_skip_keys_prevents_default(self):
        """Explicit --grouping-skip-keys prevents default preset injection."""
        argv = ["--grouping-skip-keys", "rank", "--baseline-path", "/a"]
        result = expand_preset(argv, presets=PRESETS)
        assert result == argv

    def test_unknown_preset_raises(self):
        """Unknown preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown value for --preset"):
            expand_preset(["--preset", "nonexistent"], presets=PRESETS)
