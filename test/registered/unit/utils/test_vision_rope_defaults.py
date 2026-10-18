import pytest
from transformers import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeVisionConfig,
)

from sglang.srt.utils.hf_transformers_patches import _patch_rope_parameters_validation
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestVisionRopeDefaults:
    @pytest.mark.parametrize(
        "config_cls", [Qwen3_5VisionConfig, Qwen3_5MoeVisionConfig]
    )
    def test_vision_config_keeps_native_axial_default(
        self, config_cls: type[PretrainedConfig]
    ) -> None:
        """Vision configurations resolve axial RoPE without a text context length."""
        config = config_cls()

        assert config.rope_parameters["rope_type"] == "axial"

    def test_unrelated_attribute_error_is_not_suppressed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RoPE compatibility does not hide unrelated configuration errors."""

        def fail(self: PretrainedConfig) -> None:
            raise AttributeError("unrelated_config_field")

        monkeypatch.setattr(PretrainedConfig, "standardize_rope_params", fail)
        _patch_rope_parameters_validation()

        with pytest.raises(AttributeError, match="unrelated_config_field"):
            PretrainedConfig.standardize_rope_params(
                PretrainedConfig.__new__(PretrainedConfig)
            )
