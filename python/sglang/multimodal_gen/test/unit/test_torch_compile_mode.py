import os
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinPipelineConfig,
    Flux2PipelineConfig,
    FluxPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import HunyuanConfig
from sglang.multimodal_gen.configs.pipeline_configs.mova import MOVAPipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_TI2V_5B_Config,
    WanT2V480PConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig


class TestTorchCompileModeDefaults(unittest.TestCase):
    """Verify each pipeline config has the expected torch_compile_mode."""

    def test_base_default_is_max_autotune(self):
        config = PipelineConfig()
        self.assertEqual(config.torch_compile_mode, "max-autotune-no-cudagraphs")

    # Models that should use "default" mode
    def test_flux2_uses_default(self):
        self.assertEqual(Flux2PipelineConfig().torch_compile_mode, "default")

    def test_flux2_klein_inherits_default(self):
        self.assertEqual(Flux2KleinPipelineConfig().torch_compile_mode, "default")

    def test_zimage_uses_default(self):
        self.assertEqual(ZImagePipelineConfig().torch_compile_mode, "default")

    def test_qwen_image_uses_default(self):
        self.assertEqual(QwenImagePipelineConfig().torch_compile_mode, "default")

    def test_qwen_image_edit_inherits_default(self):
        self.assertEqual(QwenImageEditPipelineConfig().torch_compile_mode, "default")

    # Models that should keep max-autotune-no-cudagraphs
    def test_flux1_uses_max_autotune(self):
        self.assertEqual(
            FluxPipelineConfig().torch_compile_mode, "max-autotune-no-cudagraphs"
        )

    def test_wan_uses_max_autotune(self):
        self.assertEqual(
            WanT2V480PConfig().torch_compile_mode, "max-autotune-no-cudagraphs"
        )

    def test_wan_ti2v_inherits_max_autotune(self):
        self.assertEqual(
            Wan2_2_TI2V_5B_Config().torch_compile_mode, "max-autotune-no-cudagraphs"
        )

    def test_mova_uses_max_autotune(self):
        self.assertEqual(
            MOVAPipelineConfig().torch_compile_mode, "max-autotune-no-cudagraphs"
        )

    def test_hunyuan_uses_max_autotune(self):
        self.assertEqual(
            HunyuanConfig().torch_compile_mode, "max-autotune-no-cudagraphs"
        )


class TestTorchCompileModeEnvOverride(unittest.TestCase):
    """Verify SGLANG_TORCH_COMPILE_MODE env var takes precedence over config."""

    def _resolve_mode(self, pipeline_config):
        """Replicate the resolution logic used in stage code."""
        env_mode = os.environ.get("SGLANG_TORCH_COMPILE_MODE")
        return env_mode if env_mode is not None else pipeline_config.torch_compile_mode

    def test_env_var_overrides_config(self):
        config = Flux2PipelineConfig()
        with patch.dict(os.environ, {"SGLANG_TORCH_COMPILE_MODE": "reduce-overhead"}):
            self.assertEqual(self._resolve_mode(config), "reduce-overhead")

    def test_env_var_not_set_uses_config(self):
        config = Flux2PipelineConfig()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SGLANG_TORCH_COMPILE_MODE", None)
            self.assertEqual(self._resolve_mode(config), "default")

    def test_empty_env_var_still_overrides(self):
        """Empty string env var should override, not fall through."""
        config = Flux2PipelineConfig()
        with patch.dict(os.environ, {"SGLANG_TORCH_COMPILE_MODE": ""}):
            self.assertEqual(self._resolve_mode(config), "")


if __name__ == "__main__":
    unittest.main()
