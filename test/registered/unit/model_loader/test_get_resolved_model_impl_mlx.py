"""Unit tests for MLX handling in get_resolved_model_impl.

The MLX backend loads models through mlx_lm and never routes inference
through sglang/srt/models or the Transformers fallback, so the
Transformers-architecture compatibility gate (which imports remote
auto_map code) must be skipped there. Regression test for the Hunyuan
auto_map crash (#32521): mlx-community/Hunyuan-7B-Instruct-4bit maps
"AutoModel" to a class name its modeling file never defines, and the
gate used to raise before the scheduler finished initializing.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _hunyuan_like_model_config():
    """Stand-in for a Hunyuan ModelConfig (unregistered arch, broken auto_map).

    Mirrors mlx-community/Hunyuan-7B-Instruct-4bit: the architecture is not
    registered in sglang or Transformers, and auto_map's "AutoModel" entry
    names a class ("HunyuanModel") the remote module does not define (the
    real class is "HunYuanModel", capital Y). No network access happens:
    the MLX path must return before the auto_map is ever imported, and the
    non-MLX gate rejects the empty-auto_map variant without imports.
    """
    hf_config = SimpleNamespace(
        architectures=["HunYuanForCausalLM"],
        model_type="hunyuan",
        auto_map={
            "AutoConfig": "configuration_hunyuan.HunyuanConfig",
            "AutoModel": "modeling_hunyuan.HunyuanModel",
            "AutoModelForCausalLM": "modeling_hunyuan.HunYuanForCausalLM",
        },
    )
    return SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,  # text-only model
        is_generation=True,
        is_multimodal=False,
        model_path="mlx-community/Hunyuan-7B-Instruct-4bit",
        revision=None,
        model_impl=ModelImpl.AUTO,
        quantization=None,
        is_embedding_gemma=False,
    )


class TestGetResolvedModelImplMlx(CustomTestCase):
    def test_mlx_skips_transformers_gate_for_unregistered_arch(self):
        model_config = _hunyuan_like_model_config()
        with patch(
            "sglang.srt.model_loader.utils.use_mlx", return_value=True
        ):
            impl = get_resolved_model_impl(model_config)
        self.assertEqual(impl, ModelImpl.SGLANG)
        self.assertNotEqual(impl, ModelImpl.TRANSFORMERS)

    def test_non_mlx_unregistered_arch_still_gated(self):
        model_config = _hunyuan_like_model_config()
        # Without "AutoModel" in auto_map the gate rejects before importing
        # any remote code, keeping this test offline.
        model_config.hf_config.auto_map = {}
        with patch(
            "sglang.srt.model_loader.utils.use_mlx", return_value=False
        ):
            with self.assertRaises(ValueError) as ctx:
                get_resolved_model_impl(model_config)
        self.assertIn("'HunYuanForCausalLM' is not a registered model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
