"""Regression tests for Cohere2MoeConfig import (sgl-project/sglang#28233).

Before the fix, applying ``huggingface_hub.dataclasses.strict`` to
``Cohere2MoeConfig`` — which is not a stdlib ``@dataclass`` — raised
``StrictDataclassDefinitionError`` at *import* time on
``transformers==5.3.0`` + ``huggingface_hub==1.9.0``, which meant that
``sglang.srt.configs`` could not be imported at all.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCohere2MoeConfig(CustomTestCase):
    def test_configs_package_imports(self):
        """Importing the configs package must not crash at module load."""
        import sglang.srt.configs  # noqa: F401

    def test_derived_defaults_num_key_value_heads(self):
        """``num_key_value_heads`` defaults to ``num_attention_heads``."""
        from sglang.srt.configs.cohere2_moe import Cohere2MoeConfig

        cfg = Cohere2MoeConfig()
        self.assertEqual(cfg.num_key_value_heads, cfg.num_attention_heads)

    def test_derived_defaults_layer_types(self):
        """``layer_types`` is auto-derived to one entry per hidden layer."""
        from sglang.srt.configs.cohere2_moe import Cohere2MoeConfig

        cfg = Cohere2MoeConfig()
        self.assertEqual(len(cfg.layer_types), cfg.num_hidden_layers)

    def test_pretrained_config_kwargs_forwarded(self):
        """Extra kwargs must flow through ``PreTrainedConfig.__init__``."""
        from sglang.srt.configs.cohere2_moe import Cohere2MoeConfig

        cfg = Cohere2MoeConfig(foo="bar")
        cfg_dict = cfg.to_dict()
        self.assertEqual(cfg_dict["model_type"], "cohere2_moe")
        self.assertEqual(cfg_dict["foo"], "bar")

    def test_use_cache_false_preserved(self):
        """``use_cache=False`` must survive ``super().__init__``.

        ``PreTrainedConfig.__init__`` re-assigns ``self.use_cache`` from its
        own keyword; if the subclass does not forward the caller's value,
        ``use_cache=False`` silently reverts to the base default ``True``.
        """
        from sglang.srt.configs.cohere2_moe import Cohere2MoeConfig

        cfg = Cohere2MoeConfig(use_cache=False)
        self.assertFalse(cfg.use_cache)


if __name__ == "__main__":
    unittest.main()
