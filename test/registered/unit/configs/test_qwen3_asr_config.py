"""Regression tests for Qwen3ASRConfig registration collision.

Before the fix, ``transformers >= 5.13.0`` registers ``qwen3_asr``
natively, and ``sglang.srt.configs.qwen3_asr``'s bare
``AutoConfig.register("qwen3_asr", Qwen3ASRConfig)`` call raised
``ValueError: 'qwen3_asr' is already used by a Transformers config, pick
another name`` at *import* time, which meant ``sglang.srt.configs`` —
and therefore the whole server — could not be imported at all on any
fresh install with a recent transformers version.
"""

import importlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3ASRConfig(CustomTestCase):
    def test_configs_package_imports(self):
        """Importing the configs package must not crash at module load."""
        import sglang.srt.configs  # noqa: F401

    def test_reimport_is_idempotent(self):
        """Re-importing/reloading qwen3_asr must not raise, even when
        transformers already registers 'qwen3_asr' natively (>=5.13.0)."""
        import sglang.srt.configs.qwen3_asr as qwen3_asr_module

        try:
            importlib.reload(qwen3_asr_module)
        except ValueError as e:
            self.fail(f"Reloading qwen3_asr raised ValueError: {e}")

    def test_qwen3_asr_config_constructs(self):
        """Qwen3ASRConfig must still construct normally after the
        registration guard is added."""
        from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig

        cfg = Qwen3ASRConfig()
        self.assertEqual(cfg.model_type, "qwen3_asr")

    def test_qwen3_asr_thinker_config_constructs(self):
        """Qwen3ASRThinkerConfig must still construct normally after the
        registration guard is added."""
        from sglang.srt.configs.qwen3_asr import Qwen3ASRThinkerConfig

        cfg = Qwen3ASRThinkerConfig()
        self.assertEqual(cfg.model_type, "qwen3_asr_thinker")


if __name__ == "__main__":
    unittest.main()
