"""Kimi-K3 / K3.1 text-only serving: --language-model-only and --language-only."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.models import kimi_k3
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKimiK3LanguageModelOnly(CustomTestCase):
    def test_kimi_k3_architectures_accept_language_model_only(self):
        # The names Kimi-K3 and Kimi-K3.1-Pro checkpoints declare in config.json.
        for arch in (
            "KimiK3ForConditionalGeneration",
            "KimiK31ForConditionalGeneration",
        ):
            self.assertIn(arch, ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES)

    def _build(self, *, language_only: bool, language_model_only: bool):
        config = SimpleNamespace(
            language_only=language_only,
            language_model_only=language_model_only,
            encoder_only=False,
            vision_config=object(),
            text_config=object(),
        )
        with (
            patch.object(kimi_k3, "KimiK3VisionTower") as tower,
            patch.object(kimi_k3, "KimiK3MultiModalProjector") as projector,
            patch.object(kimi_k3, "KimiK3LinearForCausalLM", return_value=MagicMock()),
        ):
            model = kimi_k3.KimiK3ForConditionalGeneration(config)
        return model, tower, projector

    def test_either_flag_builds_no_vision_modules(self):
        """Text-only serving must not build (or hold in HBM) the vision tower and
        projector, whichever of the two flags asks for it."""
        for flags in (
            dict(language_only=False, language_model_only=True),
            dict(language_only=True, language_model_only=False),
        ):
            with self.subTest(**flags):
                model, tower, projector = self._build(**flags)
                tower.assert_not_called()
                projector.assert_not_called()
                self.assertIsNone(model.vision_tower)
                self.assertIsNone(model.mm_projector)

    def test_multimodal_serving_still_builds_vision_modules(self):
        model, tower, projector = self._build(
            language_only=False, language_model_only=False
        )
        self.assertIs(model.vision_tower, tower.return_value)
        self.assertIs(model.mm_projector, projector.return_value)


if __name__ == "__main__":
    unittest.main()
