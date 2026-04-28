import unittest
from unittest.mock import patch

from sglang.srt.models.llava import AutoModel, LlavaForConditionalGeneration
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")


class PixtralVisionConfig:
    pass


class VoxtralRealtimeTextConfig:
    pass


class GoodConfig:
    pass


class PixtralVisionModel:
    pass


class GoodArch:
    pass


class FakeMapping:
    def __init__(self, voxtral_error):
        self.voxtral_error = voxtral_error

    def keys(self):
        return [VoxtralRealtimeTextConfig, PixtralVisionConfig, GoodConfig]

    def get(self, config_cls, default=None):
        if config_cls is VoxtralRealtimeTextConfig:
            raise self.voxtral_error
        if config_cls is PixtralVisionConfig:
            return (PixtralVisionModel,)
        if config_cls is GoodConfig:
            return GoodArch
        return default


KNOWN_VOXTRAL_ERROR = ValueError(
    "Could not find VoxtralRealtimeTextModel neither in "
    "<module 'transformers.models.voxtral_realtime'> nor in "
    "<module 'transformers'>!"
)


class TestLlavaForConditionalGeneration(CustomTestCase):
    def setUp(self):
        LlavaForConditionalGeneration._config_cls_name_to_arch_name_mapping.cache_clear()

    def _build_mapping(self, mapping):
        with patch.object(AutoModel, "_model_mapping", mapping):
            llava_model = object.__new__(LlavaForConditionalGeneration)
            return llava_model._config_cls_name_to_arch_name_mapping(AutoModel)

    @patch("sglang.srt.models.llava.logger.warning")
    def test_skip_known_broken_voxtral_automodel_mapping_entry(self, mock_warning):
        mapping = self._build_mapping(FakeMapping(KNOWN_VOXTRAL_ERROR))

        self.assertEqual(mapping[GoodConfig.__name__], GoodArch.__name__)
        self.assertEqual(
            mapping[PixtralVisionConfig.__name__], (PixtralVisionModel.__name__,)
        )
        self.assertNotIn(VoxtralRealtimeTextConfig.__name__, mapping)

        mock_warning.assert_called_once()
        self.assertEqual(
            mock_warning.call_args.args,
            (
                "Skipping broken %s mapping for config %s: %s",
                AutoModel.__name__,
                VoxtralRealtimeTextConfig.__name__,
                unittest.mock.ANY,
            ),
        )

    def test_other_voxtral_mapping_failures_still_raise(self):
        with self.assertRaisesRegex(ValueError, "some other failure"):
            self._build_mapping(FakeMapping(ValueError("some other failure")))


if __name__ == "__main__":
    unittest.main()
