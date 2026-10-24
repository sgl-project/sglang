import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import UnidentifiedImageError
from transformers import LlavaConfig

from sglang.srt.models.llava import AutoModel, LlavaForConditionalGeneration
from sglang.srt.multimodal.processors.llava import (
    LlavaImageProcessor,
    LlavaMultimodalProcessor,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


class TestLlavaImageProcessor(CustomTestCase):
    @patch("sglang.srt.multimodal.processors.llava.load_image")
    def test_preprocess_reports_invalid_media_as_client_error(self, mock_load_image):
        media_error = UnidentifiedImageError("invalid image payload")
        mock_load_image.side_effect = media_error

        with self.assertRaisesRegex(
            ValueError, "Error while processing image: invalid image payload"
        ) as raised:
            LlavaImageProcessor._preprocess_image_task(
                b"invalid",
                image_hash=1,
                processor=unittest.mock.Mock(),
            )

        self.assertIs(raised.exception.__cause__, media_error)


class TestLlavaMultimodalProcessor(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_cache_flush_reaches_the_vision_processor(self):
        """Flushes and weight updates clear the preprocess cache of the processor built
        for a Llava architecture; it must not lack the base state and raise instead."""
        server_args = ServerArgs(
            model_path="dummy",
            mm_process_config={},
            mm_preprocess_cache_size_mb=1,
            mm_processor_worker_num=1,
        )
        publish(server_args, role="tokenizer")
        processor = LlavaMultimodalProcessor(
            hf_config=LlavaConfig(),
            server_args=server_args,
            _processor=SimpleNamespace(tokenizer=None),
            transport_mode="default",
        )
        self.addCleanup(processor.shutdown)
        self.assertTrue(processor.mm_preprocess_cache.put("image", b"pixels"))

        processor.clear_preprocess_cache()

        self.assertNotIn("image", processor.mm_preprocess_cache)


if __name__ == "__main__":
    unittest.main()
