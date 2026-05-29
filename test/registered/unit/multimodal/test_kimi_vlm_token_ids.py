import unittest
from types import SimpleNamespace

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin

IMG = 163605

_expand_input_ids = BaseMultimodalProcessor._expand_input_ids


def _image(height=4, width=6):
    return SimpleNamespace(height=height, width=width)


class TestExpandInputIds(unittest.TestCase):
    def test_single_image_expands_and_preserves_text_tokens(self):
        rebuilt = _expand_input_ids([11, IMG, 22], [3], IMG)
        self.assertEqual(rebuilt, [11, IMG, IMG, IMG, 22])

    def test_multiple_images(self):
        rebuilt = _expand_input_ids([11, IMG, 22, IMG, 33], [2, 3], IMG)
        self.assertEqual(rebuilt, [11, IMG, IMG, 22, IMG, IMG, IMG, 33])

    def test_count_mismatch_returns_none(self):
        self.assertIsNone(_expand_input_ids([11, IMG, 22], [2, 3], IMG))

    def test_no_image_token_id_returns_none(self):
        self.assertIsNone(_expand_input_ids([11, 22], [3], None))


class TestResolveImageTokenCounts(unittest.TestCase):
    """The resolver is a method; call it unbound with a duck-typed `self`
    that only needs to expose `_processor`."""

    def test_base_default_uses_get_num_multimodal_tokens(self):
        def _get_num_multimodal_tokens(image_sizes):
            return SimpleNamespace(num_image_tokens=[3 for _ in image_sizes])

        fake_self = SimpleNamespace(
            _processor=SimpleNamespace(
                _get_num_multimodal_tokens=_get_num_multimodal_tokens
            )
        )
        counts = BaseMultimodalProcessor.resolve_image_token_counts(
            fake_self, [_image(), _image()]
        )
        self.assertEqual(counts, [3, 3])

    def test_base_returns_none_without_count_api(self):
        fake_self = SimpleNamespace(_processor=SimpleNamespace())
        self.assertIsNone(
            BaseMultimodalProcessor.resolve_image_token_counts(fake_self, [_image()])
        )

    def test_base_returns_none_without_images(self):
        fake_self = SimpleNamespace(_processor=SimpleNamespace())
        self.assertIsNone(
            BaseMultimodalProcessor.resolve_image_token_counts(fake_self, [])
        )

    def test_kimi_override_uses_media_tokens_calculator(self):
        def media_tokens_calculator(media):
            return media["image"].height  # arbitrary per-image count for the test

        fake_self = SimpleNamespace(
            _processor=SimpleNamespace(
                media_processor=SimpleNamespace(
                    media_tokens_calculator=media_tokens_calculator
                )
            )
        )
        counts = KimiGridMMDataMixin.resolve_image_token_counts(
            fake_self, [_image(height=2), _image(height=5)]
        )
        self.assertEqual(counts, [2, 5])

    def test_kimi_override_returns_none_without_calculator(self):
        fake_self = SimpleNamespace(_processor=SimpleNamespace())
        self.assertIsNone(
            KimiGridMMDataMixin.resolve_image_token_counts(fake_self, [_image()])
        )


if __name__ == "__main__":
    unittest.main()
