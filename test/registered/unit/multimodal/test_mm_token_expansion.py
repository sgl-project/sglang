"""CPU contracts for token-space expansion and per-item media processing.

expand_token_placeholders(IDs, spec, start_len)
  [P 80 Q P], spec for P, Q -> each replaced once, insertions not rescanned
  [history | P], start_len = len(history) -> only the last expansion used
  start_len < 0, > len(IDs) or not int -> ValueError before matching
  placeholder count != media count -> ValueError

strategy.locate_media(expanded IDs, media, spec) -> tower output spans per media
  [t I I t V V ts V V t I] -> each modality in list order; the video splits at ts
  [I I I] for two images -> one run divided by their counts
  placeholder tokens without media (prefix media not sent) -> ValueError

TokenSpaceProcessor.process(media, IDs)
  tokenizer + strategy, no HF processor -> expanded BatchFeature + mm_spans
  one item of a modality -> still merged by its merge_*_features hook
  image ok + video fails -> request fails, loaded decoder still closed
  processor_output / precomputed_embedding source -> ValueError
  bare source instead of a list -> TypeError
  strategy loading options -> loader class attributes and load kwargs
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.token_space_mm_process.expansion import (
    expand_token_placeholders,
)
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.processor import (
    TokenSpaceProcessor,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMMTokenExpansion(unittest.TestCase):
    def test_preserves_non_media_ids_and_does_not_expand_insertions_again(self):
        original_input_ids = [71, 72, 99, 80, 98, 99, 81]
        expanded_input_ids = expand_token_placeholders(
            original_input_ids,
            {99: [[99, 99], [99]], 98: [[98, 99, 98]]},
        )
        self.assertEqual(expanded_input_ids, [71, 72, 99, 99, 80, 98, 99, 98, 99, 81])
        self.assertEqual(original_input_ids, [71, 72, 99, 80, 98, 99, 81])
        self.assertEqual(
            expand_token_placeholders(original_input_ids, {}), original_input_ids
        )

    def test_suffix_takes_trailing_expansions_and_keeps_the_prefix(self):
        # [history, already expanded | PAD text] -> [history | last expansion text]
        history = [70, 99, 99, 71]
        original = history + [99, 72]
        self.assertEqual(
            expand_token_placeholders(
                original,
                {99: [[99, 99], [88, 99, 77]]},
                mm_token_expansion_start_len=len(history),
            ),
            history + [88, 99, 77, 72],
        )
        self.assertEqual(original, history + [99, 72])

    def test_validates_expansion_boundary_before_matching(self):
        input_ids = [10, 99]
        mm_token_expansion_spec = {99: [[99, 99]]}
        for start in (-1, 3, 0.5):
            with (
                self.subTest(start=start),
                self.assertRaisesRegex(ValueError, "mm_token_expansion_start_len"),
            ):
                expand_token_placeholders(input_ids, mm_token_expansion_spec, start)
        self.assertEqual(
            expand_token_placeholders(
                input_ids, mm_token_expansion_spec, len(input_ids)
            ),
            input_ids,
        )

    def test_rejects_missing_or_surplus_media(self):
        for input_ids, mm_token_expansion_spec in [
            ([], {99: [[99]]}),
            ([99], {99: []}),
            ([99, 99], {99: [[99]]}),
        ]:
            with self.subTest(input_ids=input_ids), self.assertRaises(ValueError):
                expand_token_placeholders(input_ids, mm_token_expansion_spec)


class _ClosableVideoDecoder(VideoDecoderWrapper):
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


IMAGE, VIDEO, TEXT = 3, 4, 8


class _ImageWidthStrategy(TokenSpaceProcessStrategy):
    """A day-0 model: only the required hooks, no HF processor anywhere."""

    models = ("ImageWidthForConditionalGeneration",)
    mm_item_fields = {"pixel_values": Modality.IMAGE}
    mm_feature_fields = frozenset({"pixel_values"})

    def __init__(self, hf_config, processor, **kwargs):
        super().__init__(hf_config, processor, **kwargs)
        self.mm_token_expansion_params = {"image_token_id": IMAGE}

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        pixels = torch.full((image.width, 1), float(image.width))
        return {"pixel_values": pixels, "num_image_tokens": [image.width]}

    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        return {
            IMAGE: [[IMAGE] * count for count in media_features["num_image_tokens"]]
        }


def _bare_tokenizer():
    vocab = {"[UNK]": 0, "a": 1, "b": 2, "<img>": IMAGE}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", additional_special_tokens=["<img>"]
    )


class _ImageVideoStrategy(TokenSpaceProcessStrategy):
    def __init__(self, hf_config, processor, **kwargs):
        super().__init__(hf_config, processor, **kwargs)
        self.mm_token_expansion_params = {
            "image_token_id": IMAGE,
            "video_token_id": VIDEO,
        }


_IMAGES_THEN_VIDEO = [(Modality.IMAGE,), (Modality.IMAGE,), (Modality.VIDEO,)]


class TestLocateMedia(unittest.TestCase):
    def test_each_modality_takes_its_placeholder_tokens_in_list_order(self):
        strategy = _ImageVideoStrategy(None, None)
        spans = strategy.locate_media(
            [0, IMAGE, IMAGE, 9, VIDEO, VIDEO, TEXT, VIDEO, VIDEO, 9, IMAGE],
            _IMAGES_THEN_VIDEO,
            {
                IMAGE: [[IMAGE, IMAGE], [IMAGE]],
                VIDEO: [[VIDEO, VIDEO, TEXT, VIDEO, VIDEO]],
            },
        )
        self.assertEqual(spans, [[(1, 3)], [(10, 11)], [(4, 6), (7, 9)]])
        # Adjacent images form one run; their expansions' counts divide it.
        self.assertEqual(
            strategy.locate_media(
                [IMAGE] * 3, _IMAGES_THEN_VIDEO[:2], {IMAGE: [[IMAGE, IMAGE], [IMAGE]]}
            ),
            [[(0, 2)], [(2, 3)]],
        )

    def test_rejects_placeholder_tokens_the_request_has_no_media_for(self):
        # A partial expansion whose prefix image was not sent: [history IMG | new IMG].
        with self.assertRaisesRegex(ValueError, "every media of its prefix"):
            _ImageVideoStrategy(None, None).locate_media(
                [IMAGE, 9, IMAGE], [(Modality.IMAGE,)], {IMAGE: [[IMAGE]]}
            )


class TestTokenSpaceProcessor(unittest.TestCase):
    def _processor(self, strategy_class=_ImageWidthStrategy):
        processor = TokenSpaceProcessor.for_strategy(strategy_class)(
            None,
            _bare_tokenizer(),
            processor_config=MultimodalProcessorConfig(mm_processor_worker_num=1),
        )
        self.addCleanup(processor.shutdown)
        return processor

    def test_strategy_and_tokenizer_alone_process_a_request(self):
        """A day-0 strategy needs no HF processor: a tokenizer is enough."""
        output = asyncio.run(
            self._processor().process(
                [Image.new("RGB", (2, 1))], input_ids=[1, IMAGE, 2]
            )
        )
        self.assertEqual(output["input_ids"].tolist(), [[1, IMAGE, IMAGE, 2]])
        self.assertEqual(output["pixel_values"].tolist(), [[2.0], [2.0]])
        self.assertEqual(output["mm_modalities"], [Modality.IMAGE])
        self.assertEqual(output["mm_spans"], [[(1, 3)]])

    def test_merge_hook_runs_for_a_single_item(self):
        """A one-image request must still go through merge_image_features."""

        class _TaggedMerge(_ImageWidthStrategy):
            def merge_image_features(self, item_features):
                return {**super().merge_image_features(item_features), "merged": True}

        strategy = _TaggedMerge(None, None)
        for count in (1, 2):
            with self.subTest(count=count):
                merged = strategy.merge_media(
                    [(Modality.IMAGE,)] * count,
                    [{"pixel_values": torch.ones(1, 2)}] * count,
                )
                self.assertTrue(merged["merged"])

    def test_failure_still_closes_every_loaded_decoder(self):
        # [image | decoder (unsupported video, fails)] -> the request fails after
        # both tasks finish, and the decoder opened by loading is closed.
        processor = self._processor()
        decoder = _ClosableVideoDecoder()
        loaded = ([Image.new("RGB", (2, 1))], [decoder], [])
        with (
            patch.object(
                processor, "_load_media_lists", AsyncMock(return_value=loaded)
            ),
            self.assertRaises(NotImplementedError),
        ):
            asyncio.run(processor.process(["image"], ["video"], input_ids=[1]))
        self.assertEqual(decoder.close_count, 1)

    def test_preprocessed_inputs_are_rejected(self):
        # These carry IDs an HF processor already expanded; expanding them again is wrong.
        for media_format in ("processor_output", "precomputed_embedding"):
            with (
                self.subTest(media_format=media_format),
                self.assertRaisesRegex(ValueError, "does not accept"),
            ):
                asyncio.run(
                    self._processor().process(
                        [{"format": media_format}], input_ids=[1, IMAGE]
                    )
                )

    def test_media_sources_must_be_lists(self):
        # A bare source would otherwise be iterated item by item, e.g. a URL per character.
        with self.assertRaisesRegex(TypeError, "must be a list or None"):
            asyncio.run(self._processor().process("image.png", input_ids=[1, IMAGE]))

    def test_strategy_loading_options_reach_the_loader(self):
        class CpuDecodedStrategy(_ImageWidthStrategy):
            processor_class_attributes = {"gpu_image_decode": False}
            audio_sample_rate = 24000
            discard_alpha_channel = False

        self.assertIs(
            TokenSpaceProcessor.for_strategy(CpuDecodedStrategy).gpu_image_decode,
            False,
        )
        self.assertIs(TokenSpaceProcessor.gpu_image_decode, True)

        processor = self._processor(CpuDecodedStrategy)
        load = AsyncMock(return_value=([Image.new("RGB", (1, 1))], [], []))
        with patch.object(processor, "_load_media_lists", load):
            asyncio.run(processor.process(["image"], input_ids=[1, IMAGE]))
        load.assert_awaited_once_with(
            ["image"], None, None, audio_sample_rate=24000, discard_alpha_channel=False
        )


if __name__ == "__main__":
    unittest.main()
