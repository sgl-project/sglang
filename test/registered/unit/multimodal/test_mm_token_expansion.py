"""CPU contracts for token-space expansion and per-item media processing.

partly expanded IDs + boundary -> suffix matcher -> IDs
loaded media -> process_media_async, one task per item -> BatchFeature

The matcher preserves history and never scans inserted tokens. Source configs
must align with the media, and decoders opened by loading are closed even when
an item fails.
"""

import asyncio
import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image
from transformers import ProcessorMixin

from sglang.srt.multimodal.media_processor import MultimodalProcessorMixin
from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders
from sglang.srt.multimodal.token_space.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMMTokenExpansion(unittest.TestCase):
    def test_preserves_non_media_ids_and_does_not_expand_insertions_again(self):
        original_input_ids = [71, 72, 99, 80, 98, 99, 81]
        expanded_input_ids = expand_token_placeholders(
            original_input_ids,
            [([99], [[99, 99], [99]]), ([98], [[98, 99, 98]])],
        )
        self.assertEqual(expanded_input_ids, [71, 72, 99, 99, 80, 98, 99, 98, 99, 81])
        self.assertEqual(original_input_ids, [71, 72, 99, 80, 98, 99, 81])
        self.assertEqual(
            expand_token_placeholders(original_input_ids, []), original_input_ids
        )

    def test_matches_whole_patterns_and_preserves_untouched_prefix(self):
        # [history | START PAD PAD END text] -> [history | fragment text insertion]
        # The text anchor is retained in its replacement; callers provide no positions.
        for history in ([], [70, 99, 99, 71]):
            with self.subTest(history=history):
                original = history + [80, 99, 99, 81, 72]
                media_fragments = ([[99, 99]] if history else []) + [[88, 99, 77]]
                self.assertEqual(
                    expand_token_placeholders(
                        original,
                        [
                            ([80, 99, 99, 81], media_fragments),
                            ([72], [[72, 98, 99, 98]]),
                        ],
                        mm_token_expansion_start_len=len(history),
                    ),
                    history + [88, 99, 77, 72, 98, 99, 98],
                )
                self.assertEqual(original, history + [80, 99, 99, 81, 72])

    def test_adjacent_patterns_follow_rule_order_without_overlapping(self):
        # [START PAD][START PAD][PAD] -> [fragment 0][fragment 1][bare PAD fragment]
        # Inner PADs and lower-priority START rules must not consume fragments.
        self.assertEqual(
            expand_token_placeholders(
                [80, 99, 80, 99, 99],
                [([80, 99], [[88], [89]]), ([80], []), ([99], [[77]])],
            ),
            [88, 89, 77],
        )
        # A bare START after a complete match can still use the lower-priority rule.
        self.assertEqual(
            expand_token_placeholders(
                [80, 99, 80],
                [([80, 99], [[88]]), ([80], [[77]])],
            ),
            [88, 77],
        )
        with self.assertRaisesRegex(ValueError, "patterns must not be empty"):
            expand_token_placeholders([80], [([], [[99]])])

    def test_validates_expansion_boundary_before_matching(self):
        input_ids = [10, 99]
        mm_token_expansion_spec = [([99], [[99, 99]])]
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
            ([], [([99], [[99]])]),
            ([99], [([99], [])]),
            ([80, 99, 81], [([80, 99, 81], [])]),
            ([80, 99], [([80, 99, 81], [[99]])]),
        ]:
            with self.subTest(input_ids=input_ids), self.assertRaises(ValueError):
                expand_token_placeholders(input_ids, mm_token_expansion_spec)


class _FakeMediaProcessor(ProcessorMixin):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _ClosableVideoDecoder(VideoDecoderWrapper):
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class _ImageWidthStrategy(TokenSpaceProcessStrategy):
    def process_image(self, image, processor, source_config, **kwargs):
        pixels = torch.tensor([[image.width]], dtype=torch.float32)
        return {"pixel_values": pixels, "num_image_tokens": [image.width]}

    def get_mm_token_expansion_spec(self, processor, media_features):
        return [([99], [[99] * count for count in media_features["num_image_tokens"]])]


def _init_media_host(host):
    host._tokenizer = SimpleNamespace(
        decode=Mock(side_effect=AssertionError("Caller IDs must never decode")),
        init_kwargs={},
    )
    host._processor = _FakeMediaProcessor(host._tokenizer)
    host.FEATURE_NAMES = ["pixel_values"]
    host.image_config = host.video_config = host.audio_config = {}
    host.disable_fast_image_processor = True
    host.mm_processor_executor = None
    host.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    host.token_space_process_strategy = _ImageWidthStrategy(None, host._processor)


class _MediaHost(MultimodalProcessorMixin):
    def __init__(self):
        _init_media_host(self)


class TestProcessMedia(unittest.TestCase):
    def make_host(self):
        host = _MediaHost()
        self.addCleanup(host.io_executor.shutdown)
        return host

    def test_source_configs_must_match_media_count(self):
        host = self.make_host()
        for modality in ("image", "video", "audio"):
            for configs in ([], [{}, {}]):
                with self.subTest(modality=modality, configs=configs):
                    with self.assertRaisesRegex(
                        ValueError, "Source configs must align"
                    ):
                        asyncio.run(
                            host.process_media_async(
                                **{
                                    f"{modality}s": [object()],
                                    f"{modality}_source_configs": configs,
                                }
                            )
                        )

    def test_failure_still_closes_every_loaded_decoder(self):
        # [image | decoder (unsupported video, fails)] -> the request fails after
        # both tasks finish, and the decoder opened by loading is closed.
        host = self.make_host()
        decoder = _ClosableVideoDecoder()
        with self.assertRaises(NotImplementedError):
            asyncio.run(
                host.process_media_async(
                    images=[Image.new("RGB", (2, 1))], videos=[decoder]
                )
            )
        self.assertEqual(decoder.close_count, 1)


if __name__ == "__main__":
    unittest.main()
