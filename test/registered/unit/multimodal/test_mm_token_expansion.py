"""CPU contracts for TokenSpaceProcessStrategy (no model weights).

sources -> host loader + pools -> bare media ----+
        -> outer source configs ----------------+-> member.process_item, one per item
                                                          |
                                         member.merge_media -> BatchFeature
                                                          |
native BatchFeature -> mm_token_expansion_spec             |
partly expanded IDs + boundary -> suffix matcher -> IDs    |
                                                    |     |
                              Base.sglang_post_process <--+
                                        |
                     collect -> full offsets -> serving split

raw media -> media list loader -> bare media, prompt untouched

Loaders retain the original source types and call-level sampling rate.
Serving fixtures separate and terminate media runs with ordinary tokens, matching
the legacy offset scanner contract; its circular-boundary limitation is unchanged.
The matcher preserves history and never scans inserted tokens. Serving offsets
come from the legacy binder without mutating native tensors or metadata.
Loading returns bare media and respects sample rates; outer configs remain aligned.
Processing uses HF kwargs merging; each item runs as its own task on a cloned
worker, otherwise processing runs inline like the legacy route. Decoders opened by
loading are closed after every item finishes, even when one item fails.
"""

import asyncio
import base64
import concurrent.futures
import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import BatchFeature, ProcessorMixin

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.multimodal.media_processor import get_media_source_configs
from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.multimodal.token_space.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.utils.common import ImageData, VideoData
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _process_media(processor, **kwargs):
    return asyncio.run(processor.process_media_async(**kwargs))


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

    def test_expansion_suffix_uses_trailing_media_and_preserves_prefix(self):
        # One modality's expanded fragment can contain another modality's marker.
        history = [71, 99, 99, 72, 98, 99, 98, 73]
        historical_expansion_spec = [([99], [[99, 99]]), ([98], [[98, 99, 98]])]
        mm_token_expansion_spec = [
            ([99], [[99, 99], [99, 99, 99]]),
            ([98], [[98, 99, 98]]),
        ]
        for suffix, expected_suffix in [([99, 74], [99, 99, 99, 74]), ([74], [74])]:
            with self.subTest(suffix=suffix):
                original = history + suffix
                self.assertEqual(
                    expand_token_placeholders(
                        original,
                        mm_token_expansion_spec
                        if 99 in suffix
                        else historical_expansion_spec,
                        mm_token_expansion_start_len=len(history),
                    ),
                    history + expected_suffix,
                )
        with self.assertRaises(ValueError):
            expand_token_placeholders(
                history + [98, 98], mm_token_expansion_spec, len(history)
            )


class _FakeMediaProcessor(ProcessorMixin):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)


class _ClosableVideoDecoder(VideoDecoderWrapper):
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class _TokenExpansionMember(TokenSpaceProcessStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_threads = []

    def process_image(self, image, processor, source_config, **kwargs):
        self.processing_threads.append(threading.get_ident())
        pixels = torch.tensor([[image.width]], dtype=torch.float32)
        return {"pixel_values": pixels, "num_image_tokens": [image.width]}

    def get_mm_token_expansion_spec(self, processor, media_features):
        return [([99], [[99] * count for count in media_features["num_image_tokens"]])]


class _TokenExpansionProcessor(BaseMultimodalProcessor):
    keep_mm_features_on_device = False
    precompute_hash_before_cpu_transfer = False
    use_token_space_processor = True

    def __init__(self):
        self._tokenizer = SimpleNamespace(
            decode=Mock(side_effect=AssertionError("Caller IDs must never decode")),
            encode=Mock(return_value=[10, 99, 11, 99, 11]),
            bos_token="<bos>",
            init_kwargs={},
        )
        self._processor = _FakeMediaProcessor(self._tokenizer)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=99
        )
        self.FEATURE_NAMES = ["pixel_values"]
        self.ATTR_NAME_TO_MODALITY = {"pixel_values": Modality.IMAGE}
        self.image_config = self.video_config = self.audio_config = {}
        self.disable_fast_image_processor = True
        self.mm_processor_executor = None
        self.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.token_space_process_strategy = _TokenExpansionMember(None, self._processor)

    def _finalize_mm_items(self, mm_items, *, images):
        return mm_items

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError

    def _build_mm_output(self, input_ids, media_features, mm_items):
        return MultimodalProcessorOutput(input_ids=input_ids, mm_items=mm_items)


class TestTokenSpaceProcessor(unittest.IsolatedAsyncioTestCase):
    def make_processor(self):
        processor = _TokenExpansionProcessor()
        self.addCleanup(processor.io_executor.shutdown)
        return processor

    async def test_media_list_loader_returns_bare_media(self):
        processor = self.make_processor()
        image_bytes = io.BytesIO()
        Image.new("RGBA", (3, 2), (10, 20, 30, 40)).save(image_bytes, format="PNG")
        image_url = (
            "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()
        )
        video_decoder = object()
        image_options = {"do_resize": False}
        video_options = {"frame_indices": [0, 4]}
        with patch(
            "sglang.srt.multimodal.media_processor.load_video",
            return_value=video_decoder,
        ) as load_video:
            images, videos, audios = await processor._load_media_lists(
                [ImageData(image_url, preprocess_kwargs=image_options)],
                [VideoData("/shared/video.mp4", preprocess_kwargs=video_options)],
                None,
                audio_sample_rate=None,
                discard_alpha_channel=True,
            )
        self.assertEqual(images[0].mode, "RGB")
        self.assertEqual(images[0].getpixel((0, 0)), (10, 20, 30))
        self.assertIs(videos[0], video_decoder)
        self.assertEqual(audios, [])
        load_video.assert_called_once_with(
            VideoData("/shared/video.mp4", preprocess_kwargs=video_options), None
        )
        processor._tokenizer.decode.assert_not_called()

    async def test_loader_resamples_audio_before_processing(self):
        processor = self.make_processor()
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, np.ones(480, dtype=np.float32), 48000, format="WAV")
        _, _, audios = await processor._load_media_lists(
            None,
            None,
            [audio_bytes.getvalue()],
            audio_sample_rate=None,
            discard_alpha_channel=True,
        )
        self.assertEqual(audios[0].shape, (160,))
        processor._tokenizer.decode.assert_not_called()

    async def test_loaded_media_is_reusable_without_token_processing(self):
        processor = self.make_processor()
        video_decoder = object()
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, np.ones(160, dtype=np.float32), 16000, format="WAV")
        image_sources = [
            ImageData(Image.new("RGB", (2, 1)), preprocess_kwargs={"do_resize": False}),
            Image.new("RGB", (3, 1)),
        ]
        video_sources = [
            VideoData("/shared/video.mp4", preprocess_kwargs={"nframes": 2})
        ]
        audio_sources = [audio_bytes.getvalue()]
        with patch(
            "sglang.srt.multimodal.media_processor.load_video",
            return_value=video_decoder,
        ):
            images, videos, audios = await processor._load_media_lists(
                image_sources,
                video_sources,
                audio_sources,
                audio_sample_rate=16000,
                discard_alpha_channel=True,
            )
        video_features = {
            "pixel_values_videos": torch.ones(4, 3),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "video_metadata": [{"fps": 2, "frames_indices": [0, 2]}],
        }
        audio_features = {
            "input_features": torch.ones(1, 4, 8),
            "feature_attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        with (
            patch.object(
                processor.token_space_process_strategy,
                "process_image",
                wraps=processor.token_space_process_strategy.process_image,
            ) as process_image,
            patch.object(
                processor.token_space_process_strategy,
                "process_video",
                return_value=video_features,
            ) as process_video,
            patch.object(
                processor.token_space_process_strategy,
                "process_audio",
                return_value=audio_features,
            ) as process_audio,
            patch.multiple(
                processor.token_space_process_strategy,
                get_mm_token_expansion_spec=Mock(side_effect=AssertionError),
                mm_token_expansion=Mock(side_effect=AssertionError),
            ),
        ):
            features = await processor.process_media_async(
                images=images,
                videos=videos,
                audios=audios,
                image_source_configs=get_media_source_configs(image_sources),
                video_source_configs=get_media_source_configs(video_sources),
                audio_source_configs=[{"sampling_rate": 16000}],
            )
        self.assertEqual(
            [call.args[2] for call in process_image.call_args_list],
            [{"do_resize": False}, {}],
        )
        self.assertEqual(process_video.call_args.args[2], {"nframes": 2})
        self.assertEqual(process_audio.call_args.args[2], {"sampling_rate": 16000})
        self.assertIsInstance(features, BatchFeature)
        self.assertEqual(features["pixel_values"].tolist(), [[2.0], [3.0]])
        self.assertIs(process_video.call_args.args[0], video_decoder)
        self.assertIs(process_audio.call_args.args[0], audios[0])
        for name, value in (video_features | audio_features).items():
            self.assertIs(features[name], value)
        processor._tokenizer.encode.assert_not_called()
        processor._tokenizer.decode.assert_not_called()

        # The serving orchestrator retains the same options outside its loader.
        with (
            patch(
                "sglang.srt.multimodal.media_processor.load_video",
                return_value=video_decoder,
            ),
            patch.object(
                processor, "process_media_async", AsyncMock(return_value=BatchFeature())
            ) as process_media_async,
            patch.object(processor, "_expand_token_space_media"),
        ):
            await processor.process_token_space_mm_data_async(
                input_ids=[99, 99],
                image_data=image_sources,
                request_obj=SimpleNamespace(
                    video_data=video_sources, audio_data=audio_sources
                ),
            )
        forwarded = process_media_async.call_args.kwargs
        self.assertIsInstance(forwarded["images"][0], Image.Image)
        self.assertIs(forwarded["videos"][0], video_decoder)
        self.assertIsInstance(forwarded["audios"][0], np.ndarray)
        for modality, sources in (
            ("image", image_sources),
            ("video", video_sources),
            ("audio", audio_sources),
        ):
            self.assertEqual(
                forwarded[f"{modality}_source_configs"],
                get_media_source_configs(sources),
            )

    def test_source_configs_must_match_media_count(self):
        processor = self.make_processor()
        for modality in ("image", "video", "audio"):
            for configs in ([], [{}, {}]):
                with self.subTest(modality=modality, configs=configs):
                    with self.assertRaisesRegex(
                        ValueError, "Source configs must align"
                    ):
                        _process_media(
                            processor,
                            **{
                                f"{modality}s": [object()],
                                f"{modality}_source_configs": configs,
                            },
                        )

    async def test_preprocessed_audio_does_not_resolve_sample_rate(self):
        processor = self.make_processor()
        del processor._processor.feature_extractor
        audio = {"format": "processor_output", "input_features": [1.0]}
        _, _, audios = await processor._load_media_lists(
            None, None, [audio], audio_sample_rate=None, discard_alpha_channel=True
        )
        self.assertIs(audios[0], audio)
        processor._tokenizer.decode.assert_not_called()

    async def test_async_processing_preserves_history_with_and_without_workers(self):
        # [2-token image, separator | new placeholder] -> two distinct media slots.
        for use_workers in (False, True):
            with self.subTest(use_workers=use_workers):
                processor = self.make_processor()
                if use_workers:
                    processor.mm_processor_executor = MultimodalProcessorExecutor(
                        lambda: processor._processor, 1
                    )
                    self.addCleanup(processor.mm_processor_executor.shutdown)
                result = await processor.process_token_space_mm_data_async(
                    input_ids=[10, 99, 99, 11, 99, 11],
                    image_data=[Image.new("RGB", (width, 1)) for width in (2, 3)],
                    mm_token_expansion_start_len=4,
                )
                self.assertEqual(result.input_ids, [10, 99, 99, 11, 99, 99, 99, 11])
                self.assertEqual(
                    [item.offsets for item in result.mm_items], [[(1, 2)], [(4, 6)]]
                )
                self.assertEqual(
                    [
                        worker != threading.get_ident()
                        for worker in processor.token_space_process_strategy.processing_threads
                    ],
                    [use_workers] * 2,
                )
                processor._tokenizer.decode.assert_not_called()

    async def test_text_and_ids_share_the_same_processing_pipeline(self):
        processor = self.make_processor()
        ids_result = await processor.process_token_space_mm_data_async(
            input_ids=[10, 99, 11, 99, 11],
            image_data=[Image.new("RGB", (width, 1)) for width in (2, 3)],
        )
        text_result = await processor.process_token_space_mm_data_async(
            input_text="<bos> rendered text",
            image_data=[Image.new("RGB", (width, 1)) for width in (2, 3)],
        )
        self.assertEqual(ids_result.input_ids, text_result.input_ids)
        for ids_item, text_item in zip(ids_result.mm_items, text_result.mm_items):
            torch.testing.assert_close(ids_item.feature, text_item.feature)
            self.assertEqual(ids_item.offsets, text_item.offsets)
        processor._tokenizer.encode.assert_called_once_with(
            "<bos> rendered text", add_special_tokens=False
        )
        processor._tokenizer.decode.assert_not_called()

    async def test_async_native_media_does_not_assemble_language_model_inputs(self):
        processor = self.make_processor()
        with (
            patch.multiple(
                processor.token_space_process_strategy,
                get_mm_token_expansion_spec=Mock(side_effect=AssertionError),
                mm_token_expansion=Mock(side_effect=AssertionError),
            ),
            patch.object(processor, "sglang_post_process", side_effect=AssertionError),
        ):
            output = await processor.process_media_async(
                images=[Image.new("RGB", (width, 1)) for width in (2, 3)],
            )
        self.assertIsInstance(output, BatchFeature)
        self.assertEqual(set(output), {"pixel_values", "num_image_tokens"})
        self.assertEqual(output["pixel_values"].tolist(), [[2.0], [3.0]])
        processor._tokenizer.encode.assert_not_called()
        processor._tokenizer.decode.assert_not_called()

    async def test_async_failure_still_closes_every_loaded_decoder(self):
        # [decoder (unsupported video, fails) | image] -> the request fails after
        # both tasks finish, and the decoder opened by loading is closed.
        processor = self.make_processor()
        decoder = _ClosableVideoDecoder()
        with self.assertRaises(NotImplementedError):
            await processor.process_media_async(
                images=[Image.new("RGB", (2, 1))], videos=[decoder]
            )
        self.assertEqual(decoder.close_count, 1)
        self.assertEqual(
            len(processor.token_space_process_strategy.processing_threads), 1
        )

    def test_serving_postprocessing_preserves_reusable_media_and_full_offsets(self):
        processor = self.make_processor()
        processor.token_space_process_strategy.image_config = {"crop": False}
        image_kwargs = {"size": 2}
        media = _process_media(
            processor,
            images=[Image.new("RGB", (width, 1)) for width in (2, 3)],
            images_kwargs=image_kwargs,
        )
        self.assertEqual(image_kwargs, {"size": 2})
        pixels = media["pixel_values"]
        before = pixels.clone()
        for input_ids, boundary in (
            ([99, 11, 99], 0),
            ([99, 99, 11, 99], 3),
            ([99, 99, 11, 99, 99, 99], 6),
        ):
            expanded = processor.token_space_process_strategy.mm_token_expansion(
                input_ids,
                processor.token_space_process_strategy.get_mm_token_expansion_spec(
                    processor._processor, media
                ),
                boundary,
            )
            serving = processor.sglang_post_process(expanded + [11], media)
            self.assertEqual(serving.input_ids, [99, 99, 11, 99, 99, 99, 11])
            self.assertEqual(
                [item.offsets for item in serving.mm_items], [[(0, 1)], [(3, 5)]]
            )
            torch.testing.assert_close(pixels, before)
            self.assertEqual(set(media), {"pixel_values", "num_image_tokens"})


if __name__ == "__main__":
    unittest.main()
