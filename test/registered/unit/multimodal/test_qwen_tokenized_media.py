"""CPU parity tests for the Qwen token-space strategy; real HF processors, no weights.

One row per test: what it pins, then the test.

    helper  _frame_timestamps == HF Qwen3VLProcessor
            -> test_frame_timestamps_match_hf_qwen3_vl_processor
    startup Omni keeps the outer config once serving unwraps the thinker
            -> test_omni_strategy_keeps_model_config_after_serving_unwraps_thinker
            registry picks token-space or legacy; M-RoPE hook matches legacy
            -> test_selected_token_space_processor_keeps_the_scheduler_mrope_hook
    image   BatchFeature == processor(text, images)
            -> test_token_space_batch_feature_matches_hf_processor
            partial expansion, offsets and shared source views
            -> test_real_images_partial_expansion_and_source_views
    video   IDs, grids and seconds per grid == HF
            -> test_real_video_features_timestamps_and_sampling_intervals
            each worker encodes timestamps with its own tokenizer
            -> test_worker_threads_encode_timestamps_with_their_own_tokenizer
            per-source preprocess_kwargs and serving views
            -> test_video_preprocess_kwargs_apply_per_item_and_keep_serving_views
    audio   feature mask and CNN chunk lengths
            -> test_omni_audio_mask_and_cnn_chunk_lengths
    video   the Qwen2 decoder keeps the legacy resize
            -> test_qwen2_video_reader_keeps_legacy_resize
            Omni seconds per grid follow each video's own fps
            -> test_omni_seconds_per_grid_follow_the_item_fps
    audio   use_audio_in_video is rejected
            -> test_omni_audio_in_video_is_not_implicitly_enabled
            per-source truncation, batch padding and M-RoPE lengths
            -> test_audio_preprocess_kwargs_keep_truncation_and_batch_padding
"""

import asyncio
import functools
import re
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from torchvision.transforms.functional import InterpolationMode, resize
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
    Qwen3OmniMoeProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    WhisperFeatureExtractor,
)

from sglang.srt.managers.multimodal_processor import (
    PROCESSOR_MAPPING,
    get_mm_processor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
)
from sglang.srt.multimodal.token_space_mm_process.expansion import (
    expand_token_placeholders,
)
from sglang.srt.multimodal.token_space_mm_process.models.qwen_vl import (
    QwenTokenSpaceProcessStrategy,
    _frame_timestamps,
)
from sglang.srt.multimodal.token_space_mm_process.processor import (
    TokenSpaceProcessor,
)
from sglang.srt.multimodal.token_space_mm_process.registry import (
    TOKEN_SPACE_STRATEGY_MAPPING,
)
from sglang.srt.multimodal.token_space_mm_process.sglang import (
    TokenSpaceSGLangProcessor,
)
from sglang.srt.multimodal.transport.feature_transport import (
    MultimodalFeatureTransport,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE, VIDEO, AUDIO, START, END, EOS = 101, 102, 103, 104, 105, 106
TOKEN_TEXT = {
    IMAGE: "<|image_pad|>",
    VIDEO: "<|video_pad|>",
    AUDIO: "<|audio_pad|>",
    START: "<|vision_start|>",
    END: "<|vision_end|>",
    EOS: "<|im_end|>",
}


QwenTokenSpaceCore = TokenSpaceProcessor.for_strategy(QwenTokenSpaceProcessStrategy)
QwenVLTokenSpaceProcessor = TokenSpaceSGLangProcessor.for_strategy(
    QwenTokenSpaceProcessStrategy
)


def _process_loaded(
    processor,
    *,
    images=None,
    videos=None,
    audios=None,
    preprocess_kwargs=None,
    **kwargs,
):
    # Runs the core on already loaded media; only the loader is stubbed.
    # `preprocess_kwargs` maps "images" / "videos" / "audios" to one dict per source.
    core = processor._core
    loaded = (images or [], videos or [], audios or [])
    per_source = preprocess_kwargs or {}
    sources = [
        (
            [{"preprocess_kwargs": options} for options in per_source[name]]
            if name in per_source
            else ["media"] * len(items) or None
        )
        for name, items in zip(("images", "videos", "audios"), loaded)
    ]
    with patch.object(core, "_load_media_lists", AsyncMock(return_value=loaded)):
        return asyncio.run(core.process(*sources, **kwargs))


def _process_media(processor, **kwargs):
    # The merged media fields, before expansion.
    with patch.object(
        processor._core,
        "_build_output",
        lambda *, media_items, item_features, **_: processor._core.strategy.merge_media(
            media_items, item_features
        ),
    ):
        return _process_loaded(processor, input_ids=[], **kwargs)


def _post_process(processor, input_ids, media):
    # What the core returns for already-expanded IDs, then the serving package.
    strategy = processor._core.strategy
    spec = _expansion_spec(processor, media)
    # One entry per media, in the modality order the core lists them.
    media_items = [
        (modality,)
        for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
        for _ in spec.get(
            strategy.mm_token_expansion_params[f"{modality.name.lower()}_token_id"],
            (),
        )
    ]
    ret = strategy.build_processor_output(
        media, input_ids, processor=processor._core._processor
    )
    ret["mm_modalities"] = [modality for (modality,) in media_items]
    ret["mm_spans"] = strategy.locate_media(input_ids, media_items, spec)
    mrope_positions, mrope_position_delta = strategy.compute_mrope_positions(
        input_ids, ret
    )
    if mrope_positions is not None:
        ret["mrope_positions"] = mrope_positions
        ret["mrope_position_delta"] = mrope_position_delta
    return processor._package_output(ret)


def _expansion_spec(processor, media):
    # Resolved here, so a patched tokenizer encode is the one the spec calls.
    return processor._core.strategy.get_mm_token_expansion_spec(
        media,
        encode_expansion_text=functools.partial(
            processor._core._tokenizer.encode, add_special_tokens=False
        ),
    )


def _attach_core(processor, strategy):
    # The serving wrapper over a core that shares the fixture's processor and tokenizer.
    core = object.__new__(QwenTokenSpaceCore)
    core._processor = processor._processor
    core._tokenizer = processor._tokenizer
    core.strategy = strategy
    core.mm_processor_executor = None
    core.FEATURE_NAMES = sorted(strategy.mm_feature_fields)
    core.keep_mm_features_on_device = False
    core._compute_mrope_positions = True
    processor._core = core


class _VideoDecoder(VideoDecoderWrapper):
    avg_fps = 2.0

    def __init__(self, frames, fps=2.0):
        self.frames = frames
        self.avg_fps = fps
        self.close_count = 0
        self.decode_requests = []

    def __len__(self):
        return len(self.frames)

    def get_frames_as_tensor(self, indices):
        self.decode_requests.append(list(indices))
        return self.frames[indices]

    @property
    def frame_shape(self):
        return tuple(self.frames.shape[1:3])

    def close(self):
        self.close_count += 1


@contextmanager
def _ban_text_processing(processor):
    encode = processor._tokenizer.encode

    def encode_timestamp(text, **kwargs):
        if not re.fullmatch(r"<\d+\.\d seconds>", text):
            raise AssertionError(f"Only new timestamp text may be encoded: {text}")
        return encode(text, **kwargs)

    with (
        patch.object(
            type(processor._processor),
            "__call__",
            side_effect=AssertionError("No full HF processor"),
        ),
        patch.object(
            processor._tokenizer,
            "decode",
            side_effect=AssertionError("No prompt decode"),
        ),
        patch.object(processor._tokenizer, "encode", side_effect=encode_timestamp),
    ):
        yield


def _component_config(processor):
    if processor.model_type == "qwen3_omni_moe":
        return SimpleNamespace(
            model_type=processor.model_type, thinker_config=processor.hf_config
        )
    return processor.hf_config


def _make_processor(model_type="qwen3_vl"):
    processor = object.__new__(QwenVLTokenSpaceProcessor)
    processor.model_type = model_type
    processor._tokenizer = SimpleNamespace(
        eos_token="<|im_end|>",
        eos_token_id=EOS,
        convert_ids_to_tokens=lambda ids: [TOKEN_TEXT[i] for i in ids],
        decode=Mock(side_effect=AssertionError("Caller IDs must not be decoded")),
    )
    processor._processor = SimpleNamespace(tokenizer=processor._tokenizer)
    processor._feature_transport = MultimodalFeatureTransport(
        "cpu", base_gpu_id=0, skip_mm_pool=True
    )
    processor.hf_config = SimpleNamespace(
        model_type=processor.model_type,
        vision_start_token_id=START,
        vision_end_token_id=END,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        audio_token_id=AUDIO,
        vision_config=SimpleNamespace(
            spatial_merge_size=2, temporal_patch_size=2, tokens_per_second=25
        ),
    )
    processor._disable_fast_image_processor = True
    _attach_core(
        processor,
        QwenTokenSpaceProcessStrategy(
            _component_config(processor), processor._processor
        ),
    )
    return processor


def _make_hf_processor(processor_class, patch_size, *, model_type="qwen3_vl"):
    # This tokenizer has a real noncanonical sequence: [D, escribe] decodes to
    # Describe, whose canonical encoding is one token.
    tokens = ["[UNK]", "[PAD]", "Describe", "D", "escribe"] + [
        f"unused_{i}" for i in range(5, EOS + 1)
    ]
    for token_id, text in TOKEN_TEXT.items():
        tokens[token_id] = text
    tokens.extend([f"<{tenth / 10:.1f}" for tenth in range(100)] + ["seconds>"])
    vocab = {text: token_id for token_id, text in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=[
            text for token_id, text in TOKEN_TEXT.items() if token_id != EOS
        ],
    )
    processor = _make_processor(model_type)
    processor._tokenizer = tokenizer
    processor_kwargs = {}
    if processor_class is Qwen3OmniMoeProcessor:
        for attr, value in {
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
            "audio_token": "<|audio_pad|>",
            "vision_bos_token": "<|vision_start|>",
            "vision_eos_token": "<|vision_end|>",
            "audio_bos_token": "<|vision_start|>",
            "audio_eos_token": "<|vision_end|>",
        }.items():
            setattr(tokenizer, attr, value)
        processor_kwargs["feature_extractor"] = WhisperFeatureExtractor(
            feature_size=128
        )
    vision_kwargs = {
        "patch_size": patch_size,
        "size": {
            "shortest_edge": (4 * patch_size) ** 2,
            "longest_edge": (8 * patch_size) ** 2,
        },
    }
    processor._processor = processor_class(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**vision_kwargs),
        video_processor=(
            Qwen3VLVideoProcessor
            if processor_class is Qwen3VLProcessor
            else Qwen2VLVideoProcessor
        )(**vision_kwargs),
        **processor_kwargs,
    )
    # The strategy resolves its tokenizer at construction, so rebuild it on the HF processor.
    _attach_core(
        processor,
        QwenTokenSpaceProcessStrategy(
            _component_config(processor), processor._processor
        ),
    )
    return processor


class TestQwenTokenizedMedia(CustomTestCase):
    def setUp(self):
        # Serving hooks read the published runtime config.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_frame_timestamps_match_hf_qwen3_vl_processor(self):
        """Our copy of Qwen3VLProcessor._calculate_timestamps must not drift from HF."""
        for frame_indices, fps in (
            ([0, 15, 30, 45], 30.0),
            ([0, 12, 24, 36, 48], 29.97),
            ([3], 2.0),
            (np.arange(0, 240, 7), 23.976),
        ):
            for temporal_patch_size in (1, 2, 3):
                with self.subTest(
                    frame_indices=frame_indices,
                    fps=fps,
                    temporal_patch_size=temporal_patch_size,
                ):
                    expected = Qwen3VLProcessor._calculate_timestamps(
                        None, list(frame_indices), fps, temporal_patch_size
                    )
                    actual = _frame_timestamps(
                        frame_indices=frame_indices,
                        fps=fps,
                        temporal_patch_size=temporal_patch_size,
                    )
                    self.assertEqual(actual, expected)

    def test_omni_strategy_keeps_model_config_after_serving_unwraps_thinker(self):
        fixture = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        config = SimpleNamespace(
            model_type="qwen3_omni_moe",
            architectures=["Qwen3OmniMoeForConditionalGeneration"],
            thinker_config=fixture.hf_config,
        )
        reset_context()
        self.addCleanup(reset_context)
        args = ServerArgs(
            model_path="dummy",
            mm_process_config={},
            enable_token_space_processor=True,
            disable_fast_image_processor=True,
            mm_processor_worker_num=1,
        )
        publish(args, role="test")
        with patch.dict(
            TOKEN_SPACE_STRATEGY_MAPPING,
            {
                model: QwenTokenSpaceProcessStrategy
                for model in QwenTokenSpaceProcessStrategy.models
            },
        ):
            processor = get_mm_processor(config, args, fixture._processor, None)
        self.addCleanup(processor.shutdown)
        audio = np.zeros(16000, dtype=np.float32)
        options = dict(return_attention_mask=True, truncation=False)
        actual = _process_media(processor, audios=[audio], **options)
        expected = _process_media(fixture, audios=[audio], **options)
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            self.assert_tensor_bytes_equal(actual[name], expected[name])
        self.assertEqual(
            _expansion_spec(processor, actual),
            _expansion_spec(fixture, expected),
        )

    def test_selected_token_space_processor_keeps_the_scheduler_mrope_hook(self):
        hf_processor = _make_hf_processor(Qwen3VLProcessor, 16)._processor
        config = SimpleNamespace(
            architectures=["Qwen3VLForConditionalGeneration"],
            model_type="qwen3_vl",
            image_token_id=IMAGE,
            video_token_id=VIDEO,
            vision_start_token_id=START,
            vision_end_token_id=END,
            vision_config=SimpleNamespace(
                spatial_merge_size=2, temporal_patch_size=2, tokens_per_second=25
            ),
            to_dict=lambda: {"model_type": "qwen3_vl"},
        )
        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_process_config={},
            enable_token_space_processor=True,
            mm_processor_worker_num=1,
        )
        publish(server_args, role="test")
        legacy = QwenVLImageProcessor(config, server_args, hf_processor, None)
        self.addCleanup(legacy.shutdown)
        with (
            patch.dict(
                PROCESSOR_MAPPING,
                {Qwen3VLForConditionalGeneration: QwenVLImageProcessor},
                clear=True,
            ),
            patch.dict(
                TOKEN_SPACE_STRATEGY_MAPPING,
                {"Qwen3VLForConditionalGeneration": QwenTokenSpaceProcessStrategy},
                clear=True,
            ),
        ):
            native = get_mm_processor(
                config,
                server_args,
                hf_processor,
                None,
            )
        self.addCleanup(native.shutdown)
        self.assertIs(type(native), QwenVLTokenSpaceProcessor)
        self.assertIs(type(native._core.strategy), QwenTokenSpaceProcessStrategy)
        # The token-space processor takes the request's IDs when it has them.
        self.assertTrue(native.prefer_tokenized_input)
        self.assertEqual(
            (native._core.mm_processor_worker_num, native._core.mm_io_worker_num),
            (legacy.mm_processor_worker_num, legacy.mm_io_worker_num),
        )
        # An architecture without a registered strategy keeps its legacy processor.
        with (
            patch.dict(
                PROCESSOR_MAPPING,
                {Qwen3VLForConditionalGeneration: QwenVLImageProcessor},
                clear=True,
            ),
            patch.dict(TOKEN_SPACE_STRATEGY_MAPPING, {}, clear=True),
        ):
            fallback = get_mm_processor(config, server_args, hf_processor, None)
        self.addCleanup(fallback.shutdown)
        self.assertIs(type(fallback), QwenVLImageProcessor)

        scheduler_ids = [START] + [IMAGE] * 4 + [END, START] + [IMAGE] * 8 + [END]
        scheduler_items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                model_specific_data={"image_grid_thw": torch.tensor([grid])},
            )
            for grid in ([1, 4, 4], [1, 4, 8])
        ]
        positions, delta = native.compute_mrope_positions(
            scheduler_ids, scheduler_items
        )
        expected_positions, expected_delta = legacy.compute_mrope_positions(
            scheduler_ids, scheduler_items
        )
        self.assertEqual(positions.shape, (3, len(scheduler_ids)))
        self.assert_tensor_bytes_equal(positions, expected_positions)
        self.assert_tensor_bytes_equal(delta, expected_delta)

    def assert_tensor_bytes_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            actual.contiguous().view(torch.uint8).numpy().tobytes()
            == expected.contiguous().view(torch.uint8).numpy().tobytes(),
            "Tensor bytes differ",
        )

    def test_token_space_batch_feature_matches_hf_processor(self):
        for model_type, hf_class, patch_size in (
            ("qwen2_vl", Qwen2VLProcessor, 14),
            ("qwen2_5_vl", Qwen2_5_VLProcessor, 14),
            ("qwen3_vl", Qwen3VLProcessor, 16),
            ("qwen3_5", Qwen3VLProcessor, 16),
        ):
            with self.subTest(model_type=model_type):
                processor = _make_hf_processor(
                    hf_class, patch_size, model_type=model_type
                )
                images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 128))]
                reference = processor._processor(
                    text="<|vision_start|><|image_pad|><|vision_end|>" * 2,
                    images=images,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                with _ban_text_processing(processor):
                    batch = _process_loaded(
                        processor, images=images, input_ids=[START, IMAGE, END] * 2
                    )
                # Serving also asks the core for M-RoPE, and every core output locates
                # its media, neither of which HF returns; HF's all-ones attention_mask
                # carries nothing for one unpadded request.
                reference.pop("attention_mask")
                self.assertEqual(
                    set(batch),
                    set(reference)
                    | {
                        "mrope_positions",
                        "mrope_position_delta",
                        "mm_modalities",
                        "mm_spans",
                    },
                )
                for key in reference:
                    self.assert_tensor_bytes_equal(batch[key], reference[key])

    def test_real_images_partial_expansion_and_source_views(self):
        for model_type, hf_class, patch_size in (
            ("qwen2_vl", Qwen2VLProcessor, 14),
            ("qwen2_5_vl", Qwen2_5_VLProcessor, 14),
            ("qwen3_vl", Qwen3VLProcessor, 16),
            ("qwen3_5", Qwen3VLProcessor, 16),
            ("qwen3_5_moe", Qwen3VLProcessor, 16),
        ):
            with self.subTest(model_type=model_type):
                processor = _make_hf_processor(
                    hf_class, patch_size, model_type=model_type
                )
                images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 128))]
                hf = processor._processor
                reference = hf(
                    text="<|vision_start|><|image_pad|><|vision_end|>" * 2,
                    images=images,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                original = [3, 4, EOS, EOS, START, IMAGE, END, START, IMAGE, END]
                with _ban_text_processing(processor):
                    media = _process_media(
                        processor, images=[images[0], np.asarray(images[1])]
                    )
                    expanded = expand_token_placeholders(
                        original,
                        _expansion_spec(processor, media),
                    )
                    serving = _post_process(processor, expanded, media)
                    first_end = serving.mm_items[0].offsets[0][1] + 2
                    history = expanded[:first_end]
                    partial = expand_token_placeholders(
                        history + [START, IMAGE, END],
                        _expansion_spec(processor, media),
                        len(history),
                    )
                    partial_serving = _post_process(processor, partial, media)
                self.assertEqual(
                    [expanded],
                    [original[:4] + reference["input_ids"][0].tolist()],
                )
                self.assertEqual(partial, expanded)
                self.assertEqual(
                    [item.offsets for item in partial_serving.mm_items],
                    [item.offsets for item in serving.mm_items],
                )
                torch.testing.assert_close(
                    media["pixel_values"], reference["pixel_values"]
                )
                self.assertEqual(serving.mrope_positions.shape, (3, len(expanded)))
                for item in serving.mm_items:
                    self.assertEqual(
                        item.feature.untyped_storage().data_ptr(),
                        media["pixel_values"].untyped_storage().data_ptr(),
                    )

    def test_real_video_features_timestamps_and_sampling_intervals(self):
        for model_type, hf_class in (
            ("qwen2_vl", Qwen2VLProcessor),
            ("qwen2_5_vl", Qwen2_5_VLProcessor),
            ("qwen3_vl", Qwen3VLProcessor),
            ("qwen3_5", Qwen3VLProcessor),
            ("qwen4_exp", Qwen3VLProcessor),
            ("qwen3_omni_moe", Qwen3OmniMoeProcessor),
        ):
            with self.subTest(model_type=model_type):
                processor = _make_hf_processor(
                    hf_class,
                    16 if hf_class is Qwen3VLProcessor else 14,
                    model_type=model_type,
                )
                processor._core.strategy.video_config = {"do_sample_frames": False}
                videos = [
                    torch.arange(4 * 3 * 112 * 168)
                    .remainder(251)
                    .to(torch.uint8)
                    .reshape(4, 3, 112, 168),
                    torch.ones(2, 3, 64, 128, dtype=torch.uint8),
                ]
                metadata = [
                    {"fps": 8.0, "total_num_frames": 8, "frames_indices": [0, 2, 4, 6]},
                    {"fps": 6.0, "total_num_frames": 6, "frames_indices": [0, 3]},
                ]
                kwargs = {
                    "video_metadata": metadata,
                    "do_sample_frames": False,
                    "return_metadata": True,
                    "fps": 2.0,
                }
                reference = processor._processor(
                    text="<|video_pad|><|im_end|><|video_pad|>",
                    videos=videos,
                    add_special_tokens=False,
                    return_tensors="pt",
                    **kwargs,
                )
                with _ban_text_processing(processor):
                    media = _process_media(processor, videos=videos, **kwargs)
                    expanded = expand_token_placeholders(
                        [VIDEO, EOS, VIDEO],
                        _expansion_spec(processor, media),
                    )
                self.assertEqual(expanded, reference["input_ids"][0].tolist())
                for key in ("pixel_values_videos", "video_grid_thw"):
                    self.assert_tensor_bytes_equal(media[key], reference[key])
                for key in ("second_per_grid_ts", "video_second_per_grid"):
                    if key in reference:
                        torch.testing.assert_close(
                            torch.as_tensor(media[key]), reference[key]
                        )
                with patch.object(
                    processor._core.strategy,
                    "compute_mrope_positions",
                    return_value=(None, None),
                ):
                    serving = _post_process(processor, expanded + [EOS], media)
                if hf_class is Qwen3VLProcessor:
                    self.assertEqual(
                        [len(item.offsets) for item in serving.mm_items], [2, 1]
                    )
                else:
                    self.assertEqual(len(serving.mm_items), 1)
                    self.assertEqual(len(serving.mm_items[0].offsets), 2)
                self.assert_tensor_bytes_equal(
                    torch.cat([item.feature for item in serving.mm_items]),
                    media["pixel_values_videos"],
                )
                self.assert_tensor_bytes_equal(
                    torch.cat([item.video_grid_thw for item in serving.mm_items]),
                    media["video_grid_thw"],
                )
                for item in serving.mm_items:
                    self.assertNotIn("video_metadata", item.model_specific_data)
                    self.assertIsInstance(item.video_grid_thw, torch.Tensor)

    def test_worker_threads_encode_timestamps_with_their_own_tokenizer(self):
        """Worker threads expand video timestamps with their own tokenizer clone.

        The processor's original tokenizer is shared by every thread, and a Rust
        tokenizer raises "Already borrowed" under concurrent mutable use.
        """
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        processor._core.mm_processor_executor = MultimodalProcessorExecutor(
            lambda: processor._processor, max_workers=2
        )
        self.addCleanup(processor._core.mm_processor_executor.shutdown)
        processor._core.strategy.video_config = {"do_sample_frames": False}
        videos = [
            torch.arange(4 * 3 * 112 * 168)
            .remainder(251)
            .to(torch.uint8)
            .reshape(4, 3, 112, 168),
            torch.ones(2, 3, 64, 128, dtype=torch.uint8),
        ]
        kwargs = {
            "video_metadata": [
                {"fps": 8.0, "total_num_frames": 8, "frames_indices": [0, 2, 4, 6]},
                {"fps": 6.0, "total_num_frames": 6, "frames_indices": [0, 3]},
            ],
            "do_sample_frames": False,
            "return_metadata": True,
            "fps": 2.0,
        }
        reference = processor._processor(
            text="<|video_pad|><|im_end|><|video_pad|>",
            videos=videos,
            add_special_tokens=False,
            return_tensors="pt",
            **kwargs,
        )
        original_tokenizer = processor._tokenizer
        encode = PreTrainedTokenizerFast.encode
        encode_calls = []

        def recording_encode(tokenizer, *args, **encode_kwargs):
            encode_calls.append(
                (
                    tokenizer is original_tokenizer,
                    threading.current_thread() is threading.main_thread(),
                )
            )
            return encode(tokenizer, *args, **encode_kwargs)

        with (
            patch.object(PreTrainedTokenizerFast, "encode", recording_encode),
            patch.object(
                processor._core.strategy,
                "compute_mrope_positions",
                return_value=(None, None),
            ),
        ):
            output = _process_loaded(
                processor, videos=videos, input_ids=[VIDEO, EOS, VIDEO], **kwargs
            )
        self.assertEqual(
            output["input_ids"][0].tolist(), reference["input_ids"][0].tolist()
        )
        self.assertTrue(encode_calls)
        self.assertEqual(set(encode_calls), {(False, False)})

    def test_video_preprocess_kwargs_apply_per_item_and_keep_serving_views(self):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        strategy = processor._core.strategy
        videos = [
            torch.full((frames, 3, 64, 64), value, dtype=torch.uint8)
            for frames, value in ((2, 31), (4, 127), (2, 223))
        ]
        references = [
            strategy.process_item(
                Modality.VIDEO,
                video,
                processor._processor,
                {},
                do_resize=False,
                do_sample_frames=False,
                do_normalize=normalize,
                return_tensors="pt",
            )
            for video, normalize in zip(videos, (True, False, True))
        ]
        media_features = _process_media(
            processor,
            videos=videos,
            preprocess_kwargs={
                "videos": [{"do_normalize": value} for value in (True, False, True)]
            },
            do_resize=False,
            do_sample_frames=False,
        )
        for name in ("pixel_values_videos", "video_grid_thw"):
            self.assert_tensor_bytes_equal(
                media_features[name],
                torch.cat([source[name] for source in references]),
            )
        expanded = expand_token_placeholders(
            [VIDEO, EOS, VIDEO, EOS, VIDEO],
            _expansion_spec(processor, media_features),
        )
        with patch.object(
            processor._core.strategy,
            "compute_mrope_positions",
            return_value=(None, None),
        ):
            serving = _post_process(processor, expanded, media_features)
        self.assertEqual([len(item.offsets) for item in serving.mm_items], [1, 2, 1])
        for item, source in zip(serving.mm_items, references):
            self.assert_tensor_bytes_equal(item.feature, source["pixel_values_videos"])
            self.assert_tensor_bytes_equal(
                item.video_grid_thw, source["video_grid_thw"]
            )
            self.assertEqual(
                item.feature.untyped_storage().data_ptr(),
                media_features["pixel_values_videos"].untyped_storage().data_ptr(),
            )

    def test_omni_audio_mask_and_cnn_chunk_lengths(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        processor._core.strategy.audio_config = {
            "return_attention_mask": True,
            "truncation": False,
        }
        audios = [np.zeros(3200, dtype=np.float32), np.zeros(19200, dtype=np.float32)]
        reference = processor._processor(
            text="<|audio_pad|><|im_end|><|audio_pad|>",
            audio=audios,
            add_special_tokens=False,
            return_tensors="pt",
            **processor._core.strategy.audio_config,
        )
        with _ban_text_processing(processor):
            media = _process_media(processor, audios=audios)
            expanded = expand_token_placeholders(
                [AUDIO, EOS, AUDIO],
                _expansion_spec(processor, media),
            )
        self.assertEqual(expanded, reference["input_ids"][0].tolist())
        # Each audio is extracted alone, so its valid frames match a one-audio HF call.
        for index, audio in enumerate(audios):
            single = processor._processor(
                text="<|audio_pad|>",
                audio=[audio],
                add_special_tokens=False,
                return_tensors="pt",
                **processor._core.strategy.audio_config,
            )
            width = single["input_features"].shape[-1]
            self.assert_tensor_bytes_equal(
                media["input_features"][index, :, :width], single["input_features"][0]
            )
            self.assert_tensor_bytes_equal(
                media["feature_attention_mask"][index, :width],
                single["feature_attention_mask"][0],
            )
        self.assertNotIn("attention_mask", media)
        self.assertEqual([expanded], [[AUDIO] * 3 + [EOS] + [AUDIO] * 16])

    @patch("sglang.srt.multimodal.qwen_vl_media_processing.is_cpu", return_value=True)
    def test_qwen2_video_reader_keeps_legacy_resize(self, _is_cpu):
        processor = _make_hf_processor(Qwen2VLProcessor, 14, model_type="qwen2_vl")
        processor._core.strategy.video_config = {
            "nframes": 2,
            "resized_height": 56,
            "resized_width": 56,
        }
        frames = (
            torch.arange(8 * 112 * 112 * 3)
            .remainder(251)
            .to(torch.uint8)
            .reshape(8, 112, 112, 3)
        )
        decoder = _VideoDecoder(frames, fps=8)
        reference = processor._processor.video_processor(
            [
                resize(
                    frames[[0, 7]].permute(0, 3, 1, 2),
                    [56, 56],
                    interpolation=InterpolationMode.BILINEAR,
                )
            ],
            do_sample_frames=False,
            return_tensors="pt",
        )
        output = _process_media(processor, videos=[decoder])
        self.assertEqual(decoder.close_count, 1)
        self.assertEqual(decoder.decode_requests, [[0, 7]])
        self.assertEqual(output["video_metadata"][0].frames_indices.tolist(), [0, 7])
        self.assertEqual(output["video_grid_thw"].tolist(), [[1, 4, 4]])
        for key in ("pixel_values_videos", "video_grid_thw"):
            self.assert_tensor_bytes_equal(output[key], reference[key])

    @patch("sglang.srt.multimodal.qwen_vl_media_processing.is_cpu", return_value=True)
    def test_omni_seconds_per_grid_follow_the_item_fps(self, _is_cpu):
        """A video_url fps sets both the frames sampled and the M-RoPE seconds per grid.

        Two 4 s videos at 8 fps (32 frames), temporal_patch_size 2:
            fps 4 from the item  -> 16 frames, 2 / 4.0 = 0.5 s per grid
            no fps anywhere      ->  8 frames (sampled at 2 fps),
                                     2 / 1.0 = 2.0 s per grid, as HF Omni
        """
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        frames = torch.zeros(32, 64, 64, 3, dtype=torch.uint8)
        decoders = [_VideoDecoder(frames, fps=8), _VideoDecoder(frames, fps=8)]
        output = _process_media(
            processor,
            videos=decoders,
            preprocess_kwargs={"videos": [{"fps": 4.0}, {}]},
        )
        self.assertEqual(
            [len(decoder.decode_requests[0]) for decoder in decoders], [16, 8]
        )
        self.assertEqual(output["video_second_per_grid"].tolist(), [0.5, 2.0])

    def test_omni_audio_in_video_is_not_implicitly_enabled(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        with self.assertRaisesRegex(ValueError, "use_audio_in_video"):
            _process_media(processor, videos=[object()], use_audio_in_video=True)

    def test_audio_preprocess_kwargs_keep_truncation_and_batch_padding(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        audio = np.zeros(3200, dtype=np.float32)
        audio_options = {"truncation": True, "max_length": 1600}
        first = _process_media(processor, audios=[audio], **audio_options)
        third_audio = np.full(4800, 0.25, dtype=np.float32)
        third = _process_media(processor, audios=[third_audio], **audio_options)
        with patch.object(
            processor._core.strategy,
            "process_audio",
            wraps=processor._core.strategy.process_audio,
        ) as process_audio:
            second = _process_media(
                processor,
                audios=[audio, audio, third_audio],
                preprocess_kwargs={"audios": [audio_options, {}, audio_options]},
                truncation=False,
            )
        self.assertEqual(
            [call.args[2] for call in process_audio.call_args_list],
            [audio_options, {}, audio_options],
        )
        self.assert_tensor_bytes_equal(
            third["input_features"],
            second["input_features"][2:3, ..., : third["input_features"].shape[-1]],
        )
        torch.testing.assert_close(
            first["input_features"],
            second["input_features"][:1, ..., : first["input_features"].shape[-1]],
        )
        first_spec = _expansion_spec(processor, first)
        second_spec = _expansion_spec(processor, second)
        self.assertEqual(first_spec[AUDIO][0], second_spec[AUDIO][0])
        self.assertLess(
            len(second_spec[AUDIO][0]),
            len(second_spec[AUDIO][1]),
        )
        self.assertIsInstance(second["input_features"], torch.Tensor)
        original_length = first["feature_attention_mask"].shape[-1]
        self.assertEqual(
            second["feature_attention_mask"][:1, original_length:]
            .count_nonzero()
            .item(),
            0,
        )
        expanded = expand_token_placeholders(
            [AUDIO, EOS, AUDIO, EOS, AUDIO], second_spec
        )
        with patch(
            "sglang.srt.multimodal.token_space_mm_process.models.qwen_vl.get_rope_index",
            return_value=(torch.zeros(3, 1), torch.zeros(1, 1)),
        ) as get_rope_index:
            serving = _post_process(processor, expanded + [EOS], second)
        # mRoPE gets one length per audio, not only the first split item's.
        torch.testing.assert_close(
            get_rope_index.call_args.kwargs["audio_seqlens"],
            second["feature_attention_mask"].sum(dim=1),
        )
        torch.testing.assert_close(
            torch.cat([item.feature for item in serving.mm_items]),
            second["input_features"],
        )
        torch.testing.assert_close(
            torch.cat([item.feature_attention_mask for item in serving.mm_items]),
            second["feature_attention_mask"],
        )
        for item in serving.mm_items:
            self.assertEqual(
                item.feature.untyped_storage().data_ptr(),
                second["input_features"].untyped_storage().data_ptr(),
            )
            self.assertEqual(
                item.feature_attention_mask.untyped_storage().data_ptr(),
                second["feature_attention_mask"].untyped_storage().data_ptr(),
            )


if __name__ == "__main__":
    unittest.main()
