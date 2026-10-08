"""CPU parity tests for the shared Qwen media pipeline; no model weights.

    load -> images / video decoder or frames / waveform
                            |
    Qwen token-space processor -> strategy, one item at a time -> assembled native fields
                                               |
    original IDs -> mm_token_expansion <--- pure pattern -> token sequence specs
                         |
             final IDs + native fields -> collect -> offsets -> serving split
                                               |
                                      hashes -> feature transport
                                               |
                    Qwen output builder -> deferred CUDA IPC marking + MRoPE

    startup: token-space processor if registered and supported, else legacy
    legacy / startup-selected token-space processor -> scheduler grid items -> MRoPE

Real HF processors provide feature/token references without model weights.
Partial expansion takes trailing media, preserves history and never decodes IDs.
The Qwen2 reader retains legacy resize, sampling and reader closure.
Each item is processed alone with its own source options: [A, B, C] -> A | B | C.
Model assembly keeps order, per-item resize, audio padding and shared tensor views;
an audio's valid frames match a one-audio HF call.
Serving binds offsets and splits items with the legacy binder.
Bare-placeholder serving fixtures end with EOS and stub mRoPE computation; complete
prompts exercise MRoPE without changing the legacy circular offset scanner.
Serving excludes HF VideoMetadata.
Omni startup preserves the outer model config when serving unwraps thinker_config.
"""

import asyncio
import re
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    TOKEN_SPACE_PROCESSOR_MAPPING,
    get_mm_processor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
)
from sglang.srt.multimodal.processors.qwen_vl_token_space import (
    QwenVLMultimodalProcessorOutputBuilder,
    QwenVLTokenSpaceProcessor,
)
from sglang.srt.multimodal.token_space.qwen_vl import QwenTokenSpaceProcessStrategy
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


def _process_media(processor, **kwargs):
    return asyncio.run(processor.process_media_async(**kwargs))


def _post_process(processor, input_ids, media):
    ret = processor.token_space_process_strategy.build_processor_output(
        media, input_ids, processor=processor._processor
    )
    return processor.sglang_post_process(ret)


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


def _make_processor(model_type="qwen3_vl"):
    supported = model_type in QwenTokenSpaceProcessStrategy._SUPPORTED_MODEL_TYPES
    processor = object.__new__(
        QwenVLTokenSpaceProcessor if supported else QwenVLImageProcessor
    )
    processor.model_type = model_type
    processor.use_token_space_processor = supported
    processor._tokenizer = SimpleNamespace(
        eos_token="<|im_end|>",
        eos_token_id=EOS,
        convert_ids_to_tokens=lambda ids: [TOKEN_TEXT[i] for i in ids],
        decode=Mock(side_effect=AssertionError("Caller IDs must not be decoded")),
    )
    processor._processor = SimpleNamespace(tokenizer=processor._tokenizer)
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=IMAGE, video_token_id=VIDEO, audio_token_id=AUDIO
    )
    processor.use_cuda_ipc = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.mm_feature_transport = "cpu"
    processor.FEATURE_NAMES = ["pixel_values", "pixel_values_videos", "input_features"]
    processor.ATTR_NAME_TO_MODALITY = {
        "pixel_values": Modality.IMAGE,
        "image_grid_thw": Modality.IMAGE,
        "pixel_values_videos": Modality.VIDEO,
        "video_grid_thw": Modality.VIDEO,
        "input_features": Modality.AUDIO,
        "feature_attention_mask": Modality.AUDIO,
    }
    processor.hf_config = SimpleNamespace(
        model_type=processor.model_type,
        vision_start_token_id=START,
        vision_end_token_id=END,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        audio_token_id=AUDIO,
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=25),
    )
    processor._spatial_merge_size = 2
    processor._tokens_per_second = 25
    processor.vision_start_token_id = START
    processor.vision_end_token_id = END
    processor.audio_start_token_id = None
    processor.mm_processor_executor = None
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {}
    processor.disable_fast_image_processor = True
    processor._tokenizer_auto_adds_specials = False
    processor.IM_TOKEN_ID = processor.image_token_id = IMAGE
    processor.VIDEO_TOKEN_ID = processor.video_token_id = VIDEO
    processor.audio_token_id = AUDIO
    processor.IM_START_TOKEN_ID = START
    processor.IM_END_TOKEN_ID = END
    component_config = processor.hf_config
    if model_type == "qwen3_omni_moe":
        component_config = SimpleNamespace(
            model_type=model_type, thinker_config=processor.hf_config
        )
    processor.output_builder = QwenVLMultimodalProcessorOutputBuilder(component_config)
    processor.token_space_process_strategy = (
        QwenTokenSpaceProcessStrategy(component_config, processor._processor)
        if supported
        else None
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
    return processor


class TestQwenTokenizedMedia(CustomTestCase):
    def setUp(self):
        # Serving hooks read the published runtime config.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_omni_member_keeps_model_config_after_serving_unwraps_thinker(self):
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
            TOKEN_SPACE_PROCESSOR_MAPPING,
            {
                model: QwenVLTokenSpaceProcessor
                for model in QwenVLTokenSpaceProcessor.models
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
            processor.token_space_process_strategy.get_mm_token_expansion_spec(
                processor._processor, actual
            ),
            fixture.token_space_process_strategy.get_mm_token_expansion_spec(
                fixture._processor, expected
            ),
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
            vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=25),
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
        self.assertFalse(legacy.use_token_space_processor)
        with (
            patch.dict(
                PROCESSOR_MAPPING,
                {Qwen3VLForConditionalGeneration: QwenVLImageProcessor},
                clear=True,
            ),
            patch.dict(
                TOKEN_SPACE_PROCESSOR_MAPPING,
                {Qwen3VLForConditionalGeneration: QwenVLTokenSpaceProcessor},
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
        self.assertIs(
            type(native.token_space_process_strategy), QwenTokenSpaceProcessStrategy
        )
        # The token-space processor takes the request's IDs when it has them.
        self.assertTrue(native.prefer_tokenized_input)
        self.assertEqual(
            (native.auto_mm_processor_worker_num, native.auto_mm_io_worker_num),
            (legacy.auto_mm_processor_worker_num, legacy.auto_mm_io_worker_num),
        )
        # An unsupported model type, or no registered token-space processor, keeps legacy.
        for model_type, token_space_mapping in (
            (
                "intern_s2_preview",
                {Qwen3VLForConditionalGeneration: QwenVLTokenSpaceProcessor},
            ),
            ("qwen3_vl", {}),
        ):
            fallback_config = SimpleNamespace(
                **{**vars(config), "model_type": model_type}
            )
            with (
                patch.dict(
                    PROCESSOR_MAPPING,
                    {Qwen3VLForConditionalGeneration: QwenVLImageProcessor},
                    clear=True,
                ),
                patch.dict(
                    TOKEN_SPACE_PROCESSOR_MAPPING, token_space_mapping, clear=True
                ),
            ):
                fallback = get_mm_processor(
                    fallback_config, server_args, hf_processor, None
                )
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
                    batch = asyncio.run(
                        processor.process_token_space_async(
                            images=images, input_ids=[START, IMAGE, END] * 2
                        )
                    )
                self.assertEqual(set(batch), set(reference))
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
                    expanded = processor.token_space_process_strategy.mm_token_expansion(
                        original,
                        processor.token_space_process_strategy.get_mm_token_expansion_spec(
                            processor._processor, media
                        ),
                    )
                    serving = _post_process(processor, expanded, media)
                    first_end = serving.mm_items[0].offsets[0][1] + 2
                    history = expanded[:first_end]
                    partial = processor.token_space_process_strategy.mm_token_expansion(
                        history + [START, IMAGE, END],
                        processor.token_space_process_strategy.get_mm_token_expansion_spec(
                            processor._processor, media
                        ),
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
                processor.video_config = (
                    processor.token_space_process_strategy.video_config
                ) = {"do_sample_frames": False}
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
                    expanded = processor.token_space_process_strategy.mm_token_expansion(
                        [VIDEO, EOS, VIDEO],
                        processor.token_space_process_strategy.get_mm_token_expansion_spec(
                            processor._processor, media
                        ),
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
                    processor.output_builder,
                    "_compute_mrope_positions",
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

    def test_video_source_options_apply_per_item_and_keep_serving_views(self):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        strategy = processor.token_space_process_strategy
        videos = [
            torch.full((frames, 3, 64, 64), value, dtype=torch.uint8)
            for frames, value in ((2, 31), (4, 127), (2, 223))
        ]
        references = [
            strategy.process_item(
                Modality.VIDEO,
                video,
                processor._processor,
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
            video_source_configs=[
                {"do_normalize": value} for value in (True, False, True)
            ],
            do_resize=False,
            do_sample_frames=False,
        )
        for name in ("pixel_values_videos", "video_grid_thw"):
            self.assert_tensor_bytes_equal(
                media_features[name],
                torch.cat([source[name] for source in references]),
            )
        media = media_features
        expanded = processor.token_space_process_strategy.mm_token_expansion(
            [VIDEO, EOS, VIDEO, EOS, VIDEO],
            processor.token_space_process_strategy.get_mm_token_expansion_spec(
                processor._processor, media
            ),
        )
        with patch.object(
            processor.output_builder,
            "_compute_mrope_positions",
            return_value=(None, None),
        ):
            serving = _post_process(processor, expanded, media)
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
        processor.audio_config = processor.token_space_process_strategy.audio_config = {
            "return_attention_mask": True,
            "truncation": False,
        }
        audios = [np.zeros(3200, dtype=np.float32), np.zeros(19200, dtype=np.float32)]
        reference = processor._processor(
            text="<|audio_pad|><|im_end|><|audio_pad|>",
            audio=audios,
            add_special_tokens=False,
            return_tensors="pt",
            **processor.audio_config,
        )
        with _ban_text_processing(processor):
            media = _process_media(processor, audios=audios)
            expanded = processor.token_space_process_strategy.mm_token_expansion(
                [AUDIO, EOS, AUDIO],
                processor.token_space_process_strategy.get_mm_token_expansion_spec(
                    processor._processor, media
                ),
            )
        self.assertEqual(expanded, reference["input_ids"][0].tolist())
        # Each audio is extracted alone, so its valid frames match a one-audio HF call.
        for index, audio in enumerate(audios):
            single = processor._processor(
                text="<|audio_pad|>",
                audio=[audio],
                add_special_tokens=False,
                return_tensors="pt",
                **processor.audio_config,
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
        processor.video_config = processor.token_space_process_strategy.video_config = {
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

    def test_omni_audio_in_video_is_not_implicitly_enabled(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        with self.assertRaisesRegex(ValueError, "use_audio_in_video"):
            _process_media(processor, videos=[object()], use_audio_in_video=True)

    def test_audio_source_options_keep_truncation_and_batch_padding(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        audio = np.zeros(3200, dtype=np.float32)
        audio_options = {"truncation": True, "max_length": 1600}
        first = _process_media(processor, audios=[audio], **audio_options)
        third_audio = np.full(4800, 0.25, dtype=np.float32)
        third = _process_media(processor, audios=[third_audio], **audio_options)
        with patch.object(
            processor.token_space_process_strategy,
            "process_audio",
            wraps=processor.token_space_process_strategy.process_audio,
        ) as process_audio:
            second = _process_media(
                processor,
                audios=[audio, audio, third_audio],
                audio_source_configs=[audio_options, {}, audio_options],
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
        first_replacements = (
            processor.token_space_process_strategy.get_mm_token_expansion_spec(
                processor._processor, first
            )
        )
        mm_token_expansion_spec = (
            processor.token_space_process_strategy.get_mm_token_expansion_spec(
                processor._processor, second
            )
        )
        self.assertEqual(
            first_replacements[-1][1][0], mm_token_expansion_spec[-1][1][0]
        )
        self.assertLess(
            len(mm_token_expansion_spec[-1][1][0]),
            len(mm_token_expansion_spec[-1][1][1]),
        )
        self.assertIsInstance(second["input_features"], torch.Tensor)
        original_length = first["feature_attention_mask"].shape[-1]
        self.assertEqual(
            second["feature_attention_mask"][:1, original_length:]
            .count_nonzero()
            .item(),
            0,
        )
        expanded = processor.token_space_process_strategy.mm_token_expansion(
            [AUDIO, EOS, AUDIO, EOS, AUDIO], mm_token_expansion_spec
        )
        with patch.object(
            processor.output_builder,
            "_compute_mrope_positions",
            return_value=(torch.zeros(3, 1), torch.zeros(1, 1)),
        ) as compute_mrope_positions:
            serving = _post_process(processor, expanded + [EOS], second)
        # mRoPE gets one length per audio, not only the first split item's.
        torch.testing.assert_close(
            compute_mrope_positions.call_args.kwargs["audio_feature_lengths"],
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
