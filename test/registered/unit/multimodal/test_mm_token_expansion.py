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
  CUDA initialized on rank device d -> IO threads set device d; serving keeps cuda:0

get_mm_processor -> TokenSpaceSGLangProcessor
  each strategy keeps its legacy processor's loader and worker defaults,
  except the departures listed in _LEGACY_DEFAULT_DEPARTURES
  tokenizer + strategy -> served output; caller IDs never decoded
  --allowed-media-domains / max file size -> the core's media URL policy
  adjacent images -> two offsets from mm_spans, not one scanned run
  chat turns' media + partially expanded IDs -> prefix kept, new turn expanded
"""

import asyncio
import base64
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import _collect_message_media
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_processor import (
    PROCESSOR_MAPPING,
    get_mm_processor,
    import_processors,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
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
from sglang.srt.multimodal.token_space_mm_process.registry import (
    TOKEN_SPACE_STRATEGY_MAPPING,
)
from sglang.srt.multimodal.token_space_mm_process.sglang import (
    TokenSpaceSGLangProcessor,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import configure_media_url_security
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

    def test_trainer_io_threads_decode_on_the_rank_device(self):
        """A new thread starts on cuda:0, not on the device its rank selected."""
        self.assertIsNone(self._processor().io_executor._initializer)
        with (
            patch("torch.cuda.is_initialized", return_value=True),
            patch("torch.cuda.current_device", return_value=3),
        ):
            initializer = self._processor().io_executor._initializer
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
            serving = TokenSpaceSGLangProcessor.for_strategy(_ImageWidthStrategy)(
                None, server_args, _bare_tokenizer(), None
            )
            self.addCleanup(serving.shutdown)
        self.assertIs(initializer.func, torch.cuda.set_device)
        self.assertEqual(initializer.args, (3,))
        self.assertIsNone(serving._core.io_executor._initializer)

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


# Class attributes the media loader and worker pools read.
_LOADER_ATTRIBUTES = (
    "gpu_image_decode",
    "smart_rgb_conversion",
    "video_preprocessing_device",
    "supports_mm_processor_concurrency",
)


class _StopInit(Exception):
    pass


def _legacy_defaults(processor_cls, hf_config):
    # Stops at the shared base, after a processor has applied its model defaults.
    processor = processor_cls.__new__(processor_cls)
    with (
        patch.object(BaseMultimodalProcessor, "__init__", side_effect=_StopInit),
        contextlib.suppress(_StopInit),
    ):
        processor_cls.__init__(processor, hf_config, None, None)
    defaults = {name: getattr(processor, name) for name in _LOADER_ATTRIBUTES}
    # None and an unset worker default both mean the base's own choice.
    defaults["auto_mm_processor_worker_num"] = (
        processor.auto_mm_processor_worker_num or 2
    )
    defaults["auto_mm_io_worker_num"] = processor.auto_mm_io_worker_num
    return defaults


def _token_space_defaults(strategy_class):
    core_cls = TokenSpaceProcessor.for_strategy(strategy_class)
    defaults = {name: getattr(core_cls, name) for name in _LOADER_ATTRIBUTES}
    # The wrapper's auto count when preprocessing does not compete with the scheduler.
    defaults["auto_mm_processor_worker_num"] = 2
    defaults["auto_mm_io_worker_num"] = core_cls.auto_mm_io_worker_num
    return defaults


# Deliberate departures from the legacy defaults: (model type, attribute) -> value.
_LEGACY_DEFAULT_DEPARTURES = {
    # Legacy Omni kept the base's 4 I/O workers only because it sits outside
    # the Qwen tuning branch; one strategy serves every Qwen model type.
    ("qwen3_omni_moe", "auto_mm_io_worker_num"): 16,
}


# The model types each registered architecture is served with.
_MODEL_TYPES_BY_ARCH = {
    "Qwen2VLForConditionalGeneration": ("qwen2_vl",),
    "Qwen2_5_VLForConditionalGeneration": ("qwen2_5_vl",),
    "Qwen3VLForConditionalGeneration": ("qwen3_vl",),
    "Qwen3VLMoeForConditionalGeneration": ("qwen3_vl_moe",),
    "Qwen3_5ForConditionalGeneration": ("qwen3_5",),
    "Qwen3_5MoeForConditionalGeneration": ("qwen3_5_moe",),
    "Qwen3_5ForCausalLMMTP": ("qwen3_5",),
    "Qwen3OmniMoeForConditionalGeneration": ("qwen3_omni_moe",),
    "Qwen4ExpForConditionalGeneration": ("qwen4_exp",),
}


class TestTokenSpaceSGLangProcessor(unittest.TestCase):
    def test_strategies_keep_the_legacy_processor_defaults(self):
        """A token-space model loads and schedules media like the legacy processor it replaces."""
        import_processors("sglang.srt.multimodal.processors")
        legacy = {
            model_cls.__name__: cls for model_cls, cls in PROCESSOR_MAPPING.items()
        }
        for arch, strategy_class in TOKEN_SPACE_STRATEGY_MAPPING.items():
            if arch not in legacy:
                continue
            for model_type in _MODEL_TYPES_BY_ARCH[arch]:
                hf_config = SimpleNamespace(model_type=model_type)
                hf_config.thinker_config = hf_config
                expected = _legacy_defaults(legacy[arch], hf_config)
                for (departed_type, name), value in _LEGACY_DEFAULT_DEPARTURES.items():
                    if departed_type == model_type:
                        expected[name] = value
                with self.subTest(arch=arch, model_type=model_type):
                    self.assertEqual(_token_space_defaults(strategy_class), expected)

    def test_strategy_alone_serves_a_model_without_an_hf_processor(self):
        """A strategy plus a tokenizer is enough to register, build, and serve."""
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
        config = SimpleNamespace(
            architectures=list(_ImageWidthStrategy.models),
            model_type="image_width",
            to_dict=lambda: {"model_type": "image_width"},
        )
        tokenizer = _bare_tokenizer()
        with patch.dict(
            TOKEN_SPACE_STRATEGY_MAPPING,
            {_ImageWidthStrategy.models[0]: _ImageWidthStrategy},
            clear=True,
        ):
            processor = get_mm_processor(config, server_args, tokenizer, None)
        self.addCleanup(processor.shutdown)
        self.assertIsInstance(processor, TokenSpaceSGLangProcessor)
        with patch.object(
            tokenizer,
            "decode",
            side_effect=AssertionError("Caller IDs must never decode"),
        ):
            output = asyncio.run(
                processor.process_mm_data_async(
                    image_data=[Image.new("RGB", (2, 1))], input_ids=[1, IMAGE, 2]
                )
            )
        self.assertEqual(output.input_ids, [1, IMAGE, IMAGE, 2])
        self.assertEqual([item.offsets for item in output.mm_items], [[(1, 2)]])
        self.assertEqual(output.im_token_id, IMAGE)

    def test_served_core_applies_the_server_media_url_policy(self):
        """Building a core resets the process-wide URL policy to its config's.

        Serving must pass the server's allowlist, or every media domain is allowed.
        """
        reset_context()
        self.addCleanup(reset_context)
        self.addCleanup(configure_media_url_security, [], 64)
        server_args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_process_config={},
            enable_token_space_processor=True,
            mm_processor_worker_num=1,
            allowed_media_domains=["media.example.com"],
            media_url_max_file_size_mb=7,
        )
        publish(server_args, role="test")
        with patch(
            "sglang.srt.multimodal.multimodal_processor_mixin.configure_media_url_security"
        ) as configure:
            processor = TokenSpaceSGLangProcessor.for_strategy(_ImageWidthStrategy)(
                None, server_args, _bare_tokenizer(), None
            )
        self.addCleanup(processor.shutdown)
        configure.assert_called_once_with(["media.example.com"], 7)

    def test_chat_media_reach_a_partially_expanded_request(self):
        """Chat sends every turn's image; only the new turn's placeholder expands."""
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
        processor = TokenSpaceSGLangProcessor.for_strategy(_ImageWidthStrategy)(
            None, server_args, _bare_tokenizer(), None
        )
        self.addCleanup(processor.shutdown)

        def image_part(width):
            buffer = io.BytesIO()
            Image.new("RGB", (width, 1)).save(buffer, format="PNG")
            data = base64.b64encode(buffer.getvalue()).decode()
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"},
            }

        # Turn 1's image (width 2) is already expanded; turn 2's (width 3) is not.
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": [image_part(2)]},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": [image_part(3)]},
            ],
            input_ids=[1, IMAGE, IMAGE, 2, IMAGE],
            mm_token_expansion_start_len=4,
        )
        image_data, _, _, _ = _collect_message_media(request.messages)
        obj = GenerateReqInput(
            input_ids=request.input_ids,
            image_data=image_data,
            mm_token_expansion_start_len=request.mm_token_expansion_start_len,
        )
        output = asyncio.run(
            processor.process_mm_data_async(
                image_data=obj.image_data, input_text=obj.input_ids, request_obj=obj
            )
        )
        self.assertEqual(output.input_ids, [1, IMAGE, IMAGE, 2, IMAGE, IMAGE, IMAGE])
        self.assertEqual([item.offsets for item in output.mm_items], [[(1, 2), (4, 6)]])

    def test_serving_offsets_come_from_mm_spans(self):
        """Adjacent images form one placeholder run; only mm_spans can divide it."""
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
        processor = TokenSpaceSGLangProcessor.for_strategy(_ImageWidthStrategy)(
            None, server_args, _bare_tokenizer(), None
        )
        self.addCleanup(processor.shutdown)
        output = asyncio.run(
            processor.process_mm_data_async(
                image_data=[Image.new("RGB", (2, 1)), Image.new("RGB", (3, 1))],
                input_ids=[1, IMAGE, IMAGE, 2],
            )
        )
        self.assertEqual(output.input_ids, [1] + [IMAGE] * 5 + [2])
        self.assertEqual([item.offsets for item in output.mm_items], [[(1, 2), (3, 5)]])


if __name__ == "__main__":
    unittest.main()
