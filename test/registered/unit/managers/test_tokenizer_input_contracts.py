"""Request IDs become canonical only after processor expansion and copying.

These tests exercise the tokenizer boundary with ordinary synthetic processor
outputs. They guard vocabulary validation before expansion and the distinct
embedding override, session continuation and batch metadata contracts.
"""

import asyncio
import sys
import unittest
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from transformers import (
    Gemma3Config,
    GlmImageConfig,
    LlavaConfig,
    MistralConfig,
    MllamaConfig,
    Qwen2VLConfig,
)

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.gemma3 import Gemma3SGLangImageProcessor
from sglang.srt.multimodal.processors.llava import (
    LlavaImageProcessor,
    LlavaMultimodalProcessor,
)
from sglang.srt.multimodal.processors.mlama import MllamaImageProcessor
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.parser.inkling_tokenizer import AUDIO_TOKEN_ID, IMAGE_TOKEN_ID
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _output():
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(1, 2)],
        feature=torch.ones(2, 4),
        hash=7,
    )
    item.set_pad_value()
    return MultimodalProcessorOutput(
        mm_items=[item],
        input_ids=[0, 3, 3, 15],
        padded_input_ids=[0, item.pad_value, item.pad_value, 15],
        im_token_id=3,
        token_type_ids=torch.tensor([0, 1, 1, 0]),
    )


class _Processor(BaseMultimodalProcessor):
    prefer_tokenized_input = True

    def __init__(self, output, hf_config):
        self.output = output
        self.hf_config = hf_config
        self.mm_feature_transport = "cpu"
        self.use_cuda_ipc = False
        self.use_ipc_pool_handle_cache = False

    async def process_mm_data_async(self, **kwargs):
        return self.output


class _PreprocessedHFProcessor:
    def __init__(self):
        self.tokenizer = self
        self.image_processor = SimpleNamespace(
            merge_size=1, patch_size=2, temporal_patch_size=1
        )

    def encode(self, text):
        if text == "":
            return []
        raise AssertionError("Preprocessed IDs must not be retokenized")

    def decode(self, *args, **kwargs):
        raise AssertionError("Preprocessed IDs must not be decoded")

    def convert_ids_to_tokens(self, ids):
        tokens = {
            13: "<|image_pad|>",
            14: "<|vision_start|>",
            15: "<|vision_end|>",
            16: "<|video_pad|>",
        }
        return [tokens[token_id] for token_id in ids]

    def __call__(self, *args, **kwargs):
        raise AssertionError("Preprocessed inputs must not run raw media encoding")


class TestTokenizerInputContracts(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            language_only=False,
            language_model_only=False,
            enable_tokenizer_batch_encode=False,
            mm_feature_transport="cpu",
            mm_process_config={},
            speculative_algorithm=None,
        )
        override.install()
        self.addCleanup(override.restore)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.model_config = SimpleNamespace(
            vocab_size=16,
            hf_config=SimpleNamespace(architectures=["SyntheticModel"]),
            hf_text_config=SimpleNamespace(vocab_size=16),
        )
        manager.tokenizer = None
        manager.mm_processor = None
        manager.context_len = 131072
        manager.max_req_input_len = 131071
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = True
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.rid_to_state = {}
        manager.encoder_dispatch_ready = {}
        manager.enable_trace = False
        manager.enable_metrics = False
        manager.enable_priority_scheduling = False
        manager.disaggregation_mode = "null"
        manager.auto_create_handle_loop = lambda: None
        manager.request_logger = Mock()
        manager.is_pause = False
        manager.is_pause_cond = asyncio.Condition()
        manager.model_update_lock = SimpleNamespace(reader_lock=nullcontext())
        self.sent = []

        async def send_one(tokenized):
            self.sent.append(tokenized)

        async def send_batch(tokenized):
            self.sent.extend(tokenized)

        async def wait_one(obj, request):
            yield {"rid": obj.rid}

        manager._send_one_request = send_one
        manager._send_batch_request = send_batch
        manager._wait_one_response = wait_one
        self.manager = manager

    def _generate(self, obj, output=None, *, processor=None):
        if processor is not None:
            self.manager.mm_processor = processor
        elif output is not None:
            self.manager.mm_processor = _Processor(
                output, self.manager.model_config.hf_config
            )

        async def run():
            return [response async for response in self.manager.generate_request(obj)]

        asyncio.run(run())
        return self.sent[-1]

    def test_generation_vocabulary_validation_precedes_request_state(self):
        # A prompt encoded with a different toy vocabulary must fail before
        # batch dispatch creates any live request state.
        for ids in ([0, 16], [[0, 15], [16]], [-1, 2]):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "valid range is"):
                    self._generate(GenerateReqInput(input_ids=ids))
                self.assertFalse(self.sent)
                self.assertFalse(self.manager.rid_to_state)

    def _configured_processor(self, cls, hf_config, output):
        hf_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text: []),
            image_token="<image>",
            image_token_id=getattr(hf_config, "image_token_index", None),
        )
        processor = cls(
            hf_config,
            SimpleNamespace(),
            hf_processor,
            transport_mode=None,
            skip_mm_pool=True,
        )
        owner = (
            processor.inner
            if isinstance(processor, LlavaMultimodalProcessor)
            else processor
        )
        self.addCleanup(owner.shutdown)
        # Image decoding/embedding is an external boundary of the ID contract.
        processor.process_mm_data_async = AsyncMock(return_value=output)
        return processor

    def test_glm_image_prompt_uses_text_vocabulary_and_logprobs_use_output_vocabulary(
        self,
    ):
        hf_config = GlmImageConfig(
            text_config={
                "vocab_size": 32,
                "vision_vocab_size": 16,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 2,
            }
        )
        hf_config.architectures = ["GlmImageForConditionalGeneration"]
        config = ModelConfig.__new__(ModelConfig)
        config.hf_config = hf_config
        config.hf_text_config = hf_config.text_config
        config._derive_model_shapes()
        self.manager.model_config = config
        tokenized = self._generate(GenerateReqInput(input_ids=[0, 20, 31]))
        self.assertEqual(list(tokenized.input_ids), [0, 20, 31])
        self.assertEqual(config.vocab_size, 16)
        with self.assertRaisesRegex(ValueError, "token_ids_logprob"):
            self._generate(GenerateReqInput(input_ids=[0, 20], token_ids_logprob=[20]))

    def test_configured_image_markers_reach_the_processor(self):
        for cls, hf_config in (
            (
                Gemma3SGLangImageProcessor,
                Gemma3Config(
                    text_config={"vocab_size": 16},
                    image_token_index=16,
                    boi_token_index=14,
                    eoi_token_index=15,
                ),
            ),
            (
                MllamaImageProcessor,
                MllamaConfig(
                    text_config={
                        "vocab_size": 16,
                        "bos_token_id": 0,
                        "eos_token_id": 1,
                        "pad_token_id": 2,
                    },
                    image_token_index=16,
                ),
            ),
            (
                LlavaMultimodalProcessor,
                LlavaConfig(text_config={"vocab_size": 16}, image_token_index=16),
            ),
        ):
            with self.subTest(processor=cls.__name__):
                output = _output()
                processor = self._configured_processor(cls, hf_config, output)
                tokenized = self._generate(
                    GenerateReqInput(input_ids=[0, 16, 15], image_data=["image"]),
                    processor=processor,
                )
                self.assertEqual(list(tokenized.input_ids), output.input_ids)

    def test_inkling_negative_markers_follow_the_supplied_modality(self):
        hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(decoder_dmodel=4, patch_size=40),
            audio_config=SimpleNamespace(
                decoder_dmodel=4,
                n_mel_bins=4,
                mel_vocab_size=8,
                dmel_min_value=-1.0,
                dmel_max_value=1.0,
            ),
        )
        output = _output()
        # Model registration and media encoders import optional kernels/audio
        # libraries. Run the actual constructor and marker contract with those
        # unused encoding dependencies replaced at their module boundary.
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.models.inkling": SimpleNamespace(
                        InklingForConditionalGeneration=object
                    ),
                    "sglang.srt.multimodal.inkling": SimpleNamespace(
                        InklingAudioFeatureExtractor=Mock(),
                        InklingImageProcessor=Mock(),
                        InklingProcessor=Mock(),
                    ),
                },
            ),
            envs.SGLANG_INKLING_RS_MM_PREPROCESS.override(False),
        ):
            from sglang.srt.multimodal.processors.inkling import (
                InklingMultimodalProcessor,
            )

            processor = self._configured_processor(
                InklingMultimodalProcessor, hf_config, output
            )
            tokenized = self._generate(
                GenerateReqInput(
                    input_ids=[0, IMAGE_TOKEN_ID, AUDIO_TOKEN_ID, 15],
                    image_data=["image"],
                    audio_data=["audio"],
                ),
                processor=processor,
            )
        self.assertEqual(list(tokenized.input_ids), output.input_ids)

    def test_legacy_llava_mistral_uses_the_models_implicit_marker(self):
        hf_config = MistralConfig(architectures=["LlavaMistralForCausalLM"])
        self.manager.model_config = SimpleNamespace(
            vocab_size=hf_config.vocab_size,
            hf_config=hf_config,
            hf_text_config=hf_config,
        )
        output = _output()
        # This processor preserves raw IDs; the scheduler model expands the
        # implicit image marker (32000) into image padding afterward.
        output.input_ids = None
        output.padded_input_ids = None
        processor = self._configured_processor(LlavaImageProcessor, hf_config, output)
        tokenized = self._generate(
            GenerateReqInput(input_ids=[0, 32000, 15], image_data=["image"]),
            processor=processor,
        )
        self.assertEqual(list(tokenized.input_ids), [0, 32000, 15])

    def _qwen_preprocessed_video(self):
        override = get_context().override_server_args(
            language_only=False,
            language_model_only=False,
            mm_process_config={},
            mm_feature_transport="cpu",
            mm_preprocess_cache_size_mb=0,
            mm_processor_worker_num=1,
            mm_io_worker_num=1,
            skip_tokenizer_init=False,
            speculative_algorithm=None,
        )
        override.install()
        self.addCleanup(override.restore)
        config = Qwen2VLConfig(
            text_config={
                "vocab_size": 16,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 3,
            },
            vision_config={
                "depth": 1,
                "embed_dim": 8,
                "hidden_size": 8,
                "num_heads": 2,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
            image_token_id=13,
            video_token_id=16,
            vision_start_token_id=14,
            vision_end_token_id=15,
        )
        config.architectures = ["Qwen2VLForConditionalGeneration"]
        self.manager.model_config.hf_config = config
        self.manager.model_config.hf_text_config = config.text_config
        processor = QwenVLImageProcessor(
            config,
            SimpleNamespace(),
            _PreprocessedHFProcessor(),
            transport_mode=None,
            skip_mm_pool=True,
        )
        self.addCleanup(processor.shutdown)
        self.manager.mm_processor = processor
        ids = [1, 14, 16, 16, 16, 16, 15, 2]
        payload = {
            "format": "processor_output",
            "input_ids": torch.tensor([ids]),
            "pixel_values_videos": torch.zeros(4, 12),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        return processor, ids, payload

    def test_preprocessed_video_modality_does_not_depend_on_carrier_field(self):
        processor, ids, payload = self._qwen_preprocessed_video()
        for carrier in ("image_data", "video_data"):
            with self.subTest(carrier=carrier):
                tokenized = self._generate(
                    GenerateReqInput(input_ids=ids.copy(), **{carrier: [payload]}),
                    processor=processor,
                )
                self.assertEqual(list(tokenized.input_ids), ids)
                self.assertEqual(
                    tokenized.mm_inputs.mm_items[0].modality, Modality.VIDEO
                )
                self.assertEqual(tokenized.mm_inputs.mm_items[0].offsets, [(2, 5)])
                self.assertEqual(
                    tuple(tokenized.mm_inputs.mrope_positions.shape), (3, 8)
                )
                prepared = MultimodalInputs.from_processor_output(tokenized.mm_inputs)
                self.assertEqual(prepared.mm_items[0].modality, Modality.VIDEO)

    def test_preprocessed_modality_admission_does_not_materialize_metadata(self):
        _, ids, payload = self._qwen_preprocessed_video()
        payload.update(offsets=torch.tensor([[2, 5]]), hash=torch.tensor(7))
        request = GenerateReqInput(input_ids=ids, image_data=[payload])
        request.normalize_batch_and_arguments()
        with (
            patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("metadata copy")
            ),
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("metadata read")
            ),
        ):
            self.manager._validate_generation_input_ids(request)

    def test_preprocessed_modality_isolation_between_batch_items(self):
        processor, ids, payload = self._qwen_preprocessed_video()
        request = GenerateReqInput(
            input_ids=[ids, [1, 16, 2]],
            image_data=[[payload], ["image"]],
        )
        with self.assertRaisesRegex(ValueError, "valid range is"):
            self._generate(request, processor=processor)
        self.assertFalse(self.sent)
        self.assertFalse(self.manager.rid_to_state)

    def test_llava_wrapper_keeps_its_precomputed_image_dictionary_contract(self):
        config = LlavaConfig(text_config={"vocab_size": 16}, image_token_index=16)
        processor = LlavaMultimodalProcessor(
            config,
            SimpleNamespace(),
            _PreprocessedHFProcessor(),
            transport_mode=None,
            skip_mm_pool=True,
        )
        self.addCleanup(processor.inner.shutdown)
        feature = torch.zeros(1, 3, 336, 336)
        tokenized = self._generate(
            GenerateReqInput(
                input_ids=[0, 16, 15],
                image_data=[{"format": "processor_output", "feature": feature}],
            ),
            processor=processor,
        )
        self.assertEqual(list(tokenized.input_ids), [0, 16, 15])
        self.assertIs(tokenized.mm_inputs.mm_items[0].feature, feature)
        self.assertEqual(tokenized.mm_inputs.mm_items[0].modality, Modality.IMAGE)

    def test_media_does_not_allow_undeclared_or_other_batch_item_markers(self):
        output = _output()
        hf_config = Gemma3Config(
            text_config={"vocab_size": 16},
            image_token_index=16,
            boi_token_index=14,
            eoi_token_index=15,
        )
        processor = self._configured_processor(
            Gemma3SGLangImageProcessor, hf_config, output
        )
        for request in (
            GenerateReqInput(input_ids=[0, 17], image_data=["image"]),
            GenerateReqInput(input_ids=[0, 16], audio_data=["audio"]),
            GenerateReqInput(input_ids=[[0, 16], [0, 16]], image_data=[["image"], []]),
        ):
            with self.subTest(input_ids=request.input_ids):
                with self.assertRaisesRegex(ValueError, "valid range is"):
                    self._generate(request, processor=processor)
                self.assertFalse(self.sent)
                self.assertFalse(self.manager.rid_to_state)

        self._generate(
            GenerateReqInput(
                input_ids=[[0, 15], [0, 16, 15]],
                image_data=[[], ["image"]],
            ),
            processor=processor,
        )
        self.assertEqual(
            [list(request.input_ids) for request in self.sent],
            [[0, 15], output.input_ids],
        )

    def test_batch_boundary_tokens_keep_their_order(self):
        self._generate(GenerateReqInput(input_ids=[[0, 15], [15, 0]]))
        self.assertEqual([list(obj.input_ids) for obj in self.sent], [[0, 15], [15, 0]])

    def test_validation_does_not_snapshot_batch_priority_early(self):
        self.manager.enable_priority_scheduling = True
        self.manager.default_priority_value = 5
        self._generate(GenerateReqInput(input_ids=[[0, 15], [15, 0]]))
        self.assertEqual([obj.priority for obj in self.sent], [5, 5])

    def test_processor_expansion_is_canonical_and_output_can_be_reused(self):
        output = _output()
        # A processor may already have expanded its own internal padding IDs.
        output.input_ids = list(output.padded_input_ids)
        original = list(output.input_ids)
        tokenized = self._generate(
            GenerateReqInput(input_ids=[0, 3, 15], image_data=["image"]),
            output,
        )
        self.assertEqual(list(tokenized.input_ids), original)
        self.assertEqual(tokenized.token_type_ids, [0, 1, 1, 0])
        self.assertEqual(output.input_ids, original)
        self.assertIs(tokenized.mm_inputs, output)
        output.input_ids.append(2)
        self.assertEqual(list(tokenized.input_ids), original)

    def test_embedding_override_ids_keep_their_separate_contract(self):
        self.manager.is_generation = False
        tokenized = self._generate(
            EmbeddingReqInput(
                input_ids=[1, -1, 2],
                embed_override_token_id=-1,
                embed_overrides=[torch.ones(4)],
            ),
        )
        self.assertEqual(list(tokenized.input_ids), [1, -1, 2])
        self.assertEqual(tokenized.positional_embed_overrides.positions, [1])

    def test_embedding_processor_expansion_is_retained(self):
        self.manager.is_generation = False
        output = _output()
        tokenized = self._generate(
            EmbeddingReqInput(input_ids=[0, 3, 15], image_data=["image"]),
            output,
        )
        self.assertEqual(list(tokenized.input_ids), output.input_ids)
        self.assertEqual(tokenized.token_type_ids, [0, 1, 1, 0])

    def test_empty_session_continuation_retains_normalization_contract(self):
        tokenized = self._generate(
            GenerateReqInput(input_ids=[], session_params={"id": "existing-session"})
        )
        self.assertEqual(list(tokenized.input_ids), [])
        self.assertEqual(tokenized.session_params.id, "existing-session")

    def test_precomputed_padding_survives_conversion_and_session_prefix(self):
        output = _output()
        tokenized = self._generate(
            GenerateReqInput(input_ids=[0, 3, 15], image_data=["image"]),
            output,
        )
        mm_inputs = MultimodalInputs.from_processor_output(tokenized.mm_inputs)
        req = SimpleNamespace(origin_input_ids=array("q", [7, 8]) + tokenized.input_ids)
        self.assertTrue(
            Scheduler._try_apply_padded_mm_input_ids(tokenized, req, mm_inputs)
        )
        self.assertEqual(list(req.origin_input_ids), [7, 8] + output.padded_input_ids)
        self.assertEqual(list(tokenized.input_ids), output.input_ids)

    def test_public_hash_and_namespace_fields_survive_tokenization(self):
        output = _output()
        content_hash = "sha256:" + "ab" * 32
        request = GenerateReqInput(
            input_ids=[0, 3, 15],
            image_data=["image"],
            mm_hashes=["07"],
            mm_content_hashes=[content_hash],
            cache_salt="workspace-a",
        )
        tokenized = self._generate(request, output)
        self.assertEqual(tokenized.mm_inputs.mm_items[0].hash, 7)
        self.assertEqual(request.mm_content_hashes, [content_hash])
        self.assertEqual(tokenized.cache_salt, "workspace-a")


if __name__ == "__main__":
    unittest.main()
