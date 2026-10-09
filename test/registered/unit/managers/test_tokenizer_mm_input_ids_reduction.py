"""SGLANG_MM_STRIP_PROCESSOR_INPUT_IDS omits a duplicate processor ID list.

The tokenizer copies processor-expanded IDs into the canonical request array.
These tests guard that the optional reduction only drops that consumed
duplicate, keeps padded IDs and ownership intact, and survives transport.
"""

import asyncio
import copy
import gc
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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


class TestProcessorInputIdsReduction(CustomTestCase):
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

    def _generate(self, obj, output=None, *, strip=False, processor=None):
        if processor is not None:
            self.manager.mm_processor = processor
        elif output is not None:
            self.manager.mm_processor = _Processor(
                output, self.manager.model_config.hf_config
            )

        async def run():
            return [response async for response in self.manager.generate_request(obj)]

        with envs.SGLANG_MM_STRIP_PROCESSOR_INPUT_IDS.override(strip):
            asyncio.run(run())
        return self.sent[-1]

    def test_processor_expansion_is_canonical_and_output_can_be_reused(self):
        output = _output()
        # A processor may already have expanded its own internal padding IDs.
        output.input_ids = list(output.padded_input_ids)
        original = list(output.input_ids)
        for strip in (False, True):
            with self.subTest(strip=strip):
                tokenized = self._generate(
                    GenerateReqInput(input_ids=[0, 3, 15], image_data=["image"]),
                    output,
                    strip=strip,
                )
                self.assertEqual(list(tokenized.input_ids), original)
                self.assertEqual(tokenized.token_type_ids, [0, 1, 1, 0])
                self.assertEqual(output.input_ids, original)
                self.assertIs(tokenized.mm_inputs.mm_items, output.mm_items)
                self.assertIs(
                    tokenized.mm_inputs.padded_input_ids, output.padded_input_ids
                )
                self.assertEqual(tokenized.mm_inputs.im_token_id, 3)
                if strip:
                    self.assertIsNone(tokenized.mm_inputs.input_ids)
                else:
                    self.assertIs(tokenized.mm_inputs, output)
        output.input_ids.append(2)
        self.assertEqual(list(tokenized.input_ids), original)

    def test_embedding_processor_expansion_is_retained(self):
        self.manager.is_generation = False
        output = _output()
        tokenized = self._generate(
            EmbeddingReqInput(input_ids=[0, 3, 15], image_data=["image"]),
            output,
            strip=True,
        )
        self.assertEqual(list(tokenized.input_ids), output.input_ids)
        self.assertIsNone(tokenized.mm_inputs.input_ids)
        self.assertEqual(tokenized.token_type_ids, [0, 1, 1, 0])

    def test_broadcast_materializes_stripped_output_on_both_ranks(self):
        output = _output()
        tokenized = self._generate(
            GenerateReqInput(input_ids=[0, 3, 15], image_data=["image"]),
            output,
            strip=True,
        )
        transferred = []

        def broadcast(objects, **kwargs):
            if objects[0] is None:
                objects[0] = copy.deepcopy(transferred[0])
            else:
                transferred.append(objects[0])

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.broadcast_object_list", side_effect=broadcast),
        ):
            for rank in (0, 1):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.dp_tp_cpu_group = object()
                scheduler.dp_tp_group = SimpleNamespace(
                    rank_in_group=rank, first_rank=0
                )
                scheduler.model_config = SimpleNamespace(
                    requires_mm_token_modalities=False
                )
                materialized = scheduler._process_and_broadcast_mm_inputs(
                    tokenized.mm_inputs
                )
                self.assertEqual(materialized.padded_input_ids, output.padded_input_ids)
                self.assertEqual(materialized.mm_items[0].offsets, [(1, 2)])
                self.assertEqual(materialized.mm_items[0].hash, 7)
                self.assertEqual(materialized.im_token_id, 3)

    def test_output_without_ids_keeps_the_request_ids(self):
        output = _output()
        output.input_ids = None
        tokenized = self._generate(
            GenerateReqInput(input_ids=[0, 3, 3, 15], image_data=["image"]),
            output,
            strip=True,
        )
        self.assertEqual(list(tokenized.input_ids), [0, 3, 3, 15])
        self.assertIs(tokenized.mm_inputs, output)

    def test_unconsumed_processor_ids_are_not_removed(self):
        request = GenerateReqInput(input_ids=[0, 15])
        self._generate(request)
        output = _output()
        with envs.SGLANG_MM_STRIP_PROCESSOR_INPUT_IDS.override(True):
            tokenized = self.manager._create_tokenized_object(
                request, None, request.input_ids, mm_inputs=output
            )
        self.assertEqual(list(tokenized.input_ids), [0, 15])
        self.assertIs(tokenized.mm_inputs, output)
        self.assertEqual(output.input_ids, [0, 3, 3, 15])

    def test_encoder_output_is_consumed_before_stripping(self):
        output = _output()
        override = get_context().override_server_args(
            language_only=True, encoder_transfer_backend="zmq_to_tokenizer"
        )
        override.install()
        self.addCleanup(override.restore)
        self.manager._handle_epd_disaggregation_encode_request = lambda obj: None
        self.manager.mm_receiver = SimpleNamespace(
            recv_mm_data=AsyncMock(return_value=output)
        )
        tokenized = self._generate(
            GenerateReqInput(input_ids=[0, 3, 15], image_data=["image"]),
            output,
            strip=True,
        )
        self.assertEqual(list(tokenized.input_ids), output.input_ids)
        self.assertIsNone(tokenized.mm_inputs.input_ids)
        self.assertEqual(output.input_ids, [0, 3, 3, 15])

    def test_wire_roundtrip_omits_only_the_consumed_duplicate(self):
        output = _output()
        output.input_ids = [0] * 4092 + output.input_ids
        output.padded_input_ids = [0] * 4092 + output.padded_input_ids
        output.mm_items[0].offsets = [(4093, 4094)]
        output.token_type_ids = torch.zeros(4096, dtype=torch.long)
        payloads = []
        for strip in (False, True):
            tokenized = self._generate(
                GenerateReqInput(input_ids=[0, 3, 15], image_data=["image"]),
                output,
                strip=strip,
            )
            tokenized.time_stats = None
            payloads.append(msgpack_encode(tokenized))
        decoded = msgpack_decode(payloads[1])
        self.assertEqual(list(decoded.input_ids), output.input_ids)
        self.assertIsNone(decoded.mm_inputs.input_ids)
        self.assertEqual(decoded.mm_inputs.padded_input_ids, output.padded_input_ids)
        self.assertEqual(decoded.mm_inputs.mm_items[0].offsets, [(4093, 4094)])
        self.assertLess(len(payloads[1]), len(payloads[0]))

    def test_outputs_with_attached_ownership_are_not_replaced(self):
        for attach in ("state", "finalizer"):
            with self.subTest(attach=attach):
                output = _output()
                released = []
                if attach == "state":
                    output.owner = object()
                else:
                    weakref.finalize(output, released.append, "released")
                tokenized = self._generate(
                    GenerateReqInput(input_ids=[0, 3, 15], image_data=["image"]),
                    output,
                    strip=True,
                )
                self.assertIs(tokenized.mm_inputs, output)
                self.manager.mm_processor = None
                del output
                gc.collect()
                self.assertFalse(released)
                self.assertEqual(list(tokenized.input_ids), [0, 3, 3, 15])


if __name__ == "__main__":
    unittest.main()
