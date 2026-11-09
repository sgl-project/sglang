"""CPU admission/cleanup for transferred STANDALONE metadata.

The transport and output sink are mocked. Metadata, request/token allocators,
the single-rank transfer queue, KV release and draft factory run unchanged.
"""

import dataclasses
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.utils import (
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
)
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestPdStandaloneDecodeCleanup(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_context().override_server_args(
                speculative_algorithm="STANDALONE",
                speculative_eagle_topk=1,
                disaggregation_decode_enable_radix_cache=False,
            )
        )
        self.enterContext(envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(""))
        self.enterContext(envs.SGLANG_DISAGG_CUSTOMIZED_INFO_MAX_BYTES.override(0))
        self.enterContext(envs.SGLANG_ENABLE_DISAGG_SAMPLING_MASK.override(False))
        self.enterContext(envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(0))
        # No process group or inter-rank integrity guarantee is exercised here.
        self.enterContext(patch("torch.distributed.get_world_size", return_value=1))

    def make_queue(self):
        rows = ReqToTokenPool(3, 8, "cpu", enable_memory_saver=False)
        tokens = TokenToKVPoolAllocator(12, torch.float32, "cpu", None, False)
        cache = RadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=rows,
                token_to_kv_pool_allocator=tokens,
                page_size=1,
            )
        )
        fields = dict.fromkeys(
            field.name for field in dataclasses.fields(SchedulerBatchResultProcessor)
        )
        output_streamer = Mock()
        fields.update(
            model_config=SimpleNamespace(
                vocab_size=11, self_opened_think_start_ids=None
            ),
            tree_cache=cache,
            token_to_kv_pool_allocator=tokens,
            output_streamer=output_streamer,
            disaggregation_mode=DisaggregationMode.DECODE,
            request_retirer=RequestRetirer(
                tree_cache=cache,
                beam_coordinator=None,
                hisparse_coordinator=None,
                uocr_runtime=None,
                send_abort_output=Mock(),
                stream_output=output_streamer.stream_output,
            ),
        )
        processor = SchedulerBatchResultProcessor(**fields)
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = IndexedReqList(
            ListKind.PD_RECEIVING,
            abort=queue._abort_receiving,
            req_of=lambda entry: entry.req,
        )
        queue.scheduler = SimpleNamespace(
            kv_checksum_computer=None,
            model_config=fields["model_config"],
            batch_result_processor=processor,
            request_retirer=fields["request_retirer"],
            output_streamer=fields["output_streamer"],
            enable_decode_hicache=False,
            enable_hisparse=False,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        queue.metadata_buffers = MetadataBuffers(
            size=3,
            hidden_size=4,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=0,
        )
        queue.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(3)
        queue.spec_algorithm = SpeculativeAlgorithm.STANDALONE
        queue.tree_cache = cache
        queue.enable_staging = False
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = object()
        queue.tp_rank = 0
        return queue

    def enqueue(self, queue, rid, room, *, probability=0.75, token=3, max_new=8):
        def req():
            sampling_params = SamplingParams(max_new_tokens=max_new)
            sampling_params.normalize(tokenizer=None)
            return Req(
                rid,
                "",
                array("q", [1, 2]),
                sampling_params,
                bootstrap_host="127.0.0.1",
                bootstrap_room=room,
                vocab_size=11,
            )

        slot = queue.req_to_metadata_buffer_idx_allocator.alloc()
        self.assertIsNotNone(slot)
        prefill = req()
        prefill.metadata_buffer_index = slot
        prefill.output_ids.append(4)
        prefill.output_topk_p = torch.tensor([probability])
        prefill.output_topk_index = torch.tensor([token])
        queue.metadata_buffers.set_buf(prefill)

        decode = req()
        cache = queue.tree_cache
        self.assertIsNotNone(cache.req_to_token_pool.alloc([decode]))
        locations = cache.token_to_kv_pool_allocator.alloc(2)
        self.assertIsNotNone(locations)
        cache.req_to_token_pool.write((decode.kv.req_pool_idx, slice(0, 2)), locations)
        decode.kv.set_row_len(2)
        receiver = Mock(spec=["poll", "clear"])
        receiver.poll.return_value = KVPoll.Success
        entry = DecodeRequest(decode, receiver, metadata_buffer_index=slot)
        queue.add(entry)
        return entry, receiver, locations

    def draft_input(self, queue, reqs):
        return SpeculativeAlgorithm.STANDALONE.build_disagg_draft_input(
            SimpleNamespace(
                reqs=reqs,
                device=torch.device("cpu"),
                model_config=queue.scheduler.model_config,
                enable_overlap=False,
            ),
            torch.tensor([req.output_ids[-1] for req in reqs]),
            None,
        )

    def test_bad_seed_releases_only_its_request_and_allows_slot_reuse(self):
        for probability, token in (
            (0.0, 3),
            (-0.5, 3),
            (float("nan"), 3),
            (float("inf"), 3),
            (0.5, -1),
            (0.5, 11),
        ):
            with self.subTest(probability=probability, token=token):
                queue = self.make_queue()
                bad, bad_receiver, bad_locations = self.enqueue(
                    queue, "bad", 101, probability=probability, token=token
                )
                good, good_receiver, good_locations = self.enqueue(queue, "good", 202)
                cache = queue.tree_cache
                admitted, outcomes = queue.pop_transferred_with_outcomes()
                self.assertEqual(admitted, [good.req])
                self.assertEqual(outcomes, {"bad": True, "good": False})
                self.assertEqual(list(queue.queue.items()), [])
                self.assertIsInstance(bad.req.finished_reason, FINISH_ABORT)
                self.assertEqual(bad.req.finished_reason.status_code, 500)
                self.assertIn("Invalid or unwritten", bad.req.finished_reason.message)
                self.assertEqual(list(bad.req.output_ids), [])
                self.assertIsNone(bad.req.output_topk_p)
                self.assertFalse(bad.req.kv.holds_kv)
                self.assertTrue(bad.req.kv.is_kv_released)
                self.assertTrue(good.req.kv.holds_kv)
                self.assertEqual(list(good.req.output_ids), [4])
                self.assertIsNone(good.req.finished_reason)
                for entry, receiver in ((bad, bad_receiver), (good, good_receiver)):
                    receiver.clear.assert_called_once_with()
                    self.assertIsNone(entry.kv_receiver)
                    slot = entry.metadata_buffer_index
                    self.assertEqual(
                        queue.metadata_buffers.bootstrap_room[slot, 0].item(), 0
                    )
                queue.scheduler.output_streamer.stream_output.assert_called_once_with(
                    [bad.req], False
                )
                self.assertEqual(cache.req_to_token_pool.available_size(), 2)
                self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), 10)
                free_tokens = cache.token_to_kv_pool_allocator.free_pages.tolist()
                self.assertTrue(set(bad_locations.tolist()).issubset(free_tokens))
                self.assertTrue(set(good_locations.tolist()).isdisjoint(free_tokens))
                self.assertEqual(len(free_tokens), len(set(free_tokens)))
                metadata_slots = queue.req_to_metadata_buffer_idx_allocator
                self.assertEqual(metadata_slots.available_size(), 3)
                self.assertEqual(len(set(metadata_slots.free_slots)), 3)
                # An empty poll must not release either request a second time.
                self.assertEqual(queue.pop_transferred_with_outcomes(), ([], {}))
                self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), 10)

                # Reuse the bad slot and ensure its new owner has its own seed.
                held_slots = []
                while metadata_slots.free_slots[0] != bad.metadata_buffer_index:
                    held_slots.append(metadata_slots.alloc())
                retry, _, _ = self.enqueue(
                    queue, "retry", 303, probability=0.25, token=7
                )
                self.assertEqual(retry.metadata_buffer_index, bad.metadata_buffer_index)
                self.assertEqual(
                    queue.pop_transferred_with_outcomes(),
                    ([retry.req], {"retry": False}),
                )
                for slot in held_slots:
                    metadata_slots.free(slot)
                self.assertEqual(metadata_slots.available_size(), 3)
                draft = self.draft_input(queue, [good.req, retry.req])
                torch.testing.assert_close(draft.topk_p, torch.tensor([[0.75], [0.25]]))
                torch.testing.assert_close(draft.topk_index, torch.tensor([[3], [7]]))

    def test_valid_seed_preserves_terminal_handoff(self):
        for terminal in ("max_new_tokens", "eos"):
            with self.subTest(terminal=terminal):
                queue = self.make_queue()
                entry, _, _ = self.enqueue(
                    queue,
                    "terminal",
                    101,
                    max_new=1 if terminal == "max_new_tokens" else 8,
                )
                if terminal == "eos":
                    entry.req.eos_token_ids = {4}
                admitted, outcomes = queue.pop_transferred_with_outcomes()
                self.assertEqual(admitted, [entry.req])
                self.assertEqual(outcomes, {"terminal": False})
                self.assertFalse(entry.req.finished())
                # Existing ordering builds the draft input before finish checks.
                self.assertIsNotNone(self.draft_input(queue, admitted))
                queue.scheduler.batch_result_processor.process_batch_result_prebuilt(
                    SimpleNamespace(reqs=admitted, return_logprob=False)
                )
                self.assertIsInstance(
                    entry.req.finished_reason,
                    FINISH_LENGTH
                    if terminal == "max_new_tokens"
                    else FINISH_MATCHED_TOKEN,
                )
                self.assertEqual(list(entry.req.output_ids), [4])
                self.assertTrue(entry.req.kv.is_kv_released)
                self.assertEqual(queue.tree_cache.req_to_token_pool.available_size(), 3)
                self.assertEqual(
                    queue.tree_cache.token_to_kv_pool_allocator.available_size(), 12
                )

    def test_unrelated_validation_errors_still_propagate(self):
        for error in (ValueError("conversion failure"), RuntimeError("device failure")):
            with self.subTest(error=type(error).__name__):
                queue = self.make_queue()
                entry, receiver, _ = self.enqueue(queue, "unrelated", 101)
                with patch(
                    "sglang.srt.speculative.standalone_disaggregation.torch.as_tensor",
                    side_effect=error,
                ):
                    with self.assertRaisesRegex(type(error), str(error)):
                        queue.pop_transferred_with_outcomes()
                self.assertIsNone(entry.req.finished_reason)
                self.assertEqual(list(entry.req.output_ids), [])
                self.assertEqual(list(queue.queue.items()), [entry])
                self.assertEqual(list(queue.queue), [entry.req])
                self.assertTrue(entry.req.kv.holds_kv)
                receiver.clear.assert_not_called()
                self.assertEqual(
                    queue.req_to_metadata_buffer_idx_allocator.available_size(), 2
                )


if __name__ == "__main__":
    unittest.main()
