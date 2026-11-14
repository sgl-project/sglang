"""CPU metadata handoff contracts; byte copies stand in for the PD transport.

The real buffers, decode commit, handoff-token processor and draft factory run.
No model forward, KV transfer, scheduler loop or accelerator is exercised.
"""

import ctypes
import dataclasses
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestPdStandaloneMetadata(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_context().override_server_args(
                speculative_algorithm="STANDALONE", speculative_eagle_topk=1
            )
        )
        self.enterContext(envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(""))
        self.enterContext(envs.SGLANG_DISAGG_CUSTOMIZED_INFO_MAX_BYTES.override(0))
        self.enterContext(envs.SGLANG_ENABLE_DISAGG_SAMPLING_MASK.override(False))

    def buffers(self):
        result = MetadataBuffers(
            size=3,
            hidden_size=4,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=0,
            output_dsa_topk_indices_dim=2,
        )
        self.assertEqual(result.output_ids.device.type, "cpu")
        return result

    @staticmethod
    def req(rid, room, slot=0, *, seed=None, bonus=None, hidden=None, dsa=None):
        req = Req(
            rid,
            "",
            array("q", [1, 2]),
            SamplingParams(max_new_tokens=8),
            bootstrap_host="127.0.0.1",
            bootstrap_room=room,
        )
        req.metadata_buffer_index = slot
        if bonus is not None:
            req.output_ids.append(bonus)
        if seed is not None:
            probabilities, indices = seed
            req.output_topk_p = torch.tensor(probabilities, dtype=torch.float32)
            req.output_topk_index = torch.tensor(indices, dtype=torch.int64)
        req.hidden_states_tensor = hidden
        req.output_dsa_topk_indices = dsa
        return req

    def transfer_row(self, source, destination, src_slot, dst_slot):
        """Copy registered CPU regions, including the room publication field.

        This tests the metadata layout, not the ordering/atomicity of a NIC.
        """
        src_ptrs, src_sizes, src_rows = source.get_buf_infos()
        dst_ptrs, dst_sizes, dst_rows = destination.get_buf_infos()
        self.assertEqual(src_rows, dst_rows)
        self.assertEqual(len(src_ptrs), len(dst_ptrs))
        destination.clear_slot(dst_slot)
        for src, dst, size, src_size, dst_size in zip(
            src_ptrs, dst_ptrs, src_rows, src_sizes, dst_sizes, strict=True
        ):
            self.assertLessEqual((src_slot + 1) * size, src_size)
            self.assertLessEqual((dst_slot + 1) * size, dst_size)
            ctypes.memmove(dst + dst_slot * size, src + src_slot * size, size)

    def commit(self, buffers, req, slot, *, algorithm=SpeculativeAlgorithm.STANDALONE):
        fields = dict.fromkeys(
            f.name for f in dataclasses.fields(SchedulerBatchResultProcessor)
        )
        fields["model_config"] = SimpleNamespace(self_opened_think_start_ids=None)
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.metadata_buffers = buffers
        queue.spec_algorithm = algorithm
        queue.scheduler = SimpleNamespace(
            kv_checksum_computer=None,
            model_config=SimpleNamespace(vocab_size=11),
            batch_result_processor=SchedulerBatchResultProcessor(**fields),
        )
        receiver = Mock()
        pending = DecodeRequest(
            req=req, kv_receiver=receiver, metadata_buffer_index=slot
        )
        queue._commit_transfer_to_req(pending)
        receiver.clear.assert_called_once_with()
        self.assertIsNone(pending.kv_receiver)

    def draft_input(self, reqs):
        batch = SimpleNamespace(
            reqs=reqs,
            device=torch.device("cpu"),
            enable_overlap=False,
            model_config=SimpleNamespace(vocab_size=11),
        )
        bonus = torch.tensor([req.output_ids[-1] for req in reqs])
        result = SpeculativeAlgorithm.STANDALONE.build_disagg_draft_input(
            batch, bonus, None
        )
        self.assertIsNotNone(result)
        self.assertIsNone(result.hidden_states)
        return result

    def test_registered_bytes_reach_decode_and_draft_in_request_order(self):
        source, destination = self.buffers(), self.buffers()
        specs = (("a", 101, 2, 1, 0.75, 3, 4), ("b", 202, 0, 2, 0.25, 7, 8))
        reqs = []
        for rid, room, src_slot, dst_slot, probability, token, bonus in specs:
            prefill = self.req(
                rid, room, src_slot, seed=([probability], [token]), bonus=bonus
            )
            source.set_buf(prefill)
            self.transfer_row(source, destination, src_slot, dst_slot)
            reqs.append(self.req(rid, room))
        # Commit in a different order from batch order and metadata slot order.
        for req, slot in ((reqs[1], 2), (reqs[0], 1)):
            self.commit(destination, req, slot)
            self.assertFalse(req.finished())
        result = self.draft_input(reqs)
        torch.testing.assert_close(result.topk_p, torch.tensor([[0.75], [0.25]]))
        torch.testing.assert_close(result.topk_index, torch.tensor([[3], [7]]))
        torch.testing.assert_close(result.bonus_tokens, torch.tensor([4, 8]))

    def test_committed_seed_survives_metadata_slot_reuse(self):
        source, destination = self.buffers(), self.buffers()
        source.set_buf(self.req("a", 101, seed=([0.75], [3]), bonus=4))
        self.transfer_row(source, destination, 0, 0)
        old = self.req("a", 101)
        self.commit(destination, old, 0)
        source.set_buf(self.req("b", 202, seed=([0.25], [7]), bonus=8))
        self.transfer_row(source, destination, 0, 0)
        new = self.req("b", 202)
        self.commit(destination, new, 0)
        result = self.draft_input([old, new])
        torch.testing.assert_close(result.topk_p, torch.tensor([[0.75], [0.25]]))
        torch.testing.assert_close(result.topk_index, torch.tensor([[3], [7]]))
        torch.testing.assert_close(result.bonus_tokens, torch.tensor([4, 8]))

    def test_reused_eagle_row_clears_tail_hidden_and_dsa_for_standalone(self):
        source, destination = self.buffers(), self.buffers()
        hidden, dsa = torch.arange(4, dtype=torch.float32), torch.tensor([1, 2])
        source.set_buf(
            self.req(
                "eagle", 9, seed=([0.75, 0.25], [3, 7]), bonus=4, hidden=hidden, dsa=dsa
            )
        )
        self.transfer_row(source, destination, 0, 0)
        eagle = self.req("eagle", 9)
        self.commit(destination, eagle, 0, algorithm=SpeculativeAlgorithm.EAGLE)
        torch.testing.assert_close(eagle.hidden_states_tensor, hidden)
        torch.testing.assert_close(eagle.output_dsa_topk_indices, dsa.to(torch.int32))
        torch.testing.assert_close(eagle.output_topk_p[:2], torch.tensor([0.75, 0.25]))
        source.set_buf(self.req("standalone", 10, seed=([0.5], [2]), bonus=6))
        self.transfer_row(source, destination, 0, 0)
        req = self.req("standalone", 10)
        self.commit(destination, req, 0)
        self.assertEqual(req.output_topk_p[1:].count_nonzero().item(), 0)
        self.assertEqual(req.output_topk_index[1:].count_nonzero().item(), 0)
        self.assertEqual(req.hidden_states_tensor.count_nonzero().item(), 0)
        self.assertIsNone(req.output_dsa_topk_indices)
        torch.testing.assert_close(eagle.hidden_states_tensor, hidden)
        torch.testing.assert_close(
            self.draft_input([req]).topk_index, torch.tensor([[2]])
        )

    def test_seedless_reuse_cannot_inherit_previous_seed(self):
        source, destination = self.buffers(), self.buffers()
        source.set_buf(self.req("old", 9, seed=([0.75], [3]), bonus=4))
        source.set_buf(self.req("seedless", 10, bonus=6))
        self.transfer_row(source, destination, 0, 0)
        req = self.req("seedless", 10)
        self.commit(destination, req, 0)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIn("Invalid or unwritten", req.finished_reason.message)
        self.assertEqual(list(req.output_ids), [])
        self.assertIsNone(req.output_topk_p)

    def test_valid_zero_token_is_not_an_unwritten_seed(self):
        source, destination = self.buffers(), self.buffers()
        source.set_buf(self.req("zero", 9, seed=([1.0], [0]), bonus=4))
        self.transfer_row(source, destination, 0, 0)
        req = self.req("zero", 9)
        self.commit(destination, req, 0)
        torch.testing.assert_close(
            self.draft_input([req]).topk_index, torch.tensor([[0]])
        )

    def test_room_mismatch_aborts_before_seed_is_committed(self):
        source, destination = self.buffers(), self.buffers()
        source.set_buf(self.req("wrong", 9, seed=([0.75], [3]), bonus=4))
        self.transfer_row(source, destination, 0, 0)
        req = self.req("expected", 10)
        self.commit(destination, req, 0)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(list(req.output_ids), [])
        self.assertIsNone(req.output_topk_p)

    def sender_scheduler(self, buffers):
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.disagg_metadata_buffers = buffers
        scheduler.enable_staging = False
        scheduler.disagg_prefill_pending_chunk_rids = set()
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1, translate_kv_indices_for_transfer=lambda indices: indices
        )
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[0, 1]])
        )
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            kv_manager=SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        )
        return scheduler

    def outgoing(self, rid, room, *, seed):
        req = self.req(rid, room, seed=seed, bonus=4)
        req.kv = ReqKvInfo(req_pool_idx=0, kv_allocated_len=2)
        req.extend_end = 2
        req.disagg_kv_sender = Mock()
        req.disagg_kv_sender.supports_host_tensor_indices = False
        req.disagg_kv_sender.get_max_transfer_tokens.return_value = None
        return req

    def test_bad_seed_aborts_only_that_prefill_and_never_sends_final_chunk(self):
        for kind in ("missing_index", "shape", "capacity"):
            with self.subTest(kind=kind):
                buffers = self.buffers()
                buffers.set_buf(self.req("old", 9, seed=([0.75], [3]), bonus=4))
                scheduler = self.sender_scheduler(buffers)
                bad = self.outgoing("bad", 10, seed=([0.5], [2]))
                if kind == "missing_index":
                    bad.output_topk_index = None
                elif kind == "shape":
                    bad.output_topk_index = torch.tensor([2, 3])
                else:
                    bad.output_topk_p = torch.ones(17)
                    bad.output_topk_index = torch.zeros(17, dtype=torch.int64)
                scheduler.disagg_prefill_pending_chunk_rids.add(bad.rid)
                scheduler._send_kv_chunk(bad, last_chunk=True)
                self.assertIsInstance(bad.finished_reason, FINISH_ABORT)
                self.assertEqual(bad.finished_reason.status_code, 500)
                bad.disagg_kv_sender.abort.assert_called_once_with()
                bad.disagg_kv_sender.send.assert_not_called()
                self.assertNotIn(bad.rid, scheduler.disagg_prefill_pending_chunk_rids)
                # P does not clear old rooms: failure must not publish the new room.
                self.assertEqual(buffers.bootstrap_room[0, 0].item(), 9)
                good = self.outgoing("good", 11, seed=([0.5], [2]))
                scheduler._send_kv_chunk(good, last_chunk=True)
                self.assertFalse(good.finished())
                good.disagg_kv_sender.abort.assert_not_called()
                good.disagg_kv_sender.send.assert_called_once()
                self.assertEqual(buffers.bootstrap_room[0, 0].item(), 11)

    def test_unrelated_metadata_failures_still_propagate(self):
        buffers = self.buffers()
        scheduler = self.sender_scheduler(buffers)
        for error in (
            ValueError("application failure"),
            RuntimeError("device failure"),
        ):
            with self.subTest(error=type(error).__name__):
                req = self.outgoing("runtime-error", 9, seed=([0.5], [2]))
                with patch.object(buffers, "set_buf", side_effect=error):
                    with self.assertRaisesRegex(type(error), str(error)):
                        scheduler._send_kv_chunk(req, last_chunk=True)
                req.disagg_kv_sender.abort.assert_not_called()
                req.disagg_kv_sender.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
