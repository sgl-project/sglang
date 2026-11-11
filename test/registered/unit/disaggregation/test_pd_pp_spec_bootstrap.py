"""PD PP speculation bootstraps a proposal before joining ordinary decode."""

import unittest
from contextlib import nullcontext
from itertools import product
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation import decode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardSteps
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid, slot):
    return Mock(
        rid=rid,
        owner_list=None,
        finished=Mock(return_value=False),
        kv=SimpleNamespace(req_pool_idx=slot, kv_committed_len=3),
    )


def _scheduler(new_batch, *, enabled=True):
    scheduler = decode.SchedulerDisaggregationDecodeMixin()
    scheduler._pp_spec_decode = enabled
    scheduler.chunked_req = None
    scheduler.enable_hisparse = False
    scheduler.enable_unified_memory = False
    scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
    if new_batch is not None:
        # Admission transfers ownership before returning the prebuilt batch.
        move(new_batch.reqs, scheduler.running_reqs)
    scheduler.get_new_prebuilt_batch = Mock(return_value=new_batch)
    scheduler.batch_result_processor = Mock()
    scheduler.schedule_stream = Mock()
    scheduler.forward_stream = object()
    scheduler.update_running_batch = Mock(side_effect=lambda batch: batch)
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=Mock(side_effect=lambda batch: batch)
    )
    return scheduler


def _next_batch(scheduler, running):
    # Exercise scheduling without the observability decorator's runtime context.
    with patch.object(decode, "set_schedule_time_batch"):
        return decode.SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run.__wrapped__(
            scheduler, running
        )


def _batch(req, spec_info=None):
    batch = Mock(reqs=[req], spec_info=spec_info, extend_num_tokens=12)
    batch.is_empty.return_value = False
    return batch


class TestPDPPSpecBootstrap(CustomTestCase):
    def test_fresh_request_reserves_before_launch_without_merging(self):
        req = _req("fresh", 0)
        new_batch, running = _batch(req), _batch(_req("running", 1))
        scheduler = _scheduler(new_batch)
        order = []
        scheduler.schedule_stream.wait_stream.side_effect = lambda _: order.append(
            "fence"
        )
        with patch(
            "sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode",
            side_effect=lambda *args, **kwargs: order.append("reserve"),
        ) as reserve:
            plan = _next_batch(scheduler, running)

        reserve.assert_called_once_with(new_batch, is_bootstrap=True)
        self.assertEqual(order, ["fence", "reserve"])
        self.assertIs(plan.batch_to_run, new_batch)
        self.assertIs(plan.running_batch, running)
        self.assertIn(req, scheduler.running_reqs)
        self.assertEqual(new_batch.extend_num_tokens, 0)
        running.merge_batch.assert_not_called()
        scheduler.update_running_batch.assert_not_called()
        scheduler.batch_result_processor.process_batch_result_prebuilt.assert_called_once_with(
            new_batch
        )

    def test_restored_and_ordinary_requests_merge_without_bootstrap(self):
        for enabled, restored in ((True, True), (False, False)):
            with self.subTest(enabled=enabled, restored=restored):
                req = _req("request", 0)
                relay = (
                    PPSpecRelayInput([req.rid], torch.tensor([[1, 2, 3]]))
                    if restored
                    else None
                )
                new_batch, running = _batch(req, relay), _batch(_req("running", 1))
                scheduler = _scheduler(new_batch, enabled=enabled)
                with patch(
                    "sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode"
                ) as reserve:
                    plan = _next_batch(scheduler, running)
                reserve.assert_not_called()
                running.merge_batch.assert_called_once_with(new_batch)
                scheduler.update_running_batch.assert_called_once_with(running)
                self.assertIs(plan.batch_to_run, running)
                self.assertIs(plan.running_batch, running)
                self.assertIn(req, scheduler.running_reqs)
                self.assertIs(new_batch.spec_info, relay)
                self.assertEqual(new_batch.extend_num_tokens, 12)

    def test_only_tail_drafts_and_restores_forward_state_without_target(self):
        for (enabled, is_last), unified in product(
            ((True, True), (True, False), (False, True)), (False, True)
        ):
            with self.subTest(enabled=enabled, is_last=is_last, unified=unified):
                req = _req("request", 0)
                input_ids = torch.tensor([1])
                batch = ScheduleBatch(
                    reqs=[req],
                    forward_mode=ForwardMode.PREBUILT,
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    input_ids=input_ids,
                    seq_lens=torch.tensor([3]),
                )
                scheduler = _scheduler(None, enabled=enabled)
                scheduler.enable_unified_memory = unified
                scheduler.pp_group = SimpleNamespace(is_last_rank=is_last)
                scheduler._forward_isolation = MethodType(
                    Scheduler._forward_isolation, scheduler
                )
                proposal = (torch.tensor([[1, 2, 3]]), torch.tensor([[-1, 0]]), None)
                steps = ForwardSteps([])
                steps.forward_launched()
                scheduler.req_to_token_pool = object()
                scheduler.token_to_kv_pool_allocator = SimpleNamespace(
                    forward_steps=steps
                )
                order = []
                forward_stream = object()
                forward_done = Mock()
                forward_done.record.side_effect = lambda **kwargs: order.append(
                    "record"
                )
                scheduler.device_module = SimpleNamespace(
                    Event=lambda: forward_done,
                    current_stream=lambda: forward_stream,
                    stream=lambda _: nullcontext(),
                )
                scheduler._record_unified_forward = MethodType(
                    Scheduler._record_unified_forward, scheduler
                )
                scheduler._on_step_forward_recorded = MethodType(
                    Scheduler._on_step_forward_recorded, scheduler
                )

                def draft(current):
                    order.append("draft")
                    self.assertIs(current, batch)
                    self.assertEqual(current.forward_mode, ForwardMode.DECODE)
                    self.assertIsNone(current.input_ids)
                    return proposal

                scheduler.draft_worker = SimpleNamespace(
                    draft_pp_proposal=Mock(side_effect=draft)
                )
                scheduler.run_batch = Mock()
                with patch("sglang.srt.managers.scheduler.kv_storage_of") as storage:
                    result = scheduler._run_batch_prebuilt(batch)
                if enabled and is_last and unified:
                    self.assertEqual(order, ["draft", "record"])
                    self.assertIs(result.unified_forward_done, forward_done)
                    self.assertIs(steps.pending_key()[1], forward_done)
                    self.assertTrue(
                        storage.return_value.record_forward.call_args.kwargs[
                            "token_writes_unknown"
                        ]
                    )
                    forward_done.record.assert_called_once_with(stream=forward_stream)
                else:
                    self.assertEqual(order, ["draft"] if enabled and is_last else [])
                    self.assertIsNone(result.unified_forward_done)
                    storage.assert_not_called()
                self.assertIsNone(result.next_token_ids)
                self.assertIsNone(result.accept_lens)
                self.assertEqual(req.kv.kv_committed_len, 3)
                self.assertEqual(batch.forward_mode, ForwardMode.PREBUILT)
                self.assertIs(batch.input_ids, input_ids)
                scheduler.run_batch.assert_not_called()
                if enabled and is_last:
                    scheduler.draft_worker.draft_pp_proposal.assert_called_once_with(
                        batch
                    )
                    self.assertIs(result.next_verify_chain, proposal[0])
                    self.assertIs(result.next_verify_parent_list, proposal[1])
                else:
                    scheduler.draft_worker.draft_pp_proposal.assert_not_called()
                    self.assertIsNone(result.next_verify_chain)

    def test_admission_reserves_all_microbatches_and_excludes_retired_and_host_reqs(
        self,
    ):
        running, bootstrap, waiting, transferring, host, freed = [
            _req(str(i), slot) for i, slot in enumerate((0, 1, 2, 3, 4, None))
        ]
        scheduler = _scheduler(None)
        scheduler.running_batch = SimpleNamespace(reqs=[running])
        move([running, freed], scheduler.running_reqs)
        scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
        move([bootstrap, waiting], scheduler.waiting_queue)
        move([bootstrap], scheduler.running_reqs)
        freed.finished.return_value = True
        scheduler.running_reqs._leave_finished()
        entries = [
            SimpleNamespace(req=req, host_staged=host_staged)
            for req, host_staged in (
                (transferring, False),
                (host, True),
            )
        ]
        queue = decode.DecodePreallocQueue.__new__(decode.DecodePreallocQueue)
        queue.scheduler = scheduler
        transfers = IndexedReqList(
            ListKind.PD_RECEIVING, req_of=lambda entry: entry.req
        )
        move(entries, transfers)
        queue.transfer_queue = SimpleNamespace(queue=transfers)
        queue._host_staged_admission_stats = None
        self.assertEqual(queue._active_req_count(extra_reserved_reqs=2), 6)

        scheduler._pp_spec_decode = False
        self.assertEqual(queue._active_req_count(extra_reserved_reqs=2), 5)


if __name__ == "__main__":
    unittest.main()
