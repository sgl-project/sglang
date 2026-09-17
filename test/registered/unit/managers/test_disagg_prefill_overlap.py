"""Disaggregated-prefill result ordering and transfer regressions."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from parameterized import parameterized

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDisaggPrefillOverlap(CustomTestCase):
    @parameterized.expand([(False,), (True,)])
    def test_prefill_chunk_transfers_and_delayed_sample(self, disable_prefill_overlap):
        def make_req(rid, slot, length, pending_bootstrap=False):
            return SimpleNamespace(
                rid=rid,
                kv=SimpleNamespace(req_pool_idx=slot),
                origin_input_ids=list(range(length)),
                extend_range=SimpleNamespace(end=4),
                inflight_middle_chunks=0,
                tmp_end_idx=-1,
                start_send_idx=0,
                pending_bootstrap=pending_bootstrap,
                metadata_buffer_index=slot,
                output_ids=[],
                return_logprob=False,
                return_sampling_mask=False,
                to_finish=None,
                finished_reason=None,
                grammar=None,
                time_stats=Mock(),
            )

        chunked = make_req("chunked", 0, 12)
        pending = make_req("pending", 1, 4, pending_bootstrap=True)
        parked = make_req("parked", 2, 4, pending_bootstrap=True)
        parked.output_ids = [7]
        requests = (chunked, pending, parked)
        batches = [
            ScheduleBatch(
                reqs=reqs,
                forward_mode=ForwardMode.EXTEND,
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
            for reqs in ([chunked, pending], [chunked], [chunked])
        ]
        scheduler = self.make_scheduler(batches + [None, None])
        scheduler.enable_overlap = True
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.tree_cache = object()
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(36, dtype=torch.int32).reshape(3, 12)
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1, translate_kv_indices_for_transfer=lambda indices: indices
        )
        scheduler.disagg_metadata_buffers = SimpleNamespace(set_buf=Mock())
        scheduler.disagg_prefill_bootstrap_queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[])
        )
        # A completed prefill can still await bootstrap after last_batch drains.
        scheduler.disagg_prefill_inflight_queue = [parked]
        scheduler.disagg_prefill_pending_chunk_rids = set()
        scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = None
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.batch_result_processor = SimpleNamespace(
            snapshot_auxiliary_output_starts=Mock(return_value=None),
            move_logprobs_to_cpu=Mock(),
        )
        scheduler.metrics_reporter.report_prefill_stats = Mock()
        scheduler.forward_stream = scheduler.schedule_stream = scheduler.copy_stream = (
            Mock()
        )
        scheduler.forward_stream_ctx = scheduler.copy_stream_ctx = nullcontext()
        scheduler._relay_forward_payload = Mock()
        scheduler.process_batch_result = scheduler.process_batch_result_disagg_prefill

        sent_pages = {req.rid: [] for req in requests}
        for req in requests:
            req.disagg_kv_sender = SimpleNamespace(
                req=req,
                should_send_kv_chunk=lambda count, last: count > 0 or last,
                send=lambda pages, state, num_kv_tokens, rid=req.rid: sent_pages[
                    rid
                ].append(pages.tolist()),
            )

        def finish_bootstrap(req, poll):
            self.assertEqual(poll, KVPoll.WaitingForInput)
            req.pending_bootstrap = False
            return True

        scheduler.handle_pending_bootstrap = finish_bootstrap
        plans = iter(batches + [None, None])

        def prepare_batch(**kwargs):
            batch = next(plans)
            if batch is not None:
                index = next(
                    i for i, candidate in enumerate(batches) if candidate is batch
                )
                # Admission snapshots the preceding chunk before extending the
                # shared Req; result processing must use that snapshot.
                chunked.tmp_end_idx = chunked.extend_range.end
                chunked.extend_range.end = (index + 1) * 4
                scheduler.chunked_req = chunked if index < 2 else None
                if index < 2:
                    chunked.inflight_middle_chunks += 1
            return SimpleNamespace(
                running_batch=scheduler.running_batch, batch_to_run=batch
            )

        scheduler.get_next_disagg_prefill_batch_to_run = prepare_batch
        results = []
        launch_send_offsets = []
        pending_sent_at_launch = []
        parked_sent_at_launch = []

        def run_batch(batch):
            launch_send_offsets.append(chunked.start_send_idx)
            pending_sent_at_launch.append(bool(sent_pages[pending.rid]))
            parked_sent_at_launch.append(bool(sent_pages[parked.rid]))
            result = GenerationBatchResult(
                copy_done=Mock(),
                logits_output=SimpleNamespace(
                    hidden_states=None,
                    next_token_logits=None,
                    auxiliary_device_output=None,
                    sampling_mask_output=None,
                ),
            )

            def sample():
                if result is not results[0]:
                    self.assertEqual(pending.output_ids, [7])
                result.next_token_ids = torch.full((len(batch.reqs),), 7)
                return result

            result.delay_sample_func = sample
            results.append(result)
            return result

        scheduler.run_batch = run_batch
        with (
            patch.object(
                scheduler_module.envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP,
                "get",
                return_value=disable_prefill_overlap,
            ),
            patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"),
            patch(
                "sglang.srt.disaggregation.prefill.should_force_retry",
                return_value=False,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=lambda senders, *_: [
                    (
                        KVPoll.WaitingForInput
                        if sender.req.pending_bootstrap
                        else KVPoll.Transferring
                    )
                    for sender in senders
                ],
            ),
            self.assertRaises(StopIteration),
        ):
            Scheduler.event_loop_overlap_disagg_prefill(scheduler)

        self.assertEqual(
            sent_pages[chunked.rid], [list(range(i, i + 4)) for i in (0, 4, 8)]
        )
        self.assertEqual(sent_pages[pending.rid], [list(range(12, 16))])
        self.assertEqual(sent_pages[parked.rid], [list(range(24, 28))])
        self.assertEqual(chunked.inflight_middle_chunks, 0)
        self.assertEqual(chunked.output_ids, [7])
        self.assertEqual(pending.output_ids, [7])
        self.assertFalse(pending.pending_bootstrap)
        self.assertFalse(parked.pending_bootstrap)
        self.assertEqual(list(scheduler.result_queue), [])
        scheduler.on_idle.assert_called_once()
        for result in results:
            self.assertIsNone(result.delay_sample_func)
            result.copy_done.record.assert_called_once()
            result.copy_done.synchronize.assert_called_once()
        self.assertEqual(
            launch_send_offsets, [0, 4, 8] if disable_prefill_overlap else [0, 0, 4]
        )
        self.assertEqual(
            pending_sent_at_launch,
            [False, True, True] if disable_prefill_overlap else [False, False, True],
        )
        self.assertEqual(parked_sent_at_launch, [disable_prefill_overlap, True, True])

    def make_scheduler(self, schedule):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = False
        scheduler.scheduler_stage_metrics = None
        scheduler.metrics_reporter = SimpleNamespace()
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.last_batch = None
        scheduler.chunked_req = None
        scheduler.waiting_queue = []
        scheduler.enable_staging = False
        scheduler.ingest_requests = Mock(
            side_effect=[None] * len(schedule) + [StopIteration]
        )
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=Mock(return_value=[])
        )
        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        scheduler._apply_war_barrier = Mock()
        scheduler.on_idle = Mock()
        scheduler.maybe_send_health_check_signal = Mock()
        return scheduler


if __name__ == "__main__":
    unittest.main()
