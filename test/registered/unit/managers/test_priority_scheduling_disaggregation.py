import json
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.decode import (  # noqa: E402
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class TestDisaggregationPriorityQueueing(unittest.TestCase):
    def _new_scheduler(self, disaggregation_mode: DisaggregationMode) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = disaggregation_mode
        scheduler.enable_priority_scheduling = True
        scheduler.schedule_low_priority_values_first = False
        scheduler.abort_on_priority_when_disabled = False
        scheduler.waiting_queue = []
        scheduler._prefetch_kvcache = MagicMock()
        scheduler._abort_on_queued_limit = MagicMock(return_value=False)
        scheduler.model_config = SimpleNamespace(num_key_value_heads=8)
        scheduler.disagg_prefill_bootstrap_queue = MagicMock()
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.ipc_channels = MagicMock()
        return scheduler

    def _new_req(self, priority=None):
        req = MagicMock()
        req.priority = priority
        req.rid = "req"
        req.time_stats = MagicMock()
        req.time_stats.trace_ctx = MagicMock()
        return req

    def test_prefill_mode_assigns_default_priority_before_bootstrap_queue(self):
        scheduler = self._new_scheduler(DisaggregationMode.PREFILL)
        req = self._new_req(priority=None)

        scheduler._add_request_to_queue(req)

        self.assertEqual(req.priority, -sys.maxsize - 1)
        scheduler.disagg_prefill_bootstrap_queue.add.assert_called_once_with(req, 8)
        req.time_stats.set_prefill_bootstrap_queue_entry_time.assert_called_once()

    def test_decode_mode_assigns_default_priority_before_prealloc_queue(self):
        scheduler = self._new_scheduler(DisaggregationMode.DECODE)
        req = self._new_req(priority=None)

        scheduler._add_request_to_queue(req)

        self.assertEqual(req.priority, -sys.maxsize - 1)
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_retracted=False
        )
        req.time_stats.set_decode_prealloc_queue_entry_time.assert_called_once()

    def test_priority_disabled_abort_validation_applies_to_decode_mode(self):
        scheduler = self._new_scheduler(DisaggregationMode.DECODE)
        scheduler.enable_priority_scheduling = False
        scheduler.abort_on_priority_when_disabled = True
        req = self._new_req(priority=10)

        scheduler._add_request_to_queue(req)

        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
        req.time_stats.trace_ctx.abort.assert_called_once()


class TestDecodePreallocQueuePriority(unittest.TestCase):
    def _new_decode_req(self, rid: str, priority: int, *, failed: bool = False):
        req = SimpleNamespace(
            rid=rid,
            priority=priority,
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            req_pool_idx=int(priority) % 8,
            finished_reason=FINISH_ABORT("failed") if failed else None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=8),
            cache_protected_len=0,
            time_stats=MagicMock(),
        )
        return SimpleNamespace(
            req=req,
            waiting_for_input=True,
            kv_receiver=MagicMock(),
            metadata_buffer_index=-1,
            is_rebootstrap=False,
        )

    def _new_queue(self, decode_reqs, *, low_priority_values_first: bool = False):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = list(decode_reqs)
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._allocatable_tokens = MagicMock(return_value=1000)

        def pre_alloc_mock(req, prefix_indices=None, prefix_len=0, total_prefix_len=0):
            return torch.arange(
                len(req.origin_input_ids) - prefix_len, dtype=torch.int64
            )

        queue._pre_alloc = MagicMock(side_effect=pre_alloc_mock)

        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 100
        queue.req_to_token_pool.req_to_token = torch.arange(
            8 * 16, dtype=torch.int64
        ).reshape(8, 16)

        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 100
        queue.req_to_metadata_buffer_idx_allocator.alloc.side_effect = iter(range(100))

        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 1
        queue.token_to_kv_pool_allocator.available_size.return_value = 1000
        queue.token_to_kv_pool = MagicMock()
        queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        queue.tree_cache = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_priority_scheduling = True
        scheduler.schedule_low_priority_values_first = low_priority_values_first
        scheduler.running_batch.reqs = []
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.enable_hisparse = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler
        return queue

    def test_prealloc_queue_schedules_higher_priority_values_first_by_default(self):
        reqs = [
            self._new_decode_req("low", 1),
            self._new_decode_req("high", 10),
            self._new_decode_req("mid", 5),
        ]
        queue = self._new_queue(reqs)

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated],
            [
                "high",
                "mid",
                "low",
            ],
        )
        self.assertEqual(failed, [])

    def test_prealloc_queue_can_schedule_lower_priority_values_first(self):
        reqs = [
            self._new_decode_req("mid", 5),
            self._new_decode_req("high", 10),
            self._new_decode_req("low", 1),
        ]
        queue = self._new_queue(reqs, low_priority_values_first=True)

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated],
            [
                "low",
                "mid",
                "high",
            ],
        )
        self.assertEqual(failed, [])

    def test_failed_request_indices_stay_valid_after_priority_sort(self):
        failed_low = self._new_decode_req("failed-low", 1, failed=True)
        healthy_high = self._new_decode_req("healthy-high", 10)
        queue = self._new_queue([failed_low, healthy_high])

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated], ["healthy-high"]
        )
        self.assertEqual([decode_req.req.rid for decode_req in failed], ["failed-low"])
        self.assertEqual(queue.queue, [])
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [failed_low.req], failed_low.req.return_logprob
        )


class TestDecodePreallocQueueRebootstrapPayload(unittest.TestCase):
    """The decode scheduler builds the rebootstrap ``/generate`` payload; the
    dispatch itself now lives on the kv manager (see
    ``TestCommonKVManagerPrefillRecompute``)."""

    def _sampling_params(self):
        return SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            ignore_eos=False,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )

    def _new_req(self):
        return SimpleNamespace(
            rid="rid-0",
            origin_input_ids=np.array([1, 2], dtype=np.int32),
            output_ids=[np.int32(3), np.int32(4)],
            sampling_params=self._sampling_params(),
            bootstrap_host="127.0.0.1",
            bootstrap_port=30000,
            bootstrap_room=7,
            priority=10,
            extra_key=None,
            routing_key=None,
            disagg_prefill_dp_rank=None,
        )

    def test_build_rebootstrap_payload_converts_numpy_ids_to_json_lists(self):
        req = self._new_req()

        # build_rebootstrap_payload lives on Req; exercise it unbound with a
        # namespace that carries the attributes it reads.
        payload = Req.build_rebootstrap_payload(req)

        # origin_input_ids + output_ids, coerced to plain python ints.
        self.assertEqual(payload["input_ids"], [1, 2, 3, 4])
        self.assertTrue(all(type(x) is int for x in payload["input_ids"]))
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 1)
        self.assertEqual(payload["bootstrap_room"], 7)
        # The prefill /generate URL is derived from bootstrap info on the decode
        # side, not sent in the payload; and the boundary token is replayed via
        # the decode-side override, so neither belongs in the payload.
        self.assertNotIn("pd_rebootstrap_prefill_url", payload)
        self.assertNotIn("pd_rebootstrap_forced_output_id", payload)
        # Must be JSON-serializable (numpy scalars would raise here).
        json.dumps(payload)


class TestCommonKVManagerPrefillRecompute(unittest.TestCase):
    """The kv manager owns the shared executor + HTTP session and routes any
    rebootstrap ``/generate`` failure through ``kv_receiver.abort()`` ->
    ``KVPoll.Failed`` so the scheduler's normal transfer-failure streaming runs.
    """

    def _new_manager(self):
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr._prefill_recompute_executor = None
        mgr._prefill_recompute_executor_lock = threading.Lock()
        mgr._prefill_recompute_sessions = threading.local()
        mgr.waiting_timeout = 300
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        # Only the attn-tp/attn-cp group leader on the first PP stage issues the
        # single rebootstrap /generate; default the mock manager to that leader.
        mgr.attn_tp_rank = 0
        mgr.attn_cp_rank = 0
        mgr.pp_rank = 0
        # Decode-side prefill info cache; the rebootstrap /generate URL is derived
        # from here (bootstrap_addr host + self-registered prefill_http_port)
        # instead of a router-injected pd_rebootstrap_prefill_url.
        mgr.prefill_info_table = {}
        return mgr

    def _register_prefill_info(self, mgr, bootstrap_addr, http_port):
        from sglang.srt.disaggregation.common.conn import PrefillServerInfo

        mgr.prefill_info_table[bootstrap_addr] = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=1,
            kv_cache_dtype=None,
            follow_bootstrap_room=True,
            prefill_http_port=http_port,
        )

    def _payload(self):
        return {
            "input_ids": [1, 2, 3, 4],
            "rid": "rid-0",
        }

    def test_submit_dispatches_run_to_shared_executor(self):
        mgr = self._new_manager()
        mgr._prefill_recompute_executor = MagicMock()
        receiver = MagicMock(bootstrap_room=7, bootstrap_addr="127.0.0.1:8998")
        self._register_prefill_info(mgr, "127.0.0.1:8998", 30000)

        mgr.submit_prefill_recompute(receiver, self._payload())

        mgr._prefill_recompute_executor.submit.assert_called_once()
        args = mgr._prefill_recompute_executor.submit.call_args[0]
        self.assertEqual(args[0], mgr._run_prefill_recompute)
        self.assertIs(args[1], receiver)
        # URL derived from bootstrap_addr host + registered prefill_http_port.
        self.assertEqual(args[2], "http://127.0.0.1:30000")
        receiver.abort.assert_not_called()

    def test_submit_is_noop_on_non_leader_ranks(self):
        # A retracted request is replicated across every rank in its attention
        # TP/CP group and every PP stage; only the group/first-stage leader must
        # POST the single /generate, or the prefill recomputes it once per rank.
        for attn_tp_rank, attn_cp_rank, pp_rank in (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ):
            with self.subTest(
                attn_tp_rank=attn_tp_rank,
                attn_cp_rank=attn_cp_rank,
                pp_rank=pp_rank,
            ):
                mgr = self._new_manager()
                mgr.attn_tp_rank = attn_tp_rank
                mgr.attn_cp_rank = attn_cp_rank
                mgr.pp_rank = pp_rank
                mgr._prefill_recompute_executor = MagicMock()
                receiver = MagicMock(bootstrap_room=7, bootstrap_addr="127.0.0.1:8998")
                self._register_prefill_info(mgr, "127.0.0.1:8998", 30000)

                mgr.submit_prefill_recompute(receiver, self._payload())

                mgr._prefill_recompute_executor.submit.assert_not_called()
                receiver.abort.assert_not_called()
                self.assertEqual(mgr.failure_records, {})

    def test_submit_unresolved_url_fails_via_abort(self):
        mgr = self._new_manager()
        mgr._prefill_recompute_executor = MagicMock()
        # No prefill_info registered for this bootstrap_addr -> URL unresolved.
        receiver = MagicMock(bootstrap_room=7, bootstrap_addr="127.0.0.1:8998")

        mgr.submit_prefill_recompute(receiver, self._payload())

        receiver.abort.assert_called_once()
        mgr._prefill_recompute_executor.submit.assert_not_called()
        self.assertIn(7, mgr.failure_records)

    def test_run_aborts_on_http_error(self):
        mgr = self._new_manager()
        session = MagicMock()
        session.post.return_value = SimpleNamespace(status_code=500, text="boom")
        mgr._prefill_recompute_sessions.session = session
        receiver = MagicMock(bootstrap_room=7)

        mgr._run_prefill_recompute(receiver, "http://prefill", self._payload())

        session.post.assert_called_once()
        receiver.abort.assert_called_once()
        self.assertIn(7, mgr.failure_records)

    def test_run_aborts_on_exception(self):
        mgr = self._new_manager()
        session = MagicMock()
        session.post.side_effect = RuntimeError("network down")
        mgr._prefill_recompute_sessions.session = session
        receiver = MagicMock(bootstrap_room=7)

        mgr._run_prefill_recompute(receiver, "http://prefill", self._payload())

        receiver.abort.assert_called_once()
        self.assertIn(7, mgr.failure_records)

    def test_run_success_does_not_abort(self):
        mgr = self._new_manager()
        session = MagicMock()
        session.post.return_value = SimpleNamespace(status_code=200, text="")
        mgr._prefill_recompute_sessions.session = session
        receiver = MagicMock(bootstrap_room=7)

        mgr._run_prefill_recompute(receiver, "http://prefill", self._payload())

        session.post.assert_called_once()
        receiver.abort.assert_not_called()
        self.assertEqual(mgr.failure_records, {})


class TestDecodePrebuiltPriority(unittest.TestCase):
    def test_waiting_queue_is_sorted_before_prebuilt_selection(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        original_waiting_queue = [MagicMock(rid="low"), MagicMock(rid="high")]
        scheduler.waiting_queue = original_waiting_queue
        scheduler.waiting_queue[0].priority = 1
        scheduler.waiting_queue[1].priority = 10
        scheduler.enable_priority_scheduling = True
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.batch_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock(size=1)
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.model_config = MagicMock()
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = MagicMock()
        scheduler.max_running_requests = 1
        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_radix_cache=False
        )
        scheduler.future_map = MagicMock()
        scheduler.policy = MagicMock()
        scheduler.policy.calc_priority.side_effect = lambda waiting_queue, _: (
            waiting_queue.sort(key=lambda req: -req.priority)
        )

        new_batch = MagicMock()
        with patch(
            "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
            return_value=new_batch,
        ) as init_new:
            ret = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
                scheduler, scheduler.running_batch
            )

        self.assertIs(ret, new_batch)
        scheduler.policy.calc_priority.assert_called_once_with(
            original_waiting_queue, scheduler.running_batch
        )
        selected_reqs = init_new.call_args.args[0]
        self.assertEqual([req.rid for req in selected_reqs], ["high"])
        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["low"])


if __name__ == "__main__":
    unittest.main()
