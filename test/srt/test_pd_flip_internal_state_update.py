import types
import unittest

import torch

from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.disaggregation.flip_state_machine import (
    ClusterSnapshot,
    FlipDecision,
    FlipDirection,
    FlipState,
    FlipStateMachine,
    SLOThresholdFlipEvaluator,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import (
    get_global_server_args,
    set_global_server_args_for_scheduler,
)


class TestPDFlipInternalStateUpdate(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(
            types.SimpleNamespace(
                pd_flip_prefill_slo_attainment=None,
                pd_flip_decode_slo_attainment=None,
                pd_flip_prefill_nodes=None,
                pd_flip_decode_nodes=None,
                pd_flip_slo_threshold=0.9,
                pd_flip_window_seconds=1.0,
                pd_flip_prepare_ack=False,
                pd_flip_commit_ack=False,
                pd_flip_abort=False,
                page_size=1,
                speculative_algorithm=None,
                strip_thinking_cache=False,
            )
        )

    def test_updates_pd_flip_slo_inputs_through_internal_state_path(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = types.SimpleNamespace(pp_size=1)
        scheduler.max_running_requests = 8
        scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
        scheduler.metrics_reporter = types.SimpleNamespace(
            spec_total_num_forward_ct=0,
            spec_total_num_accept_tokens=0,
        )

        Scheduler.set_internal_state(
            scheduler,
            SetInternalStateReq(
                server_args={
                    "pd_flip_prefill_slo_attainment": 0.75,
                    "pd_flip_decode_slo_attainment": 0.96,
                }
            ),
        )

        server_args = get_global_server_args()
        self.assertEqual(server_args.pd_flip_prefill_slo_attainment, 0.75)
        self.assertEqual(server_args.pd_flip_decode_slo_attainment, 0.96)

    def test_external_pd_flip_acks_are_consumed_by_callbacks(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = types.SimpleNamespace(pp_size=1, tp_rank=0)
        scheduler.is_fully_idle = lambda: True
        scheduler.max_running_requests = 8
        scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
        scheduler.metrics_reporter = types.SimpleNamespace(
            spec_total_num_forward_ct=0,
            spec_total_num_accept_tokens=0,
        )

        Scheduler.set_internal_state(
            scheduler,
            SetInternalStateReq(
                server_args={
                    "pd_flip_prepare_ack": True,
                    "pd_flip_commit_ack": True,
                }
            ),
        )

        snapshot = ClusterSnapshot(timestamp=1.0, role="decode")
        decision = FlipDecision(
            should_flip=True,
            direction=FlipDirection.D_TO_P,
            reason="prefill SLO below threshold",
        )

        self.assertTrue(Scheduler.prepare_pd_flip(scheduler, snapshot, decision))
        self.assertFalse(get_global_server_args().pd_flip_prepare_ack)
        self.assertTrue(Scheduler.commit_pd_flip(scheduler, snapshot, decision))
        self.assertFalse(get_global_server_args().pd_flip_commit_ack)

    def test_prepare_ack_waits_until_scheduler_is_idle(self):
        server_args = get_global_server_args()
        server_args.pd_flip_prepare_ack = True

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = types.SimpleNamespace(pp_size=1, tp_rank=0)
        scheduler.is_fully_idle = lambda: False
        snapshot = ClusterSnapshot(timestamp=1.0, role="decode")
        decision = FlipDecision(
            should_flip=True,
            direction=FlipDirection.D_TO_P,
            reason="prefill SLO below threshold",
        )

        self.assertFalse(Scheduler.prepare_pd_flip(scheduler, snapshot, decision))
        self.assertTrue(server_args.pd_flip_prepare_ack)

        scheduler.is_fully_idle = lambda: True
        self.assertTrue(Scheduler.prepare_pd_flip(scheduler, snapshot, decision))
        self.assertFalse(server_args.pd_flip_prepare_ack)

    def test_commit_ack_waits_until_scheduler_is_idle(self):
        server_args = get_global_server_args()
        server_args.pd_flip_commit_ack = True

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = types.SimpleNamespace(pp_size=1, tp_rank=0)
        scheduler.is_fully_idle = lambda: False
        snapshot = ClusterSnapshot(timestamp=1.0, role="decode")
        decision = FlipDecision(
            should_flip=True,
            direction=FlipDirection.D_TO_P,
            reason="prefill SLO below threshold",
        )

        self.assertFalse(Scheduler.commit_pd_flip(scheduler, snapshot, decision))
        self.assertTrue(server_args.pd_flip_commit_ack)

        scheduler.is_fully_idle = lambda: True
        self.assertTrue(Scheduler.commit_pd_flip(scheduler, snapshot, decision))
        self.assertFalse(server_args.pd_flip_commit_ack)

    def test_pd_flip_rejects_new_admission_while_draining(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_flip_state_machine = types.SimpleNamespace(state=FlipState.SAFE)
        self.assertFalse(Scheduler.pd_flip_should_reject_new_work(scheduler))

        scheduler.pd_flip_state_machine.state = FlipState.PREPARING
        self.assertTrue(Scheduler.pd_flip_should_reject_new_work(scheduler))

        scheduler.pd_flip_state_machine.state = FlipState.FLIPPING
        self.assertTrue(Scheduler.pd_flip_should_reject_new_work(scheduler))

    def test_pd_flip_rejection_streams_service_unavailable_abort(self):
        streamed = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.output_streamer = types.SimpleNamespace(
            stream_output=lambda reqs, return_logprob: streamed.append(
                (reqs, return_logprob)
            )
        )
        req = types.SimpleNamespace(rid="rid-1", return_logprob=False)

        Scheduler.reject_pd_flip_admission(scheduler, req)

        self.assertEqual(len(streamed), 1)
        self.assertFalse(streamed[0][1])
        self.assertEqual(streamed[0][0], [req])
        self.assertEqual(req.finished_reason.status_code.value, 503)
        self.assertIn("PD role flip", req.finished_reason.message)

    def test_build_snapshot_uses_runtime_pd_flip_inputs(self):
        server_args = get_global_server_args()
        server_args.pd_flip_prefill_nodes = 2
        server_args.pd_flip_decode_nodes = 4
        server_args.pd_flip_prefill_slo_attainment = 0.61
        server_args.pd_flip_decode_slo_attainment = 0.93

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = types.SimpleNamespace(
            pd_flip_prefill_nodes=None,
            pd_flip_decode_nodes=None,
            pd_flip_prefill_slo_attainment=None,
            pd_flip_decode_slo_attainment=None,
        )
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.waiting_queue = [object()]
        scheduler.running_batch = types.SimpleNamespace(reqs=[object(), object()])
        scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_inflight_queue = None
        scheduler.disagg_decode_prealloc_queue = None
        scheduler.disagg_decode_transfer_queue = None

        snapshot = Scheduler.build_pd_flip_snapshot(scheduler)

        self.assertEqual(snapshot.prefill_nodes, 2)
        self.assertEqual(snapshot.decode_nodes, 4)
        self.assertEqual(snapshot.prefill_slo_attainment, 0.61)
        self.assertEqual(snapshot.decode_slo_attainment, 0.93)
        self.assertEqual(snapshot.waiting_reqs, 1)
        self.assertEqual(snapshot.running_reqs, 2)

    def test_maybe_tick_refreshes_runtime_flip_evaluator_controls(self):
        server_args = get_global_server_args()
        server_args.pd_flip_slo_threshold = 0.8
        server_args.pd_flip_window_seconds = 0.0
        server_args.pd_flip_prefill_nodes = 1
        server_args.pd_flip_decode_nodes = 2
        server_args.pd_flip_prefill_slo_attainment = 0.85
        server_args.pd_flip_decode_slo_attainment = 0.99

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = types.SimpleNamespace(tp_rank=1)
        scheduler.server_args = server_args
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.waiting_queue = []
        scheduler.running_batch = types.SimpleNamespace(reqs=[])
        scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_inflight_queue = None
        scheduler.disagg_decode_prealloc_queue = None
        scheduler.disagg_decode_transfer_queue = None
        scheduler.pd_flip_state_machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(slo_threshold=0.9),
            min_window_seconds=10.0,
        )

        Scheduler.maybe_tick_pd_flip_state_machine(scheduler)

        self.assertEqual(scheduler.pd_flip_state_machine.state, FlipState.SAFE)
        self.assertEqual(scheduler.pd_flip_state_machine.evaluator.slo_threshold, 0.8)
        self.assertEqual(scheduler.pd_flip_state_machine.min_window_seconds, 0.0)

    def test_pd_flip_migration_source_start_exports_running_decode_manifest(self):
        from sglang.srt.managers.io_struct import PDFlipMigrationSourceStartReq

        req = types.SimpleNamespace(
            rid="rid-1",
            origin_input_ids=[1, 2, 3],
            output_ids=[4, 5],
            bootstrap_room=123,
            priority=7,
            routing_key="route-a",
            extra_key="extra-a",
            return_logprob=False,
            req_pool_idx=3,
            kv_committed_len=4,
            sampling_params=types.SimpleNamespace(to_json=lambda: {"temperature": 0.0}),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.running_batch = types.SimpleNamespace(reqs=[req])
        scheduler.waiting_queue = []
        scheduler.ps = types.SimpleNamespace(tp_rank=0)

        output = Scheduler.start_pd_flip_migration_source(
            scheduler,
            PDFlipMigrationSourceStartReq(
                session_id="session-1",
                target_url="http://decode-target",
            ),
        )

        self.assertTrue(output.success)
        self.assertEqual(output.status["state"], "source_started")
        self.assertEqual(output.status["pending_reqs"], 1)
        self.assertEqual(output.manifests[0]["rid"], "rid-1")
        self.assertEqual(output.manifests[0]["origin_input_ids"], [1, 2, 3])
        self.assertEqual(output.manifests[0]["output_ids"], [4, 5])
        self.assertEqual(output.manifests[0]["kv_committed_len"], 4)

    def test_pd_flip_source_page_indices_rejects_released_req_pool_idx(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.arange(40, dtype=torch.int32).reshape(2, 20)
        )
        scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(page_size=1)
        req = types.SimpleNamespace(rid="rid-1", req_pool_idx=None)

        with self.assertRaisesRegex(ValueError, "req_pool_idx was released"):
            Scheduler._pd_flip_source_page_indices(scheduler, req, committed_len=4)

    def test_pd_flip_source_page_indices_rejects_non_scalar_req_pool_idx(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.arange(40, dtype=torch.int32).reshape(2, 20)
        )
        scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(page_size=1)
        req = types.SimpleNamespace(
            rid="rid-1", req_pool_idx=torch.tensor([0, 1], dtype=torch.int64)
        )

        with self.assertRaisesRegex(ValueError, "non-scalar req_pool_idx"):
            Scheduler._pd_flip_source_page_indices(scheduler, req, committed_len=4)

    def test_pd_flip_deferred_release_keeps_source_req_pool_idx(self):
        calls = []
        req = types.SimpleNamespace(
            req_pool_idx=7,
            mamba_pool_idx=None,
            pd_flip_defer_kv_release=True,
            pop_overallocated_kv_cache=lambda: (0, 0),
        )
        tree_cache = types.SimpleNamespace(
            supports_mamba=lambda: False,
            cache_finished_req=lambda req, is_insert=True: calls.append(
                ("cache", is_insert)
            ),
            req_to_token_pool=types.SimpleNamespace(
                free=lambda req: calls.append(("free", req.req_pool_idx))
            ),
        )

        release_kv_cache(req, tree_cache, is_insert=False)

        self.assertEqual(calls, [])
        self.assertEqual(req.req_pool_idx, 7)
        self.assertTrue(req.pd_flip_kv_release_deferred)
        self.assertFalse(req.pd_flip_deferred_kv_release_is_insert)

    def test_pd_flip_release_source_requests_frees_deferred_finished_req(self):
        calls = []

        def cache_finished_req(req, is_insert=True):
            calls.append(("cache", is_insert))
            req.req_pool_idx = None

        req = types.SimpleNamespace(
            rid="rid-1",
            req_pool_idx=7,
            mamba_pool_idx=None,
            pd_flip_defer_kv_release=True,
            pd_flip_kv_release_deferred=True,
            pd_flip_deferred_kv_release_is_insert=False,
            finished=lambda: True,
            pop_overallocated_kv_cache=lambda: (0, 0),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = types.SimpleNamespace(
            supports_mamba=lambda: False,
            cache_finished_req=cache_finished_req,
            req_to_token_pool=types.SimpleNamespace(
                free=lambda req: calls.append(("free", req.req_pool_idx))
            ),
        )
        session = {
            "source_entries": {
                "rid-1": {
                    "req": req,
                    "metadata_index": -1,
                }
            }
        }

        Scheduler._pd_flip_release_source_requests(scheduler, session, {"rid-1"})

        self.assertEqual(calls, [("cache", False)])
        self.assertIsNone(req.req_pool_idx)
        self.assertFalse(req.pd_flip_defer_kv_release)
        self.assertFalse(req.pd_flip_kv_release_deferred)

    def test_pd_flip_prepare_waits_for_active_migration_before_ack(self):
        server_args = get_global_server_args()
        server_args.pd_flip_prepare_ack = True

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = types.SimpleNamespace(pp_size=1, tp_rank=0)
        scheduler.is_fully_idle = lambda: True
        scheduler.pd_flip_migration_session = {
            "role": "source",
            "state": "source_started",
            "pending_reqs": 1,
            "released_reqs": 0,
            "failed_reqs": 0,
        }
        snapshot = ClusterSnapshot(
            timestamp=1.0,
            role="decode",
            running_reqs=1,
        )
        decision = FlipDecision(
            should_flip=True,
            direction=FlipDirection.D_TO_P,
            reason="prefill SLO below threshold",
        )

        self.assertFalse(Scheduler.prepare_pd_flip(scheduler, snapshot, decision))
        self.assertTrue(server_args.pd_flip_prepare_ack)

        scheduler.pd_flip_migration_session.update(
            {"state": "source_released", "pending_reqs": 0, "released_reqs": 1}
        )
        idle_snapshot = ClusterSnapshot(timestamp=2.0, role="decode")
        self.assertTrue(Scheduler.prepare_pd_flip(scheduler, idle_snapshot, decision))
        self.assertFalse(server_args.pd_flip_prepare_ack)

    def test_pd_flip_internal_state_includes_migration_status(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.pd_flip_state_machine = types.SimpleNamespace(
            status=lambda: {
                "state": "preparing",
                "direction": "d_to_p",
                "active_request_migration_strategy": "drain_to_idle",
            }
        )
        scheduler.pd_flip_should_reject_new_work = lambda: True
        scheduler.pd_flip_is_idle_for_commit = lambda snapshot: False
        scheduler.build_pd_flip_snapshot = lambda: ClusterSnapshot(
            timestamp=1.0, role="decode"
        )
        scheduler.pd_flip_migration_session = {
            "role": "source",
            "state": "source_started",
            "pending_reqs": 2,
            "transferred_reqs": 0,
            "released_reqs": 0,
            "failed_reqs": 0,
            "last_error": "",
        }

        status = Scheduler.get_pd_flip_internal_state(scheduler)

        self.assertTrue(status["migration_enabled"])
        self.assertEqual(status["migration_state"], "source_started")
        self.assertEqual(status["migration_pending_reqs"], 2)
        self.assertEqual(
            status["active_request_migration_strategy"],
            "decode_to_decode_kv_transfer",
        )


if __name__ == "__main__":
    unittest.main()
