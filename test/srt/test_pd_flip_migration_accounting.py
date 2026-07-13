import types
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    PDFlipMigrationSourceFinishReq,
    PDFlipMigrationSourceStartReq,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)


class TestPDFlipMigrationAccounting(unittest.TestCase):
    @staticmethod
    def _scheduler_for_source_kv_manager():
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_flip_source_kv_manager = None
        scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            kv_manager=types.SimpleNamespace(
                kv_args=types.SimpleNamespace(page_size=1)
            )
        )
        scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(
            get_kvcache=lambda: types.SimpleNamespace(
                start_layer=0,
                end_layer=1,
                page_size=1,
                compression_ratios=None,
            )
        )
        scheduler.server_args = types.SimpleNamespace(
            disaggregation_ib_device=None
        )
        scheduler.ps = types.SimpleNamespace(gpu_id=0)
        scheduler.model_config = types.SimpleNamespace(
            get_total_num_kv_heads=lambda: 1
        )
        scheduler.transfer_backend = "fake"
        return scheduler

    def test_source_kv_manager_reuses_existing_prefill_manager(self):
        scheduler = self._scheduler_for_source_kv_manager()
        existing_manager = object()
        scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
            kv_manager=existing_manager
        )

        class ForbiddenManager:
            def __init__(self, *args, **kwargs):
                raise AssertionError("must not construct a second prefill manager")

        with patch(
            "sglang.srt.managers.scheduler.get_kv_class",
            return_value=ForbiddenManager,
        ):
            manager = Scheduler._pd_flip_get_source_kv_manager(scheduler)

        self.assertIs(manager, existing_manager)
        self.assertIs(scheduler.pd_flip_source_kv_manager, existing_manager)

    def test_source_kv_manager_prefill_queue_overrides_stale_cached_manager(self):
        scheduler = self._scheduler_for_source_kv_manager()
        stale_manager = object()
        existing_manager = object()
        scheduler.pd_flip_source_kv_manager = stale_manager
        scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
            kv_manager=existing_manager
        )

        manager = Scheduler._pd_flip_get_source_kv_manager(scheduler)

        self.assertIs(manager, existing_manager)
        self.assertIs(scheduler.pd_flip_source_kv_manager, existing_manager)

    def test_source_kv_manager_constructs_legacy_fallback_without_prefill_queue(self):
        scheduler = self._scheduler_for_source_kv_manager()
        constructed_manager = object()

        class Manager:
            def __new__(cls, *args, **kwargs):
                return constructed_manager

        with (
            patch(
                "sglang.srt.managers.scheduler.is_mla_backend",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.scheduler.get_kv_class",
                return_value=Manager,
            ),
        ):
            manager = Scheduler._pd_flip_get_source_kv_manager(scheduler)

        self.assertIs(manager, constructed_manager)
        self.assertIs(scheduler.pd_flip_source_kv_manager, constructed_manager)

    @staticmethod
    def _completed_delta_session(state, *, failed=False):
        return {
            "state": state,
            "source_entries": {
                "rid-1": {
                    "delta": {
                        "noop": False,
                        "transferred": not failed,
                        "failed": failed,
                    }
                }
            },
            "delta_transferred_rids": set() if failed else {"rid-1"},
            "delta_failed_rids": {"rid-1"} if failed else set(),
        }

    @staticmethod
    def _completed_target_session(state, *, delta=False):
        entry = {
            "phase": "transferred",
            "held": False,
            "decode_req": types.SimpleNamespace(),
        }
        session = {
            "state": state,
            "target_entries": {"rid-1": entry},
            "manifests": [{"rid": "rid-1"}],
            "transferred_rids": {"rid-1"},
            "failed_rids": set(),
            "fallback_required_rids": set(),
        }
        if delta:
            entry["delta"] = {
                "noop": False,
                "transferred": True,
                "failed": False,
            }
            session.update(
                delta_transferred_rids={"rid-1"},
                delta_failed_rids=set(),
            )
        return session

    def _checker_with_held_req(self, held_req):
        tree_cache = types.SimpleNamespace(
            protected_size=lambda: 0,
            supports_mamba=lambda: False,
        )
        pool_stats_observer = types.SimpleNamespace(
            session_held_tokens=lambda: 0,
            session_held_req_count=lambda: 0,
        )
        return SchedulerInvariantChecker(
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            disaggregation_mode=DisaggregationMode.DECODE,
            page_size=1,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=640996,
            server_args=types.SimpleNamespace(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=types.SimpleNamespace(),
            req_to_token_pool=types.SimpleNamespace(
                size=2048,
                pre_alloc_size=0,
                free_slots=[object()] * 2048,
            ),
            pool_stats_observer=pool_stats_observer,
            get_last_batch=lambda: types.SimpleNamespace(reqs=[]),
            get_running_batch=lambda: None,
            get_pd_flip_held_reqs=lambda: [held_req],
        )

    @staticmethod
    def _held_req():
        return types.SimpleNamespace(
            req_pool_idx=7,
            kv_committed_freed=False,
            kv_overallocated_freed=False,
            kv_allocated_len=7829,
            cache_protected_len=0,
            swa_evicted_seqlen=0,
        )

    @staticmethod
    def _waiting_req(
        rid="waiting-1", req_pool_idx=7, output_ids=None, kv_committed_len=4
    ):
        req = types.SimpleNamespace(
            rid=rid,
            req_pool_idx=req_pool_idx,
            origin_input_ids=[11, 12, 13],
            output_ids=[1] if output_ids is None else output_ids,
            kv_committed_len=kv_committed_len,
            pd_flip_defer_kv_release=False,
            pd_flip_force_kv_release=False,
            pd_flip_kv_release_deferred=False,
            pd_flip_deferred_kv_release_is_insert=False,
            to_finish=None,
        )
        req.finished = lambda: False
        return req

    def test_target_held_reqs_count_as_uncached_tokens(self):
        checker = self._checker_with_held_req(self._held_req())

        full_uncached, swa_uncached = checker._get_total_uncached_sizes()

        self.assertEqual(full_uncached, 7829)
        self.assertEqual(swa_uncached, 0)

    def test_idle_pool_check_accounts_for_target_held_reqs(self):
        held_req = types.SimpleNamespace(
            req_pool_idx=7,
            kv_committed_freed=False,
            kv_overallocated_freed=False,
            kv_allocated_len=7829,
            cache_protected_len=0,
            swa_evicted_seqlen=0,
        )
        checker = self._checker_with_held_req(held_req)
        pool_stats = types.SimpleNamespace(
            full_available_size=640996 - 7829,
            full_evictable_size=0,
        )

        has_leak, messages = checker._check_all_pools(pool_stats)

        self.assertFalse(has_leak, messages)

    def test_target_transferring_reqs_are_exposed_for_accounting(self):
        req = self._held_req()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_flip_migration_session = {
            "role": "target",
            "target_entries": {
                "rid-1": {
                    "phase": "transferring",
                    "held": False,
                    "decode_req": types.SimpleNamespace(req=req),
                }
            },
        }

        self.assertEqual(Scheduler._pd_flip_target_held_reqs(scheduler), [req])

    def test_req_pool_check_accounts_for_target_migration_req(self):
        held_req = self._held_req()
        checker = self._checker_with_held_req(held_req)
        checker.req_to_token_pool.free_slots = [object()] * 2047

        with patch(
            "sglang.srt.managers.scheduler_components.invariant_checker.raise_error_or_warn"
        ) as raise_or_warn:
            checker._check_req_pool()

        raise_or_warn.assert_not_called()

    def test_pd_metadata_buffers_grow_when_larger_role_initializes_later(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.req_to_metadata_buffer_idx_allocator = types.SimpleNamespace(
            size=4,
            available_size=lambda: 4,
        )
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(size=4)

        def fake_allocator(size):
            return types.SimpleNamespace(size=size, available_size=lambda: size)

        def fake_buffers(size, **kwargs):
            return types.SimpleNamespace(size=size, kwargs=kwargs)

        with (
            patch(
                "sglang.srt.managers.scheduler.ReqToMetadataIdxAllocator",
                side_effect=fake_allocator,
            ),
            patch(
                "sglang.srt.managers.scheduler.MetadataBuffers",
                side_effect=fake_buffers,
            ),
        ):
            Scheduler._init_pd_metadata_buffers(
                scheduler,
                buffer_size=16,
                hidden_size=16,
                hidden_states_dtype="float32",
            )

        self.assertEqual(scheduler.req_to_metadata_buffer_idx_allocator.size, 16)
        self.assertEqual(scheduler.disagg_metadata_buffers.size, 16)

    def test_decode_disaggregation_requests_metadata_growth_after_prefill_init(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.req_to_metadata_buffer_idx_allocator = types.SimpleNamespace(size=4)
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(size=4)
        scheduler.req_to_token_pool = types.SimpleNamespace(size=32)
        scheduler.token_to_kv_pool_allocator = types.SimpleNamespace()
        scheduler.spec_algorithm = types.SimpleNamespace(is_eagle=lambda: False)
        scheduler.attn_tp_cpu_group = object()
        scheduler.tree_cache = object()
        scheduler.ps = types.SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            gpu_id=0,
            pp_rank=0,
        )
        scheduler.server_args = types.SimpleNamespace(
            dp_size=1,
            disaggregation_bootstrap_port=8998,
            num_reserved_decode_tokens=1,
        )
        scheduler.max_total_num_tokens = 1024
        scheduler.transfer_backend = "fake"
        calls = []

        def fake_init(buffer_size, hidden_size, hidden_states_dtype):
            calls.append((buffer_size, hidden_size, hidden_states_dtype))

        with (
            patch.object(scheduler, "_init_pd_metadata_buffers", side_effect=fake_init),
            patch("sglang.srt.managers.scheduler.DecodeTransferQueue"),
            patch("sglang.srt.managers.scheduler.DecodePreallocQueue"),
        ):
            Scheduler._init_decode_disaggregation(
                scheduler,
                draft_token_to_kv_pool=None,
                model_config=types.SimpleNamespace(
                    spec_hidden_size=128,
                    dtype="float16",
                ),
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], (64, 16))
        self.assertEqual(str(calls[0][2]), "torch.float32")

    def test_hybrid_disaggregation_metadata_has_full_migration_floor(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.req_to_token_pool = types.SimpleNamespace(size=32)
        scheduler.max_running_requests = 2
        scheduler.spec_algorithm = types.SimpleNamespace(
            is_eagle=lambda: False,
            is_standalone=lambda: False,
        )
        calls = []

        def fake_init(buffer_size, hidden_size, hidden_states_dtype):
            calls.append((buffer_size, hidden_size, hidden_states_dtype))

        with patch.object(
            scheduler, "_init_pd_metadata_buffers", side_effect=fake_init
        ):
            Scheduler._init_hybrid_disaggregation_metadata(
                scheduler,
                model_config=types.SimpleNamespace(
                    spec_hidden_size=128,
                    dtype="float16",
                ),
            )

        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0][0], 1024)

    def test_classifies_waiting_reqs_with_committed_kv_as_migratable(self):
        scheduler = Scheduler.__new__(Scheduler)
        eligible = self._waiting_req("eligible")
        no_req_pool = self._waiting_req("no-pool", req_pool_idx=None)
        prompt_only = self._waiting_req("prompt-only", output_ids=[])
        no_kv = self._waiting_req("no-kv", kv_committed_len=0)
        finished = self._waiting_req("finished")
        finished.finished = lambda: True

        selected, skipped = Scheduler._pd_flip_classify_waiting_reqs(
            scheduler, [eligible, no_req_pool, prompt_only, no_kv, finished]
        )

        self.assertEqual(
            [(idx, req.rid) for idx, req in selected],
            [(0, "eligible"), (2, "prompt-only")],
        )
        self.assertEqual(
            [(item["rid"], item["reason"]) for item in skipped],
            [
                ("no-pool", "missing_req_pool_idx"),
                ("no-kv", "missing_committed_kv"),
                ("finished", "finished"),
            ],
        )

    def test_source_start_refuses_to_skip_live_waiting_reqs(self):
        scheduler = Scheduler.__new__(Scheduler)
        eligible = self._waiting_req("eligible")
        no_req_pool = self._waiting_req("no-pool", req_pool_idx=None)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.running_batch = types.SimpleNamespace(reqs=[])
        scheduler.waiting_queue = [eligible, no_req_pool]
        scheduler.server_args = types.SimpleNamespace(disaggregation_bootstrap_port=8998)
        scheduler.ps = types.SimpleNamespace(attn_dp_rank=0, dp_rank=0)
        scheduler._pd_flip_start_source_entries = (
            lambda reqs, manifests: self.fail("partial migration should not start")
        )

        output = Scheduler.start_pd_flip_migration_source(
            scheduler,
            PDFlipMigrationSourceStartReq(
                session_id="session-1",
                target_url="http://target",
                include_waiting=True,
            ),
        )

        self.assertFalse(output.success)
        self.assertIn("remaining waiting requests are not migratable", output.message)
        self.assertIn("no-pool", output.message)
        self.assertFalse(hasattr(scheduler, "pd_flip_migration_session"))

    def test_source_start_refuses_non_prefix_running_rids(self):
        scheduler = Scheduler.__new__(Scheduler)
        running_reqs = [self._waiting_req("running-1"), self._waiting_req("running-2")]
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.running_batch = types.SimpleNamespace(reqs=running_reqs)
        scheduler.waiting_queue = []
        scheduler.server_args = types.SimpleNamespace(disaggregation_bootstrap_port=8998)
        scheduler.ps = types.SimpleNamespace(attn_dp_rank=0, dp_rank=0)
        scheduler._pd_flip_start_source_entries = (
            lambda reqs, manifests: self.fail("non-prefix migration should not start")
        )

        output = Scheduler.start_pd_flip_migration_source(
            scheduler,
            PDFlipMigrationSourceStartReq(
                session_id="session-1",
                target_url="http://target",
                rids=["running-2"],
            ),
        )

        self.assertFalse(output.success)
        self.assertIn("running-batch prefix", output.message)
        self.assertFalse(hasattr(scheduler, "pd_flip_migration_session"))

    def test_source_start_includes_eligible_waiting_reqs_in_manifest(self):
        scheduler = Scheduler.__new__(Scheduler)
        waiting_req = self._waiting_req("waiting-eligible")
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.running_batch = types.SimpleNamespace(reqs=[])
        scheduler.waiting_queue = [waiting_req]
        scheduler.server_args = types.SimpleNamespace(disaggregation_bootstrap_port=8998)
        scheduler.ps = types.SimpleNamespace(attn_dp_rank=0, dp_rank=0)
        captured = {}

        def fake_start(reqs, manifests):
            captured["reqs"] = reqs
            captured["manifests"] = manifests
            return (
                {
                    "waiting-eligible": {
                        "req": waiting_req,
                        "manifest": manifests[0],
                        "metadata_index": -1,
                        "timing_debug": {},
                    }
                },
                "",
            )

        scheduler._pd_flip_start_source_entries = fake_start
        scheduler._pd_flip_source_pump_transfer = lambda session: None

        output = Scheduler.start_pd_flip_migration_source(
            scheduler,
            PDFlipMigrationSourceStartReq(
                session_id="session-1",
                target_url="http://target",
                include_waiting=True,
            ),
        )

        self.assertTrue(output.success)
        self.assertEqual([req.rid for req in captured["reqs"]], ["waiting-eligible"])
        self.assertEqual(
            captured["manifests"][0]["pd_flip_source_queue"], "waiting"
        )
        self.assertEqual(captured["manifests"][0]["pd_flip_waiting_queue_index"], 0)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_abort_restores_frozen_waiting_reqs(self):
        scheduler = Scheduler.__new__(Scheduler)
        before = self._waiting_req("before")
        waiting_req = self._waiting_req("waiting-1")
        after = self._waiting_req("after")
        scheduler.waiting_queue = [before, waiting_req, after]
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(bootstrap_room={})
        scheduler.req_to_metadata_buffer_idx_allocator = types.SimpleNamespace(
            free=lambda _: None
        )
        session = {
            "role": "source",
            "source_entries": {
                "waiting-1": {
                    "req": waiting_req,
                    "manifest": {"pd_flip_source_queue": "waiting"},
                    "metadata_index": -1,
                    "timing_debug": {},
                }
            },
            "source_waiting_reqs": [
                {"rid": "waiting-1", "req": waiting_req, "original_index": 1}
            ],
            "manifests": [{"rid": "waiting-1"}],
        }

        Scheduler._pd_flip_freeze_waiting_source_requests(scheduler, session)
        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["before", "after"])

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            Scheduler._pd_flip_abort_source_session(scheduler, session, "test abort")

        release.assert_not_called()
        self.assertEqual(
            [req.rid for req in scheduler.waiting_queue],
            ["before", "waiting-1", "after"],
        )

    def test_source_entries_allow_prompt_only_committed_kv(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hisparse = False
        scheduler.transfer_backend = "fake"
        scheduler.ps = types.SimpleNamespace(tp_rank=0, pp_rank=0)
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(bootstrap_room={})
        scheduler._pd_flip_can_use_real_migration = lambda: True
        scheduler._pd_flip_get_source_kv_manager = lambda: object()
        scheduler._pd_flip_local_bootstrap_addr = lambda _: "127.0.0.1:9999"
        scheduler._pd_flip_source_page_indices = lambda req, committed_len: [1, 2, 3]
        scheduler.req_to_metadata_buffer_idx_allocator = types.SimpleNamespace(
            alloc=lambda: 0,
            free=lambda _: None,
        )

        class Sender:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def init(self, page_count, metadata_index):
                self.page_count = page_count
                self.metadata_index = metadata_index

        req = self._waiting_req("prompt-only", output_ids=[], kv_committed_len=3)
        manifest = {
            "rid": req.rid,
            "kv_committed_len": req.kv_committed_len,
            "migration_bootstrap_room": 1,
            "pd_flip_source_queue": "waiting",
        }

        with patch("sglang.srt.managers.scheduler.get_kv_class", return_value=Sender):
            entries, message = Scheduler._pd_flip_start_source_entries(
                scheduler, [req], [manifest]
            )

        self.assertEqual(message, "")
        self.assertEqual(list(entries), ["prompt-only"])
        self.assertTrue(req.pd_flip_defer_kv_release)

    def test_finish_releases_waiting_source_without_finish_migrated(self):
        scheduler = Scheduler.__new__(Scheduler)
        waiting_req = self._waiting_req("waiting-1")
        scheduler.tree_cache = types.SimpleNamespace()
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(bootstrap_room={})
        scheduler.req_to_metadata_buffer_idx_allocator = types.SimpleNamespace(
            free=lambda _: None
        )
        session = {
            "source_entries": {
                "waiting-1": {
                    "req": waiting_req,
                    "manifest": {"pd_flip_source_queue": "waiting"},
                    "metadata_index": -1,
                    "timing_debug": {},
                }
            }
        }

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            Scheduler._pd_flip_release_source_requests(
                scheduler, session, {"waiting-1"}
            )

        release.assert_called_once_with(waiting_req, scheduler.tree_cache)
        self.assertIsNone(waiting_req.to_finish)
        self.assertFalse(waiting_req.pd_flip_defer_kv_release)

    def test_finish_refuses_running_req_that_advanced_after_snapshot(self):
        scheduler = Scheduler.__new__(Scheduler)
        running_req = self._waiting_req("running-1", kv_committed_len=5)
        session = {
            "role": "source",
            "dry_run": False,
            "manifests": [{"rid": "running-1"}],
            "transferred_rids": {"running-1"},
            "source_entries": {
                "running-1": {
                    "req": running_req,
                    "source_queue": "running",
                    "committed_len": 4,
                    "manifest": {
                        "rid": "running-1",
                        "pd_flip_source_queue": "running",
                    },
                    "timing_debug": {},
                }
            },
        }
        scheduler.pd_flip_migration_session = session
        scheduler._pd_flip_source_pump_transfer = lambda _: None
        scheduler._pd_flip_release_source_requests = (
            lambda _session, _released: self.fail(
                "source should not release stale running snapshot"
            )
        )

        output = Scheduler.finish_pd_flip_migration_source(
            scheduler,
            PDFlipMigrationSourceFinishReq(
                session_id="session-1", released_rids=["running-1"]
            ),
        )

        self.assertFalse(output.success)
        self.assertIn("advanced after migration snapshot", output.message)
        self.assertIn("running-1: 4->5", output.message)

    def test_source_start_cleans_partial_metadata_on_alloc_failure(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hisparse = False
        scheduler.transfer_backend = "fake"
        scheduler.ps = types.SimpleNamespace(tp_rank=0, pp_rank=0)
        scheduler.disagg_metadata_buffers = types.SimpleNamespace(bootstrap_room={})
        scheduler._pd_flip_can_use_real_migration = lambda: True
        scheduler._pd_flip_get_source_kv_manager = lambda: object()
        scheduler._pd_flip_local_bootstrap_addr = lambda _: "127.0.0.1:9999"
        scheduler._pd_flip_source_page_indices = lambda req, committed_len: [1]

        class Allocator:
            def __init__(self):
                self.values = [0, None]
                self.freed = []

            def alloc(self):
                return self.values.pop(0)

            def free(self, index):
                self.freed.append(index)

        allocator = Allocator()
        scheduler.req_to_metadata_buffer_idx_allocator = allocator

        senders = []

        class Sender:
            def __init__(self, **kwargs):
                self.aborted = False
                senders.append(self)

            def init(self, page_count, metadata_index):
                self.page_count = page_count
                self.metadata_index = metadata_index

            def abort(self):
                self.aborted = True

        reqs = [self._waiting_req("r1"), self._waiting_req("r2")]
        manifests = [
            {
                "rid": req.rid,
                "kv_committed_len": req.kv_committed_len,
                "migration_bootstrap_room": 1,
                "pd_flip_source_queue": "running",
            }
            for req in reqs
        ]

        with patch("sglang.srt.managers.scheduler.get_kv_class", return_value=Sender):
            entries, message = Scheduler._pd_flip_start_source_entries(
                scheduler, reqs, manifests
            )

        self.assertEqual(entries, {})
        self.assertEqual(message, "no metadata buffer available for source migration")
        self.assertEqual(allocator.freed, [0])
        self.assertTrue(senders[0].aborted)
        self.assertFalse(reqs[0].pd_flip_defer_kv_release)

    def test_delta_pump_preserves_released_source_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_delta_session("source_released")

        Scheduler._pd_flip_source_pump_delta_transfer(scheduler, session)

        self.assertEqual(session["state"], "source_released")
        self.assertEqual(session["delta_transferred_reqs"], 1)
        self.assertEqual(session["delta_pending_reqs"], 0)

    def test_delta_pump_preserves_aborted_source_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_delta_session("source_aborted")

        Scheduler._pd_flip_source_pump_delta_transfer(scheduler, session)

        self.assertEqual(session["state"], "source_aborted")
        self.assertEqual(session["delta_transferred_reqs"], 1)
        self.assertEqual(session["delta_pending_reqs"], 0)

    def test_delta_pump_completes_nonterminal_source_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_delta_session("source_delta_started")

        Scheduler._pd_flip_source_pump_delta_transfer(scheduler, session)

        self.assertEqual(session["state"], "source_delta_transferred")

    def test_delta_pump_fails_nonterminal_source_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_delta_session(
            "source_delta_started", failed=True
        )

        Scheduler._pd_flip_source_pump_delta_transfer(scheduler, session)

        self.assertEqual(session["state"], "source_failed")
        self.assertEqual(session["failed_reqs"], 1)

    def test_target_transfer_pump_preserves_active_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_target_session("active")

        Scheduler._pd_flip_target_pump_transfer(scheduler, session)

        self.assertEqual(session["state"], "active")
        self.assertEqual(session["transferred_reqs"], 1)
        self.assertEqual(session["pending_reqs"], 0)

    def test_target_transfer_pump_preserves_aborted_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_target_session("target_aborted")

        Scheduler._pd_flip_target_pump_transfer(scheduler, session)

        self.assertEqual(session["state"], "target_aborted")
        self.assertEqual(session["transferred_reqs"], 1)
        self.assertEqual(session["pending_reqs"], 0)

    def test_target_transfer_pump_completes_nonterminal_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_target_session("target_started")

        Scheduler._pd_flip_target_pump_transfer(scheduler, session)

        self.assertEqual(session["state"], "target_transferred")

    def test_target_delta_pump_preserves_active_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_target_session("active", delta=True)

        Scheduler._pd_flip_target_pump_delta_transfer(scheduler, session)

        self.assertEqual(session["state"], "active")
        self.assertEqual(session["delta_transferred_reqs"], 1)
        self.assertEqual(session["delta_pending_reqs"], 0)

    def test_target_delta_pump_preserves_aborted_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_target_session("target_aborted", delta=True)

        Scheduler._pd_flip_target_pump_delta_transfer(scheduler, session)

        self.assertEqual(session["state"], "target_aborted")
        self.assertEqual(session["delta_transferred_reqs"], 1)
        self.assertEqual(session["delta_pending_reqs"], 0)

    def test_target_delta_pump_completes_nonterminal_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        session = self._completed_target_session("target_delta_started", delta=True)

        Scheduler._pd_flip_target_pump_delta_transfer(scheduler, session)

        self.assertEqual(session["state"], "target_delta_transferred")


if __name__ == "__main__":
    unittest.main()
