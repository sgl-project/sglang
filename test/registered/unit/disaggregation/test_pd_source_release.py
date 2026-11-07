"""Native retirement entry points must preserve source KV and metadata ownership."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.source_release import SourceReleaseTracker
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    disagg_transfers_in_flight,
    validate_source_release_roster,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _Req:
    rid = "source-release"
    bootstrap_room = 42
    metadata_buffer_index = 3
    finished_reason = None


class TestPDSourceRelease(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(MooncakeKVManager)
        self.manager.source_release_tracker = SourceReleaseTracker()
        self.manager._staging_outstanding = {42: 1}
        self.manager._deferred_ack_poisoned_rooms = set()
        self.manager._deferred_ack_targets = {}
        self.manager.request_status = {42: KVPoll.Failed}
        self.manager.req_to_decode_prefix_len = {}
        self.manager.transfer_infos = {}
        self.manager.failure_lock = threading.Lock()
        self.manager.failure_records = {}
        self.manager._staging_ctx = None
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = self.manager
        sender.bootstrap_room = 42
        sender.conclude_state = None
        self.req = _Req()
        self.req.disagg_kv_sender = sender
        self.req.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.retirer = object.__new__(RequestRetirer)
        self.retirer.tree_cache = MagicMock()
        self.retirer.hisparse_coordinator = None
        self.allocator = MagicMock()
        self.unlock = patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ).start()
        self.discard = patch(
            "sglang.srt.managers.scheduler_components.request_retirer.discard_kv_cache_backup"
        ).start()
        self.addCleanup(patch.stopall)

    def retire(self):
        self.retirer.release(self.req, checkpoint=False)
        maybe_release_metadata_buffer(self.req, self.allocator)

    def collect(self, reduce_flags=lambda flags: flags):
        self.manager.source_release_tracker.collect(
            self.manager.source_release_is_safe, reduce_flags
        )

    def test_native_kv_lock_and_metadata_survive_until_drain_then_release_once(self):
        self.retire()
        self.collect()
        self.unlock.assert_not_called()
        self.discard.assert_not_called()
        self.allocator.free.assert_not_called()
        self.assertEqual(list(self.manager.source_release_tracker.requests), [self.req])
        self.assertTrue(self.req.kv.holds_kv)
        self.manager._staging_outstanding.clear()
        self.collect()
        self.unlock.assert_called_once_with(
            self.req, self.retirer.tree_cache, checkpoint=False
        )
        self.allocator.free.assert_called_once_with(3)
        self.assertEqual(self.req.metadata_buffer_index, -1)
        self.assertFalse(self.manager.source_release_tracker)
        self.collect()
        self.assertEqual(self.unlock.call_count, 1)

    def test_locally_drained_rank_still_registers_and_waits_for_peer(self):
        self.manager._staging_outstanding.clear()
        self.retire()
        self.collect(lambda flags: [0 for _ in flags])
        self.unlock.assert_not_called()
        self.allocator.free.assert_not_called()
        self.collect()
        self.unlock.assert_called_once()

    def test_unknown_legacy_batch_never_expires_or_permits_reset_or_deregistration(
        self,
    ):
        self.retire()
        self.manager._staging_outstanding.clear()
        self.manager.poison_deferred_ack_room(42)
        self.collect()
        self.collect()
        self.unlock.assert_not_called()
        self.allocator.free.assert_not_called()
        self.manager.request_status.clear()
        with self.assertRaisesRegex(RuntimeError, "still owned"):
            self.manager.require_transfer_idle()
        with self.assertRaisesRegex(RuntimeError, "still owned"):
            self.manager.deregister_buffer_to_engine()
        with self.assertRaisesRegex(RuntimeError, "still owned"):
            self.manager.teardown()

    def test_retained_room_cannot_start_a_new_lifecycle(self):
        self.retire()
        self.manager.request_status.clear()
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            self.manager.update_status(42, KVPoll.Bootstrapping)
        other = _Req()
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            self.manager.source_release_tracker.defer(other, "kv", lambda: None)

    def test_worker_bookkeeping_must_finish_before_source_release_consensus(self):
        self.retire()
        self.manager._staging_outstanding[42] = 0
        self.manager._source_worker_busy_rooms = {42: 1}
        self.collect()
        self.unlock.assert_not_called()
        self.manager._source_worker_busy_rooms.clear()
        self.manager._staging_outstanding.clear()
        self.collect()
        self.unlock.assert_called_once()

    def test_unsafe_feature_off_is_rejected_before_engine_or_registration(self):
        prefix = "sglang.srt.disaggregation.mooncake.conn"

        def initialize(manager, *args):
            manager.enable_deferred_decode_kv_release = False

        with (
            patch(
                f"{prefix}.check_mooncake_custom_mem_pool_enabled",
                return_value=(False, None),
            ),
            patch(f"{prefix}._validate_efa_allocator_compatibility"),
            patch(f"{prefix}.CommonKVManager.__init__", initialize),
            patch.object(MooncakeKVManager, "init_engine") as engine,
            patch.object(MooncakeKVManager, "register_buffer_to_engine") as register,
        ):
            with self.assertRaisesRegex(ValueError, "requires deferred"):
                MooncakeKVManager(None, None, None)
        engine.assert_not_called()
        register.assert_not_called()

    def test_partial_native_release_is_quarantined_and_never_retried(self):
        self.retire()
        self.manager._staging_outstanding.clear()
        self.unlock.side_effect = RuntimeError("possibly partial native decrement")
        with self.assertRaisesRegex(RuntimeError, "restart required"):
            self.collect()
        self.collect()
        self.unlock.assert_called_once()
        self.allocator.free.assert_not_called()
        self.assertTrue(self.manager.source_release_tracker)

    def test_peer_release_failure_blocks_future_release_even_after_local_success(self):
        self.retire()
        self.manager._staging_outstanding.clear()
        results = iter(([1], [0]))
        with self.assertRaisesRegex(RuntimeError, "restart required"):
            self.collect(lambda flags: next(results))
        self.collect()
        self.unlock.assert_called_once()
        self.assertTrue(self.manager.source_release_tracker)

    def test_foreign_thread_cannot_drop_native_request_pins(self):
        self.retire()
        failures = []

        def collect():
            try:
                self.collect()
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=collect)
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        self.unlock.assert_not_called()

    def test_chunk_abort_sender_cleanup_cannot_free_metadata_before_drain(self):
        self.req.disagg_kv_sender.abort = MagicMock()
        scheduler = SimpleNamespace(
            clear_pending_chunk_send=MagicMock(),
            req_to_metadata_buffer_idx_allocator=self.allocator,
        )
        SchedulerDisaggregationPrefillMixin.release_prefill_sender(
            scheduler, self.req, abort_sender=True
        )
        self.req.disagg_kv_sender.abort.assert_called_once()
        self.assertNotIn(42, self.manager.request_status)
        self.retirer.release(self.req, checkpoint=False)
        self.collect()
        self.unlock.assert_not_called()
        self.allocator.free.assert_not_called()

    def test_metadata_only_optimistic_rebootstrap_never_frees_the_new_slot(self):
        self.req.pending_bootstrap = True
        self.req.disagg_kv_sender.aux_index = None
        self.manager._staging_outstanding.clear()
        maybe_release_metadata_buffer(self.req, self.allocator)
        self.allocator.free.assert_called_once_with(3)
        self.assertEqual(self.req.metadata_buffer_index, -1)
        self.assertFalse(self.manager.source_release_tracker)
        self.allocator.available_size.return_value = 1
        self.allocator.alloc.return_value = 8
        queue = SimpleNamespace(req_to_metadata_buffer_idx_allocator=self.allocator)
        self.assertTrue(PrefillBootstrapQueue.ensure_metadata_buffer(queue, self.req))
        self.collect()
        self.allocator.free.assert_called_once_with(3)
        self.assertEqual(self.req.metadata_buffer_index, 8)
        # If a later owner illegally overwrites an exposed held slot, the
        # callback must quarantine rather than free that new owner's slot.
        self.req.pending_bootstrap = False
        self.req.disagg_kv_sender.aux_index = 8
        self.manager._staging_outstanding[42] = 1
        maybe_release_metadata_buffer(self.req, self.allocator)
        self.req.metadata_buffer_index = 9
        self.manager._staging_outstanding.clear()
        with self.assertRaisesRegex(RuntimeError, "restart required"):
            self.collect()
        self.allocator.free.assert_called_once_with(3)
        self.assertEqual(self.req.metadata_buffer_index, 9)

    def test_release_roster_uses_only_native_attention_groups_and_exact_order(self):
        tp, cp = object(), object()
        seen = []

        def gather(output, rooms, *, group):
            seen.append(group)
            output[:] = [rooms, rooms]

        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_gather_object", side_effect=gather),
        ):
            validate_source_release_roster((3, 42), tp, cp)
        self.assertEqual(seen, [tp, cp])
        with self.assertRaisesRegex(RuntimeError, "explicit attention"):
            validate_source_release_roster((), None, cp)

    def test_release_roster_mismatch_fails_before_native_request_unlock(self):
        self.retire()

        def gather(output, rooms, *, group):
            output[:] = [rooms, ((), False)]

        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_gather_object", side_effect=gather),
            self.assertRaisesRegex(RuntimeError, "rosters differ"),
        ):
            validate_source_release_roster((42,), object(), object())
        self.unlock.assert_not_called()

    def test_roster_failure_crosses_both_stages_of_a_two_by_two_tp_cp_grid(self):
        rosters = [(42,), (43,), (42,), (42,)]
        tp_rows = [(0, 1), (2, 3)]
        cp_columns = [(0, 2), (1, 3)]
        stages = []
        for rank in range(4):
            tp = tp_rows[rank // 2]
            cp = cp_columns[rank % 2]

            def gather(output, payload, *, group):
                stages.append((rank, group))
                output[:] = [
                    (
                        rosters[peer],
                        False
                        if group is tp
                        else rosters[(peer // 2) * 2] != rosters[(peer // 2) * 2 + 1],
                    )
                    for peer in group
                ]

            with (
                patch("torch.distributed.get_world_size", return_value=2),
                patch("torch.distributed.all_gather_object", side_effect=gather),
                self.assertRaisesRegex(RuntimeError, "rosters differ"),
            ):
                validate_source_release_roster(rosters[rank], tp, cp)
            self.assertEqual(stages[-2:], [(rank, tp), (rank, cp)])

    def test_source_quarantine_blocks_compaction_flush_and_pause_before_free(self):
        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        self.retire()
        tracker = self.manager.source_release_tracker
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(is_empty=lambda: True, reqs=[]),
            dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
            last_batch=None,
            enable_overlap=False,
            _pp_microbatches_drained=lambda: True,
            request_lists=lambda: {tracker.requests.kind: tracker.requests},
            request_receiver=SimpleNamespace(has_pending=lambda: False),
            _engine_paused=True,
            waiting_queue=[],
            uocr_runtime=None,
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=self.manager),
            disagg_prefill_transfer_queue=[],
            disagg_prefill_pending_chunk_rids=set(),
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            enable_unified_cache_external_linker=False,
        )
        scheduler.is_fully_idle = lambda **kwargs: Scheduler.is_fully_idle(
            scheduler, **kwargs
        )
        self.assertFalse(scheduler.is_fully_idle(ignore_waiting=True))
        self.assertTrue(disagg_transfers_in_flight(scheduler)())
        self.assertFalse(Scheduler.flush_cache(scheduler))
        updater = object.__new__(SchedulerWeightUpdaterManager)
        updater.scheduler = scheduler
        updater.is_fully_idle = scheduler.is_fully_idle
        updater.memory_saver_adapter = MagicMock()
        updater.flush_cache = MagicMock()
        result = updater.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
        self.assertFalse(result.success)
        updater.memory_saver_adapter.pause.assert_not_called()
        updater.flush_cache.assert_not_called()

    def test_decode_quarantine_blocks_compaction_after_receiving_queue_is_empty(self):
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            disagg_decode_prealloc_queue=SimpleNamespace(
                has_device_destinations=False, has_published_destinations=False
            ),
            disagg_decode_transfer_queue=SimpleNamespace(
                has_pending_deferred_releases=lambda: True
            ),
            decode_offload_manager=None,
        )
        self.assertTrue(disagg_transfers_in_flight(scheduler)())


if __name__ == "__main__":
    unittest.main()
