"""EPLB scheduling and recorder state specific to asynchronous Ascend migration."""

import logging
import math
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    _convert_global_physical_count_to_logical_count,
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
    _StatAccumulator,
    get_global_expert_distribution_recorder,
)
from sglang.srt.hardware_backend.npu.eplb.coordinator import (
    ControlPoint,
    Phase,
    RebalanceAborted,
    UnsafeRebalanceFailure,
)
from sglang.srt.hardware_backend.npu.eplb.placement import PlanningSpec, RecordedWindow
from sglang.srt.hardware_backend.npu.eplb.startup import create_npu_async_eplb
from sglang.srt.runtime_context import get_exec
from torch.profiler import record_function

logger = logging.getLogger(__name__)

SnapshotVote = Callable[[ControlPoint, bool, bool], tuple[bool, bool]]


class _AsyncStatAccumulator(_StatAccumulator):
    """Keep the routing row used by samples recorded before a layer moves."""

    def __init__(self, metadata, rank):
        super().__init__(metadata, rank)
        self.window_samples = 0
        self._previous_layouts = {}

    def append(self, forward_pass_id, gatherer_key, single_pass_data, outputs):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)
        self.window_samples += 1

    def before_layer_install(self, layer):
        # Each layer moves once per epoch. Reset discards its history.
        if self.window_samples:
            old_row = self._expert_location_metadata.physical_to_logical_map_cpu[layer]
            self._previous_layouts[layer] = (self.window_samples, old_row.clone())

    def reset(self):
        super().reset()
        self.window_samples = 0
        self._previous_layouts.clear()

    def _finish_snapshot(self, vote: SnapshotVote, *, snapshot_failed: bool) -> bool:
        """Clear the window everywhere and report whether planning can start."""
        local_cleared = True
        try:
            self.reset()
        except Exception:
            logger.exception("Async EPLB could not reset its recorder window")
            local_cleared = False
        all_cleared, any_failed = vote(
            ControlPoint.SNAPSHOT_COMPLETE, local_cleared, snapshot_failed
        )
        if not all_cleared:
            raise UnsafeRebalanceFailure(
                "Async EPLB could not clear the recorder on all ranks"
            )
        return not any_failed

    def _vote_on_utilization(
        self, rebalance_needed: Callable[[float], bool], vote: SnapshotVote
    ) -> tuple[bool, bool]:
        if not self._enable or math.isclose(
            get_exec().moe.eplb_min_rebalancing_utilization_threshold, 1.0
        ):
            return True, False

        # Utilization history lives on rank 0. Share its decision on Gloo.
        needed, failed = True, False
        if self._rank == 0:
            try:
                utilization = self._history.mean().get(self.window_sizes[-1], 0)
                needed = rebalance_needed(utilization)
            except Exception:
                logger.exception("Async EPLB could not check window utilization")
                failed = True
        return vote(ControlPoint.SNAPSHOT_ADMISSION, needed, failed)

    def _prepare_snapshot(
        self, counts: torch.Tensor, spec: PlanningSpec, rank: int
    ) -> RecordedWindow:
        slot_counts = torch.empty_like(counts) if rank == 0 else None
        source_totals = active_sources = None
        if source_groups := spec.source_groups:
            ranks_per_group = (
                dist.get_world_size(self._communication_group) // source_groups
            )
            source = rank // ranks_per_group
            source_totals = counts.new_zeros(
                (source_groups, *counts.shape[1:]), dtype=torch.int64
            )
            torch.sum(counts, dim=0, out=source_totals[source])
        if spec.collect_active_sources:
            # Count the ranks with local assignments before counts is reduced.
            active_sources = counts.amax(dim=-1).gt(0).to(torch.int32)
        return RecordedWindow(slot_counts, source_totals, active_sources)

    def take_rebalance_snapshot(
        self,
        spec: PlanningSpec,
        rebalance_needed: Callable[[float], bool],
        vote: SnapshotVote,
    ) -> RecordedWindow | None:
        """Return a consumed window on all ranks, or None to skip the plan."""
        needed, admission_failed = self._vote_on_utilization(rebalance_needed, vote)
        if admission_failed or not needed:
            self._finish_snapshot(vote, snapshot_failed=admission_failed)
            return None

        group = self._communication_group
        rank = dist.get_rank(group)
        window = None
        prepare_failed = False
        try:
            # A completed migration resets the window before the next epoch.
            assert not self._previous_layouts
            counts = self._global_physical_count_of_buffered_step.get_all()
            window = self._prepare_snapshot(counts, spec, rank)
        except Exception:
            logger.exception("Async EPLB could not prepare a recorder snapshot")
            prepare_failed = True
        _, any_prepare_failed = vote(
            ControlPoint.SNAPSHOT_PREPARE, not prepare_failed, prepare_failed
        )
        if any_prepare_failed:
            self._finish_snapshot(vote, snapshot_failed=True)
            return None

        assert window is not None
        # Failures inside TP collectives propagate to the serving worker.
        if window.source_totals is not None:
            dist.reduce(window.source_totals, dst=0, group=group)
        if window.active_sources is not None:
            dist.reduce(window.active_sources, dst=0, group=group)
        dist.all_reduce(counts, group=group)

        copy_failed = False
        try:
            if window.slot_counts is not None:
                window.slot_counts.copy_(counts)
        except Exception:
            logger.exception("Async EPLB could not copy the reduced recorder window")
            copy_failed = True
        # Counts now contain the group total. No rank may reuse this window.
        if not self._finish_snapshot(vote, snapshot_failed=copy_failed):
            return None
        return window if rank == 0 else RecordedWindow()

    def _logical_counts(self, physical_counts):
        logical = super()._logical_counts(physical_counts)
        if not self._previous_layouts:
            return logical
        # A manual dump can fall between layer installs. Correct only samples
        # still in the ring that predate each install, including wraparound.
        capacity = physical_counts.size(0)
        cursor = self._global_physical_count_of_buffered_step._curr_index
        for layer, (at_install, old_row) in self._previous_layouts.items():
            newer = self.window_samples - at_install
            retained = min(at_install, capacity - newer)
            if retained <= 0:
                continue
            first = (cursor - newer - retained) % capacity
            end = first + retained
            old_map = old_row.unsqueeze(0).to(physical_counts.device)
            for lo, hi in ((first, min(end, capacity)), (0, end - capacity)):
                if lo < hi:
                    rows = physical_counts[lo:hi, layer : layer + 1]
                    logical[lo:hi, layer : layer + 1].copy_(
                        _convert_global_physical_count_to_logical_count(
                            rows, 1, logical.size(-1), old_map
                        )
                    )
        return logical


class AsyncEplbRecorder(_ExpertDistributionRecorderReal):
    def __init__(self, metadata, rank):
        if metadata is None:
            raise ValueError("Async EPLB requires expert location metadata")
        super().__init__(metadata, rank, accumulator_factory=_AsyncStatAccumulator)
        # Only TopK counts describe the source rank, before dispatch.
        self.records_source_counts = all(
            isinstance(gatherer, _SelectExpertsSinglePassGatherer)
            for gatherer in self._single_pass_gatherers.values()
        )

    @property
    def window_samples(self):
        return self._accumulator.window_samples

    def before_layer_install(self, layer):
        self._accumulator.before_layer_install(layer)

    def mark_statistics_group_warmed(self):
        # NPU startup already warmed TP and emptied the allocator cache.
        self._accumulator._first_dump = False

    def snapshot_for_rebalance(self, spec, rebalance_needed, vote):
        with record_function("npu_eplb.snapshot"):
            # Per-pass gatherers reset at the next forward. The accumulator
            # owns the recorder window and clears it after taking a snapshot.
            return self._accumulator.take_rebalance_snapshot(
                spec, rebalance_needed, vote
            )


def make_prefill_release_policy(scheduler):
    """Select early result processing for the first unified EPLB epoch."""
    fallback = scheduler.should_process_last_batch_early
    manager = scheduler.tp_worker.model_runner.eplb_manager
    coordinator = manager._async_coordinator if manager is not None else None
    if coordinator is None or not coordinator.runtime.use_prefill_admission:
        return fallback

    def should_process(last_batch):
        if last_batch is not None:
            is_extend = (
                last_batch.is_extend_in_batch
                if scheduler.require_mlp_sync
                else last_batch.forward_mode.is_extend()
            )
            if is_extend and manager.should_release_first_prefill(
                has_pending_prefill=(
                    scheduler.chunked_req is not None or bool(scheduler.waiting_queue)
                ),
            ):
                return True
        return fallback(last_batch)

    return should_process


class NpuAsyncEPLBManager(EPLBManager):
    def __init__(self, **kwargs):
        self._async_coordinator = None
        super().__init__(**kwargs)

    def init_after_model_load(self, startup_weight_load):
        model = self._get_model()
        self._async_coordinator = create_npu_async_eplb(
            self._model_config, model, startup_weight_load=startup_weight_load
        )
        get_global_expert_distribution_recorder().mark_statistics_group_warmed()
        if getattr(model, "_derived_weight_cache_error", None) is None:
            model._derived_weight_cache_error = (
                "Online weight updates are unsupported with async EPLB: "
                "the migration worker holds views of the original expert storage."
            )
        if getattr(model, "_weight_export_error", None) is None:
            model._weight_export_error = (
                "Cannot export weights with async EPLB: physical expert slots can "
                "move, and the weight export does not include their routing table."
            )

    def reset_generator(self):
        if self._async_coordinator is not None:
            self._async_coordinator.reset()
            get_global_expert_distribution_recorder()._reset()
        super().reset_generator()

    def close(self) -> None:
        coordinator = self._async_coordinator
        if coordinator is not None:
            phase = coordinator.phase
            coordinator.close()
            logger.info(
                "Async EPLB drained at shutdown: epoch=%d phase=%s",
                coordinator.epoch,
                phase.name,
            )

    def should_release_first_prefill(self, *, has_pending_prefill: bool) -> bool:
        coordinator = self._async_coordinator
        if coordinator is None or coordinator.epoch != 1 or not coordinator.active:
            return False
        # Called at the same extend boundary on every rank. Keep scheduling
        # prefill before processing results while any rank has prompts pending.
        all_finished, _ = coordinator.control.vote(
            coordinator.epoch,
            coordinator.phase,
            ControlPoint.PREFILL_ADMISSION,
            not has_pending_prefill,
            False,
        )
        return all_finished

    def rebalance(self):
        if self._rebalance_disabled_reason is not None:
            yield from super().rebalance()
            return
        coordinator = self._async_coordinator
        if coordinator is None:
            raise RuntimeError(
                "Async EPLB resources were not initialized before KV sizing"
            )
        recorder = get_global_expert_distribution_recorder()
        enough_samples = (
            recorder.window_samples
            >= get_exec().moe.expert_distribution_recorder_buffer_size
        )

        def vote(point, ready, error):
            return coordinator.control.vote(
                coordinator.epoch, Phase.IDLE, point, ready, error
            )

        ready, _ = vote(ControlPoint.RECORDER_WINDOW, enough_samples, False)
        if not ready:
            return
        start = time.perf_counter()
        snapshot = recorder.snapshot_for_rebalance(
            coordinator.runtime.spec, self._check_rebalance_needed, vote
        )
        snapshot_seconds = time.perf_counter() - start
        if snapshot is None:
            return
        coordinator.control.wait_seconds = 0.0
        coordinator.start(coordinator.runtime.planner(snapshot))
        del snapshot
        try:
            while True:
                yield
                if coordinator.poll():
                    break
        except RebalanceAborted:
            self._rebalance_disabled_reason = "asynchronous planning failed"
            logger.exception("Async EPLB drained; disabling further rebalances")
        # The snapshot already started a fresh window. It is still valid if the
        # planner chose no migrations or failed before the first install.
        if coordinator.layers_installed:
            recorder._reset()
        runtime = coordinator.runtime
        logger.info(
            "Async EPLB epoch=%d layers=%d elapsed=%.3fs control_wait=%.6fs "
            "stage=%.6fs install_enqueue=%.6fs snapshot=%.6fs "
            "interleaved_layers=%d submission_wait=%.6fs",
            coordinator.epoch,
            coordinator.layers_installed,
            time.perf_counter() - start,
            coordinator.control.wait_seconds,
            runtime.stage_seconds,
            runtime.install_enqueue_seconds,
            snapshot_seconds,
            runtime.interleaved_layers,
            coordinator.submission_wait_seconds,
        )
