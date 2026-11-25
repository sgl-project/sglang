"""EPLB scheduling and recorder state specific to asynchronous Ascend migration."""

import logging
import time

import torch.distributed as dist
from torch.profiler import record_function

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderReal,
    _StatAccumulator,
    _convert_global_physical_count_to_logical_count,
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.hardware_backend.npu.eplb_async import create_npu_async_eplb
from sglang.srt.hardware_backend.npu.eplb_coordinator import (
    ControlPoint,
    Phase,
    RebalanceAborted,
)
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


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

    def snapshot_physical_counts(self):
        # A completed migration resets the window before the next epoch.
        assert not self._previous_layouts
        counts = self._global_physical_count_of_buffered_step.get_all().clone()
        dist.all_reduce(counts, group=self._communication_group)
        return counts, self._get_global_average_utilization_rate()

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

    @property
    def window_samples(self):
        return self._accumulator.window_samples

    def before_layer_install(self, layer):
        self._accumulator.before_layer_install(layer)

    def mark_statistics_group_warmed(self):
        # NPU startup already warmed TP and emptied the allocator cache.
        self._accumulator._first_dump = False

    def snapshot_for_rebalance(self, *, physical: bool):
        with record_function("npu_eplb.snapshot"):
            if physical:
                snapshot = self._accumulator.snapshot_physical_counts()
                self._reset()
                return snapshot
            snapshot = self.dump_record(output_mode="object")
            return snapshot["logical_count"], snapshot[
                "average_utilization_rate_over_window"
            ]


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
        if startup_weight_load is not None:
            raise ValueError(
                "Async EPLB requires completed weight loading before migration setup"
            )
        model = self._get_model()
        self._async_coordinator = create_npu_async_eplb(
            self._model_config,
            model.routed_experts_weights_of_layer,
            get_global_expert_location_metadata(),
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
        ready, _ = coordinator.control.vote(
            coordinator.epoch,
            Phase.IDLE,
            ControlPoint.RECORDER_WINDOW,
            enough_samples,
            False,
        )
        if not ready:
            return
        start = time.perf_counter()
        counts, utilization = recorder.snapshot_for_rebalance(
            physical=coordinator.runtime.spec.physical_count_mode is not None
        )
        snapshot_seconds = time.perf_counter() - start
        if not self._check_rebalance_needed(utilization):
            return
        coordinator.control.wait_seconds = 0.0
        coordinator.start(coordinator.runtime.planner(counts))
        del counts
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
