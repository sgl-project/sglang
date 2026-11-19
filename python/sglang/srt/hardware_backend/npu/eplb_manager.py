"""EPLB scheduling and recorder state specific to asynchronous Ascend migration."""

import logging
import time

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderReal,
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.hardware_backend.npu.eplb_async import create_npu_async_eplb
from sglang.srt.hardware_backend.npu.eplb_coordinator import Phase, RebalanceAborted
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


class AsyncEplbRecorder(_ExpertDistributionRecorderReal):
    """Count the target forwards in the current recorder window."""

    window_samples = 0

    def _on_forward_pass_end(self, forward_pass_id, outputs):
        super()._on_forward_pass_end(forward_pass_id, outputs)
        if self._recording and self._record_current_pass:
            self.window_samples += 1

    def _reset(self):
        super()._reset()
        self.window_samples = 0


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
        if getattr(model, "_derived_weight_cache_error", None) is None:
            model._derived_weight_cache_error = (
                "Online weight updates are unsupported with async EPLB: "
                "the migration worker holds views of the original expert storage."
            )

    def reset_generator(self):
        if self._async_coordinator is not None:
            self._async_coordinator.reset()
            get_global_expert_distribution_recorder()._reset()
        super().reset_generator()

    def _rebalance(self):
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
            coordinator.epoch, Phase.IDLE, -2, enough_samples, False
        )
        if not ready:
            return
        start = time.perf_counter()
        snapshot = recorder.dump_record(output_mode="object")
        if not self._check_rebalance_needed(
            snapshot["average_utilization_rate_over_window"]
        ):
            return
        coordinator.control.wait_seconds = 0.0
        coordinator.start(coordinator.runtime.planner(snapshot["logical_count"]))
        try:
            while True:
                yield
                if coordinator.poll():
                    break
        except RebalanceAborted:
            self._rebalance_disabled_reason = "asynchronous planning failed"
            logger.exception("Async EPLB drained; disabling further rebalances")
        # A partial/mixed-layout statistics window must never feed the next plan.
        recorder._reset()
        runtime = coordinator.runtime
        logger.info(
            "Async EPLB epoch=%d layers=%d elapsed=%.3fs control_wait=%.6fs "
            "stage=%.6fs install_enqueue=%.6fs",
            coordinator.epoch,
            coordinator.layers_installed,
            time.perf_counter() - start,
            coordinator.control.wait_seconds,
            runtime.stage_seconds,
            runtime.install_enqueue_seconds,
        )
