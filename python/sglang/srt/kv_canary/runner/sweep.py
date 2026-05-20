from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.plan_input import (
    AliveReqSnapshot,
    build_plan_input_radix_sweep,
    build_plan_input_running_sweep,
)
from sglang.srt.kv_canary.runner.per_forward import _is_sweep_tag

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner

logger = logging.getLogger(__name__)


class SweepOrchestrator:
    def __init__(
        self,
        *,
        owner: "CanaryRunner",
        sweep_verify_capacity: int,
    ) -> None:
        self._owner = owner
        device = owner._device
        self._verify_plan_sweep_running = VerifyPlan.allocate(
            verify_capacity=max(1, sweep_verify_capacity), device=device
        )
        self._verify_plan_sweep_radix = VerifyPlan.allocate(
            verify_capacity=max(1, sweep_verify_capacity), device=device
        )
        self._write_plan_sweep = WritePlan.allocate(write_req_capacity=1, device=device)
        self._alive_reqs_snapshot: Optional[AliveReqSnapshot] = None
        self._alive_reqs_fallback_warned: bool = False

    def set_alive_reqs_snapshot(self, snapshot: Optional[AliveReqSnapshot]) -> None:
        self._alive_reqs_snapshot = snapshot

    def maybe_run_sweep(self) -> None:
        owner = self._owner
        if owner.config.sweep_every_n_steps == 0:
            return
        if (
            owner._last_sweep_step >= 0
            and owner._step_counter - owner._last_sweep_step
            < owner.config.sweep_every_n_steps
        ):
            return
        owner._last_sweep_step = owner._step_counter

        running_snapshot = self._resolve_running_sweep_snapshot()

        for pool_idx, groups in enumerate(owner._groups_per_pool):
            for group in groups:
                window = owner._swa_window_size if group.kind is PoolKind.SWA else 0
                if running_snapshot is not None:
                    running_input = build_plan_input_running_sweep(
                        req_to_token_pool=owner._req_to_token_pool,
                        alive_reqs=running_snapshot,
                        swa_window_size=window,
                        full_to_swa_index_mapping=group.swa_index_lut,
                    )
                    owner._invoke_plan(
                        plan_input=running_input,
                        verify_plan=self._verify_plan_sweep_running,
                        write_plan=self._write_plan_sweep,
                        group=group,
                    )
                    owner._launch_endpoints(
                        pool_idx=pool_idx,
                        group=group,
                        tag_filter=_is_sweep_tag,
                        verify_plan=self._verify_plan_sweep_running,
                        forward_batch=None,
                    )

                if owner._radix_cache is not None:
                    radix_input = build_plan_input_radix_sweep(
                        radix_cache=owner._radix_cache,
                        swa_window_size=window,
                        full_to_swa_index_mapping=group.swa_index_lut,
                    )
                    owner._invoke_plan(
                        plan_input=radix_input,
                        verify_plan=self._verify_plan_sweep_radix,
                        write_plan=self._write_plan_sweep,
                        group=group,
                    )
                    owner._launch_endpoints(
                        pool_idx=pool_idx,
                        group=group,
                        tag_filter=_is_sweep_tag,
                        verify_plan=self._verify_plan_sweep_radix,
                        forward_batch=None,
                    )

        owner._sweep_passes += 1

    def _resolve_running_sweep_snapshot(self) -> Optional[AliveReqSnapshot]:
        owner = self._owner
        snapshot = self._alive_reqs_snapshot
        if snapshot is not None:
            return snapshot
        forward_batch = owner._per_forward._last_forward_batch
        if forward_batch is None:
            return None
        if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
            return None
        if not self._alive_reqs_fallback_warned:
            logger.warning(
                "kv-canary: alive_reqs_snapshot not bound; sweep falls back to forward_batch "
                "(misses paused reqs per SOT §4.1)"
            )
            self._alive_reqs_fallback_warned = True
        return AliveReqSnapshot(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
        )
