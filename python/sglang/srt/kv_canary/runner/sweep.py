from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.plan_input import build_plan_input_radix_sweep
from sglang.srt.kv_canary.runner.per_forward import _is_sweep_tag

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner

logger = logging.getLogger(__name__)


class SweepOrchestrator:
    """Only walks the radix tree. Per-forward HEAD/TAIL covers running req KV slots every step;
    sweep is purely for the radix-cached-but-not-in-running-batch slot set.

    Runs host-side eager (post-replay), kernels are NOT captured into the cuda graph — sweep
    cadence is host-side state and radix walker output size varies per cycle.
    """

    def __init__(
        self,
        *,
        owner: "CanaryRunner",
        sweep_verify_capacity: int,
    ) -> None:
        self._owner = owner
        device = owner._device
        self._verify_plan_sweep_radix = VerifyPlan.allocate(
            verify_capacity=max(1, sweep_verify_capacity), device=device
        )
        self._write_plan_sweep = WritePlan.allocate(write_req_capacity=1, device=device)

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

        if owner._radix_cache is None:
            return

        for group in owner._groups:
            window = owner._swa_window_size if group.kind is PoolKind.SWA else 0
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
                group=group,
                tag_filter=_is_sweep_tag,
                verify_plan=self._verify_plan_sweep_radix,
                forward_batch=None,
            )

        owner._sweep_passes += 1
