from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor
from sglang.srt.kv_canary.runner.swa_divergence.log import SwaDivergenceLog

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.swa_divergence.observer import (
        SwaLiveDivergenceObserver,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PendingSnapshot:
    step_counter: int
    verify_full: FutureTensor
    verify_swa: FutureTensor
    mapping_nonidentity: Optional[FutureTensor]


class SwaDivergenceStats:
    def __init__(
        self,
        *,
        device: torch.device,
        d2h_stream: torch.cuda.Stream,
        swa_live_divergence_observer: Optional["SwaLiveDivergenceObserver"] = None,
    ) -> None:
        self._device = device
        self._d2h_stream = d2h_stream
        self._swa_live_divergence_observer = swa_live_divergence_observer
        self._forward_ct: int = 0

        self._verify_full_total_device: torch.Tensor = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        self._verify_swa_total_device: torch.Tensor = torch.zeros(
            1, dtype=torch.int32, device=device
        )

        self._pending: Optional[_PendingSnapshot] = None

        self._latest_verify_full: int = 0
        self._latest_verify_swa: int = 0
        self._latest_mapping_nonidentity: int = 0

    def observe_after_invoke_plan(
        self,
        *,
        group: CanaryBufferGroup,
        verify_plan: VerifyPlan,
    ) -> None:
        if group.kind is PoolKind.FULL:
            target = self._verify_full_total_device
        else:
            target = self._verify_swa_total_device
        target.add_(verify_plan.verify_num_valid)

    def on_forward_completed(self) -> None:
        self._forward_ct += 1

    def emit_log_if_due(
        self,
        *,
        step_counter: int,
        period: int,
    ) -> None:
        if period <= 0:
            return
        if step_counter == 0 or step_counter % period != 0:
            return

        self._drain_pending_futures()
        self._stage_async_snapshots(step_counter=step_counter)

    def _drain_pending_futures(self) -> None:
        pending = self._pending
        if pending is None:
            return

        verify_full = int(pending.verify_full.wait().item())
        verify_swa = int(pending.verify_swa.wait().item())
        mapping_nonidentity = (
            int(pending.mapping_nonidentity.wait().item())
            if pending.mapping_nonidentity is not None
            else 0
        )

        self._latest_verify_full = verify_full
        self._latest_verify_swa = verify_swa
        self._latest_mapping_nonidentity = mapping_nonidentity

        logger.info(
            SwaDivergenceLog(
                forward_ct=self._forward_ct,
                verify_full=verify_full,
                verify_swa=verify_swa,
                mapping_nonidentity=mapping_nonidentity,
            ).format()
        )

        self._pending = None

    def _stage_async_snapshots(self, *, step_counter: int) -> None:
        verify_full_future = FutureTensor.device_to_host(
            src_device=self._verify_full_total_device,
            stream=self._d2h_stream,
        )
        verify_swa_future = FutureTensor.device_to_host(
            src_device=self._verify_swa_total_device,
            stream=self._d2h_stream,
        )

        mapping_future: Optional[FutureTensor] = (
            self._swa_live_divergence_observer.snapshot_nonidentity_future(
                stream=self._d2h_stream
            )
            if self._swa_live_divergence_observer is not None
            else None
        )

        self._pending = _PendingSnapshot(
            step_counter=step_counter,
            verify_full=verify_full_future,
            verify_swa=verify_swa_future,
            mapping_nonidentity=mapping_future,
        )
