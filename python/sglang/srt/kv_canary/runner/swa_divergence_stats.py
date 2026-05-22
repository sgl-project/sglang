from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor

if TYPE_CHECKING:
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

logger = logging.getLogger(__name__)

SWA_DIVERGENCE_LOG_PREFIX: str = "kv_canary swa_divergence:"

_SWA_DIVERGENCE_LINE_RE = re.compile(
    re.escape(SWA_DIVERGENCE_LOG_PREFIX)
    + r" forward_ct=(?P<forward_ct>\d+)"
    + r" verify_full=(?P<verify_full>\d+)"
    + r" verify_swa=(?P<verify_swa>\d+)"
    + r" mapping_nonidentity=(?P<mapping_nonidentity>\d+)"
    + r" swa_pool_wrap=(?P<swa_pool_wrap>\d+)"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedSwaDivergenceLine:
    forward_ct: int
    verify_full: int
    verify_swa: int
    mapping_nonidentity: int
    swa_pool_wrap: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _PendingSnapshot:
    step_counter: int
    verify_full: FutureTensor
    verify_swa: FutureTensor
    mapping_nonidentity: Optional[FutureTensor]
    wrap_count: Optional[FutureTensor]


def format_swa_divergence_line(
    *,
    forward_ct: int,
    verify_full: int,
    verify_swa: int,
    mapping_nonidentity: int,
    swa_pool_wrap: int,
) -> str:
    return (
        f"{SWA_DIVERGENCE_LOG_PREFIX} "
        f"forward_ct={forward_ct} "
        f"verify_full={verify_full} "
        f"verify_swa={verify_swa} "
        f"mapping_nonidentity={mapping_nonidentity} "
        f"swa_pool_wrap={swa_pool_wrap}"
    )


def parse_swa_divergence_line(line: str) -> Optional[ParsedSwaDivergenceLine]:
    match = _SWA_DIVERGENCE_LINE_RE.search(line)
    if match is None:
        return None
    return ParsedSwaDivergenceLine(
        forward_ct=int(match.group("forward_ct")),
        verify_full=int(match.group("verify_full")),
        verify_swa=int(match.group("verify_swa")),
        mapping_nonidentity=int(match.group("mapping_nonidentity")),
        swa_pool_wrap=int(match.group("swa_pool_wrap")),
    )


def find_last_swa_divergence_line(
    text: str,
) -> Optional[tuple[ParsedSwaDivergenceLine, str]]:
    last_match: Optional[re.Match] = None
    for match in _SWA_DIVERGENCE_LINE_RE.finditer(text):
        last_match = match
    if last_match is None:
        return None
    parsed = ParsedSwaDivergenceLine(
        forward_ct=int(last_match.group("forward_ct")),
        verify_full=int(last_match.group("verify_full")),
        verify_swa=int(last_match.group("verify_swa")),
        mapping_nonidentity=int(last_match.group("mapping_nonidentity")),
        swa_pool_wrap=int(last_match.group("swa_pool_wrap")),
    )
    return parsed, last_match.group(0)


class SwaDivergenceStats:
    def __init__(
        self,
        *,
        device: torch.device,
        d2h_stream: torch.cuda.Stream,
        swa_allocator: Optional["SWATokenToKVPoolAllocator"] = None,
    ) -> None:
        self._device = device
        self._d2h_stream = d2h_stream
        self._swa_allocator = swa_allocator
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
        wrap_count = (
            int(pending.wrap_count.wait().item())
            if pending.wrap_count is not None
            else 0
        )

        self._latest_verify_full = verify_full
        self._latest_verify_swa = verify_swa
        self._latest_mapping_nonidentity = mapping_nonidentity

        line = format_swa_divergence_line(
            forward_ct=self._forward_ct,
            verify_full=verify_full,
            verify_swa=verify_swa,
            mapping_nonidentity=mapping_nonidentity,
            swa_pool_wrap=wrap_count,
        )
        logger.info(line)

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

        counters = (
            self._swa_allocator.divergence_stats_device_tensors()
            if self._swa_allocator is not None
            else None
        )
        if counters is not None and counters.nonidentity_write_count is not None:
            mapping_future: Optional[FutureTensor] = FutureTensor.device_to_host(
                src_device=counters.nonidentity_write_count,
                stream=self._d2h_stream,
            )
            wrap_future: Optional[FutureTensor] = FutureTensor.device_to_host(
                src_device=counters.wrap_count,
                stream=self._d2h_stream,
            )
        else:
            mapping_future = None
            wrap_future = None

        self._pending = _PendingSnapshot(
            step_counter=step_counter,
            verify_full=verify_full_future,
            verify_swa=verify_swa_future,
            mapping_nonidentity=mapping_future,
            wrap_count=wrap_future,
        )
