from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor

if TYPE_CHECKING:
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

logger = logging.getLogger(__name__)

SWA_DIVERGENCE_LOG_PREFIX: str = "kv_canary swa_divergence:"


class SwaDivergenceStats:
    def __init__(
        self,
        *,
        device: torch.device,
        d2h_stream: torch.cuda.Stream,
        swa_allocator_getter: Optional[
            Callable[[], Optional["SWATokenToKVPoolAllocator"]]
        ] = None,
    ) -> None:
        self._enabled: bool = envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.get()
        self._device = device
        self._d2h_stream = d2h_stream
        self._swa_allocator_getter = swa_allocator_getter
        self._forward_ct: int = 0

        self._verify_full_total_device: Optional[torch.Tensor] = None
        self._verify_swa_total_device: Optional[torch.Tensor] = None

        self._pending_verify_full_future: Optional[FutureTensor] = None
        self._pending_verify_swa_future: Optional[FutureTensor] = None
        self._pending_mapping_nonidentity_future: Optional[FutureTensor] = None
        self._pending_wrap_count_future: Optional[FutureTensor] = None
        self._pending_step: Optional[int] = None

        self._latest_verify_full: int = 0
        self._latest_verify_swa: int = 0
        self._latest_mapping_nonidentity: int = 0

        if self._enabled:
            self._verify_full_total_device = torch.zeros(
                1, dtype=torch.int64, device=device
            )
            self._verify_swa_total_device = torch.zeros(
                1, dtype=torch.int64, device=device
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def observe_after_invoke_plan(
        self,
        *,
        group: CanaryBufferGroup,
        verify_plan: VerifyPlan,
    ) -> None:
        if not self._enabled:
            return
        if group.kind is PoolKind.FULL:
            target = self._verify_full_total_device
        else:
            target = self._verify_swa_total_device
        assert target is not None
        target.add_(verify_plan.verify_num_valid.to(torch.int64))

    def on_forward_completed(self) -> None:
        if not self._enabled:
            return
        self._forward_ct += 1

    def emit_log_if_due(
        self,
        *,
        step_counter: int,
        period: int,
    ) -> None:
        if not self._enabled:
            return
        if period <= 0:
            return
        if step_counter == 0 or step_counter % period != 0:
            return

        self._drain_pending_futures()
        self._stage_async_snapshots(step_counter=step_counter)

    def _drain_pending_futures(self) -> None:
        if self._pending_step is None:
            return

        verify_full_future = self._pending_verify_full_future
        verify_swa_future = self._pending_verify_swa_future
        mapping_future = self._pending_mapping_nonidentity_future
        wrap_future = self._pending_wrap_count_future
        assert verify_full_future is not None
        assert verify_swa_future is not None

        verify_full = int(verify_full_future.wait().item())
        verify_swa = int(verify_swa_future.wait().item())
        mapping_nonidentity = (
            int(mapping_future.wait().item()) if mapping_future is not None else 0
        )
        wrap_count = int(wrap_future.wait().item()) if wrap_future is not None else 0

        self._latest_verify_full = verify_full
        self._latest_verify_swa = verify_swa
        self._latest_mapping_nonidentity = mapping_nonidentity

        line = (
            f"{SWA_DIVERGENCE_LOG_PREFIX} "
            f"forward_ct={self._forward_ct} "
            f"verify_full={verify_full} "
            f"verify_swa={verify_swa} "
            f"mapping_nonidentity={mapping_nonidentity} "
            f"swa_pool_wrap={wrap_count}"
        )
        logger.info(line)

        self._pending_verify_full_future = None
        self._pending_verify_swa_future = None
        self._pending_mapping_nonidentity_future = None
        self._pending_wrap_count_future = None
        self._pending_step = None

    def _stage_async_snapshots(self, *, step_counter: int) -> None:
        assert self._verify_full_total_device is not None
        assert self._verify_swa_total_device is not None

        self._pending_verify_full_future = FutureTensor.device_to_host(
            src_device=self._verify_full_total_device,
            stream=self._d2h_stream,
        )
        self._pending_verify_swa_future = FutureTensor.device_to_host(
            src_device=self._verify_swa_total_device,
            stream=self._d2h_stream,
        )

        allocator = (
            self._swa_allocator_getter()
            if self._swa_allocator_getter is not None
            else None
        )
        if allocator is not None:
            self._pending_mapping_nonidentity_future = FutureTensor.device_to_host(
                src_device=allocator._nonidentity_write_count_device,
                stream=self._d2h_stream,
            )
            self._pending_wrap_count_future = FutureTensor.device_to_host(
                src_device=allocator._wrap_count_device,
                stream=self._d2h_stream,
            )
        else:
            self._pending_mapping_nonidentity_future = None
            self._pending_wrap_count_future = None

        self._pending_step = step_counter
