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
    """Env-gated per-forward observation of SWA-vs-FULL divergence signals.

    When disabled (env var unset) every public method is a no-op and no device
    tensors are allocated. Enabling the env var wires in three signals:

    - verify_full / verify_swa: cumulative ``VerifyPlan.verify_num_valid`` per
      group. Per-step the orchestrator calls ``observe_after_invoke_plan`` once
      per group, which accumulates the on-device int32 verify_num_valid into a
      group-specific int64 accumulator.
    - mapping_nonidentity: ``(full_to_swa_index_mapping[:-1] != arange).sum()``
      computed lazily on log-emit.
    - swa_pool_wrap: ``SWATokenToKVPoolAllocator.wrap_count``, read directly
      from the allocator. Always-on int counter; reader-side only.
    """

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
        full_to_swa_index_mapping: Optional[torch.Tensor],
    ) -> None:
        if not self._enabled:
            return
        if period <= 0:
            return
        if step_counter == 0 or step_counter % period != 0:
            return

        self._drain_pending_futures()
        self._stage_async_snapshots(
            step_counter=step_counter,
            full_to_swa_index_mapping=full_to_swa_index_mapping,
        )

    def _drain_pending_futures(self) -> None:
        if self._pending_step is None:
            return

        verify_full_future = self._pending_verify_full_future
        verify_swa_future = self._pending_verify_swa_future
        mapping_future = self._pending_mapping_nonidentity_future
        assert verify_full_future is not None
        assert verify_swa_future is not None
        assert mapping_future is not None

        verify_full = int(verify_full_future.wait().item())
        verify_swa = int(verify_swa_future.wait().item())
        mapping_nonidentity = int(mapping_future.wait().item())

        self._latest_verify_full = verify_full
        self._latest_verify_swa = verify_swa
        self._latest_mapping_nonidentity = mapping_nonidentity

        wrap_count = self._read_wrap_count()

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
        self._pending_step = None

    def _stage_async_snapshots(
        self,
        *,
        step_counter: int,
        full_to_swa_index_mapping: Optional[torch.Tensor],
    ) -> None:
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

        mapping_nonidentity_device = _compute_mapping_nonidentity(
            full_to_swa_index_mapping=full_to_swa_index_mapping,
            device=self._device,
        )
        self._pending_mapping_nonidentity_future = FutureTensor.device_to_host(
            src_device=mapping_nonidentity_device,
            stream=self._d2h_stream,
        )
        self._pending_step = step_counter

    def _read_wrap_count(self) -> int:
        if self._swa_allocator_getter is None:
            return 0
        allocator = self._swa_allocator_getter()
        if allocator is None:
            return 0
        return int(allocator.wrap_count)


def _compute_mapping_nonidentity(
    *,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if full_to_swa_index_mapping is None:
        return torch.zeros(1, dtype=torch.int64, device=device)

    mapping = full_to_swa_index_mapping[:-1]
    indices = torch.arange(mapping.shape[0], device=mapping.device, dtype=mapping.dtype)
    diff_count = (mapping != indices).sum().view(1).to(torch.int64)
    return diff_count
