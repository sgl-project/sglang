from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor
from sglang.srt.kv_canary.runner.swa_divergence.compute import (
    compute_swa_live_divergence,
)
from sglang.srt.kv_canary.runner.swa_divergence.log import SwaDivergenceLog

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_FULL_IDX = 0
_SWA_IDX = 1


class SwaDivergenceStats:
    def __init__(
        self,
        *,
        device: torch.device,
        d2h_stream: torch.cuda.Stream,
        swa_allocator: Optional["SWATokenToKVPoolAllocator"] = None,
        req_to_token_pool: Optional["ReqToTokenPool"] = None,
    ) -> None:
        self._d2h_stream = d2h_stream
        self._swa_allocator = swa_allocator
        self._req_to_token_pool = req_to_token_pool
        self._forward_ct: int = 0
        self._verify_total_device: torch.Tensor = torch.zeros(
            2, dtype=torch.int32, device=device
        )

    def observe_after_invoke_plan(
        self, *, group: CanaryBufferGroup, verify_plan: VerifyPlan
    ) -> None:
        idx = _FULL_IDX if group.kind is PoolKind.FULL else _SWA_IDX
        self._verify_total_device[idx].add_(verify_plan.verify_num_valid)

    def on_forward_completed(self) -> None:
        self._forward_ct += 1

    def emit_log_if_due(
        self,
        *,
        step_counter: int,
        period: int,
        forward_batch: "ForwardBatch",
    ) -> None:
        if period <= 0 or step_counter == 0 or step_counter % period != 0:
            return

        mapping_count_device: Optional[torch.Tensor] = None
        if self._swa_allocator is not None:
            with torch.cuda.stream(self._d2h_stream):
                mapping_count_device = compute_swa_live_divergence(
                    swa_allocator=self._swa_allocator,
                    req_to_token_pool=self._req_to_token_pool,
                    forward_batch=forward_batch,
                )

        verify_future = FutureTensor.device_to_host(
            src_device=self._verify_total_device, stream=self._d2h_stream
        )
        mapping_future = (
            FutureTensor.device_to_host(
                src_device=mapping_count_device, stream=self._d2h_stream
            )
            if mapping_count_device is not None
            else None
        )

        verify_totals = verify_future.wait().tolist()
        mapping_nonidentity = (
            int(mapping_future.wait().item()) if mapping_future is not None else 0
        )

        logger.info(
            SwaDivergenceLog(
                forward_ct=self._forward_ct,
                verify_full=int(verify_totals[_FULL_IDX]),
                verify_swa=int(verify_totals[_SWA_IDX]),
                mapping_nonidentity=mapping_nonidentity,
            ).format()
        )
