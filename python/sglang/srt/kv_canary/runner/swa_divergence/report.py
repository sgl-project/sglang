from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler
from sglang.srt.kv_canary.runner.swa_divergence.compute import (
    compute_swa_full_idx_divergence,
)
from sglang.srt.kv_canary.runner.swa_divergence.log import SwaDivergenceLog

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_FULL_IDX = 0
_SWA_IDX = 1


class SwaDivergenceReport:
    def __init__(
        self,
        *,
        device: torch.device,
        d2h_stream: torch.cuda.Stream,
        swa_allocator: Optional["SWATokenToKVPoolAllocator"] = None,
        req_to_token_pool: Optional["ReqToTokenPool"] = None,
    ) -> None:
        self._swa_allocator = swa_allocator
        self._req_to_token_pool = req_to_token_pool
        self._forward_ct: int = 0
        self._verify_total_count_device: torch.Tensor = torch.zeros(
            2, dtype=torch.int32, device=device
        )
        self._handler = DelayedDeviceHostHandler(d2h_stream=d2h_stream)

    def observe_after_invoke_plan(
        self, *, group: CanaryBufferGroup, verify_plan: VerifyPlan
    ) -> None:
        idx = _FULL_IDX if group.kind is PoolKind.FULL else _SWA_IDX
        self._verify_total_count_device[idx].add_(verify_plan.verify_num_valid)

    def on_forward_completed(self) -> None:
        self._forward_ct += 1

    def step(
        self,
        *,
        step_counter: int,
        period: int,
        forward_batch: "ForwardBatch",
    ) -> None:
        self._handler.step(
            compute_on_device=lambda: self._compute_on_device(
                step_counter=step_counter, period=period, forward_batch=forward_batch
            ),
            postprocess_on_host=self._postprocess_on_host,
        )

    def _compute_on_device(
        self,
        *,
        step_counter: int,
        period: int,
        forward_batch: "ForwardBatch",
    ) -> Optional[dict[str, torch.Tensor]]:
        if period <= 0 or step_counter == 0 or step_counter % period != 0:
            return None

        result: dict[str, torch.Tensor] = {
            "verify_total_count": self._verify_total_count_device,
        }
        if self._swa_allocator is not None:
            result["swa_full_idx_divergence"] = compute_swa_full_idx_divergence(
                swa_allocator=self._swa_allocator,
                req_to_token_pool=self._req_to_token_pool,
                forward_batch=forward_batch,
            )
        return result

    def _postprocess_on_host(self, host_data: dict[str, torch.Tensor]) -> None:
        verify_totals = host_data["verify_total_count"].tolist()
        swa_full_idx_divergence = (
            int(host_data["swa_full_idx_divergence"].item())
            if "swa_full_idx_divergence" in host_data
            else 0
        )
        logger.info(
            SwaDivergenceLog(
                forward_ct=self._forward_ct,
                verify_full=int(verify_totals[_FULL_IDX]),
                verify_swa=int(verify_totals[_SWA_IDX]),
                swa_full_idx_divergence=swa_full_idx_divergence,
            ).format()
        )
