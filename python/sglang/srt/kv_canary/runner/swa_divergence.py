from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.kernels.ops.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_SWA_DIVERGENCE_LOG_PREFIX: str = "kv_canary_swa_divergence="
_SWA_DIVERGENCE_LINE_RE = re.compile(re.escape(_SWA_DIVERGENCE_LOG_PREFIX) + r"(\S+)")
_FULL_IDX = 0
_SWA_IDX = 1


class SwaDivergenceReporter:
    def __init__(
        self,
        *,
        device: torch.device,
        d2h_stream: torch.cuda.Stream,
        interval: int,
        swa_allocator: Optional[SWATokenToKVPoolAllocator] = None,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
    ) -> None:
        self._interval = interval
        self._swa_allocator = swa_allocator
        self._req_to_token_pool = req_to_token_pool
        self._forward_ct: int = 0
        # Per-group running total of verify entries (shape ``[2]``, int32).
        self.verify_total_count_device: torch.Tensor = torch.zeros(
            2, dtype=torch.int32, device=device
        )
        self._handler = DelayedDeviceHostHandler(d2h_stream=d2h_stream)

    def observe_after_invoke_plan(
        self, *, group: CanaryBufferGroup, verify_plan: VerifyPlan
    ) -> None:
        idx = _FULL_IDX if group.kind is PoolKind.FULL else _SWA_IDX
        # verify_num_valid is shape [1]; slice to a length-1 view so the in-place add
        # has matching ranks (else torch refuses the broadcast into shape []).
        self.verify_total_count_device[idx : idx + 1].add_(verify_plan.verify_num_valid)

    def step(
        self,
        *,
        outer_step_counter: int,
        maybe_inaccurate_forward_batch: Optional[ForwardBatch],
    ) -> None:
        self._forward_ct += 1
        self._handler.step(
            compute_on_device=lambda: self._compute_on_device(
                outer_step_counter=outer_step_counter,
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            ),
            postprocess_on_host=self._postprocess_on_host,
        )

    def _compute_on_device(
        self,
        *,
        outer_step_counter: int,
        maybe_inaccurate_forward_batch: Optional[ForwardBatch],
    ) -> Optional[dict[str, Any]]:
        if outer_step_counter == 0 or outer_step_counter % self._interval != 0:
            return None

        result: dict[str, Any] = {
            "forward_ct": self._forward_ct,
            "verify_total_count": self.verify_total_count_device,
        }
        if (
            self._swa_allocator is not None
            and maybe_inaccurate_forward_batch is not None
        ):
            result["swa_full_idx_divergence"] = compute_swa_full_idx_divergence(
                swa_allocator=self._swa_allocator,
                req_to_token_pool=self._req_to_token_pool,
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            )
            result["swa_out_of_window_tokens"] = compute_swa_out_of_window_tokens(
                swa_allocator=self._swa_allocator,
                req_to_token_pool=self._req_to_token_pool,
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            )
        return result

    def _postprocess_on_host(self, host_data: dict[str, Any]) -> None:
        verify_totals = host_data["verify_total_count"].tolist()
        swa_full_idx_divergence = (
            int(x.item())
            if (x := host_data.get("swa_full_idx_divergence")) is not None
            else 0
        )
        swa_out_of_window_tokens = (
            int(x.item())
            if (x := host_data.get("swa_out_of_window_tokens")) is not None
            else 0
        )
        logger.info(
            SwaDivergenceLog(
                forward_ct=host_data["forward_ct"],
                verify_full=int(verify_totals[_FULL_IDX]),
                verify_swa=int(verify_totals[_SWA_IDX]),
                swa_full_idx_divergence=swa_full_idx_divergence,
                swa_out_of_window_tokens=swa_out_of_window_tokens,
            ).format()
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SwaDivergenceLog:
    forward_ct: int
    verify_full: int
    verify_swa: int
    swa_full_idx_divergence: int
    swa_out_of_window_tokens: int = 0

    def format(self) -> str:
        return _SWA_DIVERGENCE_LOG_PREFIX + json.dumps(
            asdict(self), separators=(",", ":")
        )

    @classmethod
    def parse(cls, line: str) -> Optional[SwaDivergenceLog]:
        match = _SWA_DIVERGENCE_LINE_RE.search(line)
        if match is None:
            return None
        return cls(**json.loads(match.group(1)))

    @classmethod
    def find_last(cls, text: str) -> Optional[tuple[SwaDivergenceLog, str]]:
        last_match: Optional[re.Match] = None
        for match in _SWA_DIVERGENCE_LINE_RE.finditer(text):
            last_match = match
        if last_match is None:
            return None
        return cls(**json.loads(last_match.group(1))), last_match.group(0)

    @classmethod
    def find_all(cls, text: str) -> list[tuple[SwaDivergenceLog, str]]:
        return [
            (cls(**json.loads(match.group(1))), match.group(0))
            for match in _SWA_DIVERGENCE_LINE_RE.finditer(text)
        ]


def compute_swa_out_of_window_tokens(
    *,
    swa_allocator: SWATokenToKVPoolAllocator,
    req_to_token_pool: ReqToTokenPool,
    maybe_inaccurate_forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Count tokens in the live req_to_token range whose SWA mapping is 0 (out-of-window)."""
    full_to_swa_index_mapping = swa_allocator.full_to_swa_index_mapping
    device = full_to_swa_index_mapping.device
    req_pool_indices = maybe_inaccurate_forward_batch.req_pool_indices
    seq_lens = maybe_inaccurate_forward_batch.seq_lens
    if req_pool_indices.numel() == 0:
        return torch.zeros(1, dtype=torch.int32, device=device)
    req_to_token = req_to_token_pool.req_to_token
    rows = req_to_token[req_pool_indices]
    positions = torch.arange(rows.shape[1], device=rows.device)
    mask = positions[None, :] < seq_lens[:, None]
    swa_indices = full_to_swa_index_mapping[rows]
    return ((swa_indices == 0) & mask).sum().to(torch.int32).view(1)


def compute_swa_full_idx_divergence(
    *,
    swa_allocator: SWATokenToKVPoolAllocator,
    req_to_token_pool: ReqToTokenPool,
    maybe_inaccurate_forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Count non-identity (full, swa) index pairs in the live req_to_token range."""
    full_to_swa_index_mapping = swa_allocator.full_to_swa_index_mapping
    device = full_to_swa_index_mapping.device
    req_pool_indices = maybe_inaccurate_forward_batch.req_pool_indices
    seq_lens = maybe_inaccurate_forward_batch.seq_lens

    if req_pool_indices.numel() == 0:
        return torch.zeros(1, dtype=torch.int32, device=device)

    req_to_token = req_to_token_pool.req_to_token
    rows = req_to_token[req_pool_indices]
    positions = torch.arange(rows.shape[1], device=rows.device)
    mask = positions[None, :] < seq_lens[:, None]
    swa_indices = full_to_swa_index_mapping[rows]
    # FULL pool slots beyond the sliding window have their SWA mapping written
    # to 0 (see SWATokenToKVPoolAllocator.alloc_extend); skip those so they
    # don't get counted as divergence.
    return (
        ((swa_indices != rows) & mask & (swa_indices != 0))
        .sum()
        .to(torch.int32)
        .view(1)
    )
