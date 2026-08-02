"""Position-only sink + recent-window visibility policy."""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.contracts import (
    DecisionScope,
    Granularity,
    KVStateAction,
    SelectionContext,
    SelectionEvidence,
    SelectionResult,
    SparsityCapabilities,
)
from sglang.srt.mem_cache.sparsity.policies.base import SparsityPolicy


class StreamingLLMPolicy(SparsityPolicy):
    """Expose initial sink pages and the most recent pages.

    This is visibility-only: excluded pages remain allocated in HBM. With the
    default runtime page size of one, pages are identical to tokens and the
    policy matches the usual StreamingLLM sink/window visibility pattern.
    """

    _CAPABILITIES = SparsityCapabilities(
        state_action=KVStateAction.VISIBILITY_ONLY,
        granularity=Granularity.PAGE,
        evidence=SelectionEvidence.POSITION,
        scope=DecisionScope.STEP,
        requires_request_state=False,
        supports_cuda_graph=False,
    )

    def __init__(self, config: KVSparsityConfig, device: torch.device):
        self.config = config
        self.device = device
        unknown = sorted(set(config.policy_config) - {"sink_pages", "recent_pages"})
        if unknown:
            raise ValueError(f"Unknown StreamingLLM policy field(s): {unknown}")
        self.sink_pages = self._positive_int("sink_pages", 4, allow_zero=True)
        self.recent_pages = self._positive_int("recent_pages", 1024)
        if self.sink_pages + self.recent_pages <= 0:
            raise ValueError("StreamingLLM must retain at least one page")

    def _positive_int(self, name: str, default: int, allow_zero: bool = False) -> int:
        value = self.config.policy_config.get(name, default)
        lower_bound = 0 if allow_zero else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < lower_bound:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}")
        return value

    @property
    def capabilities(self) -> SparsityCapabilities:
        return self._CAPABILITIES

    def select(self, context: SelectionContext) -> SelectionResult:
        seq_lens = context.seq_lens
        if seq_lens.ndim != 1:
            raise ValueError("seq_lens must have shape [batch]")

        page_size = self.config.page_size
        num_pages = torch.div(
            seq_lens + page_size - 1, page_size, rounding_mode="floor"
        )
        capacity = self.sink_pages + self.recent_pages
        sparse_mask = (seq_lens >= self.config.min_sparse_tokens) & (
            num_pages > capacity
        )

        sink_offsets = torch.arange(
            self.sink_pages, dtype=torch.int64, device=seq_lens.device
        )
        recent_offsets = torch.arange(
            self.recent_pages, dtype=torch.int64, device=seq_lens.device
        )
        recent_starts = num_pages - self.recent_pages

        sink = sink_offsets.unsqueeze(0).expand(seq_lens.shape[0], -1)
        recent = recent_starts.unsqueeze(1) + recent_offsets.unsqueeze(0)
        logical_indices = torch.cat((sink, recent), dim=1).to(torch.int32)
        logical_indices = torch.where(
            sparse_mask.unsqueeze(1),
            logical_indices,
            torch.full_like(logical_indices, -1),
        )

        valid_lengths = torch.where(
            sparse_mask,
            torch.full_like(seq_lens, capacity, dtype=torch.int32),
            torch.zeros_like(seq_lens, dtype=torch.int32),
        )
        final_page_padding = num_pages * page_size - seq_lens
        sparse_kv_lens = capacity * page_size - final_page_padding
        visible_kv_lens = torch.where(
            sparse_mask,
            sparse_kv_lens.to(torch.int32),
            torch.zeros_like(seq_lens, dtype=torch.int32),
        )

        return SelectionResult(
            granularity=Granularity.PAGE,
            logical_indices=logical_indices,
            valid_lengths=valid_lengths,
            visible_kv_lens=visible_kv_lens,
            sparse_mask=sparse_mask,
            layer_id=None,
            # Dense rows are either below the activation threshold or no
            # longer than the retained page budget. This is therefore a safe
            # scheduler upper bound without synchronizing seq_lens to host.
            max_visible_kv_len=max(
                capacity * page_size,
                max(self.config.min_sparse_tokens - 1, 1),
            ),
        )
