"""Shared batch-size and CUDA-graph routing for adaptive spec policies."""

from __future__ import annotations

import bisect


class AdaptiveStepRouter:
    """Route a batch size to its configured speculative-step candidates."""

    def __init__(self, bs_candidates: dict[int, list[int]]):
        if not bs_candidates:
            raise ValueError("adaptive step router requires at least one BS slot")
        self._bs_candidates = {
            batch_size: sorted(set(candidates))
            for batch_size, candidates in bs_candidates.items()
        }
        self._bs_list = sorted(self._bs_candidates)
        self._cuda_graph_bs: list[int] | None = None

    @property
    def candidate_steps(self) -> list[int]:
        return sorted(
            {step for candidates in self._bs_candidates.values() for step in candidates}
        )

    @property
    def batch_size_keys(self) -> list[int]:
        return self._bs_list

    @property
    def cuda_graph_bs(self) -> list[int] | None:
        return self._cuda_graph_bs

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def batch_size_key_for_batch(self, batch_size: int) -> int:
        """Return the configured lower-bound key after CUDA-graph padding."""
        if self._cuda_graph_bs is not None:
            graph_index = bisect.bisect_left(self._cuda_graph_bs, batch_size)
            if graph_index < len(self._cuda_graph_bs):
                batch_size = self._cuda_graph_bs[graph_index]
        key_index = bisect.bisect_right(self._bs_list, batch_size) - 1
        return self._bs_list[max(0, key_index)]

    def candidates_for_batch(self, batch_size: int) -> list[int]:
        return self._bs_candidates[self.batch_size_key_for_batch(batch_size)]

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        if self._cuda_graph_bs is None:
            return None
        return [
            batch_size
            for batch_size in self._cuda_graph_bs
            if step in self.candidates_for_batch(batch_size)
        ]
