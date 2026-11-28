"""Each request's sampling state, one table entry per request slot, addressed by
its `req_pool_idx`: its sampler params (temperature, top_p, top_k, min_p), its
penalty index, its seed, and the watermark algorithm's per-request values.

- `write` fills a request's entry at each admission; a launch reads its rows with
  one `gather`, so a step with no admission copies none of them to the device.
- Writes and gathers are enqueued on the scheduler's stream in host order: a
  launch's gather runs before a later admission rewrites a reused slot.
- A launch over the same slots as the previous gather, with no write since,
  reuses that gather's tensors: a decode batch whose requests did not change
  keeps one set of them. Callers must not write them in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import numpy as np
import torch

from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.runtime import (
    build_watermark_rows,
    num_watermark_fields,
    watermark_pool_values,
)
from sglang.srt.utils.device import async_h2d_tensor
from sglang.srt.utils.log_utils import print_warning_once

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.watermarking.interface import WatermarkRows


class SamplingRows(NamedTuple):
    """One launch's state read from its rows of `SamplingParamsPool`, in batch
    order; field names match `SamplingBatchInfo`'s. All fields but `step_penalties`
    may be the previous launch's (see the module docstring)."""

    temperatures: torch.Tensor  # [bs, 1] fp32
    top_ps: torch.Tensor  # [bs] fp32
    top_ks: torch.Tensor  # [bs] int32
    min_ps: torch.Tensor  # [bs] fp32
    is_all_greedy: bool
    is_any_greedy: bool
    need_top_p_sampling: bool
    need_top_k_sampling: bool
    need_min_p_sampling: bool
    # Whether torch_npu.npu_top_k_top_p takes every row.
    npu_top_k_top_p_eligible: bool
    # The largest top_k, or None when some row samples from the whole vocabulary.
    max_top_k: Optional[int]
    # [bs] int64 seeds, or None to sample unseeded; deterministic mode seeds every row.
    sampling_seed: Optional[torch.Tensor]
    # The penalized rows, None when no row is penalized.
    step_penalties: Optional[StepPenalties]
    # The watermark policy, None unless some row is a candidate.
    watermark: Optional[WatermarkRows]

    @classmethod
    def empty(cls, device) -> SamplingRows:
        return cls(
            temperatures=torch.empty((0, 1), dtype=torch.float32, device=device),
            top_ps=torch.empty(0, dtype=torch.float32, device=device),
            top_ks=torch.empty(0, dtype=torch.int32, device=device),
            min_ps=torch.empty(0, dtype=torch.float32, device=device),
            is_all_greedy=True,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            npu_top_k_top_p_eligible=True,
            max_top_k=None,
            # A deterministic idle batch still seeds, with no rows.
            sampling_seed=(
                torch.empty(0, dtype=torch.int64, device=device)
                if _deterministic()
                else None
            ),
            step_penalties=None,
            watermark=None,
        )


class SamplingParamsPool:
    # Fields of `table`, one row each, floats as fp32 bits and int64 values as
    # (low, high) int32 halves: one int32 gather reads them all, top_k exact.
    TEMPERATURE, TOP_P, TOP_K, MIN_P, PENALTY_INDEX, SEED_LO, SEED_HI = range(7)
    # The first of the watermark algorithm's fields, which end the table.
    WATERMARK = 7

    def __init__(
        self,
        num_rows: int,
        device,
        penalty_status_pool: Optional[PenaltyStatusPool] = None,
    ):
        # The pool the penalty indices point into; None where no request takes one.
        self.penalty_status_pool = penalty_status_pool
        self.num_watermark_fields = num_watermark_fields()
        self.table = torch.zeros(
            (self.WATERMARK + self.num_watermark_fields, num_rows),
            dtype=torch.int32,
            device=device,
        )
        floats = self.table.view(torch.float32)
        floats[self.TEMPERATURE] = 1.0
        floats[self.TOP_P] = 1.0
        self.table[self.TOP_K] = TOP_K_ALL
        self.table[self.PENALTY_INDEX] = -1
        self.top_ks_host = np.full(num_rows, TOP_K_ALL, dtype=np.int64)
        self.uses_top_p_host = np.zeros(num_rows, dtype=bool)
        self.uses_min_p_host = np.zeros(num_rows, dtype=bool)
        self.has_seed_host = np.zeros(num_rows, dtype=bool)
        self.penalty_indices_host = np.full(num_rows, -1, dtype=np.int64)
        self.watermark_candidates_host = np.zeros(num_rows, dtype=bool)
        # The last gather's slots, its rows without penalties, and its rows'
        # penalty indices; a write invalidates it.
        self._last_gather: Optional[tuple[bytes, SamplingRows, torch.Tensor]] = None

    def write(self, reqs: Sequence[Req]) -> None:
        """Fill each request's entry; runs after its penalty index is allocated."""
        slots = [r.kv.req_pool_idx for r in reqs]
        params = [r.sampling_params for r in reqs]
        top_ks = [p.top_k for p in params]
        penalty_indices = [
            -1 if r.penalty_index is None else r.penalty_index for r in reqs
        ]
        self.top_ks_host[slots] = top_ks
        self.uses_top_p_host[slots] = [p.top_p != 1.0 for p in params]
        self.uses_min_p_host[slots] = [p.min_p > 0 for p in params]
        self.has_seed_host[slots] = [p.sampling_seed is not None for p in params]
        self.penalty_indices_host[slots] = penalty_indices

        # The table's fields, then the slots in a last row: one copy writes them.
        host_values = np.empty((self.table.shape[0] + 1, len(reqs)), dtype=np.int32)
        host_values[-1] = slots
        host_floats = host_values.view(np.float32)
        host_floats[self.TEMPERATURE] = [p.temperature for p in params]
        host_floats[self.TOP_P] = [p.top_p for p in params]
        host_values[self.TOP_K] = top_ks
        host_floats[self.MIN_P] = [p.min_p for p in params]
        host_values[self.PENALTY_INDEX] = penalty_indices
        deterministic = _deterministic()
        seeds = np.array([_seed(r, deterministic) for r in reqs], dtype=np.int64)
        host_values[self.SEED_LO : self.SEED_HI + 1] = (
            seeds.view(np.int32).reshape(-1, 2).T
        )
        if self.num_watermark_fields:
            candidates, values = zip(*(watermark_pool_values(r) for r in reqs))
            self.watermark_candidates_host[slots] = candidates
            host_values[self.WATERMARK : -1] = np.array(values, dtype=np.int32).T
        values = async_h2d_tensor(torch.from_numpy(host_values), self.table.device)
        self.table.index_copy_(1, values[-1].long(), values[:-1])
        self._last_gather = None

    def gather(
        self, req_pool_indices: torch.Tensor, req_pool_indices_cpu: torch.Tensor
    ) -> SamplingRows:
        slots = req_pool_indices_cpu.numpy()
        if slots.size == 0:
            return SamplingRows.empty(self.table.device)
        key = slots.tobytes()
        if self._last_gather is None or self._last_gather[0] != key:
            self._last_gather = (key, *self._gather_rows(req_pool_indices, slots))
        _, rows, penalty_indices = self._last_gather
        # Built each step: a StepPenalties caches its step's penalties from the
        # current counts.
        return rows._replace(
            step_penalties=StepPenalties.build(
                self.penalty_status_pool,
                penalty_indices,
                self.penalty_indices_host[slots],
            )
        )

    def _gather_rows(
        self, req_pool_indices: torch.Tensor, slots: np.ndarray
    ) -> tuple[SamplingRows, torch.Tensor]:
        top_ks = self.top_ks_host[slots]
        lo, hi = int(top_ks.min()), int(top_ks.max())
        # Each row of the [rows, bs] result is a contiguous [bs] tensor.
        values = self.table.index_select(1, req_pool_indices)
        floats = values.view(torch.float32)
        seeds = None
        if _deterministic() or _any_row_with_seed(self.has_seed_host[slots], top_ks):
            seeds = (
                torch.stack((values[self.SEED_LO], values[self.SEED_HI]), dim=1)
                .view(torch.int64)
                .view(-1)
            )
        rows = SamplingRows(
            temperatures=floats[self.TEMPERATURE].view(-1, 1),
            top_ps=floats[self.TOP_P],
            top_ks=values[self.TOP_K],
            min_ps=floats[self.MIN_P],
            is_all_greedy=hi <= 1,
            is_any_greedy=lo <= 1,
            need_top_p_sampling=bool(self.uses_top_p_host[slots].any()),
            need_top_k_sampling=(lo, hi) != (TOP_K_ALL, TOP_K_ALL),
            need_min_p_sampling=bool(self.uses_min_p_host[slots].any()),
            npu_top_k_top_p_eligible=lo >= 1 and hi <= 1024,
            max_top_k=hi if lo >= 1 and hi < TOP_K_ALL else None,
            sampling_seed=seeds,
            step_penalties=None,
            watermark=build_watermark_rows(
                values[self.WATERMARK :], self.watermark_candidates_host[slots]
            ),
        )
        return rows, values[self.PENALTY_INDEX]


def _deterministic() -> bool:
    return get_exec().deterministic.enable_deterministic_inference


def _seed(req: Req, deterministic: bool) -> int:
    seed = req.sampling_params.sampling_seed
    if seed is None:
        # A deterministic run defaults to 42; otherwise the row stays random. A
        # request built outside the receiver has no filler seed.
        seed = 42 if deterministic else (req.filler_seed or 0)
    return seed


def _any_row_with_seed(has_seed: np.ndarray, top_ks: np.ndarray) -> bool:
    """Whether some row samples with its request's own seed, when deterministic
    mode is off (it seeds every row).

    False under speculative decoding, whose draft graphs have seed buffers only in
    deterministic mode. A greedy row never draws, so its seed does not count.
    """
    if not (has_seed & (top_ks > 1)).any():
        return False
    if get_spec().speculative_algorithm is not None:
        print_warning_once(
            "sampling_seed is ignored with speculative decoding unless "
            "--enable-deterministic-inference is set."
        )
        return False
    return True
