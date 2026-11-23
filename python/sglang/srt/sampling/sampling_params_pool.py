"""Each request's sampling state, one table entry per request slot, addressed by
its `req_pool_idx`: its sampler params (temperature, top_p, top_k, min_p), its
penalty index, its seed, and the watermark algorithm's per-request values.

- `write` fills a request's entry at each admission; a launch reads its rows with
  one `gather`, so a step with no admission copies none of them to the device.
- Writes and gathers are enqueued on the scheduler's stream in host order: a
  launch's gather runs before a later admission rewrites a reused slot.
- The host keeps what the batch-level flags and counts need, so a launch derives
  them without a pass over its requests or a device sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import numpy as np
import torch

from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.runtime import (
    num_watermark_fields,
    watermark_pool_values,
)
from sglang.srt.utils.device import async_h2d, async_h2d_tensor
from sglang.srt.utils.log_utils import print_warning_once

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class SamplerParams(NamedTuple):
    """One launch's sampler params; field names match `SamplingBatchInfo`'s."""

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

    @classmethod
    def empty(cls, device) -> SamplerParams:
        """The params of a batch with no rows."""
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
        )


class SamplingRows(NamedTuple):
    """One launch's rows of `SamplingParamsPool`, in batch order."""

    sampler_params: SamplerParams
    # [bs] int64 seeds, or None to sample unseeded; deterministic mode seeds every row.
    seeds: Optional[torch.Tensor]
    # [bs] int32 penalty index of each row, -1 for a row without one, and its host copy.
    penalty_indices: torch.Tensor
    penalty_indices_host: np.ndarray
    # [k, bs] the watermark algorithm's values and [bs] whether each row is a
    # candidate on the host; both None without watermarking.
    watermark_values: Optional[torch.Tensor]
    watermark_candidates_host: Optional[np.ndarray]

    @classmethod
    def empty(cls, device) -> SamplingRows:
        """The rows of a batch with no requests."""
        return cls(
            sampler_params=SamplerParams.empty(device),
            # A deterministic idle batch still seeds, with no rows.
            seeds=(
                torch.empty(0, dtype=torch.int64, device=device)
                if _deterministic()
                else None
            ),
            penalty_indices=torch.empty(0, dtype=torch.int32, device=device),
            penalty_indices_host=np.empty(0, dtype=np.int64),
            watermark_values=None,
            watermark_candidates_host=None,
        )


class SamplingParamsPool:
    # Fields of `table`, one row each, floats as fp32 bits and int64 values as
    # (low, high) int32 halves: one int32 gather reads them all, top_k exact.
    TEMPERATURE, TOP_P, TOP_K, MIN_P, PENALTY_INDEX, SEED_LO, SEED_HI = range(7)
    # The first of the watermark algorithm's fields, which end the table.
    WATERMARK = 7

    def __init__(self, num_rows: int, device):
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

        host_values = np.empty((self.table.shape[0], len(reqs)), dtype=np.int32)
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
            host_values[self.WATERMARK :] = np.array(values, dtype=np.int32).T
        rows = async_h2d(slots, dtype=torch.int64, device=self.table.device)
        values = async_h2d_tensor(torch.from_numpy(host_values), self.table.device)
        self.table.index_copy_(1, rows, values)

    def gather(
        self, req_pool_indices: torch.Tensor, req_pool_indices_cpu: torch.Tensor
    ) -> SamplingRows:
        slots = req_pool_indices_cpu.numpy()
        if slots.size == 0:
            return SamplingRows.empty(self.table.device)
        top_ks = self.top_ks_host[slots]
        lo, hi = int(top_ks.min()), int(top_ks.max())
        # Each row of the [rows, bs] result is a contiguous [bs] tensor.
        values = self.table.index_select(1, req_pool_indices)
        floats = values.view(torch.float32)
        params = SamplerParams(
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
        )
        seeds = None
        if _deterministic() or _any_row_with_seed(self.has_seed_host[slots], top_ks):
            seeds = (
                torch.stack((values[self.SEED_LO], values[self.SEED_HI]), dim=1)
                .view(torch.int64)
                .view(-1)
            )
        has_watermark = self.num_watermark_fields > 0
        return SamplingRows(
            sampler_params=params,
            seeds=seeds,
            penalty_indices=values[self.PENALTY_INDEX],
            penalty_indices_host=self.penalty_indices_host[slots],
            watermark_values=values[self.WATERMARK :] if has_watermark else None,
            watermark_candidates_host=(
                self.watermark_candidates_host[slots] if has_watermark else None
            ),
        )


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
