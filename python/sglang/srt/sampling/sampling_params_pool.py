"""Each request's sampling state, one table entry per request slot, addressed by
its `req_pool_idx`: its sampler params (temperature, top_p, top_k, min_p), its
penalty index, its seed, and the watermark algorithm's per-request values.

- `write` fills a request's entry at each admission; a launch reads its rows with
  one `gather`, so a step with no admission copies none of them to the device.
- Writes and gathers are enqueued on the scheduler's stream in host order: a
  launch's gather runs before a later admission rewrites a reused slot.
- A captured graph reads the table by its batch's slots at replay, on the
  forward stream (see `SamplingParamsTable`), so a replay stages none of them.
  Under overlap a later admission may rewrite a slot before an earlier replay
  reads it; only a finished, retracted, or aborted row's slot is reused, so the
  row it changes is discarded, and the target sampler reads the launch's gather.
- The host keeps what the batch's `SamplingFacts` and counts need, so a launch
  derives them without a pass over its requests or a device sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import numpy as np
import torch

from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_batch_info import SamplingFacts
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
    order; field names match `SamplingBatchInfo`'s."""

    temperatures: torch.Tensor  # [bs, 1] fp32
    top_ps: torch.Tensor  # [bs] fp32
    top_ks: torch.Tensor  # [bs] int32
    min_ps: torch.Tensor  # [bs] fp32
    facts: SamplingFacts
    # [bs] int64 seeds, or None to sample unseeded; deterministic mode seeds every row.
    sampling_seed: Optional[torch.Tensor]
    # The penalized rows, None when no row is penalized.
    step_penalties: Optional[StepPenalties]
    # The watermark policy, None unless some row is a candidate.
    watermark: Optional[WatermarkRows]

    @classmethod
    def empty(cls, device) -> SamplingRows:
        """The rows of a batch with no requests."""
        return cls(
            temperatures=torch.empty((0, 1), dtype=torch.float32, device=device),
            top_ps=torch.empty(0, dtype=torch.float32, device=device),
            top_ks=torch.empty(0, dtype=torch.int32, device=device),
            min_ps=torch.empty(0, dtype=torch.float32, device=device),
            facts=SamplingFacts(
                all_greedy=True,
                any_greedy=False,
                need_top_p=False,
                need_top_k=False,
                need_min_p=False,
                npu_top_k_top_p_eligible=True,
                max_top_k=None,
            ),
            # A deterministic idle batch still seeds, with no rows.
            sampling_seed=(
                torch.empty(0, dtype=torch.int64, device=device)
                if _deterministic()
                else None
            ),
            step_penalties=None,
            watermark=None,
        )


class SamplingParamsTable:
    """Each request slot's sampling state on the device, one column per slot,
    addressed by its `req_pool_idx`. The model runner builds its
    `SamplingParamsPool` before any graph capture: a graph reads its columns by
    slot index, and the pool writes them at admission."""

    # One row per field, floats as fp32 bits and int64 values as (low, high)
    # int32 halves: one int32 gather reads them all, top_k exact.
    TEMPERATURE, TOP_P, TOP_K, MIN_P, PENALTY_INDEX, SEED_LO, SEED_HI = range(7)
    # The first of the watermark algorithm's fields, which end the table.
    WATERMARK = 7

    def __init__(self, num_rows: int, device):
        self.num_watermark_fields = num_watermark_fields()
        self.values = torch.zeros(
            (self.WATERMARK + self.num_watermark_fields, num_rows),
            dtype=torch.int32,
            device=device,
        )
        floats = self.values.view(torch.float32)
        floats[self.TEMPERATURE] = 1.0
        floats[self.TOP_P] = 1.0
        self.values[self.TOP_K] = TOP_K_ALL
        self.values[self.PENALTY_INDEX] = -1
        # Slot 0 is the request table's padding slot: a graph's padded rows read it,
        # and no request is written to it. Greedy (top_k 1 at temperature 1, as a
        # greedy request is normalized), so those discarded rows draft by argmax
        # instead of sampling.
        self.values[self.TOP_K, 0] = 1

    def temperatures(self, slots: torch.Tensor) -> torch.Tensor:
        """[n] fp32, capturable."""
        return self.values[self.TEMPERATURE].view(torch.float32)[slots]

    def top_ks(self, slots: torch.Tensor) -> torch.Tensor:
        """[n] int32, capturable."""
        return self.values[self.TOP_K][slots]

    def seeds(self, slots: torch.Tensor) -> torch.Tensor:
        """[n] int64, capturable."""
        return _join_seed_halves(
            self.values[self.SEED_LO][slots], self.values[self.SEED_HI][slots]
        )


class SamplingParamsPool:
    def __init__(self, num_rows: int, device):
        self.table = SamplingParamsTable(num_rows, device)
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

        # The table's fields, then the slots in a last row: one copy writes them.
        host_values = np.empty(
            (self.table.values.shape[0] + 1, len(reqs)), dtype=np.int32
        )
        host_values[-1] = slots
        host_floats = host_values.view(np.float32)
        host_floats[self.table.TEMPERATURE] = [p.temperature for p in params]
        host_floats[self.table.TOP_P] = [p.top_p for p in params]
        host_values[self.table.TOP_K] = top_ks
        host_floats[self.table.MIN_P] = [p.min_p for p in params]
        host_values[self.table.PENALTY_INDEX] = penalty_indices
        deterministic = _deterministic()
        seeds = np.array([_seed(r, deterministic) for r in reqs], dtype=np.int64)
        host_values[self.table.SEED_LO : self.table.SEED_HI + 1] = (
            seeds.view(np.int32).reshape(-1, 2).T
        )
        if self.table.num_watermark_fields:
            candidates, values = zip(*(watermark_pool_values(r) for r in reqs))
            self.watermark_candidates_host[slots] = candidates
            host_values[self.table.WATERMARK : -1] = np.array(values, dtype=np.int32).T
        values = async_h2d_tensor(
            torch.from_numpy(host_values), self.table.values.device
        )
        self.table.values.index_copy_(1, values[-1].long(), values[:-1])

    def gather(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        # The pool the penalty indices point into; None where no request takes one.
        penalty_status_pool: Optional[PenaltyStatusPool],
    ) -> SamplingRows:
        slots = req_pool_indices_cpu.numpy()
        if slots.size == 0:
            return SamplingRows.empty(self.table.values.device)
        top_ks = self.top_ks_host[slots]
        lo, hi = int(top_ks.min()), int(top_ks.max())
        # Each row of the [rows, bs] result is a contiguous [bs] tensor.
        values = self.table.values.index_select(1, req_pool_indices)
        floats = values.view(torch.float32)
        seeds = None
        if _deterministic() or _any_row_with_seed(self.has_seed_host[slots], top_ks):
            seeds = _join_seed_halves(
                values[self.table.SEED_LO], values[self.table.SEED_HI]
            )
        return SamplingRows(
            temperatures=floats[self.table.TEMPERATURE].view(-1, 1),
            top_ps=floats[self.table.TOP_P],
            top_ks=values[self.table.TOP_K],
            min_ps=floats[self.table.MIN_P],
            facts=SamplingFacts(
                all_greedy=hi <= 1,
                any_greedy=lo <= 1,
                need_top_p=bool(self.uses_top_p_host[slots].any()),
                need_top_k=(lo, hi) != (TOP_K_ALL, TOP_K_ALL),
                need_min_p=bool(self.uses_min_p_host[slots].any()),
                npu_top_k_top_p_eligible=lo >= 1 and hi <= 1024,
                max_top_k=hi if lo >= 1 and hi < TOP_K_ALL else None,
            ),
            sampling_seed=seeds,
            step_penalties=StepPenalties.build(
                penalty_status_pool,
                values[self.table.PENALTY_INDEX],
                self.penalty_indices_host[slots],
            ),
            watermark=build_watermark_rows(
                values[self.table.WATERMARK :], self.watermark_candidates_host[slots]
            ),
        )


def _join_seed_halves(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """[n] int64 seeds from their int32 halves."""
    return torch.stack((lo, hi), dim=1).view(torch.int64).view(-1)


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
