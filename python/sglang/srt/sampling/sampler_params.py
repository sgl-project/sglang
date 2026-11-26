"""Each request's sampler params (temperature, top_p, top_k, min_p), one row per
request slot, addressed by its `req_pool_idx`.

- `write` fills a request's row at each admission; a launch reads its rows with
  `gather`, so the scheduler passes only the batch's slot indices.
- Writes and gathers are enqueued on the scheduler's stream in host order: a
  launch's gather runs before a later admission rewrites a reused slot.
- The host keeps what the batch-level flags need, so a launch derives them
  without a pass over its requests or a device sync.
- A launch over the same slots as the previous gather, with no write since,
  reuses that gather's params: a decode batch whose requests did not change
  keeps one set of tensors. Callers must not write them in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import numpy as np
import torch

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.utils.device import async_h2d, async_h2d_tensor

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


class SamplerParamsPool:
    # Rows of `table`, floats as fp32 bits: one int32 gather reads all four, top_k exact.
    TEMPERATURE, TOP_P, TOP_K, MIN_P = range(4)

    def __init__(self, num_rows: int, device):
        self.table = torch.empty((4, num_rows), dtype=torch.int32, device=device)
        floats = self.table.view(torch.float32)
        floats[self.TEMPERATURE] = 1.0
        floats[self.TOP_P] = 1.0
        self.table[self.TOP_K] = TOP_K_ALL
        floats[self.MIN_P] = 0.0
        self.top_ks_host = np.full(num_rows, TOP_K_ALL, dtype=np.int64)
        self.uses_top_p_host = np.zeros(num_rows, dtype=bool)
        self.uses_min_p_host = np.zeros(num_rows, dtype=bool)
        # The last gather's slots and result; a write invalidates it.
        self._last_gather: Optional[tuple[bytes, SamplerParams]] = None

    def write(self, reqs: Sequence[Req]) -> None:
        slots = [r.kv.req_pool_idx for r in reqs]
        params = [r.sampling_params for r in reqs]
        top_ks = [p.top_k for p in params]
        self.top_ks_host[slots] = top_ks
        self.uses_top_p_host[slots] = [p.top_p != 1.0 for p in params]
        self.uses_min_p_host[slots] = [p.min_p > 0 for p in params]

        host_values = np.empty((4, len(reqs)), dtype=np.int32)
        host_floats = host_values.view(np.float32)
        host_floats[self.TEMPERATURE] = [p.temperature for p in params]
        host_floats[self.TOP_P] = [p.top_p for p in params]
        host_values[self.TOP_K] = top_ks
        host_floats[self.MIN_P] = [p.min_p for p in params]
        rows = async_h2d(slots, dtype=torch.int64, device=self.table.device)
        values = async_h2d_tensor(torch.from_numpy(host_values), self.table.device)
        self.table.index_copy_(1, rows, values)
        self._last_gather = None

    def gather(
        self, req_pool_indices: torch.Tensor, req_pool_indices_cpu: torch.Tensor
    ) -> SamplerParams:
        slots = req_pool_indices_cpu.numpy()
        if slots.size == 0:
            return SamplerParams.empty(self.table.device)
        key = slots.tobytes()
        if self._last_gather is not None:
            last_key, last_params = self._last_gather
            if last_key == key:
                return last_params
        top_ks = self.top_ks_host[slots]
        lo, hi = int(top_ks.min()), int(top_ks.max())
        # Each row of the [4, bs] result is a contiguous [bs] tensor.
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
        self._last_gather = (key, params)
        return params
