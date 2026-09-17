"""Qualified exponent terms and conservative normalization intervals.

The serving entry checks the pinned runtime. See README.md for the bounded
input domain, binary validation and positive-sum rounding argument.
"""

import torch
import torch.distributed as dist
import triton

from sglang.srt.speculative.compact_verify.exp_dag import partial_stats
from sglang.srt.speculative.compact_verify.queue import FixedQueueVerify


class SourceBoundVerify(FixedQueueVerify):
    radius = 2**-10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.version.git_version != "cf30153c4c131c8164ee7798e5022d810682e2cb":
            raise ValueError("unreviewed Torch revision")
        if self.v != 154880 or self.tp != 4 or self.local.dtype != torch.bfloat16:
            raise ValueError("bound supports only BF16 TP4 V154880")

    def selected_probabilities(self):
        self.targets = (
            torch.cat((self.candidates[:, 1:], self.candidates[:, :1]), 1)
            .reshape(-1)
            .long()
        )
        z = self.local.view(self.r, self.lv)
        lo, hi = torch.aminmax(z, dim=-1)
        valid = torch.isfinite(lo) & torch.isfinite(hi)
        bounds = torch.stack((hi.float(), -lo.float(), (~valid).float()))
        dist.all_reduce(bounds, op=dist.ReduceOp.MAX)
        # Softmax depends on the shifted range, not absolute logit magnitude.
        # exp(-64)/154880 remains a normal FP32 probability.
        self.domain_invalid_rows = (bounds[2] != 0) | (bounds[0] + bounds[1] > 64)
        self.domain_invalid = self.domain_invalid_rows.any()
        # Use the verified exp instruction DAG and emit only tile statistics;
        # no full local FP32/FP64 exponential tensor is materialized.
        tiles = triton.cdiv(self.lv, 2048)
        partials = torch.empty((2, self.r, tiles), device=z.device, dtype=torch.float64)
        partial_stats[(self.r, tiles)](
            z,
            bounds[0],
            self.targets,
            partials,
            self.r,
            self.lv,
            tiles,
            self.rank * self.lv,
            2048,
            num_warps=8,
        )
        stats = partials.sum(-1)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        center = stats[1] / stats[0]
        self.lower = center * (1 - self.radius)
        self.upper = center * (1 + self.radius)
        return center

    def uncertainty_flags(self, p):
        q = self.q.gather(2, self.candidates[:, 1:].long()[..., None]).squeeze(-1)
        product = (self.coins[:, : self.s - 1] * q).double()
        lo = self.lower.view(self.b, self.s)[:, : self.s - 1]
        hi = self.upper.view(self.b, self.s)[:, : self.s - 1]
        flags = ~((product < lo) | (product >= hi))
        return torch.cat(
            (flags, torch.zeros((self.b, 1), device=p.device, dtype=torch.bool)), 1
        ).flatten()

    def fixed(self):
        result = super().fixed()
        invalid = result[4] | self.domain_invalid
        return (
            *(torch.where(invalid, -1, v) for v in result[:3]),
            result[3],
            invalid,
            result[5],
        )
