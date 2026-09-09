"""Shared storage and original-sampler reference for compact verification."""

import torch
import torch.distributed as dist


def load_reference():
    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
    )

    return chain_speculative_sampling_triton


class Verify:
    def __init__(self, local, q, candidates, coins, final_coins):
        self.local, self.q = local, q
        self.candidates, self.coins, self.final_coins = candidates, coins, final_coins
        self.b, self.s, self.lv = local.shape
        self.r = self.b * self.s
        self.tp, self.rank = dist.get_world_size(), dist.get_rank()
        self.v = self.lv * self.tp
        self.idx = torch.arange(self.r, device=local.device, dtype=torch.int32).view(
            self.b, self.s
        )
        self.ref = load_reference()
        self.batch_idx = torch.arange(self.b, device=local.device)
        # Acceptance at row j consumes candidate j+1; bonus row statistic unused.
        self.targets = (
            torch.cat((candidates[:, 1:], candidates[:, :1]), 1).reshape(-1).long()
        )
        self.local_ids = (self.targets - self.rank * self.lv).clamp(0, self.lv - 1)
        self.owned = (self.targets >= self.rank * self.lv) & (
            self.targets < (self.rank + 1) * self.lv
        )
        self.row_idx = torch.arange(self.r, device=local.device)

    def gather(self, local):
        shape = local.shape
        out = torch.empty((self.tp, *shape), device=local.device, dtype=local.dtype)
        dist.all_gather_into_tensor(
            out.view(-1, shape[-1]), local.reshape(-1, shape[-1]).contiguous()
        )
        return out.movedim(0, -2).reshape(*shape[:-1], self.v)

    def sample(self, p, q, candidates, idx, coins):
        b, s = candidates.shape
        predict = torch.zeros((b, s), dtype=torch.int32, device=p.device)
        accept_idx = torch.full_like(predict, -1)
        count = torch.empty(b, dtype=torch.int32, device=p.device)
        self.ref(
            predict,
            accept_idx,
            count,
            candidates,
            idx,
            None,
            None,
            coins,
            self.final_coins,
            p,
            q,
            1.0,
            1.0,
            True,
        )
        return predict, accept_idx, count

    def baseline(self):
        full = self.gather(self.local).float()
        p = torch.softmax(full, -1)
        result = self.sample(p, self.q, self.candidates, self.idx, self.coins)
        return (*result, p)
