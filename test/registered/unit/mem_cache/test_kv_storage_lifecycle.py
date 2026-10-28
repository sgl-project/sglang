"""A request's lifecycle through the unified KV storage hands out exactly what
driving the request table and the chain allocator directly did: the same rows,
state slots and token ids, in the same order, for every chain shape."""

import random
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

PAGE = 16
_SIZES = dict(page_size=PAGE, full_tokens=64 * PAGE, swa_tokens=48 * PAGE)
CHAINS = {
    "swa": dict(chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3)),
    "mamba": dict(chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3)),
    "tri": dict(
        chain="tri",
        full_layer_ids=(0, 1),
        swa_layer_ids=(2, 3),
        state_layer_ids=(0, 2),
    ),
}


def _never_short(num_tokens: int) -> None:
    pass


class _Side:
    """One copy of the pools, driven either through the storage or directly."""

    def __init__(self, chain: str, through_storage: bool):
        setup_allocator_context()
        self.parts = build_unified_bundle(
            UnifiedGeometry(device="cuda", max_num_reqs=6, **CHAINS[chain], **_SIZES)
        )
        self.storage = self.parts.storage
        self.allocator = self.storage.allocator
        self.table = self.storage.requests
        self.through_storage = through_storage

    def acquire(self, reqs):
        if self.through_storage:
            return self.storage.acquire(reqs)
        return acquire_rows(self.table, self.allocator.state, reqs)

    def extend(self, prefix_lens, seq_lens, last_loc, num_tokens):
        args = dict(
            prefix_lens=torch.tensor(prefix_lens, device="cuda"),
            prefix_lens_cpu=torch.tensor(prefix_lens),
            seq_lens=torch.tensor(seq_lens, device="cuda"),
            seq_lens_cpu=torch.tensor(seq_lens),
            last_loc=last_loc,
        )
        if self.through_storage:
            return self.storage.take_extend(
                rows=None, num_tokens=num_tokens, make_room=_never_short, **args
            )
        return self.allocator.alloc_extend(
            args["prefix_lens"],
            args["prefix_lens_cpu"],
            args["seq_lens"],
            args["seq_lens_cpu"],
            last_loc,
            num_tokens,
        )

    def decode(self, seq_lens, last_loc):
        seq = torch.tensor(seq_lens, device="cuda")
        seq_cpu = torch.tensor(seq_lens)
        if self.through_storage:
            return self.storage.take_decode(
                rows=None,
                seq_lens=seq,
                seq_lens_cpu=seq_cpu,
                last_loc=last_loc,
                make_room=_never_short,
            )
        return self.allocator.alloc_decode(seq, seq_cpu, last_loc)

    def finish(self, req, ids):
        if self.through_storage:
            self.storage.free_tokens(ids)
        else:
            self.allocator.free(ids)
        state = self.allocator.state
        if state is not None:
            state.release(self.table, req)
        if self.through_storage:
            self.storage.release_row(req)
        else:
            self.table.free(req)


def _req():
    return MagicMock(kv=ReqKvInfo())


class TestLifecycleThroughStorage(CustomTestCase):
    def _run(self, chain: str, seed: int):
        rng = random.Random(seed)
        sides = [_Side(chain, through_storage=s) for s in (True, False)]
        live = [[], []]  # per side: (req, ids, seq_len)
        for _ in range(40):
            if live[0] and (len(live[0]) >= 4 or rng.random() < 0.3):
                i = rng.randrange(len(live[0]))
                for side, held in zip(sides, live):
                    req, ids, _ = held.pop(i)
                    side.finish(req, ids)
                continue
            n = rng.randint(1, 3 * PAGE)
            if rng.random() < 0.5 and live[0]:
                # Decode one token for every live request.
                outs = []
                for side, held in zip(sides, live):
                    last = torch.stack([ids[-1] for _, ids, _ in held]).to(torch.int64)
                    seq = [s + 1 for _, _, s in held]
                    out = side.decode(seq, last)
                    outs.append(out)
                    if out is not None:
                        for j, (req, ids, s) in enumerate(held):
                            held[j] = (req, torch.cat([ids, out[j : j + 1]]), s + 1)
                self._same(outs)
                continue
            reqs = [_req(), _req()]
            rows = [side.acquire([r]) for side, r in zip(sides, reqs)]
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(
                [r.kv.mamba_pool_idx for r in reqs[:1]],
                [r.kv.mamba_pool_idx for r in reqs[1:]],
            )
            if rows[0] is None:
                continue
            last = torch.tensor([-1], device="cuda")
            outs = [side.extend([0], [n], last, n) for side in sides]
            self._same(outs)
            if outs[0] is None:
                for side, req in zip(sides, reqs):
                    side.finish(req, torch.empty(0, dtype=torch.int64, device="cuda"))
                continue
            for held, req, out in zip(live, reqs, outs):
                held.append((req, out, n))

    def _same(self, outs):
        a, b = outs
        self.assertEqual(a is None, b is None)
        if a is not None:
            self.assertTrue(torch.equal(a.cpu(), b.cpu()))

    def test_every_chain(self):
        for chain in CHAINS:
            for seed in range(3):
                with self.subTest(chain=chain, seed=seed):
                    self._run(chain, seed)


if __name__ == "__main__":
    unittest.main()
