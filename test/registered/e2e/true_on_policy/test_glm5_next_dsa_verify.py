"""TOP GLM-5.3-Flash DSA: a verify window writes the pooled index keys plain decode writes, bit for bit.

The indexer pools every 4 consecutive keys into one FP8 key (softmax over the gate scores, Hadamard rotation, UE8M0
scale). Decode compresses a group when its last key arrives (``kpool_decode_update_and_maybe_write_cache``); a verify
window writes its rows' keys to the request's tail ring and compresses the groups it completes
(``kpool_write_tail_and_maybe_compress``). Under TOP the window must compress with decode's two-pass softmax, and the
keys of rejected drafts (garbage rows past a random accept length) must leave no trace in any committed group.
"""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

KPOOL, DIM, INDEX_PAGE = 4, 128, 64
ROW_BYTES = INDEX_PAGE * (DIM + 4)
dev = "cuda"


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestVerifyCompression(unittest.TestCase):
    LENS = (70, 133, 201, 9)
    # 256-token pages per request: room for the groups garbage drafts complete past a request's end.
    PAGES = 2

    def setUp(self):
        gen = torch.Generator(device=dev).manual_seed(11)
        self.keys = [
            torch.randn(n, DIM, device=dev, generator=gen).to(torch.bfloat16)
            for n in self.LENS
        ]
        self.scores = [
            (torch.randn(n, DIM, device=dev, generator=gen) * 2).to(torch.bfloat16)
            for n in self.LENS
        ]
        self.ape = torch.randn(KPOOL, DIM, device=dev, generator=gen) * 0.5
        self.gen = gen
        R, P = len(self.LENS), self.PAGES
        # Token page c of request r (64 tokens) sits in 256-token page r * P + c // 4.
        self.block_tables = torch.tensor(
            [[(r * P + c // 4) * 4 + c % 4 for c in range(4 * P)] for r in range(R)],
            dtype=torch.int32,
            device=dev,
        )

    def pool(self, n):
        return SimpleNamespace(
            index_page_size=INDEX_PAGE,
            index_head_dim=DIM,
            index_kpool=KPOOL,
            tail_extra_slots=n,
            page_size=KPOOL * INDEX_PAGE,
        )

    def buffers(self, n):
        R = len(self.LENS)
        tail = lambda: torch.zeros(
            R + 1, KPOOL + n, DIM, dtype=torch.bfloat16, device=dev
        )  # noqa: E731
        return (
            torch.zeros(R * self.PAGES, ROW_BYTES, dtype=torch.uint8, device=dev),
            tail(),
            tail(),
        )

    def group(self, buf, r, g):
        page, off = r * self.PAGES + g // INDEX_PAGE, g % INDEX_PAGE
        return torch.cat(
            (
                buf[page, off * DIM : (off + 1) * DIM],
                buf[page, INDEX_PAGE * DIM + 4 * off : INDEX_PAGE * DIM + 4 * off + 4],
            )
        )

    def decode_all(self, n):
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            kpool_decode_update_and_maybe_write_cache,
        )

        buf, tail_k, tail_s = self.buffers(n)
        for t in range(max(self.LENS)):
            live = [r for r, L in enumerate(self.LENS) if t < L]
            kpool_decode_update_and_maybe_write_cache(
                pool=self.pool(n),
                buf=buf,
                tail_k=tail_k,
                tail_score=tail_s,
                key=torch.stack([self.keys[r][t] for r in live]),
                slot_score=torch.stack([self.scores[r][t] for r in live]),
                ape=self.ape,
                block_tables=self.block_tables[live],
                req_pool_indices=torch.tensor([r + 1 for r in live], device=dev),
                positions=torch.full((len(live),), t, device=dev),
                seq_lens=torch.full((len(live),), t + 1, dtype=torch.int32, device=dev),
                out_cache_loc=torch.full((len(live),), t + 1, device=dev),
                round_scale=True,
            )
        return buf

    def verify_all(self, n, *, two_pass, seed, decoded):
        """Verify windows of n rows until every request is committed; returns the committed groups that differ from
        decode's."""
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            kpool_max_closed_pools,
            kpool_write_tail_and_maybe_compress,
        )

        rng = random.Random(seed)
        buf, tail_k, tail_s = self.buffers(n)
        max_pools = kpool_max_closed_pools(n, KPOOL)
        pos = [0] * len(self.LENS)
        differ = []
        while any(p < L for p, L in zip(pos, self.LENS)):
            live = [r for r, L in enumerate(self.LENS) if pos[r] < L]
            accepts = [
                rng.randint(0, min(n - 1, self.LENS[r] - 1 - pos[r])) for r in live
            ]
            keys, scores, locs = [], [], []
            for r, a in zip(live, accepts):
                garbage = n - a - 1
                keys.append(
                    torch.cat(
                        (
                            self.keys[r][pos[r] : pos[r] + a + 1],
                            torch.randn(
                                garbage, DIM, device=dev, generator=self.gen
                            ).to(torch.bfloat16),
                        )
                    )
                )
                scores.append(
                    torch.cat(
                        (
                            self.scores[r][pos[r] : pos[r] + a + 1],
                            torch.randn(
                                garbage, DIM, device=dev, generator=self.gen
                            ).to(torch.bfloat16),
                        )
                    )
                )
                first = pos[r] // KPOOL
                locs.append(
                    [
                        (r * self.PAGES + g // INDEX_PAGE) * INDEX_PAGE + g % INDEX_PAGE
                        for g in range(first, first + max_pools)
                    ]
                )
            kpool_write_tail_and_maybe_compress(
                pool=self.pool(n),
                buf=buf,
                key=torch.cat(keys),
                score=torch.cat(scores),
                tail_k=tail_k,
                tail_score=tail_s,
                ape=self.ape,
                req_pool_indices=torch.tensor(
                    [r + 1 for r in live], dtype=torch.int32, device=dev
                ),
                write_start=torch.tensor(
                    [pos[r] for r in live], dtype=torch.int32, device=dev
                ),
                tail_logical_start=torch.tensor(
                    [pos[r] // KPOOL * KPOOL for r in live],
                    dtype=torch.int32,
                    device=dev,
                ),
                write_loc=torch.tensor(locs, dtype=torch.int64, device=dev),
                out_cache_loc=torch.ones(len(live) * n, dtype=torch.int64, device=dev),
                num_draft_tokens=n,
                round_scale=True,
                two_pass=two_pass,
            )
            for r, a in zip(live, accepts):
                pos[r] += a + 1
                # Every group the committed rows complete is decode's.
                for g in range(pos[r] // KPOOL):
                    if not torch.equal(
                        self.group(buf, r, g), self.group(decoded, r, g)
                    ):
                        differ.append((r, g))
        return sorted(set(differ))

    def test_two_pass_windows_write_decode_groups(self):
        for n in (2, 6):
            with self.subTest(draft_tokens=n):
                decoded = self.decode_all(n)
                self.assertEqual(
                    self.verify_all(n, two_pass=True, seed=n, decoded=decoded), []
                )
                # The native online softmax is not decode's arithmetic (reported, not asserted).
                online = self.verify_all(n, two_pass=False, seed=n, decoded=decoded)
                print(
                    f"draft tokens {n}: online-softmax verify differs from decode in {len(online)} groups"
                )


if __name__ == "__main__":
    unittest.main()
