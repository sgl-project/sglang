"""The HIP V4.1 prefill indexer's row-shard exchange: every rank ends with every row of the
selection parts attention reads, candidates stay on the rank that scored them, and the
parts exchanged always include the one attention takes."""

import itertools
import random
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip_radix
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4 import low_ratio_backend_hip as lrb
from sglang.srt.layers.attention.dsv4.v41_indexer.types import RowShard
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class ThreadGroup:
    """``world`` ranks as threads; all_gather_into_tensor completes once every rank calls."""

    def __init__(self, world: int):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=30)
        self.blocks = {}

    def for_rank(self, rank: int):
        def all_gather_into_tensor(out, inp):
            self.blocks[rank] = inp.clone()
            self.barrier.wait()
            out.copy_(torch.cat([self.blocks[r] for r in range(self.world)]))
            self.barrier.wait()

        return SimpleNamespace(all_gather_into_tensor=all_gather_into_tensor)


def run_ranks(world, fn):
    errors, threads = [], []

    def body(rank):
        try:
            fn(rank)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    for rank in range(world):
        threads.append(threading.Thread(target=body, args=(rank,)))
        threads[-1].start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), "a rank hung"
    if errors:
        raise errors[0]


class TestShardExchange(CustomTestCase):
    def exchange(self, *, world, align, parts, extend_lens, pad, topk=8, seed=0):
        """One sharded call on every rank; returns the expected rows and each rank's
        (local rows, page buffer, raw buffer) after finish."""
        gen = random.Random(seed)
        n = sum(extend_lens) + pad
        real = sum(extend_lens)
        truth = {
            p: torch.randint(0, 1 << 20, (n, topk), dtype=torch.int32)
            for p in ("page", "raw")
        }
        # rows that, like identity requests, write only their reachable columns
        short = {i: gen.randint(1, topk) for i in gen.sample(range(n), k=min(n, 5))}
        want = {}
        for p in ("page", "raw"):
            w = torch.full((n, topk), -1, dtype=torch.int32)
            for i in range(real):
                w[i, : short.get(i, topk)] = truth[p][i, : short.get(i, topk)]
            want[p] = w
        group = ThreadGroup(world)
        out = {}

        def rank_fn(rank):
            shard = RowShard(
                group=group.for_rank(rank), rank=rank, world=world, row_align=align
            )
            page = torch.full((n + 16, topk), -1, dtype=torch.int32)
            raw = torch.full((n + 16, topk), -1, dtype=torch.int32)
            ex = lrb._ShardExchange(
                shard, n, parts=parts, page_indices=page, raw_indices=raw
            )
            wp, wr = ex.selection_outputs()
            rows = shard.local_rows(n)
            for i in range(rows.start, min(rows.stop, real)):
                k = short.get(i, topk)
                wp[i, :k] = truth["page"][i, :k]
                wr[i, :k] = truth["raw"][i, :k]
            ex.start()
            ex.finish()
            out[rank] = (rows, page, raw)

        run_ranks(world, rank_fn)
        return want, n, out

    def test_every_rank_gets_every_exchanged_row(self):
        cases = itertools.product(
            (2, 4, 8), (1, 4, 16), (("page",), ("raw",), ("page", "raw"))
        )
        gen = random.Random(7)
        for world, align, parts in cases:
            for trial in range(3):
                extend_lens = [gen.randint(1, 300) for _ in range(gen.randint(1, 4))]
                pad = gen.choice([0, 1, 7, 33])
                with self.subTest(world=world, align=align, parts=parts, trial=trial):
                    want, n, out = self.exchange(
                        world=world,
                        align=align,
                        parts=parts,
                        extend_lens=extend_lens,
                        pad=pad,
                        seed=trial,
                    )
                    for rows, page, raw in out.values():
                        for name, buf in (("page", page), ("raw", raw)):
                            self.assertTrue(bool((buf[n:] == -1).all()))
                            if name in parts:
                                self.assertTrue(torch.equal(buf[:n], want[name]))
                                continue
                            # the part not exchanged keeps only this rank's rows
                            self.assertTrue(torch.equal(buf[rows], want[name][rows]))
                            others = torch.ones(n, dtype=torch.bool)
                            others[rows] = False
                            self.assertTrue(bool((buf[:n][others] == -1).all()))

    def test_empty_last_shares(self):
        # more ranks than aligned shares: the trailing ranks hold no rows
        want, n, out = self.exchange(
            world=8, align=16, parts=("raw",), extend_lens=[20], pad=0
        )
        self.assertEqual(out[7][0].start, out[7][0].stop)
        for _, _, raw in out.values():
            self.assertTrue(torch.equal(raw[:n], want["raw"]))


class TestLocalCandidates(CustomTestCase):
    def test_publication_rows_cover_each_row_once(self):
        gen = random.Random(3)
        for world, align in itertools.product((2, 4, 8), (1, 4, 16)):
            for _ in range(10):
                extend_lens = [gen.randint(1, 200) for _ in range(gen.randint(1, 5))]
                n = sum(extend_lens) + gen.choice([0, 3])
                covered = []
                for rank in range(world):
                    shard = RowShard(
                        group=None, rank=rank, world=world, row_align=align
                    )
                    rows = shard.local_rows(n)
                    lens, lo, hi, offsets = lrb._clip_requests(extend_lens, rows)
                    local = lrb._LocalCandidates.of(
                        [f"c{b}" for b in range(lo, hi)],
                        lo,
                        hi,
                        len(extend_lens),
                        shard_key=(n, rows.start, rows.stop),
                    )
                    self.assertEqual(len(local), len(extend_lens))
                    pub_rows = list(lrb._publication_rows(extend_lens, (lens, offsets)))
                    for b, (cand, r) in enumerate(zip(local, pub_rows)):
                        self.assertEqual(cand is not None, lo <= b < hi)
                        self.assertTrue(r.start >= rows.start or r.start == r.stop)
                        self.assertTrue(r.stop <= rows.stop or r.start == r.stop)
                        covered.extend(range(r.start, r.stop))
                self.assertEqual(sorted(covered), list(range(sum(extend_lens))))

    def test_unsharded_rows(self):
        self.assertEqual(
            list(lrb._publication_rows([3, 2], None)), [slice(0, 3), slice(3, 5)]
        )


class SeqLens(list):
    def max(self):
        return max(iter(self))


class TestGatherParts(CustomTestCase):
    """indexer_gather_parts / _opus_prefill_applies: attention always reads a part that
    was exchanged and never forces OPUS where it would not apply."""

    def backend(self, *, raw_buffer, uniform_fp8, q_state, ratio):
        b = object.__new__(DeepseekV4HipRadixBackend)
        b.is_draft_worker = False
        b.token_to_kv_pool = SimpleNamespace(
            request_window=None, uniform_fp8=uniform_fp8
        )
        core = SimpleNamespace(
            sparse_raw_indices=lambda r: object() if raw_buffer else None
        )
        b.forward_metadata = SimpleNamespace(
            late_layer_tail=None, indexer_gathered_parts={}, core_metadata=core
        )
        b._opus_prefill_q_ok = {} if q_state is None else {ratio: q_state}
        return b

    @staticmethod
    def batch(mode, rows, seq):
        return SimpleNamespace(
            forward_mode=mode, seq_lens_cpu=SeqLens([seq]), extend_seq_lens_cpu=[rows]
        )

    def test_attention_reads_an_exchanged_part(self):
        forced = 0
        grid = itertools.product(
            (True, False),  # OPUS sparse prefill on
            (None, True, False),  # recorded query check
            (True, False),  # raw buffer
            (False, True),  # unified triton
            (False, True),  # uniform fp8
            (1, 2),
        )
        for opus, q_state, raw_buffer, unified, uniform_fp8, ratio in grid:
            for n_idx, extra, seq in itertools.product(
                (512, 1024, 4096, 16384), (0, 1, 3000), (4096, 300_000, 1_100_000)
            ):
                b = self.backend(
                    raw_buffer=raw_buffer,
                    uniform_fp8=uniform_fp8,
                    q_state=q_state,
                    ratio=ratio,
                )
                fb = self.batch(ForwardMode.EXTEND, n_idx, seq)
                q_ok = q_state if q_state is not None else bool(n_idx % 2)
                q = torch.empty((n_idx + extra, 1, 512 if q_ok else 448)).to(
                    torch.bfloat16
                )
                with (
                    envs.SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL.override(opus),
                    patch.object(hip_radix, "is_gfx95_supported", lambda: True),
                    patch(
                        "sglang.srt.layers.attention.dsv4.unified_kv_gate."
                        "is_unified_kv_triton",
                        lambda: unified,
                    ),
                ):
                    parts = b.indexer_gather_parts(ratio, fb, n_idx)
                    applies = b._opus_prefill_applies(q, fb, ratio)
                    natural = q_ok and b._opus_prefill_rows_apply(fb, ratio, q.shape[0])
                if not raw_buffer:
                    self.assertEqual(parts, ("page",))
                    continue
                self.assertIn("raw" if applies else "page", parts)
                if len(parts) == 2:
                    self.assertEqual(applies, natural)
                forced += applies and not natural
        self.assertEqual(forced, 0)

    def test_decode_query_does_not_overwrite_the_prefill_record(self):
        b = self.backend(raw_buffer=True, uniform_fp8=False, q_state=None, ratio=1)
        bad = torch.empty((2048, 1, 448), dtype=torch.bfloat16)
        good = torch.empty((4, 1, 512), dtype=torch.bfloat16)
        with envs.SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL.override(True):
            b._opus_prefill_applies(bad, self.batch(ForwardMode.EXTEND, 2048, 4096), 1)
            self.assertIs(b._opus_prefill_q_ok[1], False)
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                b._opus_prefill_applies(good, self.batch(mode, 4, 4096), 1)
            self.assertIs(b._opus_prefill_q_ok[1], False)


if __name__ == "__main__":
    unittest.main()
