import itertools
import unittest

import torch

from sglang.kernels.ops.moe.moe_route_warp import muse_spark_route_warp
from sglang.kernels.ops.moe.muse_spark_route_warp_ring import (
    MAX_FUSED_TOKENS,
    muse_spark_ring_pick,
    muse_spark_route_warp_ring,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_TOPK = 8
_ROUTE_SCALE = 2.5
_EPS = 1e-15
_MULTIPLIER = 0.75


def _ring_flows(k, home):
    num_ranks = len(k)
    for t in range(home + 1):
        for y0 in range(k[0] + 1):
            y, prev, ok = [y0] + [0] * (num_ranks - 1), y0, True
            for r in range(num_ranks - 1, 0, -1):
                prev = max(k[r] + prev - t, 0)
                ok = ok and prev <= k[r]
                y[r] = prev
            if ok and k[0] + prev - t <= y0:
                return y
    raise AssertionError("T = home is always feasible")


def _ring_pick(rows, num_ranks, home):
    active = sorted({e for row in rows for e in row if e >= 0})
    k = [sum(1 for e in active if e // home == h) for h in range(num_ranks)]
    y = _ring_flows(k, home)
    phys, seen = {}, [0] * num_ranks
    for e in active:
        h, j = divmod(e, home)
        if seen[h] >= k[h] - y[h]:
            phys[e] = ((h - 1) % num_ranks) * 2 * home + home + j
        else:
            phys[e] = h * 2 * home + j
        seen[h] += 1
    return [[phys[e] if e >= 0 else -1 for e in row] for row in rows]


def _logits(num_tokens, num_ranks, home, hot, gen):
    logits = 4 * torch.randn(num_tokens, num_ranks * home, generator=gen, device="cuda")
    logits[:, : hot * home] += 6
    return logits


class TestMuseSparkRouteWarpRing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 90:
            raise unittest.SkipTest("The fused ring pick needs thread-block clusters")

    def _stock(self, logits, bias, num_valid=None):
        packed = torch.empty(
            logits.shape[0], _TOPK, dtype=torch.int32, device=logits.device
        )
        weights, ids = muse_spark_route_warp(
            logits,
            bias,
            _TOPK,
            _ROUTE_SCALE,
            router_output_multiplier=_MULTIPLIER,
            division_eps=_EPS,
            num_valid_tokens=num_valid,
            packed_out=packed,
        )
        return weights, ids, packed

    def _fused(self, logits, bias, num_ranks, home, num_valid=None):
        return muse_spark_route_warp_ring(
            logits,
            bias,
            _TOPK,
            _ROUTE_SCALE,
            num_ranks=num_ranks,
            home=home,
            router_output_multiplier=_MULTIPLIER,
            division_eps=_EPS,
            num_valid_tokens=num_valid,
        )

    def _check_fused(self, num_tokens, num_ranks, home, hot, num_valid=None):
        gen = torch.Generator(device="cuda").manual_seed(num_tokens * 31 + hot)
        logits = _logits(num_tokens, num_ranks, home, hot, gen)
        bias = torch.randn(num_ranks * home, generator=gen, device="cuda") / 8
        valid = (
            None
            if num_valid is None
            else torch.tensor([num_valid], dtype=torch.int32, device="cuda")
        )
        weights, ids, packed = self._fused(logits, bias, num_ranks, home, valid)
        ref_w, ref_ids, ref_packed = self._stock(logits, bias, valid)
        real = num_tokens if num_valid is None else num_valid
        # Logical ids and weights are the stock router's, bit for bit.
        self.assertTrue(torch.equal(ids, ref_ids))
        self.assertTrue(
            torch.equal(
                weights[:real].view(torch.int32), ref_w[:real].view(torch.int32)
            )
        )
        self.assertTrue(torch.equal(packed & 0xFFFF, ref_packed & 0xFFFF))
        self.assertTrue(torch.equal(packed[real:], ref_packed[real:]))
        want = _ring_pick(ref_ids[:real].tolist(), num_ranks, home)
        self.assertEqual((packed[:real] >> 16).tolist(), want)
        return ref_ids, ref_packed, packed

    def test_fused_matches_reference(self):
        for (num_ranks, home), num_tokens, hot in itertools.product(
            ((8, 32), (4, 64), (2, 64), (16, 32)),
            (1, 2, 7, 8, 9, 24, 33, 63, MAX_FUSED_TOKENS),
            (0, 1, 3),
        ):
            with self.subTest(R=num_ranks, L=home, T=num_tokens, hot=hot):
                self._check_fused(num_tokens, num_ranks, home, hot)

    def test_fused_padded_rows(self):
        for num_tokens, num_valid in ((8, 5), (24, 20), (64, 1), (40, 0)):
            with self.subTest(T=num_tokens, valid=num_valid):
                _, _, packed = self._check_fused(num_tokens, 8, 32, 1, num_valid)
                pad = packed[num_valid:]
                self.assertTrue(bool((pad == -65536).all()))

    def test_triton_matches_fused_and_reference(self):
        scratch = torch.zeros(2 * 8 * 32, dtype=torch.int32, device="cuda")
        for num_tokens, hot in itertools.product((1, 9, 64), (0, 2)):
            with self.subTest(T=num_tokens, hot=hot):
                ids, packed, fused = self._check_fused(num_tokens, 8, 32, hot)
                tri = muse_spark_ring_pick(ids, packed, scratch, num_ranks=8, home=32)
                self.assertTrue(torch.equal(tri, fused))

    def test_triton_beyond_the_fused_limit(self):
        gen = torch.Generator(device="cuda").manual_seed(7)
        for (num_ranks, home), num_tokens in itertools.product(
            ((8, 32), (16, 16), (4, 64)), (65, 72, 200, 256)
        ):
            with self.subTest(R=num_ranks, L=home, T=num_tokens):
                logits = _logits(num_tokens, num_ranks, home, 1, gen)
                bias = torch.zeros(num_ranks * home, device="cuda")
                _, ids, packed = self._stock(logits, bias)
                ids[-3:] = -1  # padded rows keep their packed value
                scratch = torch.zeros(
                    2 * num_ranks * home, dtype=torch.int32, device="cuda"
                )
                out = muse_spark_ring_pick(
                    ids, packed, scratch, num_ranks=num_ranks, home=home
                )
                self.assertTrue(torch.equal(out & 0xFFFF, packed & 0xFFFF))
                self.assertTrue(torch.equal(out[-3:], packed[-3:]))
                self.assertEqual(
                    (out[:-3] >> 16).tolist(),
                    _ring_pick(ids[:-3].tolist(), num_ranks, home),
                )

    def test_triton_arbitrary_active_sets(self):
        # Covers load vectors that routing rarely produces (empty and full homes).
        gen = torch.Generator().manual_seed(0)
        scratch = torch.zeros(2 * 8 * 32, dtype=torch.int32, device="cuda")
        for trial in range(200):
            per_home = torch.randint(0, 33, (8,), generator=gen).tolist()
            active = [
                h * 32 + j
                for h, n in enumerate(per_home)
                for j in torch.randperm(32, generator=gen)[:n].tolist()
            ]
            if not active:
                continue
            ids = torch.tensor(active, dtype=torch.int32, device="cuda").view(-1, 1)
            out = muse_spark_ring_pick(ids, ids << 16, scratch, num_ranks=8, home=32)
            with self.subTest(trial=trial, per_home=per_home):
                self.assertEqual((out >> 16).tolist(), _ring_pick(ids.tolist(), 8, 32))

    def test_cuda_graph_replay(self):
        num_ranks, home = 8, 32
        bias = torch.zeros(num_ranks * home, device="cuda")
        scratch = torch.zeros(2 * num_ranks * home, dtype=torch.int32, device="cuda")
        for num_tokens in (8, 64, 100):
            gen = torch.Generator(device="cuda").manual_seed(num_tokens)
            logits = _logits(num_tokens, num_ranks, home, 0, gen)

            def run():
                if num_tokens <= MAX_FUSED_TOKENS:
                    return self._fused(logits, bias, num_ranks, home)[2]
                _, ids, packed = self._stock(logits, bias)
                return muse_spark_ring_pick(
                    ids, packed, scratch, num_ranks=num_ranks, home=home
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            for hot in (2, 5):
                with self.subTest(T=num_tokens, hot=hot):
                    logits.copy_(_logits(num_tokens, num_ranks, home, hot, gen))
                    graph.replay()
                    torch.cuda.synchronize()
                    _, ref_ids, _ = self._stock(logits, bias)
                    self.assertEqual(
                        (out >> 16).tolist(),
                        _ring_pick(ref_ids.tolist(), num_ranks, home),
                    )


if __name__ == "__main__":
    unittest.main()
