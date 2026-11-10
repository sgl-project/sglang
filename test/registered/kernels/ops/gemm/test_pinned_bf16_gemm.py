"""Pinned cuBLASLt algorithms for BF16 linears (SGLANG_OPT_BF16_GEMM_TABLE).

Turns red if the pinned GEMM stops matching torch.mm bit for bit (descriptor
layout, transposes, workspace), if CUDA graph replay of it diverges from eager,
if the table loader accepts a table recorded for another GPU or cuBLASLt
version, keeps an entry that fails its self-check, allocates device memory at
load, self-checks on anything but the caller's weight, shares one cuBLASLt
workspace between streams, or if UnquantizedLinearMethod stops routing table
shapes through it.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

N, K = 2048, 1024
SEARCH_M = (64, 128, 256)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestPinnedBf16Gemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.gemm import pinned_bf16_gemm as lt

        cls.lt = lt
        g = torch.Generator(device="cuda").manual_seed(0)
        cls.w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
        cls.m, cls.algo, cls.ws = cls._find_exact_algo()

    @classmethod
    def _find_exact_algo(cls):
        """First split-K-free algorithm with torch.mm's accumulation order."""
        for m in SEARCH_M:
            checks = [
                cls.lt.order_check_operands(m, N, K, cls.w.device, seed)
                for seed in (1, 2)
            ]
            rows = cls.lt.enumerate_algos(m, N, K, 32 << 20)
            for row in rows[rows[:, 9] == 1][:128]:
                algo, ws = row[:8].clone(), int(row[8])

                def runner(x, w, out, algo=algo, ws=ws):
                    return cls.lt.run_algo(x, w, algo, ws, out=out)

                try:
                    if all(cls.lt.gives_torch_bits(runner, x, w) for x, w in checks):
                        return m, algo, ws
                except RuntimeError:
                    continue
        raise unittest.SkipTest("no split-K-free cuBLASLt algorithm matches torch.mm")

    def _reset(self):
        self.lt._table = None
        self.lt._unchecked.clear()
        self.lt._max_workspace_bytes = 0
        self.lt._workspaces.clear()

    def setUp(self):
        self._reset()

    def tearDown(self):
        self._reset()

    def _write_table(self, entries, **overrides):
        table = {
            "device_name": self.lt.table_device_name(),
            "cublaslt_version": self.lt.cublaslt_version(),
            self.lt.ORDER_CHECKED_KEY: True,
            "entries": entries,
        }
        table.update(overrides)
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(table, f)
        self.addCleanup(os.remove, path)
        return path

    def _entry(self, m=None):
        return {
            "n": N,
            "k": K,
            "m": self.m if m is None else m,
            "cublaslt_algo": self.algo.tolist(),
            "workspace_bytes": self.ws,
        }

    def test_order_check_rejects_other_accumulation_order(self):
        # The same products summed in another order: two half-K partial sums (as
        # split-K would), or K reversed. On plain normal inputs both round to
        # torch.mm's BF16 in nearly every output.
        def same(x, w, out):
            return torch.mm(x, w.t())

        def two_halves(x, w, out):
            h = x.shape[1] // 2
            lo = torch.mm(x[:, :h].float(), w[:, :h].float().t())
            return (lo + torch.mm(x[:, h:].float(), w[:, h:].float().t())).bfloat16()

        def reversed_k(x, w, out):
            return torch.mm(x.flip(1), w.flip(1).t())

        for m in (1, 8):
            x, w = self.lt.order_check_operands(m, N, K, self.w.device, seed=1)
            self.assertTrue(self.lt.gives_torch_bits(same, x, w))
            self.assertFalse(self.lt.gives_torch_bits(two_halves, x, w))
            self.assertFalse(self.lt.gives_torch_bits(reversed_k, x, w))

    def test_order_checked_algorithms_match_on_plain_inputs(self):
        # Every M=1 algorithm that passes the order check gives torch.mm's bits on
        # 64 plain normal inputs too. A plain-input check alone accepts algorithms
        # with another K order that differ on a few such inputs (and in serving).
        n = k = 896
        g = torch.Generator(device="cuda").manual_seed(7)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
        xs = torch.randn(64, 1, k, device="cuda", dtype=torch.bfloat16, generator=g)
        refs = [torch.mm(x, w.t()) for x in xs]
        checks = [
            self.lt.order_check_operands(1, n, k, w.device, seed) for seed in (1, 2)
        ]
        rows = self.lt.enumerate_algos(1, n, k, 32 << 20)
        accepted = 0
        for row in rows[rows[:, 9] == 1][:2048]:
            algo, ws = row[:8].clone(), int(row[8])

            def runner(x, w_, out, algo=algo, ws=ws):
                return self.lt.run_algo(x, w_, algo, ws, out=out)

            try:
                if not all(self.lt.gives_torch_bits(runner, x, w_) for x, w_ in checks):
                    continue
            except RuntimeError:
                continue
            accepted += 1
            for x, ref in zip(xs, refs):
                self.assertTrue(_same_bits(runner(x, w, None), ref), algo.tolist())
        if accepted == 0:
            self.skipTest("no M=1 algorithm has torch.mm's order on this GPU")

    def test_enumerate_reports_split_free_candidates(self):
        rows = self.lt.enumerate_algos(self.m, N, K, 32 << 20)
        self.assertEqual(rows.shape[1], 10)
        self.assertGreater(int((rows[:, 9] == 1).sum()), 0)

    def test_graph_replay_matches_eager(self):
        x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(self.m, N, device="cuda", dtype=torch.bfloat16)
        self.lt.run_algo(x, self.w, self.algo, self.ws, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.lt.run_algo(x, self.w, self.algo, self.ws, out=out)
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(_same_bits(out, torch.mm(x, self.w.t())))

    def test_table_entry_serves_its_shape_only(self):
        from sglang.srt.environ import envs

        path = self._write_table([self._entry()])
        with envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path):
            x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
            y = self.lt.pinned_mm(x, self.w)
            self.assertIsNotNone(y)
            self.assertTrue(_same_bits(y, torch.mm(x, self.w.t())))
            out = torch.empty(self.m, N, device="cuda", dtype=torch.bfloat16)
            self.assertIs(self.lt.pinned_mm(x, self.w, out=out), out)
            self.assertTrue(_same_bits(out, y))
            self.assertIsNone(self.lt.pinned_mm(x[: self.m - 1], self.w))
            wide = torch.randn(self.m, 2 * K, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(self.lt.pinned_mm(wide[:, :K], self.w))

    def test_table_for_other_cublaslt_version_is_ignored(self):
        from sglang.srt.environ import envs

        version = self.lt.cublaslt_version() + 1
        path = self._write_table([self._entry()], cublaslt_version=version)
        with envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path):
            x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(self.lt.pinned_mm(x, self.w))

    def test_table_without_order_check_marker_is_ignored(self):
        from sglang.srt.environ import envs

        for marker in (None, False):
            path = self._write_table([self._entry()], order_checked=marker)
            self._reset()
            with envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path):
                x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
                self.assertIsNone(self.lt.pinned_mm(x, self.w))
                self.assertEqual(self.lt._unchecked, {})

    def test_entry_failing_self_check_is_dropped(self):
        from sglang.srt.environ import envs

        path = self._write_table([self._entry()])
        with (
            envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path),
            mock.patch.object(self.lt, "_matches_torch", return_value=False),
        ):
            x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(self.lt.pinned_mm(x, self.w))

    def test_load_allocates_nothing_and_checks_on_real_weight(self):
        from sglang.srt.environ import envs

        path = self._write_table([self._entry()])
        x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
        self.lt.cublaslt_version()  # JIT build outside the measured window
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        self.lt._table = {}
        self.lt._load_table(path)
        self.assertEqual(torch.cuda.memory_allocated(), before)
        self.assertIn((N, K), self.lt._unchecked)

        self._reset()
        with (
            envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path),
            mock.patch.object(
                self.lt, "_matches_torch", wraps=self.lt._matches_torch
            ) as spy,
        ):
            y = self.lt.pinned_mm(x, self.w)
            self.assertEqual(spy.call_count, 1)
            self.assertIs(spy.call_args.args[0], self.w)
            self.lt.pinned_mm(x, self.w)
            self.assertEqual(spy.call_count, 1)
        self.assertTrue(_same_bits(y, torch.mm(x, self.w.t())))

    def test_unchecked_shape_is_skipped_during_capture(self):
        from sglang.srt.environ import envs

        path = self._write_table([self._entry()])
        x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
        self.lt._table = {}
        self.lt._load_table(path)
        with (
            envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ),
        ):
            self.assertIsNone(self.lt.pinned_mm(x, self.w))
        self.assertIn((N, K), self.lt._unchecked)

    def test_workspace_per_stream_sized_once(self):
        self.lt._max_workspace_bytes = 1 << 20
        device = torch.device("cuda", torch.cuda.current_device())
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        bufs = []
        for s in streams:
            with torch.cuda.stream(s):
                bufs.append(self.lt._workspace(device, 4096))
                again = self.lt._workspace(device, 1 << 20)
                self.assertEqual(again.data_ptr(), bufs[-1].data_ptr())
        self.assertNotEqual(bufs[0].data_ptr(), bufs[1].data_ptr())
        for s in streams:
            key = (device.index, s.cuda_stream)
            self.assertEqual(self.lt._workspaces[key].numel(), 1 << 20)

    def test_concurrent_streams_match_torch(self):
        from sglang.srt.environ import envs

        path = self._write_table([self._entry()])
        xs = [
            torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        refs = [torch.mm(x, self.w.t()) for x in xs]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outs = [torch.empty(self.m, N, device="cuda", dtype=torch.bfloat16) for _ in xs]
        with envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path):
            self.assertIsNotNone(self.lt.pinned_mm(xs[0], self.w))  # load + check
            torch.cuda.synchronize()
            for s in streams:
                s.wait_stream(torch.cuda.current_stream())
            for _ in range(20):
                for s, x, out in zip(streams, xs, outs):
                    with torch.cuda.stream(s):
                        self.lt.pinned_mm(x, self.w, out=out)
            torch.cuda.synchronize()
        for out, ref in zip(outs, refs):
            self.assertTrue(_same_bits(out, ref))

    def test_unquantized_linear_uses_table(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization import unquant

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(self.w, requires_grad=False)
        method = unquant.UnquantizedLinearMethod()
        x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
        path = self._write_table([self._entry()])
        with (
            envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path),
            mock.patch.object(unquant, "_use_bf16_gemm_table", True),
        ):
            self.assertIsNotNone(self.lt.pinned_mm(x, self.w))  # loads the table
            with mock.patch.object(self.lt, "run_algo", wraps=self.lt.run_algo) as spy:
                y = method.apply(layer, x)
                self.assertEqual(spy.call_count, 1)
                out = torch.empty(self.m, N, device="cuda", dtype=torch.bfloat16)
                method.apply_into(layer, x, out)
                self.assertEqual(spy.call_count, 2)
        ref = torch.nn.functional.linear(x, self.w)
        self.assertTrue(_same_bits(y, ref))
        self.assertTrue(_same_bits(out, ref))


if __name__ == "__main__":
    unittest.main()
