"""Pinned cuBLASLt algorithms for BF16 linears (SGLANG_OPT_BF16_GEMM_TABLE).

Turns red if the pinned GEMM stops matching torch.mm bit for bit (descriptor
layout, transposes, workspace), if CUDA graph replay of it diverges from eager,
if the table loader accepts a table recorded for another GPU or cuBLASLt
version, keeps an entry that fails its self-check, or if
UnquantizedLinearMethod stops routing table shapes through it.
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
        """First split-K-free algorithm that matches torch.mm on three inputs."""
        for m in SEARCH_M:
            xs = [
                torch.randn(m, K, device="cuda", dtype=torch.bfloat16) for _ in range(3)
            ]
            refs = [torch.mm(x, cls.w.t()) for x in xs]
            rows = cls.lt.enumerate_algos(m, N, K, 32 << 20)
            for row in rows[rows[:, 9] == 1][:128]:
                algo, ws = row[:8].clone(), int(row[8])
                try:
                    outs = [cls.lt.run_algo(x, cls.w, algo, ws) for x in xs]
                except RuntimeError:
                    continue
                if all(_same_bits(o, r) for o, r in zip(outs, refs)):
                    return m, algo, ws
        raise unittest.SkipTest("no split-K-free cuBLASLt algorithm matches torch.mm")

    def setUp(self):
        self.lt._table = None

    def tearDown(self):
        self.lt._table = None

    def _write_table(self, entries, **overrides):
        table = {
            "device_name": self.lt.table_device_name(),
            "cublaslt_version": self.lt.cublaslt_version(),
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

    def test_entry_failing_self_check_is_dropped(self):
        from sglang.srt.environ import envs

        path = self._write_table([self._entry()])
        with (
            envs.SGLANG_OPT_BF16_GEMM_TABLE.override(path),
            mock.patch.object(self.lt, "_matches_torch", return_value=False),
        ):
            x = torch.randn(self.m, K, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(self.lt.pinned_mm(x, self.w))

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
