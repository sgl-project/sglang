"""DeepSeek-V4.1 WO-A takes the exact FP8 weight copy for decode and target
verify of up to WO_A_FP8_MAX_ROWS rows on CUDA Blackwell, and the BF16 kernels
otherwise. Kernels are stubbed; no GPU is needed."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWoAFp8Dispatch(unittest.TestCase):
    def setUp(self):
        self.wo_a = torch.empty(2, 1024, 4096, dtype=torch.bfloat16)
        self.copy = (torch.empty(0), torch.empty(0), torch.empty(0))
        self.calls = []

        def stub(name, result):
            def run(*args, **kwargs):
                self.calls.append(name)
                return result

            return run

        out = torch.empty(0)
        platform = SimpleNamespace(is_blackwell=True, is_sm90=False)
        for target, name, value in (
            (deepseek_v4, "_is_cuda", True),
            (deepseek_v4, "get_platform", lambda: platform),
            (deepseek_v4, "wo_a_fp8_small_batch", stub("fp8", out)),
            (deepseek_v4, "wo_a_fp8_small_batch_mxfp8", stub("fp8_mxfp8", (out, out))),
            (deepseek_v4, "wo_a_bf16_gemv", stub("gemv", out)),
            (deepseek_v4, "wo_a_bf16_small_batch", stub("bf16", out)),
            (torch, "bmm", stub("bmm", None)),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _route(self, rows, *, verify=True, copy=True, mxfp8=False):
        self.calls.clear()
        result = deepseek_v4._apply_wo_a_bf16_matmul(
            torch.empty(rows, 2, 4096, dtype=torch.bfloat16),
            self.wo_a,
            is_decode=not verify,
            is_target_verify=verify,
            fuse_mxfp8_quant=mxfp8,
            fast_path=True,
            wo_a_fp8_copy=self.copy if copy else None,
        )
        return self.calls, result

    def test_copy_serves_decode_and_verify_up_to_the_row_limit(self):
        self.assertEqual(self._route(1, verify=False)[0], ["fp8"])
        for rows in (1, 8, 9, deepseek_v4.WO_A_FP8_MAX_ROWS):
            with self.subTest(rows=rows):
                self.assertEqual(self._route(rows)[0], ["fp8"])

    def test_fused_mxfp8_output(self):
        calls, result = self._route(9, mxfp8=True)
        self.assertEqual(calls, ["fp8_mxfp8"])
        self.assertIsInstance(result, Mxfp8SwizzledInput)

    def test_bf16_kernels_without_the_copy_or_above_the_limit(self):
        self.assertEqual(self._route(1, verify=False, copy=False)[0], ["gemv"])
        self.assertEqual(self._route(8, copy=False)[0], ["bf16"])
        self.assertEqual(self._route(9, copy=False)[0], ["bmm"])
        rows = deepseek_v4.WO_A_FP8_MAX_ROWS + 1
        self.assertEqual(self._route(rows)[0], ["bmm"])


if __name__ == "__main__":
    unittest.main()
