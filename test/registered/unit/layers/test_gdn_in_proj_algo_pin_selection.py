"""CPU tests for when the GDN in_proj cuBLASLt algorithm pin runs or falls back."""

import contextlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.srt.layers import gdn_in_proj_algo_pin as pin_module
from sglang.srt.layers.moe import flashinfer_tactic_pins
from sglang.test.test_utils import CustomTestCase

METADATA = {
    "flashinfer_version": "0.7.2",
    "cuda_version": "13.0",
    "cublas_version": "13.1.0",
    "gpu": "NVIDIA B200",
    "sm": "10.0",
}
# Synthetic entry; not a tuned choice.
PIN = {"n": 512, "k": 256, "m_min": 65, "m_max": 96, "heuristic_m": 128, "algo_index": 1}


class TestGdnInProjAlgoPin(CustomTestCase):
    def setUp(self):
        for cached in (flashinfer_tactic_pins.read_tuning_file, pin_module._pin_of):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)
        patcher = mock.patch.object(pin_module, "_disabled", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(pin_module, "_declined_m", set())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.weight = torch.zeros(PIN["n"], PIN["k"], dtype=torch.bfloat16)
        self.x = torch.zeros(80, PIN["k"], dtype=torch.bfloat16)

    def _file(self, document):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(document, f)
        self.addCleanup(os.unlink, f.name)
        return f.name

    @contextlib.contextmanager
    def _cuda_like(self, plan):
        """The pin's CUDA-only steps on CPU tensors: a stub plan does the GEMM."""
        with mock.patch.object(
            torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True
        ), mock.patch.object(
            torch.cuda, "device", lambda device: contextlib.nullcontext()
        ), mock.patch.object(
            torch.cuda, "current_blas_handle", return_value=0
        ), mock.patch.object(
            pin_module, "_plan", plan
        ), mock.patch.object(
            pin_module, "tuning_file_applies", return_value=True
        ):
            yield

    def _stub_plan(self):
        module = SimpleNamespace(mm_bf16_cublaslt_run_with_algo=mock.MagicMock())
        return mock.MagicMock(
            return_value=SimpleNamespace(module=module, workspace=None, algo_buf=None)
        )

    def test_parse_rejects_out_of_range_entries(self):
        self.assertEqual(
            pin_module.parse_gdn_in_proj_pin({"gdn_in_proj_cublaslt": PIN}),
            pin_module.GdnInProjAlgoPin(**PIN),
        )
        self.assertIsNone(pin_module.parse_gdn_in_proj_pin({}))
        for bad in (
            dict(PIN, m_min=97),
            dict(PIN, m_max=2048),
            dict(PIN, algo_index=100),
            dict(PIN, heuristic_m=0),
            {k: v for k, v in PIN.items() if k != "n"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pin_module.parse_gdn_in_proj_pin({"gdn_in_proj_cublaslt": bad})

    def test_pinned_gemm_runs_only_when_every_condition_holds(self):
        path = self._file({"_metadata": METADATA, "gdn_in_proj_cublaslt": PIN})
        plan = self._stub_plan()
        with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path), self._cuda_like(plan):
            out = pin_module.gdn_in_proj_pinned_gemm(self.x, self.weight)
            self.assertEqual(tuple(out.shape), (80, PIN["n"]))
            run = plan.return_value.module.mm_bf16_cublaslt_run_with_algo
            self.assertEqual(run.call_args.args[-1], PIN["algo_index"])
            for rows in (64, 97):
                outside = torch.zeros(rows, PIN["k"], dtype=torch.bfloat16)
                self.assertIsNone(pin_module.gdn_in_proj_pinned_gemm(outside, self.weight))
            narrow = self.weight[:, :128].contiguous()
            self.assertIsNone(pin_module.gdn_in_proj_pinned_gemm(self.x, narrow))
            # Batch-invariant mode overrides aten::mm, which this call would bypass.
            with mock.patch.object(
                pin_module, "is_batch_invariant_mode_enabled", return_value=True
            ):
                calls = plan.call_count
                self.assertIsNone(pin_module.gdn_in_proj_pinned_gemm(self.x, self.weight))
                self.assertEqual(plan.call_count, calls)
        self.assertEqual(run.call_count, 1)

    def test_no_entry_or_no_file_keeps_the_stock_gemm(self):
        plan = self._stub_plan()
        moe_only = self._file({"_metadata": METADATA})
        with self._cuda_like(plan):
            with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(None):
                self.assertFalse(pin_module.gdn_in_proj_pin_configured())
                self.assertIsNone(pin_module.gdn_in_proj_pinned_gemm(self.x, self.weight))
            with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(moe_only):
                self.assertFalse(pin_module.gdn_in_proj_pin_configured())
                self.assertIsNone(pin_module.gdn_in_proj_pinned_gemm(self.x, self.weight))
        plan.assert_not_called()

    def test_a_refused_plan_turns_the_pin_off_for_the_process(self):
        path = self._file({"_metadata": METADATA, "gdn_in_proj_cublaslt": PIN})
        plan = mock.MagicMock(side_effect=RuntimeError("no such algorithm"))
        with envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path), self._cuda_like(plan):
            self.assertIsNone(pin_module.gdn_in_proj_pinned_gemm(self.x, self.weight))
            self.assertIsNone(pin_module.gdn_in_proj_pinned_gemm(self.x, self.weight))
        self.assertEqual(plan.call_count, 1)


if __name__ == "__main__":
    unittest.main()
