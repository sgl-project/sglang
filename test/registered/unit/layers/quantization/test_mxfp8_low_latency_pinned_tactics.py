"""DSV4.1 low-latency MXFP8 projections: which calls take FlashInfer's
cutedsl_low_latency GEMM, and the pinned tactics the autotuner sees."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.dsv41_low_latency import mxfp8_linear
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase

WQKV_A = (1792, 5120)
OFFERED = [(256, 8, 1, 1), (128, 10, 1, 4), (256, 6, 1, 1), (256, 4, 1, 2)]


def _fp8_method(override=None):
    method = object.__new__(Fp8LinearMethod)
    method.mxfp8_cutedsl_override = override
    return method


class _Linear(LinearBase):
    def __init__(self, n, k, quant_method):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.zeros(n, k), requires_grad=False)
        self.quant_method = quant_method


def _fake_gemm_module():
    module = types.ModuleType(mxfp8_linear._GEMM_MODULE)
    module.autotune_tactics = lambda problem_mnkl, *args, **kwargs: list(OFFERED)
    return module


class TestPinnedTacticLookup(CustomTestCase):
    def test_bucket_eight_pins_on_gb300(self):
        n, k = WQKV_A
        # FlashInfer tunes 5-8 rows as bucket 8, so all of them must pin.
        for rows in (5, 6, 8):
            self.assertEqual(
                mxfp8_linear._pinned_tactic(sm=103, n=n, k=k, rows=rows),
                (128, 10, 1, 4),
            )

    def test_other_rows_shapes_and_gpus_do_not_pin(self):
        n, k = WQKV_A
        for rows in (1, 2, 4, 9):
            self.assertIsNone(mxfp8_linear._pinned_tactic(sm=103, n=n, k=k, rows=rows))
        self.assertIsNone(mxfp8_linear._pinned_tactic(sm=100, n=n, k=k, rows=6))
        self.assertIsNone(mxfp8_linear._pinned_tactic(sm=103, n=n, k=4096, rows=6))

    def test_table_holds_the_three_tp4_projections(self):
        # wqkv_a, wq_b and wo_b of DeepSeek-V4.1-Flash at TP4.
        self.assertEqual(
            sorted(mxfp8_linear.PINNED_TACTICS),
            [(103, 1792, 5120), (103, 5120, 2048), (103, 8192, 1280)],
        )


class TestPinnedTacticsHook(CustomTestCase):
    def setUp(self):
        self.module = _fake_gemm_module()
        self.original = self.module.autotune_tactics
        for patcher in (
            patch.dict(sys.modules, {mxfp8_linear._GEMM_MODULE: self.module}),
            override_platform(device_sm=103),
        ):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)

    def test_pinned_problem_offers_only_the_pin(self):
        n, k = WQKV_A
        with mxfp8_linear.pinned_tactics():
            # The runner passes (N, M, K, L).
            self.assertEqual(
                self.module.autotune_tactics((n, 6, k, 1)), [(128, 10, 1, 4)]
            )
            self.assertEqual(self.module.autotune_tactics((n, 4, k, 1)), OFFERED)
        self.assertIs(self.module.autotune_tactics, self.original)

    def test_pin_missing_from_offer_falls_back_to_tuning(self):
        n, k = 5120, 2048
        self.module.autotune_tactics = lambda *a, **kw: [(256, 8, 1, 1)]
        with mxfp8_linear.pinned_tactics():
            self.assertEqual(
                self.module.autotune_tactics((n, 8, k, 1)), [(256, 8, 1, 1)]
            )

    def test_restored_after_exception(self):
        with self.assertRaises(RuntimeError):
            with mxfp8_linear.pinned_tactics():
                raise RuntimeError
        self.assertIs(self.module.autotune_tactics, self.original)

    def test_every_pass_pinned_keeps_the_pin_for_any_first_row_count(self):
        """FlashInfer files a bucket at its first tuning and never re-tunes it, so
        whatever row count a pass tunes first, bucket 8 must already hold the pin."""
        n, k = WQKV_A
        for first_rows in range(1, 9):
            with self.subTest(first_rows=first_rows):
                cache = {}
                with mxfp8_linear.pinned_tactics():
                    _tune_like_flashinfer(self.module, cache, n=n, k=k, rows=first_rows)
                with mxfp8_linear.pinned_tactics():
                    tactic = _tune_like_flashinfer(self.module, cache, n=n, k=k, rows=8)
                self.assertEqual(tactic, (128, 10, 1, 4))


def _tune_like_flashinfer(module, cache, *, n, k, rows):
    """FlashInfer 0.7.0.post1 for <= 8 rows: a call tunes every power of two below
    `rows` and `rows`, each filed under the next power of two; a filed bucket is
    not re-tuned, and the cached tactic must be one the call offers."""
    profiles = sorted({1 << i for i in range(rows.bit_length())} | {rows})
    for m in profiles:
        bucket = 1 << (m - 1).bit_length()
        # The first offered tactic stands in for the fastest one.
        cache.setdefault((n, k, bucket), module.autotune_tactics((n, m, k, 1))[0])
    tactic = cache[(n, k, 1 << (rows - 1).bit_length())]
    if tactic not in module.autotune_tactics((n, rows, k, 1)):
        raise ValueError(f"Invalid low-latency GEMM tactic: {tactic}")
    return tactic


class TestInstall(CustomTestCase):
    def test_only_fp8_linears_take_the_override(self):
        fp8 = _Linear(16, 128, _fp8_method())
        unquant = _Linear(16, 128, UnquantizedLinearMethod())
        with patch.object(mxfp8_linear, "supported", return_value=True):
            mxfp8_linear.install((fp8, unquant))
        self.assertIs(fp8.quant_method.mxfp8_cutedsl_override, mxfp8_linear._linear)
        self.assertFalse(hasattr(unquant.quant_method, "mxfp8_cutedsl_override"))

    def test_without_the_backend_nothing_is_installed(self):
        fp8 = _Linear(16, 128, _fp8_method())
        with patch.object(mxfp8_linear, "supported", return_value=False):
            mxfp8_linear.install((fp8,))
        self.assertIsNone(fp8.quant_method.mxfp8_cutedsl_override)


class TestLinearRouting(CustomTestCase):
    def setUp(self):
        self.gemm = MagicMock(return_value="low_latency")
        patcher = patch(
            "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_blockscaled_linear",
            self.gemm,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.layer = _Linear(256, 256, _fp8_method())

    def _call(self, x):
        return mxfp8_linear._linear(self.layer, x, torch.ones(4, 4), None)

    def test_one_to_eight_rows_take_the_low_latency_gemm(self):
        for rows in (1, 6, 8):
            self.assertEqual(self._call(torch.zeros(rows, 256)), "low_latency")
        # Rows are counted over the leading dims of a 3-D input.
        self.assertEqual(self._call(torch.zeros(2, 3, 256)), "low_latency")
        kwargs = self.gemm.call_args.kwargs
        self.assertEqual(kwargs["backend"], "cutedsl_low_latency")
        self.assertEqual(kwargs["weight_scale"].shape, (16,))
        self.assertIsNone(kwargs["input_scale"])

    def test_pre_quantized_input_passes_its_flat_scales(self):
        x = (torch.zeros(6, 256), torch.ones(6, 8))
        self.assertEqual(self._call(x), "low_latency")
        kwargs = self.gemm.call_args.kwargs
        self.assertIs(kwargs["input"], x[0])
        self.assertEqual(kwargs["input_scale"].shape, (48,))

    def test_declines_outside_the_kernel_contract(self):
        cases = {
            "9 rows": torch.zeros(9, 256),
            "0 rows": torch.zeros(0, 256),
            "K % 128": torch.zeros(6, 192),
            "non-contiguous": torch.zeros(256, 6).T,
        }
        for name, x in cases.items():
            with self.subTest(name):
                self.assertIsNone(self._call(x))
        self.gemm.assert_not_called()


class TestAutotune(CustomTestCase):
    def test_one_eight_row_call_per_installed_shape_in_sorted_order(self):
        calls = []

        def method(installed=True):
            m = _fp8_method(mxfp8_linear._linear if installed else None)
            m.apply = lambda layer, x: calls.append(
                (tuple(layer.weight.shape), x.shape)
            )
            return m

        model = nn.ModuleList(
            [
                _Linear(8192, 1280, method()),
                _Linear(1792, 5120, method()),
                _Linear(8192, 1280, method()),  # same shape: tuned once
                _Linear(4096, 4096, method(installed=False)),
            ]
        )
        mxfp8_linear.autotune(model, dtype=torch.bfloat16)
        # Sorted, so every rank of the tactic-timing reduce issues the same calls.
        self.assertEqual(
            calls,
            [((1792, 5120), (8, 5120)), ((8192, 1280), (8, 1280))],
        )


if __name__ == "__main__":
    unittest.main()
