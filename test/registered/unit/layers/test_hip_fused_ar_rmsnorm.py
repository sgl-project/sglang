import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.communication.hip_fused_ar_rmsnorm import (
    _ZEROS,
    _aiter_fused_ar_rms,
    aiter_ar_uses_1stage,
    try_fused_ar_rmsnorm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHipFusedArRmsnorm(CustomTestCase):
    def test_aiter_ar_1stage_cutoff_matches_80kib(self):
        # Combined fused-front c2 is 6*3584*2 = 43 KiB (1-stage AR).
        c2 = torch.empty(6, 3584, dtype=torch.bfloat16)
        self.assertTrue(aiter_ar_uses_1stage(c2))
        # c4 is 12*3584*2 = 86 KiB (2-stage AR).
        c4 = torch.empty(12, 3584, dtype=torch.bfloat16)
        self.assertFalse(aiter_ar_uses_1stage(c4))
        c8 = torch.empty(24, 3584, dtype=torch.bfloat16)
        self.assertFalse(aiter_ar_uses_1stage(c8))

    def test_helper_skips_1stage_sizes_unless_forced(self):
        x = torch.zeros(6, 3584, dtype=torch.bfloat16)
        weight = torch.ones(3584, dtype=torch.bfloat16)
        with (
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.envs.SGLANG_ROCM_FUSED_AR_RMSNORM.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm._aiter_fused_ar_rms"
            ) as fused,
        ):
            self.assertIsNone(try_fused_ar_rmsnorm(x, weight, 1e-6))
            fused.assert_not_called()
            try_fused_ar_rmsnorm(x, weight, 1e-6, use_1stage=True)
            fused.assert_called_once()

    def test_helper_stays_off_when_not_hip(self):
        x = torch.zeros(2, 3584)
        weight = torch.ones(3584)
        with (
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.is_hip",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm._aiter_fused_ar_rms"
            ) as fused,
        ):
            self.assertIsNone(try_fused_ar_rmsnorm(x, weight, 1e-6, use_1stage=True))
            fused.assert_not_called()

    def test_helper_stays_off_when_flag_disabled(self):
        x = torch.zeros(2, 3584)
        weight = torch.ones(3584)
        with (
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.envs.SGLANG_ROCM_FUSED_AR_RMSNORM.get",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm._aiter_fused_ar_rms"
            ) as fused,
        ):
            self.assertIsNone(try_fused_ar_rmsnorm(x, weight, 1e-6, use_1stage=True))
            fused.assert_not_called()

    def _run_with_cap(self, x, weight, max_tokens, **kwargs):
        with (
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.envs.SGLANG_ROCM_FUSED_AR_RMSNORM.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.envs."
                "SGLANG_ROCM_FUSED_AR_RMSNORM_MAX_TOKENS.get",
                return_value=max_tokens,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm._aiter_fused_ar_rms"
            ) as fused,
        ):
            try_fused_ar_rmsnorm(x, weight, 1e-6, **kwargs)
            return fused.called

    def test_max_tokens_caps_the_fusion(self):
        # 24 rows = c8, well inside the 2-stage regime the fusion targets.
        x = torch.zeros(24, 3584, dtype=torch.bfloat16)
        weight = torch.ones(3584, dtype=torch.bfloat16)
        for max_tokens, expect_fused in ((0, True), (32, True), (16, True), (8, False)):
            with self.subTest(max_tokens=max_tokens):
                self.assertEqual(
                    self._run_with_cap(x, weight, max_tokens, num_norm_rows=16),
                    expect_fused,
                )

    def test_max_tokens_counts_model_tokens_not_buffer_rows(self):
        # The combined front passes num_norm_rows=num_tokens over a buffer
        # holding several rows per token; the cap follows the tokens.
        x = torch.zeros(24, 3584, dtype=torch.bfloat16)
        weight = torch.ones(3584, dtype=torch.bfloat16)
        self.assertTrue(self._run_with_cap(x, weight, 16, num_norm_rows=8))
        # Without num_norm_rows the rows are the tokens.
        self.assertFalse(self._run_with_cap(x, weight, 16))

    def test_helper_fail_closed_on_missing_communicator(self):
        x = torch.zeros(2, 3584)
        weight = torch.ones(3584)
        with (
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.envs.SGLANG_ROCM_FUSED_AR_RMSNORM.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm._aiter_fused_ar_rms",
                side_effect=RuntimeError("no custom AR"),
            ),
        ):
            self.assertIsNone(try_fused_ar_rmsnorm(x, weight, 1e-6, use_1stage=True))

    def test_helper_reuses_cached_zeros_residual_and_forwards_1stage(self):
        _ZEROS.clear()
        x = torch.zeros(2, 3584, dtype=torch.bfloat16)
        weight = torch.ones(3584, dtype=torch.bfloat16)
        normed = torch.ones_like(x)
        reduced = torch.full_like(x, 2)
        captured = {}

        def _fused(
            *,
            x,
            residual,
            weight,
            eps,
            use_1stage,
            residual_out,
            out,
            num_norm_rows,
            skip_residual,
        ):
            captured["residual"] = residual
            captured["use_1stage"] = use_1stage
            captured["eps"] = eps
            captured["skip_residual"] = skip_residual
            captured["residual_out"] = residual_out
            captured["num_norm_rows"] = num_norm_rows
            captured["out"] = out
            return normed, reduced

        with (
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.envs.SGLANG_ROCM_FUSED_AR_RMSNORM.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm._aiter_fused_ar_rms",
                side_effect=_fused,
            ),
        ):
            first = try_fused_ar_rmsnorm(x, weight, 1e-5, use_1stage=True)
            residual_ptr = captured["residual"].data_ptr()
            second = try_fused_ar_rmsnorm(x, weight, 1e-5, use_1stage=True)

        self.assertIs(first[0], normed)
        self.assertIs(first[1], reduced)
        self.assertIs(second[0], normed)
        self.assertEqual(captured["use_1stage"], True)
        self.assertEqual(captured["eps"], 1e-5)
        self.assertEqual(captured["residual"].data_ptr(), residual_ptr)
        self.assertTrue(torch.count_nonzero(captured["residual"]) == 0)
        self.assertFalse(captured["skip_residual"])

    def test_helper_2stage_skips_residual_and_writes_inplace(self):
        x = torch.zeros(24, 3584, dtype=torch.bfloat16)
        weight = torch.ones(3584, dtype=torch.bfloat16)
        normed = torch.ones(8, 3584, dtype=torch.bfloat16)
        captured = {}

        def _fused(
            *,
            x,
            residual,
            weight,
            eps,
            use_1stage,
            residual_out,
            out,
            num_norm_rows,
            skip_residual,
        ):
            captured["use_1stage"] = use_1stage
            captured["skip_residual"] = skip_residual
            captured["residual_out"] = residual_out
            captured["num_norm_rows"] = num_norm_rows
            captured["out"] = out
            captured["residual_is_inp"] = residual.data_ptr() == x.data_ptr()
            return normed, residual_out if residual_out is not None else x

        with (
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm.envs.SGLANG_ROCM_FUSED_AR_RMSNORM.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_fused_ar_rmsnorm._aiter_fused_ar_rms",
                side_effect=_fused,
            ),
        ):
            got = try_fused_ar_rmsnorm(x, weight, 1e-5, num_norm_rows=8)

        self.assertIs(got[0], normed)
        self.assertIs(got[1], x)
        self.assertEqual(captured["use_1stage"], False)
        self.assertTrue(captured["skip_residual"])
        self.assertIs(captured["residual_out"], x)
        self.assertEqual(captured["num_norm_rows"], 8)
        self.assertTrue(captured["residual_is_inp"])
        self.assertEqual(tuple(captured["out"].shape), (8, 3584))

    def test_combined_layout_keeps_shared_rows_unnormed(self):
        n, dim = 2, 8
        reduced = torch.arange(3 * n * dim, dtype=torch.float32).reshape(3 * n, dim)
        weight = torch.ones(dim)
        rms = reduced * torch.rsqrt(reduced.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        rms = rms * weight
        latent = rms[:n]
        shared = reduced[n:]
        self.assertEqual(tuple(latent.shape), (n, dim))
        self.assertEqual(tuple(shared.shape), (2 * n, dim))
        self.assertFalse(torch.allclose(shared, rms[n:]))

    def test_aiter_dispatch_forwards_k3_kernel_options(self):
        ca_comm = Mock(spec=["disabled", "custom_fused_ar_rms", "_IS_CAPTURING"])
        ca_comm.disabled = False
        ca_comm._IS_CAPTURING = False
        x = torch.zeros(24, 3584, dtype=torch.bfloat16)
        w = torch.ones(3584, dtype=torch.bfloat16)
        out_buf = x[:8]
        with patch(
            "sglang.srt.distributed.get_tp_group",
            return_value=Mock(ca_comm=ca_comm),
        ):
            _aiter_fused_ar_rms(
                x=x,
                residual=x,
                weight=w,
                eps=1e-6,
                use_1stage=False,
                residual_out=x,
                out=out_buf,
                num_norm_rows=8,
                skip_residual=True,
            )
        kwargs = ca_comm.custom_fused_ar_rms.call_args.kwargs
        self.assertIs(ca_comm.custom_fused_ar_rms.call_args.args[4], False)
        self.assertIs(kwargs["residual_out"], x)
        self.assertIs(kwargs["out"], out_buf)
        self.assertEqual(kwargs["num_norm_rows"], 8)
        self.assertTrue(kwargs["skip_residual"])


if __name__ == "__main__":
    unittest.main()
