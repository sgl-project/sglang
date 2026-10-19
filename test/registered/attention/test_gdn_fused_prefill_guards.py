import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import gdn_fused_prefill_aiter as adapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestPrefillRowsMatchRealTokens(CustomTestCase):
    """The fused GDN prefill kernel has no DP-padding trim of its own, so it must
    decline any batch whose projected rows exceed the real varlen tokens -- else
    it folds pad rows into the conv/SSM state or leaves the output tail unwritten.
    Guards the two decline conditions (global count below rows; cu_seqlens[-1],
    i.e. sum(extend_seq_lens), not equal to rows) against a rewrite that lets a
    padded batch through.
    """

    def test_unpadded_batch_is_accepted(self):
        self.assertTrue(
            adapter.prefill_rows_match_real_tokens(
                num_rows=10,
                global_num_token_non_padded_cpu=10,
                extend_seq_lens_cpu=[4, 6],
            )
        )

    def test_dp_padded_global_below_rows_declines(self):
        self.assertFalse(
            adapter.prefill_rows_match_real_tokens(
                num_rows=12,
                global_num_token_non_padded_cpu=10,
                extend_seq_lens_cpu=[4, 6],
            )
        )

    def test_cu_seqlens_total_below_rows_declines(self):
        # sum(extend_seq_lens) == cu_seqlens[-1] == 10, but 10 rows were projected
        # to 12: the padded tail would be uninitialized.
        self.assertFalse(
            adapter.prefill_rows_match_real_tokens(
                num_rows=12,
                global_num_token_non_padded_cpu=None,
                extend_seq_lens_cpu=[4, 6],
            )
        )

    def test_missing_extend_lens_declines(self):
        self.assertFalse(
            adapter.prefill_rows_match_real_tokens(
                num_rows=10,
                global_num_token_non_padded_cpu=None,
                extend_seq_lens_cpu=None,
            )
        )


class TestOutProjAcceptsGroup128Fp8(CustomTestCase):
    """The kernel epilogue emits group-128 FP8 activations with fp32 [M, K//128]
    scales, which only a block-FP8 GEMM with a K-block of 128 consumes directly.
    Guards that MXFP8 (wants group-32 e8m0 scales) and any other K-block are NOT
    fed the FP8 stash, so they take the bf16 re-quant path instead of a silently
    mismatched GEMM.
    """

    def _fp8_method(self, *, block_quant, use_mxfp8, weight_block_size):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        method = object.__new__(Fp8LinearMethod)
        method.block_quant = block_quant
        method.use_mxfp8 = use_mxfp8
        method.weight_block_size = weight_block_size
        return method

    def test_block_fp8_k128_accepts(self):
        method = self._fp8_method(
            block_quant=True, use_mxfp8=False, weight_block_size=[128, 128]
        )
        self.assertTrue(adapter.out_proj_accepts_group128_fp8(method))

    def test_mxfp8_declines(self):
        method = self._fp8_method(
            block_quant=True, use_mxfp8=True, weight_block_size=[1, 32]
        )
        self.assertFalse(adapter.out_proj_accepts_group128_fp8(method))

    def test_non_128_kblock_declines(self):
        method = self._fp8_method(
            block_quant=True, use_mxfp8=False, weight_block_size=[128, 64]
        )
        self.assertFalse(adapter.out_proj_accepts_group128_fp8(method))

    def test_per_tensor_fp8_declines(self):
        method = self._fp8_method(
            block_quant=False, use_mxfp8=False, weight_block_size=None
        )
        self.assertFalse(adapter.out_proj_accepts_group128_fp8(method))

    def test_non_fp8_method_declines(self):
        self.assertFalse(adapter.out_proj_accepts_group128_fp8(object()))


class TestCoveredEarlyDeclines(CustomTestCase):
    """covered() screens the two contract items AITER's own predicate cannot see
    before it ever reaches the kernel. Guards that a non-SiLU output gate and a
    missing conv bias decline (and never fall through to a raise).
    """

    def _tensor(self):
        return torch.zeros(1)

    def test_non_silu_activation_declines(self):
        ok, reason = adapter.covered(
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),  # conv_bias present
            self._tensor(),  # A_log
            self._tensor(),  # dt_bias
            self._tensor(),  # norm_weight
            "gelu",
            torch.float8_e4m3fn,
        )
        self.assertFalse(ok)
        self.assertIn("SiLU", reason)

    def test_missing_conv_bias_declines(self):
        ok, reason = adapter.covered(
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            self._tensor(),
            None,  # conv_bias missing
            self._tensor(),
            self._tensor(),
            self._tensor(),
            "silu",
            torch.float8_e4m3fn,
        )
        self.assertFalse(ok)
        self.assertIn("conv bias", reason)


class TestCoveredVectorDtypeContract(CustomTestCase):
    """Once AITER's predicate passes, covered() fixes the per-vector dtype the
    kernel indexes off the pointer: A_log fp32 (full-precision decay), dt_bias and
    norm_weight bf16. Guards against loosening any of them back (e.g. an fp32
    dt_bias that the kernel would reject) or drifting the element counts. _ops is
    stubbed so AITER's own predicate is not needed on a CPU runner.
    """

    V_HEADS = 8
    HEAD_V = 128

    def _delta_state(self):
        # covered() reads shape[1]=v_heads and shape[2]=head_v_dim off delta_state.
        return torch.empty((1, self.V_HEADS, self.HEAD_V, self.HEAD_V))

    def _covered(self, *, a_log, dt_bias, norm_weight):
        t = torch.zeros(1)
        with patch.object(
            adapter, "_ops", return_value=(None, lambda *a, **k: (True, ""))
        ):
            return adapter.covered(
                t,
                t,
                t,
                self._delta_state(),
                t,
                t,
                t,
                t,
                t,  # tensors up to conv_bias
                a_log,
                dt_bias,
                norm_weight,
                "silu",
                torch.float8_e4m3fn,
            )

    def _vec(self, n, dtype):
        return torch.zeros(n, dtype=dtype)

    def test_real_contract_is_covered(self):
        ok, reason = self._covered(
            a_log=self._vec(self.V_HEADS, torch.float32),
            dt_bias=self._vec(self.V_HEADS, torch.bfloat16),
            norm_weight=self._vec(self.HEAD_V, torch.bfloat16),
        )
        self.assertTrue(ok, reason)

    def test_fp32_dt_bias_declines(self):
        ok, reason = self._covered(
            a_log=self._vec(self.V_HEADS, torch.float32),
            dt_bias=self._vec(self.V_HEADS, torch.float32),
            norm_weight=self._vec(self.HEAD_V, torch.bfloat16),
        )
        self.assertFalse(ok)
        self.assertIn("dt_bias", reason)

    def test_bf16_a_log_declines(self):
        ok, reason = self._covered(
            a_log=self._vec(self.V_HEADS, torch.bfloat16),
            dt_bias=self._vec(self.V_HEADS, torch.bfloat16),
            norm_weight=self._vec(self.HEAD_V, torch.bfloat16),
        )
        self.assertFalse(ok)
        self.assertIn("A_log", reason)

    def test_fp32_norm_weight_declines(self):
        ok, reason = self._covered(
            a_log=self._vec(self.V_HEADS, torch.float32),
            dt_bias=self._vec(self.V_HEADS, torch.bfloat16),
            norm_weight=self._vec(self.HEAD_V, torch.float32),
        )
        self.assertFalse(ok)
        self.assertIn("norm_weight", reason)

    def test_wrong_element_count_declines(self):
        ok, reason = self._covered(
            a_log=self._vec(self.V_HEADS + 1, torch.float32),
            dt_bias=self._vec(self.V_HEADS, torch.bfloat16),
            norm_weight=self._vec(self.HEAD_V, torch.bfloat16),
        )
        self.assertFalse(ok)
        self.assertIn("A_log", reason)


if __name__ == "__main__":
    unittest.main()
