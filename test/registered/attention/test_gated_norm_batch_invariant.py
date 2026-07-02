import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.fla.layernorm_gated import (
    calc_rows_per_block,
    rms_norm_gated,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="stage-b", runner_config="1-gpu-large")


class TestGatedNormBatchInvariant(unittest.TestCase):
    """The GDN gated RMSNorm must be batch-size invariant under true-on-policy RL.

    Single-token decode (M=num_v_heads) and Megatron full-sequence teacher-forcing
    (M=seq_len*num_v_heads) must produce bit-identical output for the same row.
    calc_rows_per_block previously scaled ROWS_PER_BLOCK with M, and the 2D-tile fp32
    reduction rounded ~1 bf16 ULP differently for rows_per_block=1 vs 4, diverging
    post_norm_out and cascading down the residual stream.
    """

    def test_rows_per_block_constant_in_batch_invariant_mode(self):
        dev = torch.device("cuda")
        with set_batch_invariant_mode(True):
            vals = {calc_rows_per_block(M, dev) for M in (12, 24, 128, 512, 6144)}
        self.assertEqual(len(vals), 1, f"rows_per_block varies with M: {vals}")

    def test_decode_row_matches_full_sequence(self):
        dev = "cuda"
        HV, D = 12, 128
        with set_batch_invariant_mode(True):
            torch.manual_seed(0)
            weight = torch.randn(D, device=dev, dtype=torch.bfloat16)
            row = 120
            seq = 512
            big_x = torch.randn(seq * HV, D, device=dev, dtype=torch.bfloat16)
            big_z = torch.randn(seq * HV, D, device=dev, dtype=torch.bfloat16)
            y_full = rms_norm_gated(
                x=big_x, weight=weight, bias=None, z=big_z, eps=1e-6,
                norm_before_gate=True, is_rms_norm=True, activation="swish",
            )
            x1 = big_x[row * HV : row * HV + HV].contiguous()
            z1 = big_z[row * HV : row * HV + HV].contiguous()
            y_one = rms_norm_gated(
                x=x1, weight=weight, bias=None, z=z1, eps=1e-6,
                norm_before_gate=True, is_rms_norm=True, activation="swish",
            )
            self.assertTrue(
                torch.equal(y_one, y_full[row * HV : row * HV + HV]),
                msg="decode (M=12) norm not bit-identical to full-sequence (M=6144) row",
            )


if __name__ == "__main__":
    unittest.main()
