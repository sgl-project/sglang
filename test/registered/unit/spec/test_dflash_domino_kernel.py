"""Correctness test for the DFLASH Domino full-vocab fused scoring kernel.

The Domino rollout selects each draft token as the argmax of
``base_logits + domino_bias`` over the whole vocabulary, where
``domino_bias = SiLU(z_proj + s_proj) @ fc2.T + fc2_bias``. ``fused_silu_fc2_argmax``
fuses that into one Triton matmul + reduction. This test pins the fused result
to a dense ``torch`` reference so the first PR's "full-vocab, no candidate-pool
approximation" claim is verified, not assumed.

A mutation check perturbs ``base_logits`` at the reference-selected token and
asserts the fused argmax moves, so the equivalence is not vacuous.
"""

import unittest

import torch

from sglang.srt.speculative.domino_kernels import fused_silu_fc2_argmax
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "Domino kernel requires CUDA")
class TestDominoFusedScoringKernel(CustomTestCase):
    def _make_inputs(self, B, M, V, has_bias, dtype, seed):
        torch.manual_seed(seed)
        dev = "cuda"
        z = torch.randn(B, M, device=dev, dtype=dtype)
        s = torch.randn(B, M, device=dev, dtype=dtype)
        fc2_w = torch.randn(V, M, device=dev, dtype=dtype) * 0.1
        fc2_b = torch.randn(V, device=dev, dtype=dtype) * 0.1 if has_bias else None
        base = torch.randn(B, V, device=dev, dtype=dtype)
        return z, s, fc2_w, fc2_b, base

    def _run_fused(self, z, s, fc2_w, fc2_b, base, block_v=512):
        B = z.shape[0]
        V = base.shape[1]
        num_v_blocks = (V + block_v - 1) // block_v
        out_val = torch.empty(B, num_v_blocks, dtype=torch.float32, device=z.device)
        out_idx = torch.empty(B, num_v_blocks, dtype=torch.int32, device=z.device)
        final_token = torch.empty(B, dtype=torch.long, device=z.device)
        fused_silu_fc2_argmax(
            z_proj=z,
            s_proj=s,
            fc2_weight=fc2_w,
            fc2_bias=fc2_b,
            base_logits=base,
            out_val=out_val,
            out_idx=out_idx,
            final_token=final_token,
            block_v=block_v,
        )
        torch.cuda.synchronize()
        return final_token

    def test_matches_dense_reference(self):
        # Qwen3-class vocab + a couple of smaller shapes, across dtypes.
        cases = [
            (1, 256, 151936, True, torch.bfloat16, 0),
            (1, 256, 151936, False, torch.bfloat16, 1),
            (4, 256, 151936, True, torch.bfloat16, 2),
            (1, 256, 151936, True, torch.float16, 3),
            (2, 256, 4096, True, torch.float32, 4),
            (1, 256, 50000, False, torch.float32, 5),
        ]
        for B, M, V, has_bias, dtype, seed in cases:
            with self.subTest(B=B, V=V, has_bias=has_bias, dtype=dtype):
                z, s, fc2_w, fc2_b, base = self._make_inputs(
                    B, M, V, has_bias, dtype, seed
                )
                # Dense reference: SiLU(z+s) @ fc2.T + fc2_bias + base_logits,
                # then full-vocab argmax.
                mid = torch.nn.functional.silu(z + s)
                bias = torch.nn.functional.linear(mid, fc2_w, fc2_b)
                logits = (base + bias.to(base.dtype)).float()
                ref = torch.argmax(logits, dim=-1).to(torch.long)
                got = self._run_fused(z, s, fc2_w, fc2_b, base)
                # Argmax must match where there is no tie; if it differs, the two
                # tokens must carry the same (max) logit value within tolerance.
                ref_val = logits.gather(1, ref.view(-1, 1)).squeeze(1)
                got_val = logits.gather(1, got.view(-1, 1)).squeeze(1)
                self.assertTrue(
                    torch.allclose(ref_val, got_val, atol=1e-3, rtol=0),
                    f"fused argmax logit value diverged: ref={ref_val} got={got_val}",
                )

    def test_mutation_changes_argmax(self):
        # Make the fused selection sensitive: spike base_logits at a fixed token
        # for batch row 0 and assert the fused argmax follows it.
        z, s, fc2_w, fc2_b, base = self._make_inputs(
            1, 256, 4096, True, torch.float32, 7
        )
        spiked = 1234
        base[0, spiked] += 1e4
        got = self._run_fused(z, s, fc2_w, fc2_b, base)
        self.assertEqual(int(got[0].item()), spiked)


if __name__ == "__main__":
    unittest.main()
