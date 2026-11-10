"""Dense (non-absorbed) Triton prefill over a materialized prefix + chunk.

Covers ``AttnForwardMethod.MHA_ONE_SHOT`` for Kimi-K3 on the triton backend,
where the cached prefix is up-projected to the 192/128 MHA shape and attended
in one pass instead of running the 576/512 absorbed kernel.
"""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import (
    can_use_dense_prefill_fp8,
    dense_prefill_attention_fwd,
)
from sglang.srt.environ import envs
from sglang.srt.utils import get_device, is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

H_Q, D_QK, D_V = 12, 192, 128
FP8 = torch.float8_e4m3fn


def _reference(q, k, v, qo_indptr, kv_indptr, scale, is_causal):
    """Bottom-right aligned causal attention, one sequence at a time, in fp32."""
    out = torch.empty(q.shape[0], H_Q, D_V, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[0], H_Q, dtype=torch.float32, device=q.device)
    for i in range(len(qo_indptr) - 1):
        q_lo, q_hi = int(qo_indptr[i]), int(qo_indptr[i + 1])
        k_lo, k_hi = int(kv_indptr[i]), int(kv_indptr[i + 1])
        q_len, kv_len = q_hi - q_lo, k_hi - k_lo
        scores = (
            torch.matmul(
                q[q_lo:q_hi].float().transpose(0, 1),
                k[k_lo:k_hi].float().transpose(0, 1).transpose(1, 2),
            )
            * scale
        )
        if is_causal:
            # Query m sits at absolute position (kv_len - q_len) + m.
            q_pos = torch.arange(q_len, device=q.device)[:, None] + (kv_len - q_len)
            k_pos = torch.arange(kv_len, device=q.device)[None, :]
            scores = scores.masked_fill(q_pos < k_pos, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[q_lo:q_hi] = torch.matmul(
            probs, v[k_lo:k_hi].float().transpose(0, 1)
        ).transpose(0, 1)
        lse[q_lo:q_hi] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    return out, lse


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "Kimi-K3 dense Triton prefill requires gfx950"
)
class TestKimiK3TritonDensePrefill(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.device = get_device()
        self.scale = D_QK**-0.5

    def _run(self, q_lens, prefix_lens, *, mode="bf16", is_causal=True, want_lse=False):
        device = self.device
        kv_lens = [q + p for q, p in zip(q_lens, prefix_lens)]
        qo_indptr = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=device)
        kv_indptr = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.tensor(q_lens, device=device).cumsum(0)
        kv_indptr[1:] = torch.tensor(kv_lens, device=device).cumsum(0)
        total_q, total_kv = int(qo_indptr[-1]), int(kv_indptr[-1])

        q = torch.randn(total_q, H_Q, D_QK, dtype=torch.bfloat16, device=device) * 0.25
        k = torch.randn(total_kv, H_Q, D_QK, dtype=torch.bfloat16, device=device) * 0.25
        v = torch.randn(total_kv, H_Q, D_V, dtype=torch.bfloat16, device=device) * 0.25

        # Quantize before taking the reference so the comparison isolates the
        # kernel from the cast: an FP8 run should match FP8 inputs exactly.
        if mode == "fp8":
            q, k, v = q.to(FP8), k.to(FP8), v.to(FP8)
        elif mode == "mixed":
            # What forward_mha_rocm actually hands over on gfx95 with MXFP4
            # kv_b_proj weights: k/v already e4m3 from the fused up-projection,
            # q still bf16.
            k, v = k.to(FP8), v.to(FP8)
        ref_out, ref_lse = _reference(
            q.float() if mode == "bf16" else q.to(FP8).float(),
            k.float() if mode == "bf16" else k.float(),
            v.float() if mode == "bf16" else v.float(),
            qo_indptr,
            kv_indptr,
            self.scale,
            is_causal,
        )

        out = torch.empty(total_q, H_Q, D_V, dtype=torch.bfloat16, device=device)
        lse = (
            torch.empty(total_q, H_Q, dtype=torch.float32, device=device)
            if want_lse
            else None
        )
        dense_prefill_attention_fwd(
            q,
            k,
            v,
            out,
            qo_indptr,
            kv_indptr,
            max(q_lens),
            sm_scale=self.scale,
            is_causal=is_causal,
            lse=lse,
        )
        return out, lse, ref_out, ref_lse

    def test_causal_shapes(self):
        # Prefix lengths deliberately straddle the BLOCK_N=64 boundary: the
        # kernel splits its KV sweep into an unmasked interior and a masked
        # tail at a BLOCK_N multiple, so an off-by-one there is only visible
        # when the prefix is not a clean multiple.
        cases = [
            ([128], [0]),  # no prefix: every block is diagonal
            ([7], [0]),  # q shorter than BLOCK_M
            ([1], [1000]),  # single query, long prefix
            ([128], [63]),
            ([128], [64]),
            ([128], [65]),
            ([256], [1025]),
            ([100, 37, 256, 51], [0, 500, 300, 77]),  # ragged, mixed prefixes
        ]
        for q_lens, prefix_lens in cases:
            with self.subTest(q=q_lens, prefix=prefix_lens):
                out, _, ref, _ = self._run(q_lens, prefix_lens)
                torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    def test_non_causal(self):
        out, _, ref, _ = self._run([128, 64], [256, 130], is_causal=False)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    def test_lse_matches_reference(self):
        out, lse, ref, ref_lse = self._run([192, 64], [300, 129], want_lse=True)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        # LSE is what the chunked-KV merge would consume, so it has to be a
        # natural log in absolute terms, not just proportional.
        torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=1e-3)

    def test_fp8_matches_quantized_reference(self):
        out, _, ref, _ = self._run([128, 64], [512, 77], mode="fp8")
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)

    def test_mixed_bf16_query_fp8_kv(self):
        # Both operands of a tl.dot must share a dtype, so the wrapper has to
        # promote q rather than compile a bf16 x fp8 pair. Regression guard:
        # this shape reaches the kernel straight from forward_mha_rocm.
        out, _, ref, _ = self._run([128, 64], [512, 77], mode="mixed")
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)

    def test_fp8_gate(self):
        device = self.device
        q = torch.empty(1, H_Q, D_QK, dtype=torch.bfloat16, device=device)
        k = torch.empty(1, H_Q, D_QK, dtype=torch.bfloat16, device=device)
        v = torch.empty(1, H_Q, D_V, dtype=torch.bfloat16, device=device)

        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(True):
            self.assertTrue(
                can_use_dense_prefill_fp8(q, k, v, is_causal=True, logit_cap=0.0)
            )
            # Non-causal and logit-capped softmax are outside the gate, and an
            # already-quantized input must not be cast a second time.
            self.assertFalse(
                can_use_dense_prefill_fp8(q, k, v, is_causal=False, logit_cap=0.0)
            )
            self.assertFalse(
                can_use_dense_prefill_fp8(q, k, v, is_causal=True, logit_cap=1.0)
            )
            self.assertFalse(
                can_use_dense_prefill_fp8(
                    q.to(FP8), k, v, is_causal=True, logit_cap=0.0
                )
            )
        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(False):
            self.assertFalse(
                can_use_dense_prefill_fp8(q, k, v, is_causal=True, logit_cap=0.0)
            )


if __name__ == "__main__":
    unittest.main()
