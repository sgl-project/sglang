"""Tests for AttnForwardMethod.MHA_CHUNKED_KV on the aiter (ROCm) backend.

Chunked-prefix attention splits the cached prefix into chunks, attends each
separately, and merges the per-chunk partial softmaxes. That is only correct if
three things line up, and each has already been a bug:

1. merge_state must be callable on ROCm. sgl_kernel exports a merge_state_v2
   wrapper there but never registers the op, so the CUDA-gated import leaves the
   name undefined and the chunked path dies with NameError.
2. The lse layout must match. aiter's flash_attn_varlen_func returns the softmax
   lse heads-first [num_heads, num_tokens]; merge_state sizes its grid off the
   output's token/head axes and so indexes [num_tokens, num_heads].
3. The causal flags must be right per slice: causal within the new tokens,
   NON-causal against a prefix chunk (which lies entirely in the queries' past).
   Getting this backwards produces plausible-but-wrong output, not a crash.

The check is end-to-end on the math: chunk + merge must reproduce a single-shot
attention over the whole context. Shapes are Kimi-K3-at-TP8 (12 heads, qk 192,
v 128), which is where the gqa=12 path actually runs.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

NUM_HEADS = 12
QK_HEAD_DIM = 192
V_HEAD_DIM = 128
NUM_Q_TOKENS = 512
SOFTMAX_SCALE = 0.08
# bf16 reassociation floor: splitting the KV stream reorders accumulation, which
# shows up around 3e-3 regardless of implementation.
REL_L2_TOL = 5e-3


class TestAiterMHAChunkedKV(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("GPU required")
        try:
            from aiter import flash_attn_varlen_func
        except ImportError:
            raise unittest.SkipTest("aiter required (ROCm backend test)")
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
            merge_state_v2,
        )

        # staticmethod: a plain function assigned to a class attribute would be
        # bound, and self would arrive as the first kernel argument.
        cls.fa = staticmethod(flash_attn_varlen_func)
        cls.merge = staticmethod(merge_state_v2)
        cls.device = "cuda"

    def _rand(self, *shape):
        return torch.randn(*shape, device=self.device, dtype=torch.bfloat16)

    def _cu_seqlens(self, total):
        return torch.tensor([0, total], device=self.device, dtype=torch.int32)

    def _attend(self, q, k, v, cu_q, cu_kv, max_q, max_kv, causal, return_lse):
        """One slice, mirroring AiterAttnBackend._forward_extend_mha_chunk."""
        out = self.fa(
            q,
            k,
            v,
            cu_q,
            cu_kv,
            max_q,
            max_kv,
            softmax_scale=SOFTMAX_SCALE,
            causal=causal,
            return_lse=return_lse,
        )
        if not return_lse:
            return out
        o, lse = out
        # aiter is heads-first; merge_state wants [num_tokens, num_heads].
        return o, lse.transpose(0, 1).contiguous()

    def _chunked_attention(self, q, k_prefix, v_prefix, k_new, v_new, chunk_lens):
        cu_q = self._cu_seqlens(NUM_Q_TOKENS)
        # Phase 1: the new tokens, causal, seeding the accumulator.
        acc_o, acc_lse = self._attend(
            q, k_new, v_new, cu_q, cu_q, NUM_Q_TOKENS, NUM_Q_TOKENS, True, True
        )
        # Phase 2: one non-causal pass per prefix chunk, merged in.
        offset = 0
        for chunk_len in chunk_lens:
            o, lse = self._attend(
                q,
                k_prefix[offset : offset + chunk_len],
                v_prefix[offset : offset + chunk_len],
                cu_q,
                self._cu_seqlens(chunk_len),
                NUM_Q_TOKENS,
                chunk_len,
                False,
                True,
            )
            merged_o = torch.empty_like(acc_o)
            merged_lse = torch.empty_like(acc_lse)
            self.merge(o, lse, acc_o, acc_lse, merged_o, merged_lse)
            acc_o, acc_lse = merged_o, merged_lse
            offset += chunk_len
        return acc_o

    def _single_shot_attention(self, q, k_prefix, v_prefix, k_new, v_new):
        """Reference: one causal pass over [prefix ; new tokens]."""
        prefix_len = k_prefix.shape[0]
        return self._attend(
            q,
            torch.cat([k_prefix, k_new]),
            torch.cat([v_prefix, v_new]),
            self._cu_seqlens(NUM_Q_TOKENS),
            self._cu_seqlens(prefix_len + NUM_Q_TOKENS),
            NUM_Q_TOKENS,
            prefix_len + NUM_Q_TOKENS,
            True,
            False,
        )

    def _assert_matches_single_shot(self, chunk_lens):
        torch.manual_seed(0)
        prefix_len = sum(chunk_lens)
        q = self._rand(NUM_Q_TOKENS, NUM_HEADS, QK_HEAD_DIM)
        k_prefix = self._rand(prefix_len, NUM_HEADS, QK_HEAD_DIM)
        v_prefix = self._rand(prefix_len, NUM_HEADS, V_HEAD_DIM)
        k_new = self._rand(NUM_Q_TOKENS, NUM_HEADS, QK_HEAD_DIM)
        v_new = self._rand(NUM_Q_TOKENS, NUM_HEADS, V_HEAD_DIM)

        chunked = self._chunked_attention(
            q, k_prefix, v_prefix, k_new, v_new, chunk_lens
        )
        reference = self._single_shot_attention(q, k_prefix, v_prefix, k_new, v_new)

        self.assertTrue(torch.isfinite(chunked).all(), "chunked output has NaN/inf")
        rel_l2 = (
            (chunked.float() - reference.float()).norm() / reference.float().norm()
        ).item()
        self.assertLess(
            rel_l2,
            REL_L2_TOL,
            f"chunks={chunk_lens}: rel L2 {rel_l2:.3e} exceeds {REL_L2_TOL:.0e}",
        )

    def test_multiple_chunks_match_single_shot(self):
        """Merge math: iterated merges must equal one pass over the union.

        Red if the merge is skipped/misordered, if the accumulator is not
        threaded through the loop, if the per-slice causal flags are swapped, or
        if the lse layout handed to merge_state stops matching what it indexes
        (a layout flip corrupts the second and later merges).
        """
        self._assert_matches_single_shot([1024, 1024, 1024])

    def test_uneven_final_chunk(self):
        """Length arithmetic: the last chunk is shorter than the rest.

        Red if a chunk's kv extent is taken from the chunk size rather than the
        chunk's own length -- which the equal-length case cannot see.
        """
        self._assert_matches_single_shot([1024, 1024, 379])


if __name__ == "__main__":
    unittest.main()
