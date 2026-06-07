# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Unit tests for P-EAGLE parallel speculative decoding (no server required).

Covers:
  - SpeculativeAlgorithm.PEAGLE / PEAGLE_DSL enum registration and routing
  - fused_parallel_draft_input Triton kernel vs. PyTorch reference
  - Output shape contract: [batch*K, hidden_dim]
  - P-EAGLE invariants: pos-0 seq-specific, pos-1..K-1 shared context
  - Sync-free DSL: no .item() call in the fused kernel path
"""

import unittest

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-test-1-gpu-small")


# ---------------------------------------------------------------------------
# Algorithm registration tests (no GPU needed)
# ---------------------------------------------------------------------------


class TestPEagleRegistration(unittest.TestCase):
    """PEAGLE / PEAGLE_DSL enum registration and spec_info routing."""

    def test_peagle_algorithm_registered(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        peagle = SpeculativeAlgorithm.from_string("PEAGLE")
        peagle_dsl = SpeculativeAlgorithm.from_string("PEAGLE_DSL")
        self.assertEqual(peagle, SpeculativeAlgorithm.PEAGLE)
        self.assertEqual(peagle_dsl, SpeculativeAlgorithm.PEAGLE_DSL)

    def test_peagle_is_eagle(self):
        """PEAGLE variants must satisfy is_eagle() to share EAGLE-3 infrastructure."""
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertTrue(SpeculativeAlgorithm.PEAGLE.is_eagle())
        self.assertTrue(SpeculativeAlgorithm.PEAGLE_DSL.is_eagle())

    def test_peagle_is_peagle(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertTrue(SpeculativeAlgorithm.PEAGLE.is_peagle())
        self.assertTrue(SpeculativeAlgorithm.PEAGLE_DSL.is_peagle())
        self.assertFalse(SpeculativeAlgorithm.EAGLE.is_peagle())
        self.assertFalse(SpeculativeAlgorithm.EAGLE3.is_peagle())

    def test_peagle_is_some(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertTrue(SpeculativeAlgorithm.PEAGLE.is_some())
        self.assertTrue(SpeculativeAlgorithm.PEAGLE_DSL.is_some())

    def test_peagle_dsl_threshold_default(self):
        import dataclasses

        from sglang.srt.server_args import ServerArgs

        fields = {f.name: f for f in dataclasses.fields(ServerArgs)}
        self.assertIn("peagle_dsl_threshold", fields)
        self.assertEqual(fields["peagle_dsl_threshold"].default, 2.0)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sgl_kernel"),
        reason="sgl_kernel not installed locally; passes in CI",
    )
    def test_peagle_worker_class_resolved(self):
        from sglang.srt.speculative.p_eagle_worker import PEAGLEDSLWorker, PEAGLEWorker
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertIs(SpeculativeAlgorithm.PEAGLE.get_worker_cls(), PEAGLEWorker)
        self.assertIs(SpeculativeAlgorithm.PEAGLE_DSL.get_worker_cls(), PEAGLEDSLWorker)


# ---------------------------------------------------------------------------
# Kernel correctness tests (GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedDraftInputKernel(unittest.TestCase):
    """fused_parallel_draft_input Triton kernel correctness."""

    def _make(self, B, K, H, vocab=1024, device="cuda"):
        torch.manual_seed(42)
        h = torch.randn(B, H, dtype=torch.float16, device=device)
        emb = torch.randn(vocab, H, dtype=torch.float16, device=device)
        tok = torch.randint(0, vocab, (B,), dtype=torch.int64, device=device)
        hs = h.mean(0)
        return h, emb, tok, hs, 2  # mask_token_id=2

    def _run(self, B, K, H):
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
            fused_parallel_draft_input_torch,
        )

        h, emb, tok, hs, mid = self._make(B, K, H)
        out_triton = fused_parallel_draft_input(h, emb, tok, hs, mid, K)
        out_torch = fused_parallel_draft_input_torch(h, emb, tok, hs, mid, K)
        self.assertEqual(out_triton.shape, (B * K, H))
        torch.testing.assert_close(
            out_triton.float(), out_torch.float(), atol=1e-3, rtol=1e-3
        )

    def test_shape_and_correctness_B1_K4_small(self):
        self._run(1, 4, 512)

    def test_shape_and_correctness_B4_K4_standard(self):
        self._run(4, 4, 2048)

    def test_shape_and_correctness_B8_K6_llama(self):
        self._run(8, 6, 4096)

    def test_position0_uses_seq_specific_h_fused(self):
        """pos=0 row for each seq must equal h_fused[seq] + embed[last_token]."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, K, H, vocab = 4, 3, 256, 128
        h = torch.eye(B, H, dtype=torch.float16, device="cuda")
        emb = torch.zeros(vocab, H, dtype=torch.float16, device="cuda")
        tok = torch.zeros(B, dtype=torch.int64, device="cuda")
        hs = torch.zeros(H, dtype=torch.float16, device="cuda")

        out = fused_parallel_draft_input(h, emb, tok, hs, 0, K)
        for seq in range(B):
            torch.testing.assert_close(
                out[seq * K].float(), h[seq].float(), atol=1e-3, rtol=1e-3
            )

    def test_positions_gt0_use_shared_context(self):
        """pos>0 rows must equal h_shared + embed[MASK] (same across all seqs)."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, K, H, vocab = 4, 4, 256, 128
        h = torch.randn(B, H, dtype=torch.float16, device="cuda")
        emb = torch.zeros(vocab, H, dtype=torch.float16, device="cuda")
        tok = torch.zeros(B, dtype=torch.int64, device="cuda")
        hs = torch.ones(H, dtype=torch.float16, device="cuda")

        out = fused_parallel_draft_input(h, emb, tok, hs, 0, K)
        expected = hs.float()  # h_shared + embed[0]=0
        for seq in range(B):
            for pos in range(1, K):
                torch.testing.assert_close(
                    out[seq * K + pos].float(), expected, atol=1e-3, rtol=1e-3
                )

    def test_no_item_call_sync_free_dsl(self):
        """fused_parallel_draft_input must not call .item() (CPU-GPU sync)."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        original_item = torch.Tensor.item

        def _no_item(self):
            raise AssertionError(
                "fused_parallel_draft_input called .item() — violates sync-free contract"
            )

        torch.Tensor.item = _no_item
        try:
            h, emb, tok, hs, mid = self._make(4, 4, 512)
            out = fused_parallel_draft_input(h, emb, tok, hs, mid, 4)
            self.assertEqual(out.shape, (16, 512))
        finally:
            torch.Tensor.item = original_item

    def test_varying_K(self):
        """Kernel must handle K=1..8 without shape errors."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, H = 2, 1024
        for K in [1, 2, 4, 6, 8]:
            h, emb, tok, hs, mid = self._make(B, K, H)
            out = fused_parallel_draft_input(h, emb, tok, hs, mid, K)
            self.assertEqual(out.shape, (B * K, H), f"K={K}")
