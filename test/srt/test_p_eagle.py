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
Tests for P-EAGLE parallel speculative decoding.

Covers:
  - fused_parallel_draft_input kernel vs. PyTorch reference
  - Output shape contracts [batch*K, hidden_dim]
  - sync-free DSL: verifies no .item() call in the critical path
  - SpeculativeAlgorithm.PEAGLE / PEAGLE_DSL enum registration
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_cuda_available() -> bool:
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Algorithm registration tests (no GPU needed)
# ---------------------------------------------------------------------------


def test_peagle_algorithm_registered():
    """PEAGLE and PEAGLE_DSL must be valid SpeculativeAlgorithm members."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    peagle = SpeculativeAlgorithm.from_string("PEAGLE")
    peagle_dsl = SpeculativeAlgorithm.from_string("PEAGLE_DSL")

    assert peagle == SpeculativeAlgorithm.PEAGLE
    assert peagle_dsl == SpeculativeAlgorithm.PEAGLE_DSL


def test_peagle_is_eagle():
    """PEAGLE variants must satisfy is_eagle() to share EAGLE-3 infrastructure."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    assert SpeculativeAlgorithm.PEAGLE.is_eagle()
    assert SpeculativeAlgorithm.PEAGLE_DSL.is_eagle()


def test_peagle_is_peagle():
    """is_peagle() must be True for PEAGLE variants and False for others."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    assert SpeculativeAlgorithm.PEAGLE.is_peagle()
    assert SpeculativeAlgorithm.PEAGLE_DSL.is_peagle()
    assert not SpeculativeAlgorithm.EAGLE.is_peagle()
    assert not SpeculativeAlgorithm.EAGLE3.is_peagle()
    assert not SpeculativeAlgorithm.NGRAM.is_peagle()


def test_peagle_is_some():
    """PEAGLE must be a 'some' (active) speculative algorithm."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    assert SpeculativeAlgorithm.PEAGLE.is_some()
    assert SpeculativeAlgorithm.PEAGLE_DSL.is_some()
    assert not SpeculativeAlgorithm.NONE.is_some()


def test_peagle_dsl_threshold_default():
    """peagle_dsl_threshold field must exist in ServerArgs with default 2.0."""
    import dataclasses

    from sglang.srt.server_args import ServerArgs

    fields = {f.name: f for f in dataclasses.fields(ServerArgs)}
    assert (
        "peagle_dsl_threshold" in fields
    ), "peagle_dsl_threshold not found in ServerArgs"
    assert fields["peagle_dsl_threshold"].default == 2.0


# ---------------------------------------------------------------------------
# Kernel correctness tests (GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("K", [1, 3, 4])
@pytest.mark.parametrize("hidden_dim", [512, 2048])
def test_fused_kernel_matches_torch_reference(batch_size, K, hidden_dim):
    """
    Triton fused_parallel_draft_input must match the PyTorch reference
    to within fp16 precision (atol=1e-3).
    """
    from sglang.srt.speculative.triton_ops.fused_draft_input import (
        fused_parallel_draft_input,
        fused_parallel_draft_input_torch,
    )

    device = torch.device("cuda")
    vocab_size = 1024
    mask_token_id = 2

    torch.manual_seed(42)
    h_fused = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device)
    embed_table = torch.randn(
        vocab_size, hidden_dim, dtype=torch.float16, device=device
    )
    last_tokens = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int64, device=device
    )
    h_shared = h_fused.mean(dim=0)

    out_triton = fused_parallel_draft_input(
        h_fused=h_fused,
        embed_table=embed_table,
        last_tokens=last_tokens,
        h_shared=h_shared,
        mask_token_id=mask_token_id,
        K=K,
    )
    out_torch = fused_parallel_draft_input_torch(
        h_fused=h_fused,
        embed_table=embed_table,
        last_tokens=last_tokens,
        h_shared=h_shared,
        mask_token_id=mask_token_id,
        K=K,
    )

    assert out_triton.shape == (
        batch_size * K,
        hidden_dim,
    ), f"Expected shape {(batch_size * K, hidden_dim)}, got {out_triton.shape}"
    assert out_torch.shape == (batch_size * K, hidden_dim)

    torch.testing.assert_close(
        out_triton.float(), out_torch.float(), atol=1e-3, rtol=1e-3
    )


@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA required")
def test_fused_kernel_output_shape():
    """Output shape is exactly [batch*K, hidden_dim] for all valid (batch, K) combos."""
    from sglang.srt.speculative.triton_ops.fused_draft_input import (
        fused_parallel_draft_input,
    )

    device = torch.device("cuda")
    hidden_dim = 1024
    vocab_size = 512
    mask_token_id = 1

    for batch_size, K in [(1, 1), (8, 4), (32, 6)]:
        h_fused = torch.randn(
            batch_size, hidden_dim, dtype=torch.float16, device=device
        )
        embed_table = torch.randn(
            vocab_size, hidden_dim, dtype=torch.float16, device=device
        )
        last_tokens = torch.zeros(batch_size, dtype=torch.int64, device=device)
        h_shared = h_fused.mean(0)

        out = fused_parallel_draft_input(
            h_fused=h_fused,
            embed_table=embed_table,
            last_tokens=last_tokens,
            h_shared=h_shared,
            mask_token_id=mask_token_id,
            K=K,
        )
        assert out.shape == (
            batch_size * K,
            hidden_dim,
        ), f"batch={batch_size} K={K}: expected ({batch_size * K}, {hidden_dim}), got {out.shape}"


@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA required")
def test_position0_is_seq_specific():
    """
    Position 0 rows of each sequence must differ when h_fused rows differ.
    This validates that the kernel uses per-sequence h_fused[seq] for pos=0.
    """
    from sglang.srt.speculative.triton_ops.fused_draft_input import (
        fused_parallel_draft_input,
    )

    device = torch.device("cuda")
    batch_size, K, hidden_dim, vocab_size = 4, 3, 256, 128

    h_fused = torch.eye(batch_size, hidden_dim, dtype=torch.float16, device=device)
    embed_table = torch.zeros(
        vocab_size, hidden_dim, dtype=torch.float16, device=device
    )
    last_tokens = torch.zeros(batch_size, dtype=torch.int64, device=device)
    h_shared = torch.zeros(hidden_dim, dtype=torch.float16, device=device)

    out = fused_parallel_draft_input(
        h_fused=h_fused,
        embed_table=embed_table,
        last_tokens=last_tokens,
        h_shared=h_shared,
        mask_token_id=0,
        K=K,
    )
    # pos=0 rows: out[seq*K + 0] should equal h_fused[seq] + embed[0] = h_fused[seq]
    for seq in range(batch_size):
        row = out[seq * K]  # position 0 for this sequence
        expected = h_fused[seq].float()
        torch.testing.assert_close(row.float(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA required")
def test_positions_gt0_are_shared():
    """
    All position > 0 rows across all sequences should be identical
    when h_shared is the same (validates shared-context logic).
    """
    from sglang.srt.speculative.triton_ops.fused_draft_input import (
        fused_parallel_draft_input,
    )

    device = torch.device("cuda")
    batch_size, K, hidden_dim, vocab_size = 4, 4, 256, 128

    h_fused = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device)
    embed_table = torch.zeros(
        vocab_size, hidden_dim, dtype=torch.float16, device=device
    )
    last_tokens = torch.zeros(batch_size, dtype=torch.int64, device=device)
    h_shared = torch.ones(hidden_dim, dtype=torch.float16, device=device)

    out = fused_parallel_draft_input(
        h_fused=h_fused,
        embed_table=embed_table,
        last_tokens=last_tokens,
        h_shared=h_shared,
        mask_token_id=0,
        K=K,
    )
    # All pos > 0 rows should equal h_shared + embed[mask=0] = h_shared
    expected_shared = h_shared.float()
    for seq in range(batch_size):
        for pos in range(1, K):
            row = out[seq * K + pos].float()
            torch.testing.assert_close(row, expected_shared, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Sync-free DSL: verify no .item() call in the critical kernel path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA required")
def test_dsl_kernel_no_item_call(monkeypatch):
    """
    The DSL sampling kernel must not call .item() (which causes CPU-GPU sync).

    We exercise _draft_sample_with_dsl_kernel — the function that actually
    implements sync-free early exit — rather than fused_parallel_draft_input,
    so the monkeypatched .item() will catch any host synchronization in the
    critical DSL path.
    """
    from sglang.srt.speculative.p_eagle_worker import PEAGLEDSLWorker

    device = torch.device("cuda")
    batch_size, K, vocab_size = 4, 3, 256

    def _no_item(self):
        raise AssertionError(
            "_draft_sample_with_dsl_kernel called .item() — this causes "
            "CPU-GPU sync and violates the sync-free DSL contract."
        )

    monkeypatch.setattr(torch.Tensor, "item", _no_item)

    # Build a minimal PEAGLEDSLWorker instance (no model required)
    worker = object.__new__(PEAGLEDSLWorker)
    worker._dsl_continue_buf = torch.ones(batch_size, dtype=torch.bool, device=device)
    worker.dsl_threshold = 2.0

    all_logits = torch.randn(batch_size, K, vocab_size, dtype=torch.float32, device=device)

    # Should not raise — any .item() call would propagate AssertionError
    tokens, scores = worker._sample_with_dsl_kernel(all_logits, batch_size, K)

    assert tokens.shape == (batch_size, K)
    assert scores.shape == (batch_size, K)
