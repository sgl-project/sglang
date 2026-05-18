"""Correctness tests for the optimized embedding_lora_a kernel.

Compares :func:`sglang.srt.lora.triton_ops.embedding_lora_a.embedding_lora_a_fwd`
output against the reference implementation
:func:`sglang.test.lora_utils.reference_embedding_lora_a_shrink` across
various batch sizes, ranks, and edge cases.

Style follows ``test/registered/lora/test_fused_moe_lora_kernel.py``.
"""

import random
import sys
from typing import List

import pytest
import torch

from sglang.srt.lora.triton_ops.embedding_lora_a import embedding_lora_a_fwd
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import set_random_seed
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import reference_embedding_lora_a_shrink


register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")

DEVICE = "cuda:0"
SEED = 42
VOCAB_SIZE = 32000
DTYPE = torch.float16


# ==============================================================================
# HELPERS
# ==============================================================================

def create_batch_info(
    seq_lengths: List[int],
    lora_assignments: List[int],
    lora_ranks: List[int],
    device: str,
) -> LoRABatchInfo:
    """Build a LoRABatchInfo for the triton (non-chunked) backend.

    For the triton backend every segment == one request, so num_segments == bs
    and permutation is identity.
    """
    bs = len(seq_lengths)
    total_tokens = sum(seq_lengths)
    num_loras = len(lora_ranks)

    seg_lens = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=bs,
        num_segments=bs,
        seg_indptr=seg_indptr,
        weight_indices=torch.tensor(
            lora_assignments, dtype=torch.int32, device=device
        ),
        lora_ranks=torch.tensor(lora_ranks, dtype=torch.int32, device=device),
        scalings=torch.ones(num_loras, dtype=torch.float32, device=device),
        max_len=max(seq_lengths) if seq_lengths else 0,
        seg_lens=seg_lens,
        permutation=torch.arange(total_tokens, dtype=torch.int32, device=device),
    )


def create_embedding_weights(
    lora_ranks: List[int],
    max_rank: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Create LoRA A embedding weights of shape (num_loras, max_rank, vocab_size)."""
    num_loras = len(lora_ranks)
    weights = torch.zeros(
        num_loras,
        max_rank,
        vocab_size,
        dtype=dtype,
        device=device,
    )

    for i, rank in enumerate(lora_ranks):
        if rank > 0:
            weights[i, :rank, :] = torch.randn(
                rank, vocab_size, dtype=dtype, device=device
            )

    return weights


def run_kernel_and_reference(
    seq_lengths: List[int],
    lora_assignments: List[int],
    lora_ranks: List[int],
    max_rank: int,
    vocab_size: int = VOCAB_SIZE,
    dtype: torch.dtype = DTYPE,
    device: str = DEVICE,
):
    """Run the Triton kernel and the PyTorch reference, return both outputs."""
    total_tokens = sum(seq_lengths)
    input_ids = torch.randint(
        0,
        vocab_size,
        (total_tokens,),
        dtype=torch.int64,
        device=device,
    )
    weights = create_embedding_weights(
        lora_ranks, max_rank, vocab_size, dtype, device
    )
    batch_info = create_batch_info(seq_lengths, lora_assignments, lora_ranks, device)

    # --- Triton kernel under test ---
    kernel_output = embedding_lora_a_fwd(input_ids, weights, batch_info, vocab_size)

    # --- PyTorch reference (same one used by test_chunked_sgmv_backend) ---
    reference_output = reference_embedding_lora_a_shrink(
        input_ids,
        weights,
        torch.tensor(lora_assignments, dtype=torch.int32, device="cpu"),
        torch.tensor(seq_lengths, dtype=torch.int32, device="cpu"),
        batch_info.lora_ranks.detach().cpu(),
        batch_info.scalings.detach().cpu(),
        vocab_size,
    )

    return kernel_output, reference_output


# ==============================================================================
# TESTS — basic operations
# ==============================================================================

@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
@pytest.mark.parametrize("max_rank", [8, 32])
@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_basic_batch_sizes(batch_size, max_rank, device, seed):
    """Test basic embedding LoRA A for various batch sizes."""
    set_random_seed(seed)
    seq_lengths = [random.randint(1, 128) for _ in range(batch_size)]
    lora_ranks = [max_rank]
    lora_assignments = [0] * batch_size

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths,
        lora_assignments,
        lora_ranks,
        max_rank,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("max_rank", [16, 64])
@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_decode_mode(batch_size, max_rank, device, seed):
    """Test with seq_len=1 (decode mode)."""
    set_random_seed(seed)
    seq_lengths = [1] * batch_size
    lora_ranks = [max_rank]
    lora_assignments = [0] * batch_size

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths,
        lora_assignments,
        lora_ranks,
        max_rank,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


# ==============================================================================
# TESTS — rank parallelization (the core optimization)
# ==============================================================================

@pytest.mark.parametrize("max_rank", [16, 64, 128])
@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_rank_single_block(max_rank, device, seed):
    """rank <= 128 (BLOCK_RANK) — single rank block, baseline behavior."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[32, 64],
        lora_assignments=[0, 1],
        lora_ranks=[max_rank, max_rank],
        max_rank=max_rank,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("max_rank", [256, 384, 512])
@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_rank_multiple_blocks(max_rank, device, seed):
    """rank > 128 — multiple rank blocks parallelized in grid dim 2."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[32, 16],
        lora_assignments=[0, 1],
        lora_ranks=[max_rank, max_rank],
        max_rank=max_rank,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("max_rank", [100, 200, 300])
@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_rank_not_multiple_of_block(max_rank, device, seed):
    """rank not divisible by BLOCK_RANK=128 — tests mask correctness."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[32, 16],
        lora_assignments=[0, 0],
        lora_ranks=[max_rank],
        max_rank=max_rank,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


# ==============================================================================
# TESTS — mixed configurations
# ==============================================================================

@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_mixed_ranks_across_adapters(device, seed):
    """Different adapters use different ranks."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[32, 16, 64, 8],
        lora_assignments=[0, 1, 2, 0],
        lora_ranks=[64, 256, 128],
        max_rank=256,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_uniform_adapter(device, seed):
    """All sequences share the same adapter."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[32, 64, 16, 48],
        lora_assignments=[0, 0, 0, 0],
        lora_ranks=[256],
        max_rank=256,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


# ==============================================================================
# TESTS — edge cases
# ==============================================================================

@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_rank_zero_skipped(device, seed):
    """Adapter with rank=0 should produce zero output for its segment."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[32, 16],
        lora_assignments=[0, 1],
        lora_ranks=[0, 128],
        max_rank=128,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_all_rank_zero(device, seed):
    """All adapters have rank=0 — output should be all zeros."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[32, 16],
        lora_assignments=[0, 1],
        lora_ranks=[0, 0],
        max_rank=1,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("seed", [SEED])
def test_unequal_segment_lengths(device, seed):
    """Segments with very different lengths."""
    set_random_seed(seed)

    kernel_out, ref_out = run_kernel_and_reference(
        seq_lengths=[1, 512, 3, 128],
        lora_assignments=[0, 1, 0, 1],
        lora_ranks=[128, 256],
        max_rank=256,
        device=device,
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
 