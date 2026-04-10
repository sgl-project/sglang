import pytest
import torch

from sglang.jit_kernel.apply_token_bitmask_inplace import (
    apply_token_bitmask_inplace_jit,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-kernel-unit-1-gpu-large")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
BITS_PER_BLOCK = 32


def _reference_apply(logits, bitmask, indices=None):
    """Pure-Python reference: unpack int32 bitmask and mask logits to -inf."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        bitmask = bitmask.unsqueeze(0)

    batch = logits.size(0)
    vocab = logits.size(1)
    bm_cols = bitmask.size(1)

    rows = list(range(batch)) if indices is None else indices.tolist()
    out = logits.clone()
    for row_idx, batch_idx in enumerate(rows):
        bm_row = row_idx if indices is None else row_idx
        for v in range(min(vocab, bm_cols * BITS_PER_BLOCK)):
            word = v // BITS_PER_BLOCK
            bit = v % BITS_PER_BLOCK
            allowed = (bitmask[bm_row, word].item() >> bit) & 1
            if not allowed:
                out[batch_idx, v] = float("-inf")
    return out


# ---------------------------------------------------------------------------
# Basic: matches sgl-kernel AOT test
# ---------------------------------------------------------------------------


def test_basic_bitmask():
    bool_mask = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool
    )
    logits = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=torch.float32,
    )
    expected = torch.where(bool_mask, logits, torch.tensor(float("-inf")))

    logits_gpu = logits.to("cuda")
    bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
    apply_token_bitmask_inplace_jit(logits_gpu, bitmask)
    torch.cuda.synchronize()
    torch.testing.assert_close(logits_gpu, expected.to("cuda"))


# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtypes(dtype):
    torch.manual_seed(42)
    vocab_size = 256
    bm_width = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK

    logits = torch.randn(vocab_size, dtype=dtype, device="cuda")
    bitmask_bits = torch.randint(0, 2, (vocab_size,), dtype=torch.int32)
    packed = torch.zeros(bm_width, dtype=torch.int32)
    for i in range(vocab_size):
        packed[i // BITS_PER_BLOCK] |= bitmask_bits[i].item() << (
            i % BITS_PER_BLOCK
        )
    bitmask = packed.to("cuda")

    expected = logits.clone()
    for i in range(vocab_size):
        if not bitmask_bits[i]:
            expected[i] = float("-inf")

    apply_token_bitmask_inplace_jit(logits, bitmask)
    torch.cuda.synchronize()
    torch.testing.assert_close(logits, expected)


# ---------------------------------------------------------------------------
# Batched (2D) inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_batched(dtype, batch_size):
    torch.manual_seed(0)
    vocab_size = 512
    bm_width = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK

    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device="cuda")
    bitmask = torch.randint(
        -(2**31), 2**31 - 1, (batch_size, bm_width), dtype=torch.int32, device="cuda"
    )

    expected = _reference_apply(logits.cpu(), bitmask.cpu()).to("cuda")

    apply_token_bitmask_inplace_jit(logits, bitmask)
    torch.cuda.synchronize()
    torch.testing.assert_close(logits, expected)


# ---------------------------------------------------------------------------
# With indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_with_indices(dtype):
    torch.manual_seed(1)
    batch_size = 8
    vocab_size = 256
    bm_width = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK

    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device="cuda")
    bitmask = torch.randint(
        -(2**31), 2**31 - 1, (batch_size, bm_width), dtype=torch.int32, device="cuda"
    )
    indices = torch.tensor([1, 4, 7], dtype=torch.int32, device="cuda")

    expected = logits.clone()
    for batch_idx in indices.tolist():
        for v in range(vocab_size):
            word = v // BITS_PER_BLOCK
            bit = v % BITS_PER_BLOCK
            allowed = (bitmask[batch_idx, word].item() >> bit) & 1
            if not allowed:
                expected[batch_idx, v] = float("-inf")

    apply_token_bitmask_inplace_jit(logits, bitmask, indices)
    torch.cuda.synchronize()
    torch.testing.assert_close(logits, expected)


# ---------------------------------------------------------------------------
# Large vocab (128k) — representative of modern LLMs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_vocab(dtype):
    torch.manual_seed(7)
    batch_size = 4
    vocab_size = 128256  # Llama-3 vocab size
    bm_width = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK

    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device="cuda")
    bitmask = torch.randint(
        -(2**31), 2**31 - 1, (batch_size, bm_width), dtype=torch.int32, device="cuda"
    )

    logits_ref = logits.clone()
    apply_token_bitmask_inplace_jit(logits, bitmask)
    torch.cuda.synchronize()

    for b in range(batch_size):
        for v in range(vocab_size):
            word = v // BITS_PER_BLOCK
            bit = v % BITS_PER_BLOCK
            allowed = (bitmask[b, word].item() >> bit) & 1
            if allowed:
                assert logits[b, v] == logits_ref[b, v], (
                    f"Allowed token modified at [{b},{v}]"
                )
            else:
                assert logits[b, v] == float("-inf"), (
                    f"Masked token not -inf at [{b},{v}]"
                )


# ---------------------------------------------------------------------------
# All-allowed and all-masked edge cases
# ---------------------------------------------------------------------------


def test_all_allowed():
    logits = torch.randn(64, dtype=torch.float32, device="cuda")
    bitmask = torch.full((2,), 0xFFFFFFFF, dtype=torch.int32, device="cuda")
    expected = logits.clone()
    apply_token_bitmask_inplace_jit(logits, bitmask)
    torch.cuda.synchronize()
    torch.testing.assert_close(logits, expected)


def test_all_masked():
    logits = torch.randn(64, dtype=torch.float32, device="cuda")
    bitmask = torch.zeros(2, dtype=torch.int32, device="cuda")
    apply_token_bitmask_inplace_jit(logits, bitmask)
    torch.cuda.synchronize()
    assert (logits == float("-inf")).all()


# ---------------------------------------------------------------------------
# Cross-validation against sgl-kernel AOT (if available)
# ---------------------------------------------------------------------------


def test_cross_validate_aot():
    try:
        from sgl_kernel import apply_token_bitmask_inplace_cuda
    except ImportError:
        pytest.skip("sgl_kernel not installed, skipping AOT cross-validation")

    torch.manual_seed(99)
    batch_size, vocab_size = 8, 4096
    bm_width = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        logits_jit = torch.randn(
            batch_size, vocab_size, dtype=dtype, device="cuda"
        )
        logits_aot = logits_jit.clone()
        bitmask = torch.randint(
            -(2**31),
            2**31 - 1,
            (batch_size, bm_width),
            dtype=torch.int32,
            device="cuda",
        )

        apply_token_bitmask_inplace_jit(logits_jit, bitmask)
        apply_token_bitmask_inplace_cuda(logits_aot, bitmask)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            logits_jit,
            logits_aot,
            msg=f"JIT vs AOT mismatch for {dtype}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
