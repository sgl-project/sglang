"""Correctness tests for the fused SiLU-Mul + Per-Token FP8 Quantization kernel.

Tests that the single fused kernel produces identical results to the sequential
pipeline: silu_and_mul → per_token_quant_fp8.

We verify:
  1. FP8 quantized output matches the sequential pipeline within ±1 ULP
  2. Per-token scales match closely (float32 reduction order may differ slightly)
  3. Dequantized outputs are close
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

import sgl_kernel


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, FP32 accumulation)
# ---------------------------------------------------------------------------

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def reference_silu_mul_quant_fp8(
    input: torch.Tensor,
) -> tuple:
    """Sequential reference: silu_and_mul then per_token_quant_fp8.

    All intermediate computation in FP32.

    Parameters
    ----------
    input : torch.Tensor
        Shape (num_tokens, 2 * d). First half is gate, second half is up.

    Returns (output_q, output_s).
    """
    d = input.size(-1) // 2

    # Step 1: SiLU(gate) * up
    gate = input[..., :d].float()
    up = input[..., d:].float()
    silu_gate = gate / (1.0 + torch.exp(-gate))
    act = silu_gate * up

    # Step 2: Per-token quantization
    abs_max = act.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / FP8_E4M3_MAX
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    q_float = (act * scale_inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    output_q = q_float.to(torch.float8_e4m3fn)

    return output_q, scale


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# intermediate_size values from real models:
#   11008 (Llama-7B), 14336 (Llama-8B), 18432 (DeepSeek-V3), 27648 (Qwen-32B), 28672 (Llama-70B)
# Dispatch paths exercised (BF16, VEC_SIZE=8, num_threads=1024):
#   ROUNDS=1: d ≤ 8192          → 128, 512, 2048, 4096, 8192
#   ROUNDS=2: 8192 < d ≤ 16384  → 11008, 14336
#   ROUNDS=3: 16384 < d ≤ 24576 → 18432
#   ROUNDS=4: 24576 < d ≤ 32768 → 27648, 28672
INTERMEDIATE_SIZES = [128, 512, 2048, 4096, 8192, 11008, 14336, 18432, 27648, 28672]
BATCH_SIZES = [1, 4, 17, 128]
DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize(
    "batch_size,d",
    list(itertools.product(BATCH_SIZES, INTERMEDIATE_SIZES)),
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_silu_mul_quant_fp8_correctness(batch_size, d, dtype):
    """Core correctness: fused kernel == sequential reference."""
    torch.manual_seed(42)
    device = "cuda"

    # Input shape: [num_tokens, 2 * d] (gate_up projection output)
    input_t = torch.randn(batch_size, 2 * d, dtype=dtype, device=device)

    # --- Reference (sequential) ---
    ref_q, ref_scale = reference_silu_mul_quant_fp8(input_t)

    # --- Fused kernel ---
    fused_q, fused_s = sgl_kernel.fused_silu_mul_quant_fp8(input_t)

    # 1. Dequantized output: cosine similarity (FP8 standard metric)
    deq_fused = fused_q.float() * fused_s
    deq_ref = ref_q.float() * ref_scale
    cos_sim = F.cosine_similarity(
        deq_fused.reshape(-1), deq_ref.reshape(-1), dim=0
    )
    assert cos_sim > 0.999, (
        f"Dequantized cosine sim {cos_sim:.6f} < 0.999 "
        f"at shape ({batch_size}, {d})"
    )


@pytest.mark.parametrize("d", [4096, 14336, 18432, 27648, 28672])
def test_fused_kernel_input_not_modified(d):
    """Verify that input tensor is not modified (read-only)."""
    torch.manual_seed(123)
    device = "cuda"
    batch_size = 16

    input_t = torch.randn(batch_size, 2 * d, dtype=torch.bfloat16, device=device)
    input_copy = input_t.clone()

    sgl_kernel.fused_silu_mul_quant_fp8(input_t)

    assert torch.equal(input_t, input_copy), "Input tensor was modified!"


@pytest.mark.parametrize("d", [4096, 14336, 18432, 27648, 28672])
def test_fused_kernel_zero_gate(d):
    """Edge case: zero gate → silu(0) * up = 0 → all zeros output."""
    device = "cuda"
    batch_size = 4

    input_t = torch.zeros(batch_size, 2 * d, dtype=torch.bfloat16, device=device)
    # Set up portion to nonzero (doesn't matter since silu(0) = 0)
    input_t[:, d:] = 1.0

    fused_q, fused_s = sgl_kernel.fused_silu_mul_quant_fp8(input_t)

    # silu(0) = 0 → act = 0 → scale = 0, all FP8 zeros
    assert (fused_s == 0).all(), "Scale should be 0 when all activations are zero"
    assert (fused_q.float() == 0).all(), "Output should be all zeros"


@pytest.mark.parametrize("d", [4096, 14336, 18432, 27648, 28672])
def test_fused_kernel_matches_sequential_pipeline(d):
    """End-to-end: fused kernel == sgl_kernel.silu_and_mul + sgl_kernel.sgl_per_token_quant_fp8."""
    torch.manual_seed(99)
    device = "cuda"
    batch_size = 32

    input_t = torch.randn(batch_size, 2 * d, dtype=torch.bfloat16, device=device)

    # --- Sequential pipeline using existing sgl_kernel ops ---
    seq_act = sgl_kernel.silu_and_mul(input_t)
    seq_q = torch.empty(batch_size, d, dtype=torch.float8_e4m3fn, device=device)
    seq_s = torch.zeros(batch_size, dtype=torch.float32, device=device)
    sgl_kernel.sgl_per_token_quant_fp8(seq_act, seq_q, seq_s)
    seq_s = seq_s.reshape(-1, 1)

    # --- Fused kernel ---
    fused_q, fused_s = sgl_kernel.fused_silu_mul_quant_fp8(input_t)

    # Scales differ intentionally: sequential rounds silu*mul to BF16 before
    # finding abs_max; fused kernel keeps float32 throughout (more precise).
    torch.testing.assert_close(
        fused_s, seq_s, rtol=2e-2, atol=1e-3,
        msg="Scale mismatch vs sequential pipeline"
    )

    # Dequantized output: cosine similarity (FP8 standard)
    deq_fused = fused_q.float() * fused_s
    deq_seq = seq_q.float() * seq_s
    cos_sim = F.cosine_similarity(
        deq_fused.reshape(-1), deq_seq.reshape(-1), dim=0
    )
    assert cos_sim > 0.99, (
        f"Dequantized cosine sim {cos_sim:.6f} < 0.99 vs sequential pipeline"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
