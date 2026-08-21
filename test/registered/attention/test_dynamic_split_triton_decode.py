"""Correctness tests for Triton Decode Attention with Dynamic Split-K.

Verifies numerical parity between dynamic split-K Triton decoding kernel
and standard PyTorch eager reference across varying context lengths (256 ~ 8192).

Tested on NVIDIA GPUs (CUDA 12.x / Triton 3.x).
"""

import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd_grouped
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.test.ci.ci_register import register_cuda_ci

# Register for SGLang 1-GPU PR CI pipeline
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def torch_eager_decode_attention(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Standard PyTorch Eager Reference for Grouped-Query Attention Decoding."""
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    group_size = num_heads // num_kv_heads
    outputs = []
    for b in range(batch_size):
        start_idx = int(kv_indptr[b].item())
        end_idx = int(kv_indptr[b + 1].item())
        cur_indices = kv_indices[start_idx:end_idx]
        k_b = k_buffer[cur_indices].float()
        v_b = v_buffer[cur_indices].float()
        q_b = q[b].float().view(num_kv_heads, group_size, head_dim)
        scores = torch.einsum("gsh,lgh->gsl", q_b, k_b) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out_b = torch.einsum("gsl,lgh->gsh", probs, v_b).reshape(num_heads, head_dim)
        outputs.append(out_b)
    return torch.stack(outputs, dim=0)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("context_len", [256, 2048, 8192])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim",
    [
        (32, 8, 128),  # Llama-3.1-8B geometry
        (28, 4, 128),  # Qwen2.5-7B geometry
    ],
)
@torch.inference_mode()
def test_dynamic_split_k_decode_accuracy(
    batch_size: int,
    context_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Verifies numerical accuracy of dynamic Split-K against PyTorch eager baseline."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    device, dtype = "cuda", torch.float16
    sm_scale = 1.0 / math.sqrt(head_dim)
    max_kv_splits = 32
    device_core_count = torch.cuda.get_device_properties(0).multi_processor_count

    torch.cuda.empty_cache()
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    total_tokens = context_len * batch_size
    k_buffer = torch.randn(
        total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_buffer = torch.randn(
        total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype
    )

    kv_indptr = torch.arange(
        0, total_tokens + 1, context_len, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_tokens, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)

    # Dynamic hardware-aware split calculation
    num_kv_splits = torch.empty(batch_size, dtype=torch.int32, device=device)
    get_num_kv_splits_triton[(1,)](
        num_kv_splits,
        seq_lens,
        batch_size,
        1,
        num_heads,
        num_kv_heads,
        max_kv_splits,
        device_core_count,
        MAX_NUM_SEQ=32,
    )

    attn_logits = torch.empty(
        batch_size,
        num_heads,
        max_kv_splits,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(
        batch_size, num_heads, max_kv_splits, dtype=torch.float32, device=device
    )
    out_triton = torch.empty_like(q)

    decode_attention_fwd_grouped(
        q,
        k_buffer,
        v_buffer,
        out_triton,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        1.0,
        page_size=1,
    )

    ref_out = torch_eager_decode_attention(
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        sm_scale,
    ).to(dtype)

    assert not torch.isnan(out_triton).any(), "NaN detected in Triton output"
    assert not torch.isinf(out_triton).any(), "Inf detected in Triton output"

    cos_sim = torch.nn.functional.cosine_similarity(
        out_triton.flatten().to(torch.float32),
        ref_out.flatten().to(torch.float32),
        dim=0,
    )
    assert cos_sim.item() > 0.99, f"Cosine similarity too low: {cos_sim.item()}"
    assert torch.allclose(
        out_triton, ref_out, atol=3e-2, rtol=3e-2
    ), "Numerical mismatch against eager reference"

    torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
