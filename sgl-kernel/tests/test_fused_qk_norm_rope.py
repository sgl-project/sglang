import pytest
import torch
from sgl_kernel import fused_qk_norm_rope as sgl_fused_qk_norm_rope

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_is_xpu = is_xpu()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def torch_ref_rms_norm_rope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    base,
    is_neox,
    position_ids,
):
    """
    PyTorch reference implementation of RMSNorm+RoPE for verification.

    Uses SGLang's own RMSNorm and RotaryEmbedding modules to ensure consistency
    with the expected behavior of the fused kernel.

    Args:
        qkv: Combined QKV tensor of shape [num_tokens, hidden_size]
        num_heads_q: Number of query heads
        num_heads_k: Number of key heads
        num_heads_v: Number of value heads (unused for normalization/RoPE but needed for tensor splitting)
        head_dim: Dimension of each head
        eps: Epsilon value for RMS normalization
        q_weight: RMSNorm weights for query [head_dim]
        k_weight: RMSNorm weights for key [head_dim]
        base: Base value for RoPE calculations
        is_neox: Whether to use NeoX style RoPE
        position_ids: Position IDs for RoPE of shape [num_tokens]

    Returns:
        Combined tensor with Q and K parts normalized and RoPE applied
    """
    # Get input shape information
    num_tokens = qkv.shape[0]
    hidden_size = qkv.shape[1]

    # Calculate dimensions for Q, K, V segments
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim

    # Verify dimensions match
    assert (
        hidden_size == q_size + k_size + v_size
    ), f"Hidden size {hidden_size} doesn't match Q+K+V dimensions {q_size + k_size + v_size}"

    # Split the tensor into Q, K, V parts
    q = qkv[:, :q_size]
    k = qkv[:, q_size : q_size + k_size]
    v = qkv[:, q_size + k_size :]

    rotary_emb = get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=8192,
        base=10000,
        is_neox_style=is_neox,
        rope_scaling=None,
        dual_chunk_attention_config=None,
    )
    rotary_emb = rotary_emb.to(qkv.device)

    # Create and apply RMSNorm modules with custom weights
    q_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device).to(qkv.dtype)
    q_norm.weight.data.copy_(q_weight)
    k_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device).to(qkv.dtype)
    k_norm.weight.data.copy_(k_weight)

    q_by_head = q.reshape(-1, head_dim)
    q_by_head = q_norm(q_by_head)
    k_by_head = k.reshape(-1, head_dim)
    k_by_head = k_norm(k_by_head)
    q = q_by_head.view(q.shape)
    k = k_by_head.view(k.shape)

    [q_rope, k_rope] = rotary_emb(
        position_ids,
        q,
        k,
        fused_set_kv_buffer_arg=None,
    )

    # Combine Q, K, V back together
    result = torch.cat([q_rope, k_rope, v], dim=1)

    return result


head_dims = [64, 128]
# (Q heads, K heads, V heads)
num_heads_groups = [
    (16, 8, 8),  # Qwen3-0.6B, Qwen3-1.7B
    (32, 8, 8),  # Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B
    (40, 8, 8),  # Qwen3-14B
    (64, 8, 8),  # Qwen3-32B, Qwen3-235B-A22B
]
num_tokens_list = [1, 3, 8, 32, 256]
is_neox_list = [False, True]
dtypes = [torch.bfloat16]


@pytest.mark.skipif(not _is_cuda, reason="Skipping CUDA/ROCm only tests.")
@pytest.mark.parametrize("head_dim", head_dims)
@pytest.mark.parametrize("num_heads_group", num_heads_groups)
@pytest.mark.parametrize("num_tokens", num_tokens_list)
@pytest.mark.parametrize("is_neox", is_neox_list)
@pytest.mark.parametrize("dtype", dtypes)
def test_fused_qk_norm_rope(head_dim, num_heads_group, num_tokens, is_neox, dtype):
    """
    Test the fused QK RMSNorm + RoPE operation with various configurations.

    This test verifies that the fused kernel correctly applies:
    1. RMSNorm to both query (Q) and key (K) portions of the QKV tensor
    2. Rotary Position Embeddings (RoPE) to the normalized Q and K
    3. Leaves the value (V) portion unchanged

    Args:
        head_dim: Dimension of each attention head
        num_heads_group: Tuple of (num_heads_q, num_heads_k, num_heads_v)
        num_tokens: Number of tokens to process
        dtype: Data type (float16 or bfloat16)
    """
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    device = "cuda"
    torch_dtype = dtype

    # Unpack head counts
    num_heads_q, num_heads_k, num_heads_v = num_heads_group

    # Calculate total hidden dimension
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    # Generate random inputs directly as 2D [num_tokens, hidden_size]
    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch_dtype, device=device)
    qkv_copy = qkv.clone()

    # Generate position IDs with +100 offset to test decoding scenarios
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device) + 100

    # Generate random weights for RMSNorm
    q_weight = torch.randn(head_dim, dtype=torch_dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=torch_dtype, device=device) * 5.0

    # Set RMSNorm and RoPE parameters
    eps = 1e-5
    base = 10000.0

    factor, low, high, attention_factor = 1.0, 0, 0, 1.0
    # Run the custom fusedQKNormRope operation
    sgl_fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
    )
    output = qkv  # This op is inplace

    # Compute reference output using TensorRT-LLM modules
    ref_output = torch_ref_rms_norm_rope(
        qkv_copy,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
    )

    # Compare outputs from custom kernel vs reference implementation
    torch.testing.assert_close(
        output,
        ref_output,
        rtol=5e-2,
        atol=1e-1,
    )
