import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="TileLang GDN kernels require an NVIDIA V100",
)


def _inputs(lengths, state_dtype=torch.float32):
    torch.manual_seed(17)
    q_heads, value_heads, key_dim, value_dim = 4, 12, 128, 128
    tokens = sum(lengths)
    mixed_qkv = (
        torch.randn(
            tokens,
            2 * q_heads * key_dim + value_heads * value_dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.15
    ).contiguous()
    gate_a = (
        torch.randn(tokens, value_heads, device="cuda", dtype=torch.float16) * 0.2
    ).contiguous()
    gate_b = (torch.randn_like(gate_a) * 0.2).contiguous()
    a_log = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.float16) * 0.1
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state = (
        torch.randn(
            5,
            value_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.01
    ).to(state_dtype)
    return (
        mixed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
        state,
    )


def _reference(
    mixed_qkv,
    gate_a,
    gate_b,
    a_log,
    dt_bias,
    cu_seqlens,
    initial_state,
    state_indices,
):
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
        fused_gdn_gating,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    q_heads, value_heads, key_dim, value_dim = 4, 12, 128, 128
    tokens = mixed_qkv.shape[0]
    q, k, v = torch.split(
        mixed_qkv,
        [q_heads * key_dim, q_heads * key_dim, value_heads * value_dim],
        dim=-1,
    )
    q = q.view(1, tokens, q_heads, key_dim)
    k = k.view(1, tokens, q_heads, key_dim)
    v = v.view(1, tokens, value_heads, value_dim)
    gate, beta = fused_gdn_gating(a_log, gate_a, gate_b, dt_bias)
    return fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        scale=1 / math.sqrt(key_dim),
        initial_state=initial_state[state_indices].clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens.to(torch.int64),
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.parametrize(
    ("lengths", "implementation", "state_dtype"),
    [
        ([73, 439], "packed_recurrent_gdn_sm70", torch.float32),
        ([503, 521], "packed_chunked_gdn_sm70", torch.float32),
        ([73, 439], "packed_recurrent_gdn_sm70", torch.float16),
        ([503, 521], "packed_chunked_gdn_sm70", torch.float16),
    ],
)
def test_tilelang_packed_gdn_matches_recurrent_reference(
    lengths, implementation, state_dtype
):
    pytest.importorskip("tilelang")
    from sglang.srt.layers.attention.linear.kernels import gdn_chunked_tilelang

    args = _inputs(lengths, state_dtype)
    (
        mixed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
        initial_state,
    ) = args
    state = initial_state.clone()
    function = getattr(gdn_chunked_tilelang, implementation)
    output, checkpoints = function(
        mixed_qkv,
        gate_a,
        gate_b,
        q_heads=4,
        value_heads=12,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=128**-0.5,
        state=state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
    )
    expected_output, expected_state = _reference(
        mixed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        cu_seqlens,
        initial_state,
        state_indices,
    )

    assert checkpoints is None
    torch.testing.assert_close(
        output.float(), expected_output.float(), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        state[state_indices],
        expected_state.to(state_dtype),
        rtol=2e-4 if state_dtype == torch.float32 else 1e-3,
        atol=6e-5 if state_dtype == torch.float32 else 1e-3,
    )
    torch.testing.assert_close(
        state[[0, 2, 4]], initial_state[[0, 2, 4]], rtol=0, atol=0
    )


def test_tilelang_recurrent_gdn_writes_chunk_entry_checkpoints():
    pytest.importorskip("tilelang")
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
        fused_gdn_gating,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )
    from sglang.srt.layers.attention.linear.kernels.gdn_chunked_tilelang import (
        packed_recurrent_gdn_sm70,
    )

    args = _inputs([73, 184])
    (
        mixed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
        initial_state,
    ) = args
    state = initial_state.clone()
    output, checkpoints = packed_recurrent_gdn_sm70(
        mixed_qkv,
        gate_a,
        gate_b,
        q_heads=4,
        value_heads=12,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=128**-0.5,
        state=state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        store_checkpoints=True,
    )

    q, k, v = torch.split(mixed_qkv, [512, 512, 1536], dim=-1)
    q = q.view(1, -1, 4, 128)
    k = k.view(1, -1, 4, 128)
    v = v.view(1, -1, 12, 128)
    gate, beta = fused_gdn_gating(a_log, gate_a, gate_b, dt_bias)
    expected_output = torch.empty_like(output)
    expected_checkpoints = []
    final_states = []
    for sequence in range(2):
        begin = int(cu_seqlens[sequence])
        end = int(cu_seqlens[sequence + 1])
        current = initial_state[state_indices[sequence]].unsqueeze(0).clone()
        for chunk_begin in range(begin, end, 64):
            chunk_end = min(chunk_begin + 64, end)
            expected_checkpoints.append(current[0].half())
            chunk_output, current = fused_recurrent_gated_delta_rule(
                q[:, chunk_begin:chunk_end],
                k[:, chunk_begin:chunk_end],
                v[:, chunk_begin:chunk_end],
                gate[:, chunk_begin:chunk_end],
                beta[:, chunk_begin:chunk_end],
                scale=128**-0.5,
                initial_state=current,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            expected_output[:, chunk_begin:chunk_end].copy_(chunk_output)
        final_states.append(current[0])

    torch.testing.assert_close(
        output.float(), expected_output.float(), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        checkpoints[0],
        torch.stack(expected_checkpoints),
        rtol=2e-4,
        atol=6e-5,
    )
    torch.testing.assert_close(
        state[state_indices],
        torch.stack(final_states),
        rtol=2e-4,
        atol=6e-5,
    )
