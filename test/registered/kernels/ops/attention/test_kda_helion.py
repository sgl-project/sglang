from __future__ import annotations

import inspect
import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda as triton_chunk_kda
from sglang.kernels.ops.attention.helion.kda_decode import (
    helion_fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.attention.helion.kda_prefill import (
    chunk_kda as helion_chunk_kda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_public_signatures_match_triton() -> None:
    decode_signature = inspect.signature(fused_recurrent_kda_packed_decode)
    helion_decode_signature = inspect.signature(
        helion_fused_recurrent_kda_packed_decode
    )
    decode_parameters = list(decode_signature.parameters.values())
    helion_decode_parameters = list(helion_decode_signature.parameters.values())
    assert [parameter.name for parameter in helion_decode_parameters] == [
        parameter.name for parameter in decode_parameters
    ] + ["lower_bound"]
    assert [parameter.kind for parameter in helion_decode_parameters[:-1]] == [
        parameter.kind for parameter in decode_parameters
    ]
    assert [parameter.default for parameter in helion_decode_parameters[:-1]] == [
        parameter.default for parameter in decode_parameters
    ]
    lower_bound_parameter = helion_decode_parameters[-1]
    assert lower_bound_parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert lower_bound_parameter.default is None

    prefill_signature = inspect.signature(triton_chunk_kda)
    helion_prefill_signature = inspect.signature(helion_chunk_kda)
    assert list(helion_prefill_signature.parameters) == list(
        prefill_signature.parameters
    )
    assert [
        parameter.kind for parameter in helion_prefill_signature.parameters.values()
    ] == [parameter.kind for parameter in prefill_signature.parameters.values()]
    assert [
        parameter.default for parameter in helion_prefill_signature.parameters.values()
    ] == [parameter.default for parameter in prefill_signature.parameters.values()]


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_packed_decode_contract(state_dtype: torch.dtype) -> None:
    torch.manual_seed(123)
    batch, q_heads, v_heads, key_dim, value_dim = 3, 2, 4, 128, 128
    pool_size = 7
    mixed_qkv = torch.randn(
        batch,
        2 * q_heads * key_dim + v_heads * value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = torch.randn(batch, v_heads * key_dim, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(batch, v_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(v_heads * key_dim, device="cuda", dtype=torch.float32)
    state = (
        torch.randn(
            pool_size,
            v_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.01
    )
    indices = torch.tensor([5, -1, 2], device="cuda", dtype=torch.int32)
    triton_state = state.clone()
    helion_state = state.clone()
    triton_out = mixed_qkv.new_empty(batch, 1, v_heads, value_dim)
    helion_out = torch.empty_like(triton_out)

    fused_recurrent_kda_packed_decode(
        mixed_qkv,
        gate,
        beta,
        a_log,
        dt_bias,
        key_dim**-0.5,
        triton_state,
        triton_out,
        indices,
        True,
    )
    result, result_state = helion_fused_recurrent_kda_packed_decode(
        mixed_qkv,
        gate,
        beta,
        a_log,
        dt_bias,
        key_dim**-0.5,
        helion_state,
        helion_out,
        indices,
        True,
    )

    assert result.data_ptr() == helion_out.data_ptr()
    assert result_state.data_ptr() == helion_state.data_ptr()
    torch.testing.assert_close(helion_out, triton_out, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(helion_state, triton_state, atol=2e-2, rtol=1e-2)
    assert torch.count_nonzero(helion_out[1]).item() == 0
    untouched = torch.tensor([0, 1, 3, 4, 6], device="cuda")
    assert torch.equal(helion_state[untouched], state[untouched])


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_packed_decode_lower_bound_contract(state_dtype: torch.dtype) -> None:
    torch.manual_seed(321)
    batch, q_heads, v_heads, key_dim, value_dim = 3, 2, 4, 128, 128
    pool_size = 7
    mixed_qkv = torch.randn(
        batch,
        2 * q_heads * key_dim + v_heads * value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = torch.randn(batch, v_heads * key_dim, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(batch, v_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(v_heads * key_dim, device="cuda", dtype=torch.float32)
    state = (
        torch.randn(
            pool_size,
            v_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.01
    )
    indices = torch.tensor([5, -1, 2], device="cuda", dtype=torch.int32)
    reference_state = state.clone()
    helion_state = state.clone()
    reference_out = mixed_qkv.new_zeros(batch, 1, v_heads, value_dim)
    helion_out = torch.empty_like(reference_out)
    scale = key_dim**-0.5
    lower_bound = -5.0

    heads_per_q = v_heads // q_heads
    q, k, v = mixed_qkv.split(
        [q_heads * key_dim, q_heads * key_dim, v_heads * value_dim], dim=-1
    )
    q = q.float().view(batch, q_heads, key_dim)
    k = k.float().view(batch, q_heads, key_dim)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(heads_per_q, dim=1)
    k = k.repeat_interleave(heads_per_q, dim=1)
    v = v.float().view(batch, v_heads, value_dim)
    raw_gate = gate.float().view(batch, v_heads, key_dim)
    raw_gate = raw_gate + dt_bias.view(1, v_heads, key_dim)
    A = torch.exp(a_log.float()).view(1, v_heads, 1)
    decay = torch.exp(lower_bound * torch.sigmoid(A * raw_gate))
    beta_value = torch.sigmoid(beta.float())
    for batch_idx, state_idx in enumerate(indices.tolist()):
        if state_idx < 0:
            continue
        current_state = reference_state[state_idx].float()
        current_state = current_state * decay[batch_idx, :, None, :]
        residual = v[batch_idx] - (current_state * k[batch_idx, :, None, :]).sum(-1)
        residual = residual * beta_value[batch_idx, :, None]
        current_state = current_state + residual[..., None] * k[batch_idx, :, None, :]
        reference_out[batch_idx, 0] = (
            current_state * (q[batch_idx] * scale)[:, None, :]
        ).sum(-1)
        reference_state[state_idx] = current_state

    result, result_state = helion_fused_recurrent_kda_packed_decode(
        mixed_qkv,
        gate,
        beta,
        a_log,
        dt_bias,
        scale,
        helion_state,
        helion_out,
        indices,
        True,
        lower_bound,
    )

    assert result.data_ptr() == helion_out.data_ptr()
    assert result_state.data_ptr() == helion_state.data_ptr()
    torch.testing.assert_close(helion_out, reference_out, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(helion_state, reference_state, atol=2e-2, rtol=1e-2)
    assert torch.count_nonzero(helion_out[1]).item() == 0


def _compare_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triton_state = state.clone()
    helion_state = state.clone()
    triton_v = v.clone()
    helion_v = v.clone()
    triton_out, triton_chunks = triton_chunk_kda(
        q,
        k,
        triton_v,
        gate,
        beta,
        initial_state=triton_state,
        initial_state_indices=indices,
        output_intermediate_states=True,
        **kwargs,
    )
    helion_out, helion_chunks = helion_chunk_kda(
        q,
        k,
        helion_v,
        gate,
        beta,
        initial_state=helion_state,
        initial_state_indices=indices,
        output_intermediate_states=True,
        **kwargs,
    )

    assert helion_out.data_ptr() == helion_v.data_ptr()
    torch.testing.assert_close(helion_out, triton_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(helion_chunks, triton_chunks, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(helion_state, triton_state, atol=2e-2, rtol=2e-2)
    return helion_out, helion_chunks, helion_state


def test_fixed_partial_prefill_and_state_pool_contract() -> None:
    torch.manual_seed(789)
    batch, tokens, heads, key_dim, value_dim = 2, 17, 2, 32, 32
    q = torch.randn(batch, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(
        batch, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16
    )
    gate = torch.randn_like(q) * 0.2
    beta = torch.rand(batch, tokens, heads, device="cuda")
    a_log = torch.full([heads], -2.0, device="cuda")
    dt_bias = torch.zeros(heads * key_dim, device="cuda")
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state = torch.randn(5, heads, value_dim, key_dim, device="cuda") * 0.01

    _, _, helion_state = _compare_prefill(
        q,
        k,
        v,
        gate,
        beta,
        state,
        indices,
        use_qk_l2norm_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
    )

    untouched = torch.tensor([0, 2, 4], device="cuda")
    assert torch.equal(helion_state[untouched], state[untouched])


@pytest.mark.parametrize(
    "newton_schulz",
    [False, True],
    ids=["forward-substitution", "newton-schulz"],
)
def test_packed_varlen_safe_gate_contract(newton_schulz: bool) -> None:
    torch.manual_seed(1011)
    lengths = [1, 15, 17]
    tokens, heads, key_dim, value_dim = sum(lengths), 2, 32, 32
    q = torch.randn(1, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(q) * 0.2
    beta = torch.rand(1, tokens, heads, device="cuda")
    a_log = torch.full([heads], -2.0, device="cuda")
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    indices = torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32)
    state = torch.randn(6, heads, value_dim, key_dim, device="cuda") * 0.01

    _compare_prefill(
        q,
        k,
        v,
        gate,
        beta,
        state,
        indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        lower_bound=-0.01,
        newton_schulz=newton_schulz,
    )


def test_newton_schulz_uses_stable_subchunk_gates() -> None:
    torch.manual_seed(1117)
    tokens, heads, key_dim, value_dim = 64, 1, 32, 32
    q = torch.nn.functional.normalize(
        torch.randn(1, tokens, heads, key_dim, device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, heads, key_dim, device="cuda"), dim=-1
    ).bfloat16()
    v = torch.randn(1, tokens, heads, value_dim, device="cuda").bfloat16()
    # A chunk-global reference would form exp2(63 * 2 * RCP_LN2), which
    # overflows FP32. The 16-token anchors keep every matrix factor finite.
    gate = torch.full(
        (1, tokens, heads, key_dim), -2.0, device="cuda", dtype=torch.float32
    )
    beta = torch.full((1, tokens, heads), 0.5, device="cuda")
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    state = torch.zeros(1, heads, value_dim, key_dim, device="cuda")

    _compare_prefill(
        q,
        k,
        v,
        gate,
        beta,
        state,
        indices,
        cu_seqlens=cu_seqlens,
        newton_schulz=True,
    )


def test_fp16_preactivated_gate_with_bf16_state_contract() -> None:
    torch.manual_seed(1213)
    batch, tokens, heads, key_dim, value_dim = 1, 17, 1, 32, 32
    q = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1
    ).half()
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1
    ).half()
    v = torch.randn(batch, tokens, heads, value_dim, device="cuda", dtype=torch.float16)
    gate = -torch.rand(batch, tokens, heads, key_dim, device="cuda") * 0.01
    beta = torch.rand(batch, tokens, heads, device="cuda")
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    state = (
        torch.randn(
            3,
            heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.01
    )

    output, chunks, _ = _compare_prefill(
        q,
        k,
        v,
        gate,
        beta,
        state,
        indices,
    )
    assert output.dtype == torch.float16
    assert chunks.dtype == torch.float16


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_packed_varlen_prefill_contract(state_dtype: torch.dtype) -> None:
    torch.manual_seed(456)
    lengths = [65, 31]
    tokens, heads, key_dim, value_dim = sum(lengths), 2, 128, 128
    q = torch.randn(1, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(q)
    beta = torch.sigmoid(
        torch.randn(1, tokens, heads, device="cuda", dtype=torch.float32)
    )
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads * key_dim, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor([0, lengths[0], tokens], device="cuda", dtype=torch.int32)
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state = (
        torch.randn(
            5,
            heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.01
    )
    triton_state = state.clone()
    helion_state = state.clone()

    triton_out, triton_chunks = triton_chunk_kda(
        q,
        k,
        v.clone(),
        gate,
        beta,
        initial_state=triton_state,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        dt_bias=dt_bias,
        output_intermediate_states=True,
    )
    helion_out, helion_chunks = helion_chunk_kda(
        q,
        k,
        v.clone(),
        gate,
        beta,
        initial_state=helion_state,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        dt_bias=dt_bias,
        output_intermediate_states=True,
    )

    torch.testing.assert_close(helion_out, triton_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(helion_chunks, triton_chunks, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(helion_state, triton_state, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
