from __future__ import annotations

import inspect
import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
    fused_recurrent_linear_replayssm_decode,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda as triton_chunk_kda
from sglang.test.ci.ci_register import register_cuda_ci

try:
    import helion  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "helion":
        raise
    HELION_AVAILABLE = False
else:
    HELION_AVAILABLE = True
    from sglang.kernels.ops.attention.helion.kda_decode import (
        helion_fused_recurrent_kda_packed_decode,
    )
    from sglang.kernels.ops.attention.helion.kda_prefill import (
        _intra_matrices_wide,
        _l2norm_qk,
    )
    from sglang.kernels.ops.attention.helion.kda_prefill import (
        chunk_kda as helion_chunk_kda,
    )
    from sglang.kernels.ops.attention.helion.kda_replayssm import (
        helion_fused_recurrent_kda_replayssm_decode,
    )

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not HELION_AVAILABLE,
    reason="helion is not installed",
)

_DECODE_STATE_ATOL = {
    torch.float32: 1e-5,
    torch.bfloat16: 2e-3,
    torch.float16: 5e-4,
}


def test_public_signatures_match_triton() -> None:
    decode_signature = inspect.signature(fused_recurrent_kda_packed_decode)
    helion_decode_signature = inspect.signature(
        helion_fused_recurrent_kda_packed_decode
    )
    decode_parameters = list(decode_signature.parameters.values())
    helion_decode_parameters = list(helion_decode_signature.parameters.values())
    assert [parameter.name for parameter in helion_decode_parameters] == [
        parameter.name for parameter in decode_parameters
    ]
    assert [parameter.kind for parameter in helion_decode_parameters] == [
        parameter.kind for parameter in decode_parameters
    ]
    assert [parameter.default for parameter in helion_decode_parameters] == [
        parameter.default for parameter in decode_parameters
    ]

    replay_parameters = list(
        inspect.signature(fused_recurrent_linear_replayssm_decode).parameters.values()
    )
    helion_replay_parameters = list(
        inspect.signature(
            helion_fused_recurrent_kda_replayssm_decode
        ).parameters.values()
    )
    shared_replay_parameter_count = 15
    assert (
        helion_replay_parameters[:shared_replay_parameter_count]
        == replay_parameters[:shared_replay_parameter_count]
    )
    assert [
        parameter.name
        for parameter in helion_replay_parameters[shared_replay_parameter_count:]
    ] == ["lower_bound"]

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
    torch.testing.assert_close(helion_out, triton_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        helion_state,
        triton_state,
        atol=_DECODE_STATE_ATOL[state_dtype],
        rtol=1e-4,
    )
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
    torch.testing.assert_close(helion_out, reference_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        helion_state,
        reference_state,
        atol=_DECODE_STATE_ATOL[state_dtype],
        rtol=1e-4,
    )
    assert torch.count_nonzero(helion_out[1]).item() == 0


# `v_heads` selects the tuned config: <= KDA_SMALL_VALUE_HEAD_THRESHOLD picks the
# small-head bf16 schedule, above it the wide bf16 one. Cover both.
@pytest.mark.parametrize(
    ("state_dtype", "lower_bound", "v_heads"),
    [
        (torch.float32, None, 4),
        (torch.bfloat16, None, 4),
        (torch.bfloat16, -5.0, 4),
        (torch.bfloat16, None, 16),
    ],
    ids=["fp32", "bf16-small-head", "bf16-small-head-lower-bound", "bf16"],
)
def test_replayssm_decode_contract(
    state_dtype: torch.dtype, lower_bound: float | None, v_heads: int
) -> None:
    """Match Triton ring writes, forced flushes, and natural flushes."""
    batch, q_heads, key_dim, value_dim = 3, 2, 128, 128
    cache_length, pool_size = 4, 5
    scale = key_dim**-0.5
    torch.manual_seed(721)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.3
    dt_bias = torch.randn(v_heads, key_dim, device="cuda", dtype=torch.float32) * 0.1
    initial = torch.randn(
        pool_size,
        v_heads,
        value_dim,
        key_dim,
        device="cuda",
        dtype=state_dtype,
    )
    triton_state = initial.clone()
    helion_state = initial.clone()
    triton_d = torch.zeros(
        pool_size,
        v_heads,
        cache_length,
        value_dim,
        device="cuda",
        dtype=state_dtype,
    )
    helion_d = triton_d.clone()
    triton_k = torch.zeros(
        pool_size,
        q_heads,
        cache_length,
        key_dim,
        device="cuda",
        dtype=state_dtype,
    )
    helion_k = triton_k.clone()
    triton_g = torch.zeros(
        pool_size,
        v_heads,
        cache_length,
        key_dim,
        device="cuda",
        dtype=torch.float32,
    )
    helion_g = triton_g.clone()
    indices = torch.tensor([3, -1, 1], device="cuda", dtype=torch.int32)
    write_pos = torch.zeros(batch, device="cuda", dtype=torch.int32)

    for step in range(7):
        generator = torch.Generator(device="cuda").manual_seed(900 + step)
        mixed_qkv = torch.randn(
            batch,
            2 * q_heads * key_dim + v_heads * value_dim,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = (
            torch.randn(
                batch,
                v_heads,
                key_dim,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.5
        )
        beta = torch.randn(
            batch,
            v_heads,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        triton_out = torch.empty(
            batch, 1, v_heads, value_dim, device="cuda", dtype=torch.bfloat16
        )
        helion_out = torch.empty_like(triton_out)
        force_flush = (
            torch.ones(batch, device="cuda", dtype=torch.int32) if step == 2 else None
        )

        # Triton ReplaySSM is the protocol oracle for the unbounded gate. Its
        # bounded-gate path is unsupported, so that case uses the full recurrent
        # update as a stronger state oracle and checks the checkpoint on flushes.
        if lower_bound is None:
            fused_recurrent_linear_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=gate,
                b=beta,
                A_log=a_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=triton_state,
                d_cache=triton_d,
                k_cache=triton_k,
                g_cache=triton_g,
                out=triton_out,
                ssm_state_indices=indices,
                write_pos=write_pos,
                force_flush=force_flush,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
                nk=2,
            )
        else:
            fused_recurrent_kda_packed_decode(
                mixed_qkv=mixed_qkv,
                a=gate.view(batch, -1),
                b=beta,
                A_log=a_log,
                dt_bias=dt_bias.view(-1),
                scale=scale,
                initial_state=triton_state,
                out=triton_out,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                lower_bound=lower_bound,
            )
        helion_fused_recurrent_kda_replayssm_decode(
            mixed_qkv=mixed_qkv,
            a=gate,
            b=beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=helion_state,
            d_cache=helion_d,
            k_cache=helion_k,
            g_cache=helion_g,
            out=helion_out,
            ssm_state_indices=indices,
            write_pos=write_pos,
            force_flush=force_flush,
            use_qk_l2norm_in_kernel=True,
            lower_bound=lower_bound,
        )

        torch.testing.assert_close(helion_out, triton_out, atol=5e-4, rtol=1e-2)
        assert torch.count_nonzero(helion_out[1]).item() == 0
        is_flush = force_flush is not None or write_pos[0].item() == cache_length - 1
        if is_flush:
            torch.testing.assert_close(
                helion_state.float(),
                triton_state.float(),
                atol=4e-3,
                rtol=1e-2,
            )
            write_pos.zero_()
        else:
            write_pos.add_(1)

    assert torch.equal(helion_state[4], initial[4])


@pytest.mark.parametrize(
    ("write_pos_values", "force_flush_values", "flushed_rows"),
    [
        ([0, 2, 3], None, [False, False, True]),
        ([1, 2, 1], [1, 0, 1], [True, False, True]),
    ],
    ids=["divergent-natural-flush", "mixed-forced-flush"],
)
def test_replayssm_per_row_flush_contract(
    write_pos_values: list[int],
    force_flush_values: list[int] | None,
    flushed_rows: list[bool],
) -> None:
    """Keep each row's cursor and partial-ring flush decision independent."""
    batch, q_heads, v_heads, key_dim, value_dim = 3, 2, 4, 128, 128
    cache_length = 4
    torch.manual_seed(977)
    mixed_qkv = torch.randn(
        batch,
        2 * q_heads * key_dim + v_heads * value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = torch.randn(
        batch,
        v_heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    beta = torch.randn(batch, v_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.2
    dt_bias = torch.randn(v_heads, key_dim, device="cuda", dtype=torch.float32) * 0.1
    initial = (
        torch.randn(
            batch,
            v_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    d_cache = (
        torch.randn(
            batch,
            v_heads,
            cache_length,
            value_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    k_cache = torch.randn(
        batch,
        q_heads,
        cache_length,
        key_dim,
        device="cuda",
        dtype=torch.float32,
    )
    g_cache = (
        -torch.rand(
            batch,
            v_heads,
            cache_length,
            key_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.1
    )
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    write_pos = torch.tensor(write_pos_values, device="cuda", dtype=torch.int32)
    force_flush = (
        None
        if force_flush_values is None
        else torch.tensor(force_flush_values, device="cuda", dtype=torch.int32)
    )

    triton_state = initial.clone()
    helion_state = initial.clone()
    triton_d, helion_d = d_cache.clone(), d_cache.clone()
    triton_k, helion_k = k_cache.clone(), k_cache.clone()
    triton_g, helion_g = g_cache.clone(), g_cache.clone()
    triton_out = torch.empty(
        batch, 1, v_heads, value_dim, device="cuda", dtype=torch.bfloat16
    )
    helion_out = torch.empty_like(triton_out)

    common_args = dict(
        mixed_qkv=mixed_qkv,
        a=gate,
        b=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        ssm_state_indices=indices,
        write_pos=write_pos,
        force_flush=force_flush,
        use_qk_l2norm_in_kernel=True,
    )
    fused_recurrent_linear_replayssm_decode(
        **common_args,
        initial_state=triton_state,
        d_cache=triton_d,
        k_cache=triton_k,
        g_cache=triton_g,
        out=triton_out,
        is_kda=True,
        nk=2,
    )
    helion_fused_recurrent_kda_replayssm_decode(
        **common_args,
        initial_state=helion_state,
        d_cache=helion_d,
        k_cache=helion_k,
        g_cache=helion_g,
        out=helion_out,
    )

    torch.testing.assert_close(helion_out, triton_out, atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(helion_state, triton_state, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(helion_d, triton_d, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(helion_k, triton_k, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(helion_g, triton_g, atol=1e-5, rtol=1e-4)
    for row, flushed in enumerate(flushed_rows):
        if flushed:
            assert not torch.equal(helion_state[row], initial[row])
        else:
            assert torch.equal(helion_state[row], initial[row])


def test_replayssm_cuda_graph_replay_with_strided_state() -> None:
    """Keep cursor branches dynamic and preserve envelope-strided state I/O."""
    batch, q_heads, v_heads, key_dim, value_dim = 2, 2, 4, 128, 128
    cache_length, pool_size = 4, 3
    state_size = v_heads * value_dim * key_dim
    slot_stride = state_size + 257
    storage = torch.empty(pool_size * slot_stride, device="cuda", dtype=torch.float32)
    state = torch.as_strided(
        storage,
        (pool_size, v_heads, value_dim, key_dim),
        (slot_stride, value_dim * key_dim, key_dim, 1),
    )
    torch.manual_seed(811)
    initial = torch.randn_like(state)
    state.copy_(initial)
    mixed_qkv = torch.randn(
        batch,
        2 * q_heads * key_dim + v_heads * value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = torch.randn(
        batch,
        v_heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    beta = torch.randn(batch, v_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(v_heads, key_dim, device="cuda", dtype=torch.float32)
    d_cache = torch.zeros(
        pool_size,
        v_heads,
        cache_length,
        value_dim,
        device="cuda",
        dtype=torch.float32,
    )
    k_cache = torch.zeros(
        pool_size,
        q_heads,
        cache_length,
        key_dim,
        device="cuda",
        dtype=torch.float32,
    )
    g_cache = torch.zeros(
        pool_size,
        v_heads,
        cache_length,
        key_dim,
        device="cuda",
        dtype=torch.float32,
    )
    indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    write_pos = torch.zeros(batch, device="cuda", dtype=torch.int32)
    force_flush = torch.zeros(batch, device="cuda", dtype=torch.int32)
    output = torch.empty(
        batch, 1, v_heads, value_dim, device="cuda", dtype=torch.bfloat16
    )

    def run_helion() -> None:
        helion_fused_recurrent_kda_replayssm_decode(
            mixed_qkv=mixed_qkv,
            a=gate,
            b=beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=state,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=output,
            ssm_state_indices=indices,
            write_pos=write_pos,
            force_flush=force_flush,
            use_qk_l2norm_in_kernel=True,
        )

    run_helion()
    torch.cuda.synchronize()
    state.copy_(initial)
    d_cache.zero_()
    k_cache.zero_()
    g_cache.zero_()
    write_pos.fill_(1)
    force_flush.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_helion()

    state.copy_(initial)
    d_cache.zero_()
    k_cache.zero_()
    g_cache.zero_()
    write_pos.fill_(1)
    force_flush.copy_(torch.tensor([1, 0], device="cuda", dtype=torch.int32))
    reference_state = initial.clone()
    reference_out = torch.empty_like(output)
    fused_recurrent_linear_replayssm_decode(
        mixed_qkv=mixed_qkv,
        a=gate,
        b=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=reference_state,
        d_cache=d_cache.clone(),
        k_cache=k_cache.clone(),
        g_cache=g_cache.clone(),
        out=reference_out,
        ssm_state_indices=indices,
        write_pos=write_pos,
        force_flush=force_flush,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        nk=2,
    )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, reference_out, atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(state, reference_state, atol=2e-3, rtol=1e-3)


def _compare_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    beta_is_raw: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.size(-1)
    if scale is None:
        scale = key_dim**-0.5

    reference_q = q.float()
    reference_k = k.float()
    if use_qk_l2norm_in_kernel:
        reference_q = reference_q / torch.sqrt(
            (reference_q * reference_q).sum(-1, keepdim=True) + 1e-6
        )
        reference_k = reference_k / torch.sqrt(
            (reference_k * reference_k).sum(-1, keepdim=True) + 1e-6
        )
        reference_q = reference_q.to(q.dtype).float()
        reference_k = reference_k.to(k.dtype).float()

    reference_gate = gate.float()
    if A_log is not None:
        if dt_bias is not None:
            reference_gate = reference_gate + dt_bias.view(1, 1, heads, key_dim)
        a = torch.exp(A_log.float()).view(1, 1, heads, 1)
        if lower_bound is not None:
            reference_gate = lower_bound * torch.sigmoid(a * reference_gate)
        else:
            reference_gate = -a * torch.nn.functional.softplus(reference_gate)

    reference_state = state.clone()
    reference_out = torch.empty_like(v)
    q_rows = reference_q.view(batch * tokens, heads, key_dim)
    k_rows = reference_k.view(batch * tokens, heads, key_dim)
    v_rows = v.view(batch * tokens, heads, value_dim).float()
    gate_rows = reference_gate.view(batch * tokens, heads, key_dim)
    reference_beta = beta.float().sigmoid() if beta_is_raw else beta
    beta_rows = reference_beta.view(batch * tokens, heads).float()
    out_rows = reference_out.view(batch * tokens, heads, value_dim)

    if cu_seqlens is None:
        sequence_bounds = [
            (sequence * tokens, (sequence + 1) * tokens) for sequence in range(batch)
        ]
        chunks_per_sequence = (tokens + 63) // 64
        reference_chunks = torch.empty(
            batch,
            chunks_per_sequence,
            heads,
            value_dim,
            key_dim,
            device=q.device,
            dtype=v.dtype,
        )
    else:
        offsets = cu_seqlens.tolist()
        sequence_bounds = list(zip(offsets, offsets[1:]))
        total_chunks = sum((end - begin + 63) // 64 for begin, end in sequence_bounds)
        reference_chunks = torch.empty(
            1,
            total_chunks,
            heads,
            value_dim,
            key_dim,
            device=q.device,
            dtype=v.dtype,
        )

    global_chunk = 0
    for sequence, (begin, end) in enumerate(sequence_bounds):
        state_index = indices[sequence].item()
        current_state = reference_state[state_index].float()
        for local_chunk, chunk_begin in enumerate(range(begin, end, 64)):
            chunk_index = local_chunk if cu_seqlens is None else global_chunk
            chunk_batch = sequence if cu_seqlens is None else 0
            reference_chunks[chunk_batch, chunk_index] = current_state.to(v.dtype)
            if cu_seqlens is not None:
                global_chunk += 1
            for token in range(chunk_begin, min(chunk_begin + 64, end)):
                current_state = current_state * torch.exp(gate_rows[token])[:, None, :]
                residual = v_rows[token] - (
                    current_state * k_rows[token][:, None, :]
                ).sum(-1)
                residual = residual * beta_rows[token][:, None]
                current_state = current_state + (
                    residual[:, :, None] * k_rows[token][:, None, :]
                )
                output = (current_state * (q_rows[token] * scale)[:, None, :]).sum(-1)
                out_rows[token] = output.to(v.dtype)
        reference_state[state_index] = current_state.to(state.dtype)

    helion_state = state.clone()
    helion_v = v.clone()
    helion_out, helion_chunks = helion_chunk_kda(
        q,
        k,
        helion_v,
        gate,
        beta,
        initial_state=helion_state,
        initial_state_indices=indices,
        output_intermediate_states=True,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
    )

    assert helion_out.data_ptr() == helion_v.data_ptr()
    torch.testing.assert_close(helion_out, reference_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(helion_chunks, reference_chunks, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(helion_state, reference_state, atol=2e-2, rtol=2e-2)
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


def test_raw_beta_prefill_contract() -> None:
    torch.manual_seed(811)
    batch, tokens, heads, key_dim, value_dim = 2, 17, 2, 32, 32
    q = torch.randn(batch, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    # Keep the unnormalized recurrence numerically contractive while still
    # exercising the no-QK-L2-normalization path. Unit-scale random keys make
    # (I - beta * k k^T) expansive and obscure the raw-beta contract with
    # exponentially amplified BF16 round-off.
    k = torch.randn_like(q) * 0.05
    v = torch.randn(
        batch, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16
    )
    # Keep the recurrent decay contractive so the raw-beta check measures the
    # sigmoid conversion instead of amplifying BF16 round-off exponentially.
    gate = -torch.rand_like(q) * 0.2
    raw_beta = torch.linspace(
        -2,
        2,
        steps=batch * tokens * heads,
        device="cuda",
    ).reshape(batch, tokens, heads)
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state = torch.randn(5, heads, value_dim, key_dim, device="cuda") * 0.01

    _compare_prefill(
        q,
        k,
        v,
        gate,
        raw_beta,
        state,
        indices,
        beta_is_raw=True,
    )


@pytest.mark.parametrize("is_varlen", [False, True], ids=["fixed", "varlen"])
def test_single_token_prefill_does_not_poison_later_shapes(
    is_varlen: bool,
) -> None:
    """Keep a size-one first trace from specializing later prefill calls."""
    torch.manual_seed(991)
    heads, key_dim, value_dim = 2, 32, 32
    a_log = torch.full([heads], -2.0, device="cuda")
    dt_bias = torch.zeros(heads * key_dim, device="cuda")
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)

    _l2norm_qk.reset()
    try:
        for tokens in (1, 3):
            q = torch.randn(
                1, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn_like(q)
            v = torch.randn(
                1, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16
            )
            gate = torch.randn_like(q) * 0.2
            beta = torch.rand(1, tokens, heads, device="cuda")
            state = (
                torch.randn(
                    1, heads, value_dim, key_dim, device="cuda", dtype=torch.float32
                )
                * 0.01
            )
            cu_seqlens = (
                torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
                if is_varlen
                else None
            )

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
                dt_bias=dt_bias,
            )
    finally:
        _l2norm_qk.reset()


@pytest.mark.parametrize("is_varlen", [False, True], ids=["fixed", "varlen"])
def test_prefill_ignores_padded_gate_rows(is_varlen: bool) -> None:
    torch.manual_seed(997)
    tokens, padded_tokens, heads, key_dim, value_dim = 51, 64, 2, 32, 32
    q = torch.randn(1, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    gate = -torch.rand(
        1, padded_tokens, heads, key_dim, device="cuda", dtype=torch.float32
    )
    beta = torch.rand(1, padded_tokens, heads, device="cuda")
    gate[:, tokens:] = 1e4
    beta[:, tokens:] = 1e4
    cu_seqlens = (
        torch.tensor([0, 17, tokens], device="cuda", dtype=torch.int32)
        if is_varlen
        else None
    )
    indices = (
        torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        if is_varlen
        else torch.zeros(1, device="cuda", dtype=torch.int32)
    )
    initial_state = torch.randn(
        indices.numel(),
        heads,
        value_dim,
        key_dim,
        device="cuda",
        dtype=torch.float32,
    )

    def run(
        gate_input: torch.Tensor, beta_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        output, chunks = helion_chunk_kda(
            q,
            k,
            v.clone(),
            gate_input,
            beta_input,
            initial_state=state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            output_intermediate_states=True,
        )
        return output, chunks, state

    trimmed = run(gate[:, :tokens], beta[:, :tokens])
    padded = run(gate, beta)
    for padded_value, trimmed_value in zip(padded, trimmed):
        assert torch.equal(padded_value, trimmed_value)

    short_inputs = (
        (gate[:, : tokens - 1], beta[:, :tokens]),
        (gate[:, :tokens], beta[:, : tokens - 1]),
    )
    for short_gate, short_beta in short_inputs:
        with pytest.raises(ValueError, match="g and beta must cover every q token"):
            run(short_gate, short_beta)


def test_prefill_uses_stable_subchunk_gates() -> None:
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

    output, chunks, final_state = _compare_prefill(
        q,
        k,
        v,
        gate,
        beta,
        state,
        indices,
        cu_seqlens=cu_seqlens,
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(chunks).all()
    assert torch.isfinite(final_state).all()


@pytest.mark.parametrize("is_varlen", [False, True])
def test_prefill_diagonal_uses_midpoint_gate_anchor(is_varlen: bool) -> None:
    """Prevent leading-edge anchoring from saturating the +/-126 gate clamp."""
    tokens, heads, key_dim = 16, 1, 32
    q = torch.full(
        (1, tokens, heads, key_dim),
        key_dim**-0.5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = q.clone()
    cumulative_gate = -10.0 * torch.arange(tokens, device="cuda", dtype=torch.float32)
    gate = cumulative_gate.view(1, tokens, 1, 1).expand_as(q).float()
    beta = torch.ones(1, tokens, heads, device="cuda")
    if is_varlen:
        metadata = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        chunk_indices = torch.tensor([[0, 0]], device="cuda", dtype=torch.int32)
    else:
        metadata = torch.empty(0, device="cuda", dtype=torch.int32)
        chunk_indices = torch.empty(0, 2, device="cuda", dtype=torch.int32)

    aqk, _ = _intra_matrices_wide(
        q,
        k,
        gate,
        beta,
        metadata,
        chunk_indices,
        1.0,
        is_varlen=is_varlen,
    )

    qk = q[0, :, 0].float() @ k[0, :, 0].float().T
    gate_delta = cumulative_gate[:, None] - cumulative_gate[None, :]
    causal = (
        torch.arange(tokens, device="cuda")[:, None]
        >= torch.arange(tokens, device="cuda")[None, :]
    )
    expected = torch.where(causal, qk * torch.exp2(gate_delta), 0.0)
    actual = aqk[0, :, 0, :tokens].float()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


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


@pytest.mark.parametrize(
    ("state_dtype", "lower_bound"),
    [
        (torch.float32, None),
        (torch.bfloat16, -5.0),
        (torch.float16, None),
    ],
    ids=["fp32", "bf16-lower-bound", "fp16"],
)
def test_packed_varlen_prefill_contract(
    state_dtype: torch.dtype, lower_bound: float | None
) -> None:
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
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
