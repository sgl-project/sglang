"""Correctness test for the SM100 CuTe DSL GDN prefill kernel.

Ported from vLLM PR https://github.com/vllm-project/vllm/pull/43273.
Validates ``chunk_gated_delta_rule_cutedsl`` against the
``fused_recurrent_gated_delta_rule`` Triton reference.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

# CuteDSL prefill kernel only exists on Blackwell. Single-GPU kernel-unit
# suite is the right slot (matches existing jit_kernel test_*.py pattern).
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
    pytest.skip(
        "GDN CuteDSL prefill requires CUDA SM10x (Blackwell).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule,
)
from sglang.kernels.ops.attention.fla.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.linear.gdn_blackwell import (  # noqa: E402
    chunk_gated_delta_rule_cutedsl,
    prepare_metadata_cutedsl,
)


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_chunk_cutedsl_correctness(num_seqs: int, state_dtype: torch.dtype):
    # Fixed per-case seed: the mean-error assertions sit close to the observed
    # error distribution (state_error.mean() threshold 6e-4 vs ~6.3e-4 seen on
    # unlucky draws in CI), so unseeded inputs make the test flaky.
    torch.manual_seed(num_seqs)
    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to(device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128
    dtype = torch.bfloat16

    q = torch.randn(
        1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
    q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    a = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    b = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)

    # Match upstream FLA GatedDeltaNet synthetic init.
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())
    initial_state = (
        torch.randn(
            num_seqs,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )

    # Metadata kernel matches the FLA reference helpers.
    chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, total_tokens)
    torch.cuda.synchronize()

    expected_indices = prepare_chunk_indices(cu_seqlens, 64)
    expected_offsets = prepare_chunk_offsets(cu_seqlens, 64)
    total_chunks = int(expected_offsets[-1].item())

    torch.testing.assert_close(chunk_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(chunk_indices[:total_chunks], expected_indices)

    # Reference: token-by-token recurrent kernel returns (o, final_state).
    # Recurrent path needs float32 state, so cast initial_state for the call.
    ref_o, ref_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.to(torch.float32),
        output_final_state=True,
        cu_seqlens=cu_seqlens.to(torch.int64),
        use_qk_l2norm_in_kernel=False,
    )

    actual_core_attn_out = torch.empty(
        total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
    actual_o, actual_state = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        core_attn_out=actual_core_attn_out,
    )
    torch.cuda.synchronize()

    o_error = (actual_o.float() - ref_o.float()).abs()
    state_error = (
        actual_state.float() - ref_state.to(actual_state.dtype).float()
    ).abs()
    assert o_error.max().item() < 2e-3
    assert o_error.mean().item() < 6e-5
    assert state_error.max().item() < 2e-2
    assert state_error.mean().item() < 6e-4
    core_attn_out_error = (
        actual_core_attn_out.float() - actual_o.squeeze(0).float()
    ).abs()
    assert core_attn_out_error.max().item() == 0

    no_buffer_o, no_buffer_state = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    torch.cuda.synchronize()

    no_buffer_o_error = (no_buffer_o.float() - ref_o.float()).abs()
    no_buffer_state_error = (
        no_buffer_state.float() - ref_state.to(no_buffer_state.dtype).float()
    ).abs()
    buffer_o_error = (no_buffer_o.float() - actual_o.float()).abs()
    buffer_state_error = (
        no_buffer_state.float() - actual_state.to(no_buffer_state.dtype).float()
    ).abs()
    assert no_buffer_o_error.max().item() < 2e-3
    assert no_buffer_o_error.mean().item() < 6e-5
    assert no_buffer_state_error.max().item() < 2e-2
    assert no_buffer_state_error.mean().item() < 6e-4
    assert buffer_o_error.max().item() == 0
    assert buffer_state_error.max().item() == 0


def test_gdn_chunk_cutedsl_newton_schulz_precision():
    """Regression test for precision loss in the Newton-Schulz inverse.

    The nearly collinear K vectors produce numerically sensitive 16x16
    diagonal blocks in I + A. A single-BF16 Newton-Schulz operand loses
    enough precision to produce a large error, while the compensated
    BF16x2 implementation stays close to the recurrent reference.
    """

    dtype = torch.bfloat16
    total_tokens = 64

    # Match Qwen3.5-35B-A3B GDN shapes.
    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128

    # A compact, deterministic near-collinear pattern. Each row contains
    # two non-zero coordinates and is normalized before conversion to BF16.
    #
    # Repeating the 16-row pattern four times targets all four 16x16
    # diagonal blocks processed by the Newton-Schulz phase of one
    # 64-token GDN chunk.
    k_pattern_2d = torch.tensor(
        [
            [0.96875000, 0.25390625],
            [0.96875000, 0.25390625],
            [0.97265625, 0.23925781],
            [0.98437500, 0.18261719],
            [0.95312500, 0.29882812],
            [0.95312500, 0.30273438],
            [0.92187500, 0.38476562],
            [0.97656250, 0.22265625],
            [0.90625000, 0.42578125],
            [0.92187500, 0.38867188],
            [0.85156250, 0.52343750],
            [0.78906250, 0.61718750],
            [0.36523438, 0.92968750],
            [0.01489258, 1.00000000],
            [-0.06347656, 0.99609375],
            [0.26953125, 0.96484375],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    k_pattern_2d = F.normalize(
        k_pattern_2d,
        p=2,
        dim=-1,
    ).to(dtype)

    k_base = torch.zeros(
        total_tokens,
        head_k_dim,
        device="cuda",
        dtype=dtype,
    )
    k_base[:, :2] = k_pattern_2d.repeat(4, 1)

    # Use the same normalized K pattern on every QK head. This keeps the
    # test deterministic and exercises the exact Qwen3.5 TP1 head shape.
    k = (
        k_base[None, :, None, :]
        .expand(
            1,
            total_tokens,
            num_k_heads,
            head_k_dim,
        )
        .contiguous()
    )
    q = k.clone()

    # Deterministic V values avoid dependence on CUDA RNG implementation.
    v = torch.linspace(
        -0.5,
        0.5,
        steps=total_tokens * num_v_heads * head_v_dim,
        device="cuda",
        dtype=torch.float32,
    ).reshape(
        1,
        total_tokens,
        num_v_heads,
        head_v_dim,
    ).to(dtype)

    # These values are within the GDN kernel contract:
    #   g is a small negative decay
    #   beta is close to, but below, one
    #
    # Together with the correlated K vectors, they form a precision-
    # sensitive I + A without requiring model weights.
    g = torch.full(
        (1, total_tokens, num_v_heads),
        -1.0e-4,
        device="cuda",
        dtype=torch.float32,
    )
    beta = torch.full(
        (1, total_tokens, num_v_heads),
        0.99,
        device="cuda",
        dtype=torch.float32,
    )

    initial_state = torch.zeros(
        1,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        device="cuda",
        dtype=torch.float32,
    )

    cu_seqlens = torch.tensor(
        [0, total_tokens],
        device="cuda",
        dtype=torch.int32,
    )
    chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
        cu_seqlens,
        total_tokens,
    )

    ref_o, ref_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens.to(torch.int64),
        use_qk_l2norm_in_kernel=False,
    )

    actual_o, actual_state = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    torch.cuda.synchronize()

    # Keep explicit finite checks so a regression reports the numerical
    # failure directly instead of only producing an assert_close message
    # containing NaN/Inf.
    assert torch.isfinite(ref_o).all()
    assert torch.isfinite(ref_state).all()
    assert torch.isfinite(actual_o).all()
    assert torch.isfinite(actual_state).all()

    torch.testing.assert_close(
        actual_o.float(),
        ref_o.float(),
        atol=5.0e-2,
        rtol=5.0e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        ref_state.float(),
        atol=5.0e-2,
        rtol=5.0e-2,
    )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_chunk_cutedsl_pool_mode_matches_dense(state_dtype: torch.dtype):
    """Pool mode (initial_state_indices) must reproduce the dense gather/scatter
    path bit-for-bit: same o, same final-state rows written in place at the
    indexed pool slots, and every other pool row untouched."""
    torch.manual_seed(11)
    num_seqs = 5
    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to(device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128
    dtype = torch.bfloat16

    q = torch.randn(
        1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
    q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    a = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    b = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())
    h0_dense = (
        torch.randn(
            num_seqs,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )

    # Same states scattered into a larger pool at shuffled slots.
    num_slots = 64
    pool = (
        torch.randn(
            num_slots,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )
    slots = torch.randperm(num_slots, device="cuda")[:num_seqs].to(torch.int32)
    pool[slots.long()] = h0_dense
    pool_before = pool.clone()

    chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, total_tokens)

    o_dense, ht_dense = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0_dense.clone(),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o_pool, ht_pool = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=pool,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_state_indices=slots,
    )
    torch.cuda.synchronize()

    # Same kernels and math; only the state addressing differs -> bit-identical.
    assert ht_pool is pool
    assert torch.equal(o_pool, o_dense)
    assert torch.equal(pool[slots.long()], ht_dense)
    untouched = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    untouched[slots.long()] = False
    assert torch.equal(pool[untouched], pool_before[untouched])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
