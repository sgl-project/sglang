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

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
    pytest.skip(
        "GDN CuteDSL prefill requires CUDA SM10x (Blackwell).",
        allow_module_level=True,
    )

from sglang.srt.layers.attention.fla.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.layers.attention.fla.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.linear.kernels.gdn_blackwell import (  # noqa: E402
    chunk_gated_delta_rule_cutedsl,
    prepare_metadata_cutedsl,
)


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_chunk_cutedsl_correctness(num_seqs: int, state_dtype: torch.dtype):
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


def _run(kernel, inp, *, out=None, prep=None):
    ssm = inp["ssm_states"].clone()
    o, s1, s2 = kernel.extend(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        query_start_loc=inp["query_start_loc"],
        out=out,
        prep=prep,
    )
    torch.cuda.synchronize()
    assert s1 is None and s2 is None
    return o, ssm


def _build_extend_inputs(num_seqs: int):
    """Synthetic GDN extend inputs mirroring the kernel test's init."""
    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to(device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128
    dtype = torch.bfloat16

    # extend() l2-normalizes q/k internally, so pass raw (un-normalized) q/k.
    q = torch.randn(
        1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
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
    # Log-space gate + sigmoid beta, matching fused_gdn_gating's output contract.
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())

    # One SSM slot per sequence plus a trailing sentinel slot (extend maps a
    # cache index of -1 to the last slot).
    ssm_states = (
        torch.randn(
            num_seqs + 1,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
    )
    cache_indices = torch.arange(num_seqs, device="cuda", dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=cu_seqlens,
        total_tokens=total_tokens,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
    )


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
def test_cutedsl_extend_direct_write(num_seqs: int):
    """`CuteDSLGDNKernel.extend(out=...)` must (1) produce bit-identical
    output and SSM-state writeback as the fresh-allocation path, and (2) return
    a tensor that aliases the supplied buffer so the caller can skip its copy."""
    from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
        CuteDSLGDNKernel,
    )

    kernel = CuteDSLGDNKernel()
    if not kernel.supports_prefill:
        pytest.skip("CuteDSL GDN prefill unsupported on this device")

    inp = _build_extend_inputs(num_seqs)

    o_ref, ssm_ref = _run(kernel, inp)

    out_buf = torch.empty(
        1,
        inp["total_tokens"],
        inp["num_v_heads"],
        inp["head_v_dim"],
        device="cuda",
        dtype=o_ref.dtype,
    )
    o_buf, ssm_buf = _run(kernel, inp, out=out_buf)

    # (1) direct-write is numerically identical to fresh-allocation.
    assert (o_buf.float() - o_ref.float()).abs().max().item() == 0
    assert (ssm_buf - ssm_ref).abs().max().item() == 0
    # (2) the returned output aliases the caller's buffer (data_ptr identity is
    # exactly what lets `unified_linear_attention_with_output` skip its copy).
    assert o_buf.data_ptr() == out_buf.data_ptr()
    assert (out_buf.squeeze(0) - o_buf.squeeze(0)).abs().max().item() == 0

    # A mis-shaped out buffer must fail loud, not silently corrupt.
    with pytest.raises(AssertionError):
        _run(
            kernel,
            inp,
            out=out_buf[:, :-1] if inp["total_tokens"] > 1 else out_buf.unsqueeze(0),
        )


@pytest.mark.parametrize("total_tokens", [1, 3, 127, 1024, 4096])
def test_l2norm_qk_fusion_bitexact(total_tokens: int):
    """The fused q/k l2norm must be bit-identical (0-ULP) to the two
    separate ``fla.l2norm.l2norm_fwd`` calls it replaces, including token counts
    that are not multiples of the BT=16 row block."""
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd  # reference
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd_qk

    num_k_heads, head_k_dim, dtype = 4, 128, torch.bfloat16
    q = torch.randn(
        1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)

    old_q = l2norm_fwd(q[0].contiguous())
    old_k = l2norm_fwd(k[0].contiguous())
    new_q, new_k = l2norm_fwd_qk(q[0].contiguous(), k[0].contiguous())
    torch.cuda.synchronize()

    assert (new_q.float() - old_q.float()).abs().max().item() == 0
    assert (new_k.float() - old_k.float()).abs().max().item() == 0
    assert new_q.dtype == dtype and new_k.dtype == dtype
    assert tuple(new_q.shape) == (total_tokens, num_k_heads, head_k_dim)
    assert tuple(new_k.shape) == (total_tokens, num_k_heads, head_k_dim)


def test_l2norm_qk_fusion_launch_count():
    """The fusion must actually collapse 2 kernel launches into 1."""
    from torch.profiler import ProfilerActivity, profile

    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd_qk

    q = torch.randn(4096, 4, 128, device="cuda", dtype=torch.bfloat16).reshape(-1, 128)
    k = torch.randn_like(q)

    def count(name_substr, fn):
        fn()
        torch.cuda.synchronize()  # warmup: excludes Triton JIT-compile launches
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return sum(e.count for e in prof.key_averages() if name_substr in e.key)

    old = count("l2norm_fwd_kernel", lambda: (l2norm_fwd(q), l2norm_fwd(k)))
    new = count("l2norm_fwd_qk_kernel", lambda: l2norm_fwd_qk(q, k))
    assert old == 2 and new == 1


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
@pytest.mark.parametrize("pad_last", [False, True])
def test_cutedsl_extend_prep_hoist_equiv(num_seqs: int, pad_last: bool):
    """Precomputed (hoisted) prep passed via extend(prep=...) must yield
    bit-identical core-attn output AND ssm-state writeback vs the per-layer
    recompute path (prep=None), for single & multi-request, incl. -1 padding
    (remapped to sentinel slot N-1)."""
    from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
        CuteDSLGDNKernel,
    )

    kernel = CuteDSLGDNKernel()
    if not kernel.supports_prefill:
        pytest.skip("CuteDSL GDN prefill unsupported on this device")

    inp = _build_extend_inputs(num_seqs)
    if pad_last:
        # Poison the last row: extend()/build_extend_prep must remap -1 to the
        # trailing sentinel slot (ssm_states.shape[0] - 1 == num_seqs).
        inp["cache_indices"][-1] = -1

    prep = kernel.build_extend_prep(
        head_k_dim=inp["k"].shape[-1],
        query_start_loc=inp["query_start_loc"],
        cache_indices=inp["cache_indices"],
        ssm_states=inp["ssm_states"],
        total_seq_len=inp["total_tokens"],
    )

    # The hoisted ssm_cache_indices must equal the in-kernel where() remap.
    _, ssm_cache_indices, _, _ = prep
    expected = torch.where(
        inp["cache_indices"] >= 0,
        inp["cache_indices"],
        inp["ssm_states"].shape[0] - 1,
    ).to(torch.long)
    assert torch.equal(ssm_cache_indices, expected)

    o_ref, ssm_ref = _run(kernel, inp)  # per-layer recompute path
    o_hoist, ssm_hoist = _run(kernel, inp, prep=prep)  # hoisted reuse

    assert (o_hoist.float() - o_ref.float()).abs().max().item() == 0
    assert (ssm_hoist - ssm_ref).abs().max().item() == 0


def test_cutedsl_extend_prep_hoist_launch_count():
    """Reusing hoisted prep across N layers collapses the
    prepare_metadata_cutedsl (PrepMetaKernel) call from N -> 1."""
    from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
        CuteDSLGDNKernel,
    )

    kernel = CuteDSLGDNKernel()
    if not kernel.supports_prefill:
        pytest.skip("CuteDSL GDN prefill unsupported on this device")

    inp = _build_extend_inputs(5)
    kernel._ensure_extend_loaded(inp["k"].shape[-1])  # load _prepare_meta_fn

    calls = {"n": 0}
    orig = kernel._prepare_meta_fn

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    kernel._prepare_meta_fn = counting

    def one_layer(prep_arg):
        kernel.extend(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=inp["g"],
            beta=inp["beta"],
            ssm_states=inp["ssm_states"].clone(),
            cache_indices=inp["cache_indices"],
            query_start_loc=inp["query_start_loc"],
            prep=prep_arg,
        )

    N_LAYERS = 45

    # Per-layer recompute path: one prepare_meta per layer.
    calls["n"] = 0
    for _ in range(N_LAYERS):
        one_layer(None)
    torch.cuda.synchronize()
    assert calls["n"] == N_LAYERS

    # Hoist path: build once, reuse -> exactly one prepare_meta for all layers.
    calls["n"] = 0
    prep = kernel.build_extend_prep(
        head_k_dim=inp["k"].shape[-1],
        query_start_loc=inp["query_start_loc"],
        cache_indices=inp["cache_indices"],
        ssm_states=inp["ssm_states"],
        total_seq_len=inp["total_tokens"],
    )
    for _ in range(N_LAYERS):
        one_layer(prep)
    torch.cuda.synchronize()
    assert calls["n"] == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
