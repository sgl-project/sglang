"""Tests for the GDN FlashInfer prefill wrapper optimizations.

Covered:
- fused packed prologue (single launch, 0-ULP vs the established kernels)
- direct-write output (extend_fused(out=...) writes via FlashInfer's output= param)
"""

import math

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (  # noqa: E402
    FlashInferGDNKernel,
    is_flashinfer_gdn_prefill_available,
)

if not is_flashinfer_gdn_prefill_available():
    pytest.skip("FlashInfer GDN prefill unavailable", allow_module_level=True)


def _build_extend_inputs(kernel: FlashInferGDNKernel, num_seqs: int):
    """Synthetic GDN extend inputs matching production layouts."""
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
    mixed_qkv = torch.cat(
        [
            q[0].reshape(total_tokens, -1),
            k[0].reshape(total_tokens, -1),
            v[0].reshape(total_tokens, -1),
        ],
        dim=-1,
    )
    shape = dict(
        num_q_heads=num_k_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_q_dim=head_k_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )

    # Mirror the production MambaSlotAllocator layout: slot 0 is the reserved
    # dummy pad target (negative indices clamp to it on SM100) and live
    # requests get slots 1..N. Using slot 0 for a live row would give
    # index_copy_ duplicate indices when a -1 pad row is present, which is
    # nondeterministic and never happens in production.
    state_dtype = torch.bfloat16 if kernel.use_state_pool else torch.float32
    ssm_states = (
        torch.randn(
            num_seqs + 1,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )
    cache_indices = torch.arange(1, num_seqs + 1, device="cuda", dtype=torch.int32)
    return dict(
        mixed_qkv=mixed_qkv,
        a=a[0],
        b=b[0],
        A_log=A_log,
        dt_bias=dt_bias,
        shape=shape,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=cu_seqlens,
        total_tokens=total_tokens,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
    )


def _run_fused(kernel, inp, *, out=None, no_prefix=False):
    ssm = inp["ssm_states"].clone()
    o, s1, s2 = kernel.extend_fused(
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        **inp["shape"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        query_start_loc=inp["query_start_loc"],
        out=out,
        no_prefix=no_prefix,
    )
    torch.cuda.synchronize()
    assert s1 is None and s2 is None
    return o, ssm


@pytest.mark.parametrize("num_seqs", [1, 5, 64])
def test_flashinfer_extend_fused_direct_write(num_seqs: int):
    """extend_fused(out=...) must produce bit-identical output and state
    writeback as the fresh-allocation path, and (2) return a tensor aliasing
    the supplied buffer so the caller can skip its copy."""
    kernel = FlashInferGDNKernel()
    inp = _build_extend_inputs(kernel, num_seqs)

    o_ref, ssm_ref = _run_fused(kernel, inp)

    out_buf = torch.empty(
        1,
        inp["total_tokens"],
        inp["num_v_heads"],
        inp["head_v_dim"],
        device="cuda",
        dtype=o_ref.dtype,
    )
    o_buf, ssm_buf = _run_fused(kernel, inp, out=out_buf)

    assert (o_buf.float() - o_ref.float()).abs().max().item() == 0
    assert (ssm_buf - ssm_ref).abs().max().item() == 0
    assert o_buf.data_ptr() == out_buf.data_ptr()
    assert (out_buf.squeeze(0) - o_buf.squeeze(0)).abs().max().item() == 0

    with pytest.raises(AssertionError):
        _run_fused(
            kernel,
            inp,
            out=out_buf[:, :-1] if inp["total_tokens"] > 1 else out_buf.unsqueeze(0),
        )


@pytest.mark.parametrize("num_seqs", [1, 5])
def test_flashinfer_extend_fused_no_prefix_gather_skip(num_seqs: int):
    """extend_fused(no_prefix=True) skips the pool gather and zero-seeds.

    (1) On a cleared pool (the real no-prefix state: freed slots are zeroed) it
    must be bit-identical to the gather path. (2) With poisoned pool rows it
    must IGNORE them — fresh prefills never read slot residue. A regression
    re-introducing the gather on the no_prefix path breaks (2)."""
    kernel = FlashInferGDNKernel()
    inp = _build_extend_inputs(kernel, num_seqs)
    inp["ssm_states"].zero_()  # no-prefix reality: slots are cleared

    o_gather, ssm_gather = _run_fused(kernel, inp)

    o_np, ssm_np = _run_fused(kernel, inp, no_prefix=True)
    assert (o_np.float() - o_gather.float()).abs().max().item() == 0
    assert (ssm_np - ssm_gather).abs().max().item() == 0

    # Poison the live rows: no_prefix must not read them.
    inp["ssm_states"].fill_(float("nan"))
    o_poison, ssm_poison = _run_fused(kernel, inp, no_prefix=True)
    assert (o_poison.float() - o_gather.float()).abs().max().item() == 0
    assert not torch.isnan(o_poison.float()).any()
    active = inp["cache_indices"].to(torch.int64)
    assert (ssm_poison[active] - ssm_gather[active]).abs().max().item() == 0
    assert torch.isfinite(ssm_poison[active]).all()


def _bits_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    assert x.shape == y.reshape(x.shape).shape
    itype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
    return bool(
        (
            x.contiguous().view(itype) == y.reshape(x.shape).contiguous().view(itype)
        ).all()
    )


def _make_fused_inputs(total_tokens: int, seed: int, strided_ab: bool = False):
    """Adversarial fused inputs: wide magnitudes exercise both softplus branches
    (threshold 20), sigmoid saturation, and exp under/overflow."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Hk, Hv, D = 4, 16, 128
    mixed_qkv = (
        torch.randn((total_tokens, 2 * Hk * D + Hv * D), generator=gen, device="cuda")
        .mul(2.0)
        .to(torch.bfloat16)
    )
    if total_tokens > 2:
        mixed_qkv[1] = 0  # all-zero q/k rows: the 0/sqrt(eps) norm path
    ab_cols = 2 * Hv if strided_ab else Hv
    a_buf = (
        torch.randn((total_tokens, ab_cols), generator=gen, device="cuda") * 12.0
    ).to(torch.bfloat16)
    b_buf = (
        torch.randn((total_tokens, ab_cols), generator=gen, device="cuda") * 4.0
    ).to(torch.bfloat16)
    a = a_buf[:, :Hv]  # row-strided view when strided_ab
    b = b_buf[:, :Hv]
    A_log = torch.randn((Hv,), generator=gen, device="cuda") * 1.5
    dt_bias = torch.randn((Hv,), generator=gen, device="cuda") * 6.0
    return mixed_qkv, a, b, A_log, dt_bias


def _fused_ref_chain(mixed_qkv, a, b, A_log, dt_bias):
    """The established split, gating, and two-normalization chain."""
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

    q, k, v = fused_qkv_split_gdn_prefill(mixed_qkv, 4, 4, 16, 128, 128, 128)
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    q_norm = l2norm_fwd(q[0].contiguous())
    k_norm = l2norm_fwd(k[0].contiguous())
    return q_norm, k_norm, v[0], torch.exp(g), beta


@pytest.mark.parametrize("total_tokens", [1, 7, 127, 1021, 1024, 16384])
def test_gdn_prefill_fused_bitexact(total_tokens: int):
    """The axis-stacked fused kernel must be bit-identical (0-ULP) to the real
    split -> gating -> l2norm chain on all five outputs. This is the
    load-bearing guard for layout-specific lowering and Triton-upgrade drift
    (the manual-pointer norm loads equal the block-ptr original as a lowering
    property, not a language contract)."""
    from sglang.jit_kernel.triton.gdn_prefill_fused import gdn_prefill_fused

    mixed_qkv, a, b, A_log, dt_bias = _make_fused_inputs(
        total_tokens, seed=1234 + total_tokens, strided_ab=(total_tokens == 1021)
    )
    q_ref, k_ref, v_ref, alpha_ref, beta_ref = _fused_ref_chain(
        mixed_qkv, a, b, A_log, dt_bias
    )
    q_norm, k_norm, v, alpha, beta = gdn_prefill_fused(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        num_qk_heads=4,
        num_v_heads=16,
        head_qk_dim=128,
        head_v_dim=128,
    )
    torch.cuda.synchronize()

    assert _bits_equal(q_norm[0].view(-1, 128), q_ref.view(-1, 128))
    assert _bits_equal(k_norm[0].view(-1, 128), k_ref.view(-1, 128))
    assert _bits_equal(v[0], v_ref)
    assert _bits_equal(alpha, alpha_ref)
    assert _bits_equal(beta, beta_ref)


def test_flashinfer_extend_fused_matches_extend():
    """The fused route must match the canonical log-g ``extend`` API."""
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

    kernel = FlashInferGDNKernel()
    T = 940
    mixed_qkv, a, b, A_log, dt_bias = _make_fused_inputs(T, seed=77)
    shape = dict(
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=16,
        head_q_dim=128,
        head_k_dim=128,
        head_v_dim=128,
    )
    cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    state_dtype = torch.bfloat16 if kernel.use_state_pool else torch.float32
    pool = torch.zeros(2, 16, 128, 128, device="cuda", dtype=state_dtype)
    idx = torch.arange(1, 2, device="cuda", dtype=torch.int32)

    # Canonical public path: raw split, log-space gate, q/k normalized internally.
    q_raw, k_raw, v_raw = fused_qkv_split_gdn_prefill(
        mixed_qkv,
        shape["num_q_heads"],
        shape["num_k_heads"],
        shape["num_v_heads"],
        shape["head_q_dim"],
        shape["head_k_dim"],
        shape["head_v_dim"],
    )
    g_log, beta_e = fused_gdn_gating(A_log, a, b, dt_bias)
    pool_e = pool.clone()
    o_e, _, _ = kernel.extend(
        q=q_raw,
        k=k_raw,
        v=v_raw,
        g=g_log,
        beta=beta_e,
        ssm_states=pool_e,
        cache_indices=idx,
        query_start_loc=cu,
    )

    # Production fast path: packed/raw inputs remain atomic inside FlashInfer.
    pool_p = pool.clone()
    out_p = torch.empty(1, T, 16, 128, device="cuda", dtype=torch.bfloat16)
    o_p, s1, s2 = kernel.extend_fused(
        mixed_qkv,
        a,
        b,
        **shape,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=pool_p,
        cache_indices=idx,
        query_start_loc=cu,
        out=out_p,
    )
    torch.cuda.synchronize()
    assert s1 is None and s2 is None

    assert _bits_equal(o_p[0], o_e[0])
    assert _bits_equal(pool_p, pool_e)
    assert _bits_equal(out_p[0], o_e[0])
    assert o_p.data_ptr() == out_p.data_ptr()  # direct-write preserved


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
