"""Tests for the GDN FlashInfer prefill wrapper optimizations.

Ports of the CuteDSL-path optimizations (see test_gdn_prefill_cutedsl.py):
- fused q/k l2norm (single launch, 0-ULP vs two l2norm_fwd calls)
- direct-write output (extend(out=...) writes via FlashInfer's output= param)
- hoisted layer-invariant prep (extend(prep=...) reuses pool-gather indices)
"""

import math

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (  # noqa: E402
    FlashInferGDNKernel,
    is_flashinfer_gdn_prefill_available,
)

if not is_flashinfer_gdn_prefill_available():
    pytest.skip("FlashInfer GDN prefill unavailable", allow_module_level=True)


def _build_extend_inputs(num_seqs: int):
    """Synthetic GDN extend inputs (same construction as the CuteDSL test)."""
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
    g_log = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    # FlashInferGDNKernel.extend consumes alpha = exp(g) (extend_gate_form ==
    # "exp"; produced in serving by fused_gdn_gating(exp_gate=True)).
    g = torch.exp(g_log)
    beta = torch.sigmoid(b.float())

    # Mirror the production MambaSlotAllocator layout: slot 0 is the reserved
    # dummy pad target (negative indices clamp to it on SM100) and live
    # requests get slots 1..N. Using slot 0 for a live row would give
    # index_copy_ duplicate indices when a -1 pad row is present, which is
    # nondeterministic and never happens in production.
    ssm_states = (
        torch.randn(
            num_seqs + 1,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    cache_indices = torch.arange(1, num_seqs + 1, device="cuda", dtype=torch.int32)
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


@pytest.mark.parametrize("num_seqs", [1, 5, 64])
def test_flashinfer_extend_direct_write(num_seqs: int):
    """extend(out=...) must (1) produce bit-identical output and SSM-state
    writeback as the fresh-allocation path, and (2) return a tensor aliasing
    the supplied buffer so the caller can skip its copy."""
    kernel = FlashInferGDNKernel()
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

    assert (o_buf.float() - o_ref.float()).abs().max().item() == 0
    assert (ssm_buf - ssm_ref).abs().max().item() == 0
    assert o_buf.data_ptr() == out_buf.data_ptr()
    assert (out_buf.squeeze(0) - o_buf.squeeze(0)).abs().max().item() == 0

    with pytest.raises(AssertionError):
        _run(
            kernel,
            inp,
            out=out_buf[:, :-1] if inp["total_tokens"] > 1 else out_buf.unsqueeze(0),
        )


@pytest.mark.parametrize("num_seqs", [1, 5, 64])
@pytest.mark.parametrize("pad_last", [False, True])
def test_flashinfer_extend_prep_hoist_equiv(num_seqs: int, pad_last: bool):
    """Precomputed (hoisted) prep passed via extend(prep=...) must yield
    bit-identical output AND ssm-state writeback vs the per-layer recompute
    path (prep=None), incl. -1 padding (clamped to reserved slot 0 on SM100)."""
    kernel = FlashInferGDNKernel()
    inp = _build_extend_inputs(num_seqs)
    if pad_last:
        inp["cache_indices"][-1] = -1

    prep = kernel.build_extend_prep(
        head_k_dim=inp["k"].shape[-1],
        query_start_loc=inp["query_start_loc"],
        cache_indices=inp["cache_indices"],
        ssm_states=inp["ssm_states"],
        total_seq_len=inp["total_tokens"],
    )

    (ssm_cache_indices,) = prep
    if kernel.use_state_pool:
        expected = inp["cache_indices"].clamp(min=0).to(torch.int64)
    else:
        expected = torch.where(
            inp["cache_indices"] >= 0,
            inp["cache_indices"],
            inp["ssm_states"].shape[0] - 1,
        ).to(torch.int64)
    assert torch.equal(ssm_cache_indices, expected)

    o_ref, ssm_ref = _run(kernel, inp)
    o_hoist, ssm_hoist = _run(kernel, inp, prep=prep)

    assert (o_hoist.float() - o_ref.float()).abs().max().item() == 0
    assert (ssm_hoist - ssm_ref).abs().max().item() == 0


@pytest.mark.parametrize("num_seqs", [1, 5])
def test_flashinfer_extend_no_prefix_gather_skip(num_seqs: int):
    """extend(no_prefix=True) skips the SSM pool gather and zero-seeds in-kernel.

    (1) On a cleared pool (the real no-prefix state: freed slots are zeroed) it
    must be bit-identical to the gather path. (2) With poisoned pool rows it
    must IGNORE them — fresh prefills never read slot residue. A regression
    re-introducing the gather on the no_prefix path breaks (2)."""
    kernel = FlashInferGDNKernel()
    inp = _build_extend_inputs(num_seqs)
    inp["ssm_states"].zero_()  # no-prefix reality: slots are cleared

    o_gather, ssm_gather = _run(kernel, inp)

    def run_no_prefix():
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
            no_prefix=True,
        )
        torch.cuda.synchronize()
        assert s1 is None and s2 is None
        return o, ssm

    o_np, ssm_np = run_no_prefix()
    assert (o_np.float() - o_gather.float()).abs().max().item() == 0
    assert (ssm_np - ssm_gather).abs().max().item() == 0

    # Poison the live rows: no_prefix must not read them.
    inp["ssm_states"].fill_(float("nan"))
    o_poison, _ = run_no_prefix()
    assert (o_poison.float() - o_gather.float()).abs().max().item() == 0
    assert not torch.isnan(o_poison.float()).any()


def test_fused_gdn_gating_exp_gate_parity():
    """fused_gdn_gating(exp_gate=True) must equal torch.exp of the log-space
    output elementwise, with beta byte-identical. Guards the in-kernel exp fold
    the FlashInfer prefill path relies on (extend_gate_form == "exp"): a gate-
    form regression here silently corrupts every FlashInfer prefill."""
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating

    T, Hv = 1024, 16
    a = torch.randn(T, Hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(T, Hv, device="cuda", dtype=torch.bfloat16)
    A_log = torch.log(
        torch.empty(Hv, device="cuda", dtype=torch.float32).uniform_(0, 16)
    )
    dt_bias = torch.rand(Hv, device="cuda", dtype=torch.float32)

    g_log, beta_log = fused_gdn_gating(A_log, a, b, dt_bias)
    alpha, beta_exp = fused_gdn_gating(A_log, a, b, dt_bias, exp_gate=True)
    torch.cuda.synchronize()

    assert torch.equal(beta_log, beta_exp)
    assert (alpha - torch.exp(g_log)).abs().max().item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
