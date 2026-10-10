"""Correctness tests for the DeepSeek-V4 sparse paged-prefill Triton kernel,
covering the gfx1250 launch-config selection.

``_sparse_attn_v4_paged_prefill_triton`` picks BLOCK_H / BLOCK_K / num_warps /
num_stages per architecture: gfx1250 uses (64, 16, 4, 2), every other target
keeps the original (16, 16-or-32, 8, no-pipelining). Those constants only
change how the kernel is *scheduled* — the online-softmax accumulation is
tile-size invariant, so every config must produce the same numbers.

The tests therefore:
  - check the kernel against a dense torch reference (gather the indexed KV
    rows, masked softmax, weighted sum), and
  - check every launch config against each other on identical inputs, which is
    what actually guards the tuning: a reviewer changing the constants gets a
    failure here rather than silently different attention output.

Shapes cover both regions the kernel walks (paged ``unified_kv`` prefix and
flat per-fwd ``kv`` extend), ``-1`` sentinels in both index lists, H not a
multiple of BLOCK_H, and T on both sides of the point where the gfx1250
config was measured to win.
"""

from __future__ import annotations

import sys

import pytest
import torch
import triton

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import (
    _sparse_attn_v4_paged_prefill_kernel,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

# (BLOCK_H, BLOCK_K, num_warps, num_stages)
SHIPPED_CONFIG = (16, 16, 8, None)
GFX1250_CONFIG = (64, 16, 4, 2)
LAUNCH_CONFIGS = [SHIPPED_CONFIG, GFX1250_CONFIG, (32, 32, 8, 2)]

D = 512  # DSV4-Flash head_dim; >= 256 so the shipped branch picks BLOCK_K=16
NEG_LARGE = -1.0e38


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP required")


def _make_inputs(T, H, n_prefix, n_extend, n_slots=1024, seed=0, device="cuda"):
    """Build one prefill batch. ``-1`` entries exercise the sentinel path."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = torch.bfloat16

    q = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=dt
    )
    unified_kv = torch.randn(n_slots, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=dt
    )
    kv = torch.randn(n_slots, D, generator=g, dtype=torch.float32).to(
        device=device, dtype=dt
    )
    attn_sink = torch.zeros(H, device=device, dtype=torch.float32)

    ip = torch.randint(0, n_slots, (T * n_prefix,), generator=g, dtype=torch.int32)
    ie = torch.randint(0, n_slots, (T * n_extend,), generator=g, dtype=torch.int32)
    # sprinkle sentinels: every 7th prefix slot and every 5th extend slot
    ip[::7] = -1
    ie[::5] = -1

    return dict(
        q=q,
        unified_kv=unified_kv,
        kv=kv,
        attn_sink=attn_sink,
        kv_indices_prefix=ip.to(device),
        kv_indptr_prefix=(torch.arange(T + 1, dtype=torch.int32) * n_prefix).to(device),
        kv_indices_extend=ie.to(device),
        kv_indptr_extend=(torch.arange(T + 1, dtype=torch.int32) * n_extend).to(device),
        T=T,
        H=H,
    )


def _run(inp, block_h, block_k, num_warps, num_stages):
    q = inp["q"]
    T, H = inp["T"], inp["H"]
    out = torch.empty_like(q)
    kwargs = dict(
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    if num_stages is not None:
        kwargs["num_stages"] = num_stages

    _sparse_attn_v4_paged_prefill_kernel[(T, triton.cdiv(H, block_h))](
        q,
        inp["unified_kv"],
        inp["kv_indices_prefix"],
        inp["kv_indptr_prefix"],
        inp["kv"],
        inp["kv_indices_extend"],
        inp["kv_indptr_extend"],
        inp["attn_sink"],
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        inp["unified_kv"].stride(0),
        inp["unified_kv"].stride(1),
        inp["kv"].stride(0),
        inp["kv"].stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        1.0 / (D**0.5),
        **kwargs,
    )
    return out


def _reference(inp):
    """Dense fp32 reference: gather both KV regions per token, masked softmax."""
    q = inp["q"].float()
    ukv, kv = inp["unified_kv"].float(), inp["kv"].float()
    T, H = inp["T"], inp["H"]
    scale = 1.0 / (D**0.5)
    out = torch.empty(T, H, D, device=q.device, dtype=torch.float32)

    ip, pp = inp["kv_indices_prefix"], inp["kv_indptr_prefix"]
    ie, pe = inp["kv_indices_extend"], inp["kv_indptr_extend"]

    for t in range(T):
        rows = []
        sl = ip[pp[t] : pp[t + 1]]
        rows.append(ukv[sl[sl >= 0].long()])
        sl = ie[pe[t] : pe[t + 1]]
        rows.append(kv[sl[sl >= 0].long()])
        k = torch.cat(rows, dim=0)  # [N, D]
        scores = (q[t] @ k.T) * scale  # [H, N]
        p = torch.softmax(scores.float(), dim=-1)
        out[t] = p @ k
    return out


@pytest.mark.parametrize("block_h,block_k,num_warps,num_stages", LAUNCH_CONFIGS)
def test_matches_dense_reference(block_h, block_k, num_warps, num_stages):
    """Every launch config reproduces a dense torch softmax-attention."""
    _skip_if_unavailable()
    inp = _make_inputs(T=8, H=64, n_prefix=64, n_extend=16)
    got = _run(inp, block_h, block_k, num_warps, num_stages).float()
    want = _reference(inp)
    # bf16 inputs accumulated in fp32; tolerance covers the input rounding only
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("T", [1, 5, 64, 512])
@pytest.mark.parametrize("H", [64, 48])
def test_launch_configs_agree(T, H):
    """The launch config must not change results.

    H=48 is deliberately not a multiple of 64, so BLOCK_H=64 exercises the
    head-mask tail. T spans both sides of the size where the gfx1250 config
    was measured to win.
    """
    _skip_if_unavailable()
    inp = _make_inputs(T=T, H=H, n_prefix=48, n_extend=16, seed=T + H)
    ref = _run(inp, *SHIPPED_CONFIG).float()
    for cfg in LAUNCH_CONFIGS[1:]:
        got = _run(inp, *cfg).float()
        torch.testing.assert_close(
            got, ref, rtol=1e-2, atol=1e-2, msg=f"config {cfg} disagrees with shipped"
        )


def test_all_sentinel_rows_are_finite():
    """A token whose index lists are entirely ``-1`` must not emit NaN/Inf.

    The kernel seeds the running max with a large negative sentinel; if a token
    contributes no KV rows at all, the softmax denominator stays at zero and a
    naive implementation divides by it.
    """
    _skip_if_unavailable()
    inp = _make_inputs(T=4, H=64, n_prefix=32, n_extend=8, seed=7)
    inp["kv_indices_prefix"][:] = -1
    inp["kv_indices_extend"][:] = -1
    for cfg in LAUNCH_CONFIGS:
        out = _run(inp, *cfg)
        assert torch.isfinite(out.float()).all(), (
            f"config {cfg} produced non-finite output"
        )


def test_wrapper_dispatch_matches_arch():
    """The public wrapper must launch with the constants its arch branch claims.

    Capture the kwargs the wrapper hands to the Triton launcher and compare
    against the expected tuple, so a future edit that changes one branch
    without the other is caught. Both branches are asserted from a single run:
    the non-gfx1250 constants are checked by monkeypatching the module flag.
    """
    _skip_if_unavailable()
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import paged_prefill as m

    inp = _make_inputs(T=4, H=64, n_prefix=32, n_extend=8, seed=3)
    seen = {}

    class _Spy:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                seen["cfg"] = (
                    kwargs["BLOCK_H"],
                    kwargs["BLOCK_K"],
                    kwargs["num_warps"],
                    kwargs.get("num_stages"),
                )
                return _sparse_attn_v4_paged_prefill_kernel[grid](*args, **kwargs)

            return call

    orig_kernel, orig_flag = m._sparse_attn_v4_paged_prefill_kernel, m._is_gfx1250
    try:
        m._sparse_attn_v4_paged_prefill_kernel = _Spy()
        for flag, expected in ((True, GFX1250_CONFIG), (False, SHIPPED_CONFIG)):
            m._is_gfx1250 = flag
            m._sparse_attn_v4_paged_prefill_triton(
                inp["q"],
                inp["unified_kv"],
                inp["kv_indices_prefix"],
                inp["kv_indptr_prefix"],
                inp["kv"],
                inp["kv_indices_extend"],
                inp["kv_indptr_extend"],
                inp["attn_sink"],
                1.0 / (D**0.5),
            )
            assert seen["cfg"] == expected, (
                f"_is_gfx1250={flag} launched {seen['cfg']}, expected {expected}"
            )
    finally:
        m._sparse_attn_v4_paged_prefill_kernel = orig_kernel
        m._is_gfx1250 = orig_flag


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
