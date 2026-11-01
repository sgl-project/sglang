"""Correctness harness for the gated causal conv1d CuTe DSL kernel (`cutedsl_gated_conv`).

Validates the kernel for kernel widths {3,4,7} -- K=7 (e.g. LFM3's conv_L_cache=7) is
beyond what the sgl_kernel C++ causal_conv1d (width 2-4) and the shared Triton kernel
(width 2-5, partial) cover. The *reference* semantics are pinned to the torch
`Lfm3ShortConv._conv_prefill` / `_conv_decode` (which match HF `slow_forward`), and the
kernel is checked against that reference for prefill (varlen, with/without initial
state) and decode.

Run:
    python -m pytest test/registered/kernels/ops/mamba/test_cutedsl_gated_conv.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


# ---------------------------------------------------------------------------
# Reference semantics (the contract the CuTe kernel must satisfy)
# ---------------------------------------------------------------------------
def ref_conv_decode(
    Bx: torch.Tensor,  # (B, dim) one new token per request
    conv_state: torch.Tensor,  # (num_slots, dim, K-1) holds last K-1 inputs, mutated
    weight: torch.Tensor,  # (dim, K)
    bias: torch.Tensor | None,  # (dim,) or None
    slot_idx: torch.Tensor,  # (B,) int32 -> row in conv_state
) -> torch.Tensor:
    """Single-token rolling update. Mirrors Lfm3ShortConv._conv_decode."""
    K = weight.shape[-1]
    sel = conv_state[slot_idx].to(Bx.dtype)  # (B, dim, K-1)
    window = torch.cat([sel, Bx.unsqueeze(-1)], dim=-1)  # (B, dim, K)
    out = (window * weight.to(Bx.dtype).unsqueeze(0)).sum(-1)  # (B, dim)
    if bias is not None:
        out = out + bias.to(out.dtype)
    conv_state[slot_idx] = window[:, :, 1:].to(conv_state.dtype)
    return out


def ref_conv_prefill(
    Bx_t: torch.Tensor,  # (dim, total_tokens) channel-major, varlen-concat
    conv_state: torch.Tensor,  # (num_slots, dim, K-1), mutated
    weight: torch.Tensor,  # (dim, K)
    bias: torch.Tensor | None,  # (dim,) or None
    query_start_loc: torch.Tensor,  # (B+1,) int32 cumulative token offsets
    slot_idx: torch.Tensor,  # (B,) int32
    has_initial_state: torch.Tensor,  # (B,) bool
) -> torch.Tensor:
    """Per-request causal depthwise conv. Mirrors Lfm3ShortConv._conv_prefill +
    the has_initial_state handling from Lfm2ShortConv (chunked-prefill safe)."""
    dim, K = weight.shape
    w = weight.to(Bx_t.dtype).unsqueeze(1)  # (dim,1,K)
    b = bias.to(Bx_t.dtype) if bias is not None else None
    starts = query_start_loc.tolist()
    slots = slot_idx.tolist()
    init = has_initial_state.tolist()
    out_chunks = []
    for i, slot in enumerate(slots):
        s, e = starts[i], starts[i + 1]
        if s == e:
            continue
        seg = Bx_t[:, s:e]  # (dim, T_i)
        if init[i]:
            state_i = conv_state[slot].to(Bx_t.dtype)  # (dim, K-1)
        else:
            state_i = torch.zeros(dim, K - 1, device=Bx_t.device, dtype=Bx_t.dtype)
        cat = torch.cat([state_i, seg], dim=-1)  # (dim, K-1+T_i)
        y = F.conv1d(cat.unsqueeze(0), w, bias=b, groups=dim)[0]  # (dim, T_i)
        out_chunks.append(y)
        new_state = cat[:, -(K - 1) :]
        if new_state.shape[-1] < K - 1:
            new_state = F.pad(new_state, (K - 1 - new_state.shape[-1], 0))
        conv_state[slot] = new_state.to(conv_state.dtype)
    return torch.cat(out_chunks, dim=-1) if out_chunks else Bx_t.new_zeros(dim, 0)


# ---------------------------------------------------------------------------
# Kernel-under-test hook (CuTe DSL kernel plugs in here once written)
# ---------------------------------------------------------------------------
def _cute_kernel_supported():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] in (9, 10, 12)


def _load_cute_kernel():
    from sglang.kernels.ops.mamba.cutedsl_gated_conv import (
        gated_causal_conv1d_decode,
        gated_causal_conv1d_prefill,
    )

    return gated_causal_conv1d_prefill, gated_causal_conv1d_decode


# ---------------------------------------------------------------------------
# A *second*, independent reference (pure einsum sliding window) to prove the
# F.conv1d-based reference is itself correct — reference self-consistency.
# ---------------------------------------------------------------------------
def naive_decode(Bx, conv_state, weight, bias, slot_idx):
    K = weight.shape[-1]
    B, dim = Bx.shape
    out = torch.empty(B, dim, device=Bx.device, dtype=torch.float32)
    for bi in range(B):
        win = torch.cat(
            [conv_state[slot_idx[bi]].float(), Bx[bi].float().unsqueeze(-1)], dim=-1
        )  # (dim, K)
        o = (win * weight.float()).sum(-1)
        if bias is not None:
            o = o + bias.float()
        out[bi] = o
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("K", [3, 4, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_reference_decode_self_consistent(K, dtype):
    torch.manual_seed(0)
    dim, B, slots = 32, 5, 8
    dev = "cuda"
    Bx = torch.randn(B, dim, device=dev, dtype=dtype)
    state = torch.randn(slots, dim, K - 1, device=dev, dtype=dtype)
    w = torch.randn(dim, K, device=dev, dtype=dtype)
    bias = torch.randn(dim, device=dev, dtype=dtype)
    idx = torch.randperm(slots, device=dev)[:B].to(torch.int32)
    ref = ref_conv_decode(Bx, state.clone(), w, bias, idx)
    naive = naive_decode(Bx, state, w, bias, idx)
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(ref.float(), naive, atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("K", [3, 4, 7])
@pytest.mark.parametrize("with_init", [False, True])
def test_reference_prefill_matches_decode_rollout(K, with_init):
    """Prefill of T tokens for one request must equal T sequential decode steps."""
    torch.manual_seed(1)
    dim, T = 16, 11
    dev = "cuda"
    dt = torch.float32  # exactness check in fp32
    Bx_t = torch.randn(dim, T, device=dev, dtype=dt)
    w = torch.randn(dim, K, device=dev, dtype=dt)
    bias = torch.randn(dim, device=dev, dtype=dt)
    init_state = (
        torch.randn(1, dim, K - 1, device=dev, dtype=dt)
        if with_init
        else torch.zeros(1, dim, K - 1, device=dev, dtype=dt)
    )
    qsl = torch.tensor([0, T], device=dev, dtype=torch.int32)
    idx = torch.tensor([0], device=dev, dtype=torch.int32)
    hi = torch.tensor([with_init], device=dev)

    pre_state = init_state.clone()
    pre_out = ref_conv_prefill(Bx_t, pre_state, w, bias, qsl, idx, hi)  # (dim, T)

    # sequential decode rollout
    dec_state = init_state.clone()
    dec_outs = []
    for t in range(T):
        o = ref_conv_decode(Bx_t[:, t].unsqueeze(0), dec_state, w, bias, idx)
        dec_outs.append(o[0])
    dec_out = torch.stack(dec_outs, dim=-1)  # (dim, T)

    torch.testing.assert_close(pre_out, dec_out, atol=1e-4, rtol=1e-4)
    # state after prefill == state after rollout
    torch.testing.assert_close(pre_state, dec_state, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _cute_kernel_supported(), reason="needs sm90/sm100/sm120 CUDA")
@pytest.mark.parametrize("K", [3, 4, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cute_decode_matches_reference(K, dtype):
    prefill_k, decode_k = _load_cute_kernel()
    torch.manual_seed(2)
    dim, B, slots = 64, 6, 10
    dev = "cuda"
    # Gated kernel: take packed proj = [B_gate | C_gate | x], emit C_gate * conv(B_gate*x).
    B_gate = torch.randn(B, dim, device=dev, dtype=dtype)
    C_gate = torch.randn(B, dim, device=dev, dtype=dtype)
    x = torch.randn(B, dim, device=dev, dtype=dtype)
    proj = torch.cat([B_gate, C_gate, x], dim=-1).contiguous()  # native: [B | C | x]
    Bx = B_gate * x  # what the reference conv consumes
    state = torch.randn(slots, dim, K - 1, device=dev, dtype=dtype)
    w = torch.randn(dim, K, device=dev, dtype=dtype)
    bias = torch.randn(dim, device=dev, dtype=dtype)
    idx = torch.randperm(slots, device=dev)[:B].to(torch.int32)

    ref_conv = ref_conv_decode(Bx, state.clone(), w, bias, idx)
    ref = C_gate * ref_conv  # external output gate
    got = decode_k(proj, state.clone(), w, bias, idx)  # fused inside the kernel
    atol = 8e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=2e-2)


@pytest.mark.skipif(not _cute_kernel_supported(), reason="needs sm90/sm100/sm120 CUDA")
@pytest.mark.parametrize("K", [3, 4, 7])
@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cute_prefill_matches_reference(K, with_init, dtype):
    prefill_k, decode_k = _load_cute_kernel()
    torch.manual_seed(3)
    dim, slots = 48, 8
    dev = "cuda"
    seqlens = [5, 1, 13, 8]  # varlen batch
    B = len(seqlens)
    qsl = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], device=dev, dtype=torch.int32
    )
    T = qsl[-1].item()
    # Gated kernel: token-major proj = [B_gate | C_gate | x], emits C_gate*conv(B_gate*x).
    B_gate = torch.randn(T, dim, device=dev, dtype=dtype)
    C_gate = torch.randn(T, dim, device=dev, dtype=dtype)
    x = torch.randn(T, dim, device=dev, dtype=dtype)
    proj = torch.cat(
        [B_gate, C_gate, x], dim=-1
    ).contiguous()  # native: [B | C | x]  # (T, 3*dim)
    Bx_t = (B_gate * x).t().contiguous()  # (dim, T) for the reference
    w = torch.randn(dim, K, device=dev, dtype=dtype)
    bias = torch.randn(dim, device=dev, dtype=dtype)
    state = torch.randn(slots, dim, K - 1, device=dev, dtype=dtype)
    idx = torch.randperm(slots, device=dev)[:B].to(torch.int32)
    hi = torch.tensor([with_init] * B, device=dev)

    ref_state = state.clone()
    ref_conv = ref_conv_prefill(Bx_t, ref_state, w, bias, qsl, idx, hi)  # (dim, T)
    ref = C_gate * ref_conv.t()  # (T, dim) with output gate applied
    got_state = state.clone()
    got = prefill_k(proj, got_state, w, bias, qsl, idx, hi)  # (T, dim) gated
    atol = 8e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=2e-2)
    torch.testing.assert_close(
        got_state.float(), ref_state.float(), atol=atol, rtol=1e-2
    )


@pytest.mark.skipif(not _cute_kernel_supported(), reason="needs sm90/sm100/sm120 CUDA")
def test_cute_prefill_across_chunk_boundary_with_initial_state():
    """A request spanning several 256-token chunks must seed its first chunk from the
    prior conv state and leave its last K-1 inputs as the new state."""
    prefill_k, _ = _load_cute_kernel()
    torch.manual_seed(4)
    dim, slots, K = 256, 4, 7
    dev = "cuda"
    dtype = torch.float32
    seqlens = [600, 3]
    qsl = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], device=dev, dtype=torch.int32
    )
    T = qsl[-1].item()
    B_gate, C_gate, x = (torch.randn(T, dim, device=dev, dtype=dtype) for _ in range(3))
    proj = torch.cat([B_gate, C_gate, x], dim=-1).contiguous()
    Bx_t = (B_gate * x).t().contiguous()
    w = torch.randn(dim, K, device=dev, dtype=dtype)
    state = torch.randn(slots, dim, K - 1, device=dev, dtype=dtype)
    idx = torch.tensor([2, 0], device=dev, dtype=torch.int32)
    hi = torch.tensor([True, True], device=dev)

    ref_state = state.clone()
    ref = C_gate * ref_conv_prefill(Bx_t, ref_state, w, None, qsl, idx, hi).t()
    got_state = state.clone()
    got = prefill_k(proj, got_state, w, None, qsl, idx, hi)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got_state, ref_state, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
