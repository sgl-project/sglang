"""Width-generic gated causal conv1d (CuTe DSL).

Fuses a depthwise causal conv1d with its B/C gates: the input is packed
`[B_gate | C_gate | x]`, so the kernel forms `Bx = B_gate * x`, runs the depthwise
conv over the K-tap causal window, and multiplies the result by `C_gate` -- no
transpose or permutation. Width-generic in K: the in-tree sgl_kernel C++
causal_conv1d (widths 2-4) and the shared Triton kernel (widths 2-5) don't cover
wider taps such as K=7 (e.g. LFM3's conv_L_cache=7), which this kernel serves. The
conv is channel-parallel and memory-bound, so this is a SIMT kernel (no MMA/TMA): one
thread owns one channel, and a warp's threads are adjacent channels, so each per-token
read `proj[t, d..d+31]` is one coalesced transaction. Each thread keeps the K-1 causal
window for its channel in registers (no shared memory, no barrier).

Layout is token-major (T, dim) with `dim` contiguous -- sglang's native hidden-state
layout, so no transpose is needed. Streamed tensors use the same descriptor idiom as
the in-tree cutedsl_gdn / cutedsl_kda kernels:
`from_dlpack(t.detach(), assumed_align=32).mark_layout_dynamic(leading_dim=...)`. The
conv-state cache is left static so its slot offset folds to a compile-time stride and
the int64 slot keeps the offset 64-bit (the cache can exceed 2**31 elements).

Math is done in fp32 -- the operands are bf16/fp16, and fp32 accumulation matches the
torch reference, whose `F.conv1d` / `.sum` reduce bf16 inputs in fp32 -- and the
result is cast back to the tensor dtype.

  decode : (B, dim) one token/req + (slots, dim, K-1) state -> (B, dim), state updated
  prefill: (T, dim) varlen token-major + state + has_initial_state -> (T, dim)
Validated against the PyTorch reference in tests/test_cutedsl_gated_conv.py.
"""

from __future__ import annotations

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)

# Single compile cache for both ops, mirroring cutedsl_gdn / cutedsl_kda's
# `_compiled_kernels`. `_compiled` is the one place that calls `cute.compile`.
_compiled_kernels: dict = {}


def _compiled(key, host_fn, *compile_args):
    kernel = _compiled_kernels.get(key)
    if kernel is None:
        kernel = cute.compile(host_fn, *compile_args, options="--opt-level 3")
        _compiled_kernels[key] = kernel
        logger.info(f"CuTe DSL gated conv kernel compiled: {key}")
    return kernel


def _zero_bias(dim: int, device, dtype) -> torch.Tensor:
    # Checkpoints with conv_bias=False (e.g. LFM3) give the wrappers bias=None; they
    # pass this cached zero tensor (allocated once per (dim, device, dtype)) instead --
    # the kernel keeps an unconditional `+ bias` so it stays branch-free, and a
    # conv_bias=True checkpoint still works by passing a real bias.
    key = (dim, str(device), dtype)
    b = _zero_bias_cache.get(key)
    if b is None:
        b = torch.zeros(dim, device=device, dtype=dtype)
        _zero_bias_cache[key] = b
    return b


_zero_bias_cache: dict = {}


@cute.kernel
def _decode_kernel(
    mProj: cute.Tensor,  # (B, 3*dim)        packed in_proj output [B_gate | C_gate | x]
    mW: cute.Tensor,  # (dim, K)          conv taps, static layout
    mBias: cute.Tensor,  # (dim,)            conv bias (zeros when conv_bias=False)
    mOut: cute.Tensor,  # (B, dim)          C_gate-gated conv output
    mState: cute.Tensor,  # (slots, dim, K-1) rolling Bx window, static layout, mutated
    mSlot: cute.Tensor,  # (B,) int64        per-request state-slot index
    dim: cutlass.Int32,
    SLOTS: cutlass.Constexpr,
    K: cutlass.Constexpr,
    BLOCK: cutlass.Constexpr,
):
    """One thread per channel for a one-token-per-request decode step.

    For its (slot, channel) the thread loads the K-1 cached Bx values and the K conv
    taps into fp32 registers, forms `Bx = B_gate * x` for the new token, computes the
    depthwise dot (+bias), multiplies by `C_gate`, writes the output, and rolls the
    state forward to `[old[1:], Bx]`. mProj columns are [0:dim)=B_gate, [dim:2dim)=C_gate,
    [2dim:3dim)=x. Slots that are padded/out-of-range (<0 or >=SLOTS) are skipped.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bx, bb, _ = cute.arch.block_idx()
    d = bx * BLOCK + tidx  # channel
    cute.arch.griddepcontrol_wait()
    slot = mSlot[bb]
    if d < dim and slot >= 0 and slot < SLOTS:
        mss = mState[slot, None, None]  # (dim, K-1) at the 64-bit slot offset
        wv = cute.make_rmem_tensor(K, cutlass.Float32)
        for k in cutlass.range_constexpr(K):
            wv[k] = mW[d, k].to(cutlass.Float32)
        win = cute.make_rmem_tensor(K - 1, cutlass.Float32)
        for j in cutlass.range_constexpr(K - 1):
            win[j] = mss[d, j].to(cutlass.Float32)
        bxv = mProj[bb, d].to(cutlass.Float32) * mProj[bb, 2 * dim + d].to(
            cutlass.Float32
        )
        cgate = mProj[bb, dim + d].to(cutlass.Float32)
        acc = mBias[d].to(cutlass.Float32)
        for j in cutlass.range_constexpr(K - 1):
            acc = acc + win[j] * wv[j]
        acc = acc + bxv * wv[K - 1]
        mOut[bb, d] = (acc * cgate).to(mOut.element_type)
        for j in cutlass.range_constexpr(K - 2):
            mss[d, j] = win[j + 1].to(mState.element_type)
        mss[d, K - 2] = bxv.to(mState.element_type)
    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _decode_host(
    mProj,
    mW,
    mBias,
    mOut,
    mState,
    mSlot,
    dim: cutlass.Int32,
    B: cutlass.Int32,
    SLOTS: cutlass.Constexpr,
    K: cutlass.Constexpr,
    BLOCK: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    grid_d = (dim + BLOCK - 1) // BLOCK
    _decode_kernel(mProj, mW, mBias, mOut, mState, mSlot, dim, SLOTS, K, BLOCK).launch(
        grid=[grid_d, B, 1],
        block=[BLOCK, 1, 1],
        stream=stream,
        use_pdl=True,
    )


def gated_causal_conv1d_decode(proj, conv_state, weight, bias, slot_idx, block=128):
    """C-gated depthwise conv decode (one token per request).

    Args:
        proj: (B, 3*dim) packed in_proj output [B_gate | C_gate | x].
        conv_state: (slots, dim, K-1) per-request rolling Bx window, mutated in place.
        weight: (dim, K) conv taps.
        bias: (dim,) conv bias, or None (conv_bias=False checkpoints).
        slot_idx: (B,) state-slot index per request.
        block: threads per block (channels per block).

    Returns:
        (B, dim) the C_gate-gated conv output. Bx=B_gate*x and the C_gate output gate
        are fused into the kernel.
    """
    B = proj.shape[0]
    dim = weight.shape[0]
    K = weight.shape[-1]
    slots = conv_state.shape[0]
    out = proj.new_empty(B, dim)
    bias_t = bias if bias is not None else _zero_bias(dim, proj.device, proj.dtype)
    slot64 = slot_idx.detach().to(torch.int64)
    mProj = from_dlpack(proj.detach(), assumed_align=32).mark_layout_dynamic(
        leading_dim=1
    )
    mW = from_dlpack(weight.detach())
    mB = from_dlpack(bias_t.detach(), assumed_align=32).mark_layout_dynamic(
        leading_dim=0
    )
    mO = from_dlpack(out.detach(), assumed_align=32).mark_layout_dynamic(leading_dim=1)
    mS = from_dlpack(conv_state.detach())
    mSlot = from_dlpack(slot64, assumed_align=32).mark_layout_dynamic(leading_dim=0)
    # Launch on torch's current stream (correct ordering + cuda-graph capture).
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = ("decode", dim, K, slots, block, proj.dtype, conv_state.dtype)
    _compiled(
        key, _decode_host, mProj, mW, mB, mO, mS, mSlot, dim, B, slots, K, block, stream
    )(mProj, mW, mB, mO, mS, mSlot, dim, B, stream)
    return out


@cute.kernel
def _prefill_kernel(
    mProj: cute.Tensor,  # (T_total, 3*dim)  packed in_proj output [B_gate | C_gate | x]
    mW: cute.Tensor,  # (dim, K)          conv taps
    mBias: cute.Tensor,  # (dim,)            conv bias (zeros when conv_bias=False)
    mOut: cute.Tensor,  # (T_total, dim)    C_gate-gated conv output
    mState: cute.Tensor,  # (slots, dim, K-1) static, mutated
    mInit: cute.Tensor,  # (B, dim, K-1)     pre-launch snapshot of each request's state
    mQsl: cute.Tensor,  # (B+1,) int32      cumulative per-request token offsets
    mSlot: cute.Tensor,  # (B,) int64        per-request state-slot index
    mHasInit: cute.Tensor,  # (B,) int          1 if conv_state holds a real prefix
    dim: cutlass.Int32,
    SLOTS: cutlass.Constexpr,
    K: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    BLOCK_D: cutlass.Constexpr,
):
    """One thread per channel, gridded over (d-tile, token-chunk, request) for varlen
    prefill.

    Each thread streams CHUNK tokens of one request for its channel, keeping the K-1
    causal window of `Bx = B_gate * x` in registers; per-token reads coalesce across
    the warp (dim contiguous). The window is seeded from conv_state when has_initial
    (chunked prefill / carried prefix) else zero, and the chunk holding the request's
    last token writes the final window back as the new state. mProj columns are
    [0:dim)=B_gate, [dim:2dim)=C_gate, [2dim:3dim)=x. Out-of-range slots are skipped.
    """
    tidx, _, _ = cute.arch.thread_idx()  # channel within the d-tile
    bdx, bc, bb = cute.arch.block_idx()  # d-tile, token-chunk, request
    d = bdx * BLOCK_D + tidx
    cute.arch.griddepcontrol_wait()
    s = mQsl[bb]
    e = mQsl[bb + 1]
    treq = e - s
    c0 = bc * CHUNK  # first token of this chunk within the request
    slot = mSlot[bb]
    if d < dim and c0 < treq and slot >= 0 and slot < SLOTS:
        mss = mState[slot, None, None]  # (dim, K-1) at int64 slot offset
        minit = mInit[bb, None, None]
        has_init = mHasInit[bb]
        wv = cute.make_rmem_tensor(K, cutlass.Float32)
        for k in cutlass.range_constexpr(K):
            wv[k] = mW[d, k].to(cutlass.Float32)
        bias = mBias[d].to(cutlass.Float32)
        win = cute.make_rmem_tensor(K - 1, cutlass.Float32)
        for j in cutlass.range_constexpr(K - 1):
            r = c0 - (K - 1) + j  # req-token index for window slot j (oldest first)
            hv = cutlass.Float32(0.0)
            if r >= 0:
                hv = mProj[s + r, d].to(cutlass.Float32) * mProj[s + r, 2 * dim + d].to(
                    cutlass.Float32
                )  # B_gate*x
            elif has_init != 0:
                hv = minit[d, (K - 1) + r].to(
                    cutlass.Float32
                )  # before request -> initial state
            win[j] = hv
        cend = c0 + CHUNK
        if cend > treq:
            cend = treq
        t = c0
        while t < cend:
            bxv = mProj[s + t, d].to(cutlass.Float32) * mProj[s + t, 2 * dim + d].to(
                cutlass.Float32
            )  # B_gate*x
            cgate = mProj[s + t, dim + d].to(cutlass.Float32)
            acc = bias
            for j in cutlass.range_constexpr(K - 1):
                acc = acc + win[j] * wv[j]
            acc = acc + bxv * wv[K - 1]
            mOut[s + t, d] = (acc * cgate).to(mOut.element_type)
            for j in cutlass.range_constexpr(K - 2):
                win[j] = win[j + 1]
            win[K - 2] = bxv
            t = t + 1
        if cend == treq:
            for j in cutlass.range_constexpr(K - 1):
                mss[d, j] = win[j].to(mState.element_type)
    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _prefill_host(
    mProj,
    mW,
    mBias,
    mOut,
    mState,
    mInit,
    mQsl,
    mSlot,
    mHasInit,
    dim: cutlass.Int32,
    n_chunks: cutlass.Int32,
    B: cutlass.Int32,
    SLOTS: cutlass.Constexpr,
    K: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    BLOCK_D: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    grid_d = (dim + BLOCK_D - 1) // BLOCK_D
    _prefill_kernel(
        mProj,
        mW,
        mBias,
        mOut,
        mState,
        mInit,
        mQsl,
        mSlot,
        mHasInit,
        dim,
        SLOTS,
        K,
        CHUNK,
        BLOCK_D,
    ).launch(
        grid=[grid_d, n_chunks, B], block=[BLOCK_D, 1, 1], stream=stream, use_pdl=True
    )


def gated_causal_conv1d_prefill(
    proj,
    conv_state,
    weight,
    bias,
    query_start_loc,
    slot_idx,
    has_initial_state,
    chunk=256,
    block_d=128,
):
    """C-gated depthwise conv prefill (varlen, token-major).

    Args:
        proj: (T_total, 3*dim) packed in_proj output [B_gate | C_gate | x].
        conv_state: (slots, dim, K-1) per-request rolling Bx window, mutated in place.
        weight: (dim, K) conv taps.
        bias: (dim,) conv bias, or None (conv_bias=False checkpoints).
        query_start_loc: (B+1,) cumulative token offsets [0, t0, t0+t1, ...].
        slot_idx: (B,) state-slot index per request.
        has_initial_state: (B,) 1 if conv_state holds a real prefix (chunked prefill).
        chunk: tokens streamed per block along the token axis.
        block_d: threads per block (channels per d-tile).

    Returns:
        (T_total, dim) the C_gate-gated conv output. Bx=B_gate*x and the C_gate output
        gate are fused into the kernel.
    """
    T = proj.shape[0]
    dim = weight.shape[0]
    K = weight.shape[-1]
    B = query_start_loc.shape[0] - 1
    slots = conv_state.shape[0]
    out = proj.new_empty(T, dim)
    bias_t = bias if bias is not None else _zero_bias(dim, proj.device, proj.dtype)
    n_chunks = (
        T + chunk - 1
    ) // chunk  # upper bound (any single req <= T); idle chunks early-exit
    qsl32 = query_start_loc.detach().to(torch.int32)
    slot64 = slot_idx.detach().to(torch.int64)
    hi32 = has_initial_state.detach().to(torch.int32)
    # The last chunk's CTA writes the final state while chunk 0 of the same request
    # reads the initial one, with no inter-CTA ordering; read from a snapshot.
    init_state = conv_state.detach()[slot64.clamp(0, slots - 1)]
    mProj = from_dlpack(proj.detach(), assumed_align=32).mark_layout_dynamic(
        leading_dim=1
    )
    mW = from_dlpack(weight.detach(), assumed_align=32).mark_layout_dynamic(
        leading_dim=1
    )
    mB = from_dlpack(bias_t.detach(), assumed_align=32).mark_layout_dynamic(
        leading_dim=0
    )
    mO = from_dlpack(out.detach(), assumed_align=32).mark_layout_dynamic(leading_dim=1)
    mS = from_dlpack(conv_state.detach())
    mInit = from_dlpack(init_state, assumed_align=32).mark_layout_dynamic(leading_dim=2)
    mQ = from_dlpack(qsl32, assumed_align=32).mark_layout_dynamic(leading_dim=0)
    mSlot = from_dlpack(slot64, assumed_align=32).mark_layout_dynamic(leading_dim=0)
    mHI = from_dlpack(hi32, assumed_align=32).mark_layout_dynamic(leading_dim=0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = ("prefill", dim, K, slots, chunk, block_d, proj.dtype, conv_state.dtype)
    _compiled(
        key,
        _prefill_host,
        mProj,
        mW,
        mB,
        mO,
        mS,
        mInit,
        mQ,
        mSlot,
        mHI,
        dim,
        n_chunks,
        B,
        slots,
        K,
        chunk,
        block_d,
        stream,
    )(mProj, mW, mB, mO, mS, mInit, mQ, mSlot, mHI, dim, n_chunks, B, stream)
    return out
