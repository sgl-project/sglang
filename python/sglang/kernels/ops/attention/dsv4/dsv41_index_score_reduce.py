"""Fused indexer score reduction for the DSv4.1 low-ratio indexer -- fusion step 2.

One launch that turns the raw per-head logits ``[t, H, n]`` into the masked
per-token scores ``[t, n]`` the top-k consumes: ReLU, the per-head weight
multiply, the reduction over H, the widening to fp32, and the visible-length
mask, all in the Unified Buffer.

Why the einsum stays in torch
-----------------------------
The step this fuses is the tail of ``DeepseekV41Indexer.scores``::

    s = torch.einsum("bhd,nd->bhn", q, k)          # [t, H, n] bf16
    s = (s.relu() * weights.unsqueeze(-1)).sum(dim=1)
    return s.float()                               # [t, n] fp32

The einsum is deliberately NOT part of this kernel, and folding it in later
would be a regression, not an improvement. triton-ascend only reaches the
Vector unit on this backend: ``tl.dot`` was measured at 686 GFLOP/s -- the
Vector fp32 rate -- and never dispatched to Cube. This einsum is 128 MACs per
output element against a 2-byte store, so on Vector it costs roughly 190x what
writing the output costs; handing it to the aclnn BMM that ``torch.einsum``
already dispatches to is the only way it runs on Cube. None of the eleven
Ascend triton ops in this tree uses ``tl.dot``, and this one must not either.

What is left after the einsum is pure bandwidth, and it is the expensive half
in launches and traffic. Eager walks the full ``[t, H, n]`` tensor three times
(``relu``, ``* weights``, ``sum``) plus two more full-width ``[t, n]`` passes
(``.float()``, ``masked_fill``). At ``H = 64`` and ``n = 524288`` a single
``t = 8`` chunk is 537 MB per full-width pass; the fused kernel reads it once
and writes the 16 MB fp32 result: 5 launches -> 1, ~1.6 GB -> ~553 MB.

The mask is fused because it is the caller's next line, not a separate concern:
both NPU arch paths follow ``scores()`` with ``j >= lens`` -> ``-inf`` (A3 in
``_low_ratio_index_topk_torch_a3``, A5 inside ``topk_from_scores``), so it is
another full-width read-modify-write of the same tensor that just left the UB.

Numerical contract -- where bf16 rounds
---------------------------------------
The eager chain passes through bf16 at three points; two of them round, and
this kernel reproduces both, in order:

1. ``s.relu()`` is bf16 -- but ReLU never rounds, so widening the loaded bf16
   to fp32 first is bit-identical.
2. ``* weights`` is a bf16 x bf16 multiply whose product **lands back in
   bf16**. The kernel takes the product in fp32 and rounds it to bf16 once,
   which is the same correctly-rounded result.
3. ``.sum(dim=1)`` reduces those bf16 products and produces a **bf16** result;
   ``.float()`` only widens it afterwards. So there is a rounding to bf16
   after the reduction, before the fp32 output -- dropping it would be a
   different (more accurate, and wrong) function.

The accumulator inside point 3 was measured, not assumed. On torch 2.13,
``x.sum(1)`` for a bf16 ``x`` is bit-identical to ``x.float().sum(1).bfloat16()``
over every shape and magnitude tried, and is NOT the elementwise bf16 running
sum (which differs by up to 32.0 at H=64). torch accumulates a bf16 reduction
in fp32 and rounds once at the end; the kernel does exactly that -- an fp32
``tl.sum`` over the H lane, then one ``.to(bfloat16)``, then ``.to(float32)``.

The only freedom left is the *order* of the fp32 accumulation, which torch does
not pin either. It is invisible here: over 250k outputs at H=64, spanning six
magnitude scales, random head permutations, three block shapes, and an
adversarial 60-binade exponent spread, the blocked order and torch's order gave
zero differing bf16 results -- 64 products of 8-bit significands sum exactly in
fp32's 24. ``mirror_index_score_reduce`` reproduces the kernel's blocked order
so a host without a card can hold that to bit-equality; see
``test/manual/dsv41_indexer_step2/``.

Do not fold the weights into q
------------------------------
``weights_proj`` is a bias-free ``ReplicatedLinear`` with no activation, so a
head weight can be negative, and ``relu(s) * w != relu(s * w)`` there (measured
gap 26.5 on a random bf16 case). Scaling q by w before the einsum is therefore
not an algebraic identity; the ReLU has to see the unweighted logit.

triton-ascend contracts honoured here (each one learned on card; see
``dsv41_index_k_dequant.py`` and ``low_ratio_compress.py``):

* jit code resolves names from module globals only -- the kernel is defined at
  module level under the import guard and is self-contained, with no jit-to-jit
  helper calls.
* the grid is one dimensional; the ``(token, n-block)`` pair is recovered by
  div/mod inside the kernel.
* no ``tl.cumsum``, no ``tl.dot``, no loop-carried scalars.
* the token index is widened to int64 before it scales a stride: at
  ``n = 524288`` and ``H = 64`` the ``[t, H, n]`` element count passes 2^31
  around ``t = 64``, and an int32 offset would wrap.
* ``BLOCK_N`` is picked to fit the Unified Buffer by the same arithmetic
  ``low_ratio_compress._choose_block_n`` uses (180 KB on NPU -> 64 at
  ``H = 64``). It is duplicated rather than imported so this file stays a
  standalone copy for op-bench to sync verbatim; keep the two in step.
* an empty batch is short-circuited in the wrapper. ``t = 0`` (or ``n = 0``)
  makes the grid ``(0,)``, which on A5 is a launch error (``coreDim is
  invalid``, EE1003) rather than a no-op.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORTABLE = True
except Exception:  # no triton: the module must still import for the gate
    _TRITON_IMPORTABLE = False


# Mirrors low_ratio_compress._NPU_UB_BUDGET / _GPU_UB_BUDGET.
_NPU_UB_BUDGET = 180_000
_GPU_UB_BUDGET = 1_000_000


def index_score_reduce_available() -> bool:
    """True only if the jit kernel below can actually be launched.

    ``import triton`` succeeding is not the same question. ``sglang/__init__``
    installs a triton stub (``_platform_stubs.py``) on hosts without triton so
    that kernel modules stay importable; under it ``@triton.jit`` is a
    pass-through decorator, so the "kernel" is a plain function and the launch
    ``_kernel[grid](...)`` fails with ``'function' object is not subscriptable``
    rather than falling back. Ask the decorated object whether it is
    subscriptable -- that is the property the launch needs.
    """
    return _TRITON_LAUNCHABLE


def _choose_block_n(h: int, device: torch.device, requested: int = 64) -> int:
    """Largest power-of-two BLOCK_N whose ``[BLOCK_H, BLOCK_N]`` tile group fits.

    Same arithmetic as ``low_ratio_compress._choose_block_n``, duplicated to
    keep this file standalone (see the module docstring), with the head lane
    standing in for the feature lane: the tiles here are ``[BLOCK_H, BLOCK_N]``,
    not ``[BLOCK_N, BLOCK_D]``. ``num_buffers`` is the conservative live-tile
    count that file settled on; erring small is the safe direction -- a UB
    overflow is a compile refusal, not a wrong answer.
    """
    block_h = triton.next_power_of_2(h)
    budget = _NPU_UB_BUDGET if device.type == "npu" else _GPU_UB_BUDGET
    num_buffers = 8
    max_bn = budget // (num_buffers * block_h * 4)
    bn = 1
    while bn * 2 <= min(requested, max_bn):
        bn *= 2
    return max(1, bn)


if _TRITON_IMPORTABLE:

    @triton.jit(do_not_specialize=["N"])
    def _dsv41_index_score_reduce_kernel(
        s_ptr,  # [T, H, N] bf16 raw per-head logits (last stride 1)
        w_ptr,  # [T, H] bf16 per-head weights, contiguous
        lens_ptr,  # [T] int32 visible length (unread when HAS_LENS is False)
        out_ptr,  # [T, N] fp32 out, contiguous
        N,
        stride_st,  # element stride of the token axis of s
        stride_sh,  # element stride of the head axis of s
        NUM_N_BLOCKS,
        H: tl.constexpr,
        HAS_LENS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        # One dimensional grid: the (token, n-block) pair is recovered here.
        t = pid // NUM_N_BLOCKS
        nb = pid % NUM_N_BLOCKS
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_n = offs_n < N
        offs_h = tl.arange(0, BLOCK_H)
        valid_h = offs_h < H

        # int64 before the stride multiply -- [T, H, N] passes 2^31 elements at
        # production n, and an int32 token offset would wrap silently.
        t64 = t.to(tl.int64)
        s = tl.load(
            s_ptr + t64 * stride_st + offs_h[:, None] * stride_sh + offs_n[None, :],
            mask=valid_h[:, None] & valid_n[None, :],
            other=0.0,
        )
        w = tl.load(w_ptr + t64 * H + offs_h, mask=valid_h, other=0.0)

        # ReLU never rounds, so taking it on the widened value is bit-identical
        # to the bf16 `s.relu()`; the product is the one that has to land back
        # in bf16, exactly once, as `s.relu() * weights` does.
        p = tl.maximum(s.to(tl.float32), 0.0) * w.to(tl.float32)[:, None]
        p = p.to(tl.bfloat16).to(tl.float32)
        # Masked-off head lanes contributed an exact 0 above, so they do not
        # perturb the fp32 accumulation.
        acc = tl.sum(p, axis=0)
        # `.sum(dim=1)` yields bf16 and `.float()` only widens it: the fp32
        # accumulator rounds to bf16 once here, before the fp32 store.
        out = acc.to(tl.bfloat16).to(tl.float32)

        if HAS_LENS:
            # The caller's next line, folded in: positions at or past the
            # visible compressed length are unreachable.
            ln = tl.load(lens_ptr + t).to(tl.int32)
            out = tl.where(offs_n < ln, out, float("-inf"))

        tl.store(out_ptr + t64 * N + offs_n, out, mask=valid_n)

    # See index_score_reduce_available: the stub's @triton.jit hands back the
    # plain function, which has no __getitem__ and so cannot be launched.
    _TRITON_LAUNCHABLE = hasattr(_dsv41_index_score_reduce_kernel, "__getitem__")
else:
    _TRITON_LAUNCHABLE = False


def _validated(
    s_raw: torch.Tensor, weights: torch.Tensor, lens: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int, int]:
    """Shape/dtype check plus the row-contiguity the pointer arithmetic assumes."""
    if s_raw.ndim != 3:
        raise ValueError(f"s_raw must be [t, H, n], got {tuple(s_raw.shape)}")
    t, h, n = s_raw.shape
    if tuple(weights.shape) != (t, h):
        raise ValueError(
            f"weights must be [{t}, {h}] to match s_raw, got {tuple(weights.shape)}"
        )
    if lens is not None and tuple(lens.shape) != (t,):
        raise ValueError(f"lens must be [{t}], got {tuple(lens.shape)}")
    # The kernel indexes the n lane with a unit stride; the token and head
    # lanes are free, so an einsum output that is already row-contiguous costs
    # no copy here whatever its outer layout.
    if n > 0 and s_raw.stride(-1) != 1:
        s_raw = s_raw.contiguous()
    weights = weights.contiguous()
    if lens is not None:
        lens = lens.to(torch.int32).contiguous()
    return s_raw, weights, lens, t, h, n


def index_score_reduce(
    s_raw: torch.Tensor,
    weights: torch.Tensor,
    lens: Optional[torch.Tensor] = None,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """Masked per-token indexer scores ``[t, n]`` fp32, in one launch.

    ``s_raw`` is the raw ``torch.einsum("bhd,nd->bhn", q, k)`` output, bf16
    ``[t, H, n]`` -- the einsum stays in torch on purpose (module docstring).
    ``weights`` is the bf16 ``[t, H]`` head weighting.

    ``lens`` decides whether the visible-length mask is applied at all:

    * a ``[t]`` integer tensor -- positions ``>= lens[row]`` are set to
      ``-inf``, bit-identical to the callers' ``s.masked_fill(j[None, :] >=
      lens[:, None], -inf)``.
    * ``None`` -- no mask, bit-identical to a bare
      ``(s.relu() * weights.unsqueeze(-1)).sum(dim=1).float()``. Reserved
      capability: both DSv4.1 NPU paths mask today, but the reduction is
      meaningful without one and should not need a sentinel ``lens`` to run.
    """
    s_raw, weights, lens, t, h, n = _validated(s_raw, weights, lens)
    out = torch.empty((t, n), dtype=torch.float32, device=s_raw.device)
    # Ahead of the triton check on purpose: an idle step must return the same
    # shape and dtype as a busy one, and a (0,) grid is a launch error on A5
    # (coreDim is invalid, EE1003), not a no-op. Validation is host-side torch,
    # so a bad shape is a ValueError with or without triton.
    if t == 0 or n == 0:
        return out
    if not _TRITON_LAUNCHABLE:
        raise RuntimeError("triton not available (or only the sglang stub is)")

    block_h = triton.next_power_of_2(h)
    if block_n is None:
        block_n = _choose_block_n(h, s_raw.device)
    num_n_blocks = triton.cdiv(n, block_n)
    _dsv41_index_score_reduce_kernel[(t * num_n_blocks,)](
        s_raw,
        weights,
        # The pointer is unread when HAS_LENS is False, but triton still needs
        # a tensor argument; reuse the weights rather than allocating a dummy.
        lens if lens is not None else weights,
        out,
        n,
        s_raw.stride(0),
        s_raw.stride(1),
        num_n_blocks,
        H=h,
        HAS_LENS=lens is not None,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
    )
    return out


def mirror_index_score_reduce(
    s_raw: torch.Tensor,
    weights: torch.Tensor,
    lens: Optional[torch.Tensor] = None,
    block_n: int = 64,
    block_h: Optional[int] = None,
) -> torch.Tensor:
    """Pure-torch replica of the KERNEL's arithmetic, for off-device validation.

    Deliberately not the reference implementation: it reduces in the kernel's
    tile order (fp32 accumulation per ``[block_h, block_n]`` tile, one bf16
    rounding of the product inside the tile and one of the total after it) so a
    host without a card can prove the kernel's math equals the eager chain's.
    ``block_h`` defaults to the whole head lane, which is what the kernel does
    at ``H = 64``; passing a smaller one exercises the accumulation order the
    kernel would use if H ever outgrew a single tile. Change it in the same
    commit as the kernel body.
    """
    if s_raw.ndim != 3:
        raise ValueError(f"s_raw must be [t, H, n], got {tuple(s_raw.shape)}")
    t, h, n = s_raw.shape
    out = torch.empty((t, n), dtype=torch.float32, device=s_raw.device)
    if t == 0 or n == 0:
        return out
    if block_h is None:
        block_h = h
    for ti in range(t):
        for n0 in range(0, n, block_n):
            cols = slice(n0, min(n0 + block_n, n))
            acc = torch.zeros(
                cols.stop - cols.start, dtype=torch.float32, device=s_raw.device
            )
            for h0 in range(0, h, block_h):
                sv = s_raw[ti, h0 : h0 + block_h, cols].to(torch.float32)
                wv = weights[ti, h0 : h0 + block_h].to(torch.float32)
                p = (sv.clamp(min=0.0) * wv.unsqueeze(-1)).to(torch.bfloat16)
                acc = acc + p.to(torch.float32).sum(0)
            out[ti, cols] = acc.to(torch.bfloat16).to(torch.float32)
    if lens is not None:
        j = torch.arange(n, device=s_raw.device)
        out = out.masked_fill(j[None, :] >= lens.to(torch.int64)[:, None], -torch.inf)
    return out
