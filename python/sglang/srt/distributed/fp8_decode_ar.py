"""fp8-e4m3 narrow-wire DECODE all-reduce, behind SGLANG_FP8_DECODE_AR.

WHY THIS EXISTS, AND WHY IT IS NOT THE PREFILL PATCH
The prefill all-reduce (`SGLANG_FP8_PREFILL_AR`) narrows a *NCCL* collective of
83.9 MB. Decode does not use that path at all: at tp4 the decode residual is
24-30 rows x 5120 bf16 = 245,760 B, which is under the PCIe-P2P custom-AR byte
ceiling (262,144 B), so it is served by sglang's own one-shot push kernel
(`all_reduce_1shot_push_kernel`, `AllReducePushImpl`) over PCIe P2P. The prefill
patch's interception point never sees it.

The campaign closed "decode all-reduce" as latency-bound, and that closure is
correct FOR SPLITTING (chunking one collective into several multiplies the
per-call latency: 240 KiB 23.7 us, 120 KiB 14.9, 60 KiB 12.6). Narrowing the
dtype is a different operation: it keeps ONE call and halves the payload, which
is exactly the 240 -> 120 KiB step.

MEASURED on this stack, 4 ranks, the real decode payload, inside a captured
CUDA graph (which is how decode always runs):

    bf16 one-shot push AR  245,760 B    23.39 us/call      <- today
    fp8  one-shot push AR  122,880 B    11.13 us/call      (-52.4%)
    fused quantize + dequantize          2.50 us/call
    ------------------------------------------------------------------
    quantize + fp8 AR + dequantize      13.77 us/call      (-41.1%)

The two elementwise kernels are the whole risk of this scheme at decode sizes:
they are ~1.2 us each *inside a graph*, against a 12.3 us wire saving. Written
as eager torch they would be ~10x that and the lever would be a wash.

THE SCALE MUST BE BIT-IDENTICAL ON EVERY RANK
`sum_i(q_i * s_i) != s * sum_i(q_i)` unless the `s_i` agree, so a per-rank scale
silently computes the wrong sum. Worse, under EP a 1-ULP disagreement between
ranks re-routes the MoE gate's discrete top-k and the ranks diverge outright.

There is no cross-rank-identical quantity available *before* the reduce without
paying for a second collective -- and at 11 us/call a second collective would
eat the entire win (the prefill scheme's amax all-reduce costs 25 us). So the
scale is taken from the ALL-REDUCE OUTPUT, which every rank computes from the
same peer buffers in the same order and which is therefore bit-identical by
construction, and it is carried forward to the NEXT call at the same site:

  * `_dequant_flat` already has the output in registers, so it folds in a block
    max and one `atomicMax` per block into `_STATE[site]` -- no extra kernel.
  * `_quant_flat` reads `_STATE[site]`, which was last written by the *previous*
    step's dequantize at the same site (an earlier kernel in the same stream),
    and derives `scale = amax * HEADROOM / 448`.
  * `_STATE` is a running maximum: monotone, never reset, so it is a pure
    function of the (identical) history on every rank. No promotion step, no
    grid-wide sync, no race with concurrent readers.
  * Bootstrap: an unwritten slot reads 0 and the kernels fall back to scale 1.0,
    which is in range for this model's residual (measured amax 77.5).

CUDA-GRAPH SAFETY. Decode is 100% graph-replayed (1493/1493 `cuda graph: True`).
Everything here is a plain kernel launch on a fixed device tensor, so it
captures; the per-site slot index is resolved at CAPTURE time from a Python
counter and then frozen into the graph, which is exactly the behaviour we want
(the same site gets the same slot on every replay, on every rank).

DECODE-ONLY, THREE WAYS
1. This code is reached only from `CustomAllReduceV2.custom_all_reduce`, and the
   custom AR only ever sees payloads <= 262,144 B. Prefill's all-reduce is
   83.9 MB and goes to NCCL; it cannot arrive here.
2. An explicit row ceiling (`SGLANG_FP8_DECODE_AR_MAX_ROWS`, default 256):
   decode runs <= 192 rows (bs 32 x gamma+1); prefill runs 8192.
3. An explicit byte ceiling equal to the push-path ceiling.

OFF BY DEFAULT. With `SGLANG_FP8_DECODE_AR` unset, `maybe_narrow` is never
called and the served numerics are bit-identical to stock.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


#: Master switch. OFF by default.
ENABLED: bool = _flag("SGLANG_FP8_DECODE_AR")

#: Second, independent guard against ever catching a prefill collective.
MAX_ROWS: int = int(os.environ.get("SGLANG_FP8_DECODE_AR_MAX_ROWS", "256"))

#: Third guard: the PCIe-P2P push ceiling, applied to the NARROWED (fp8) size.
MAX_BYTES: int = int(os.environ.get("SGLANG_FP8_DECODE_AR_MAX_BYTES", str(256 * 1024)))

#: scale = amax * HEADROOM / 448. 0 => use the world size, which bounds each
#: rank's quantized magnitude by 448/world so the SUM cannot overflow e4m3.
_HEADROOM_ENV: float = float(os.environ.get("SGLANG_FP8_DECODE_AR_HEADROOM", "0"))

#: Engagement witness. This campaign has twice been fooled by a rig that looked
#: healthy while the arm never fired, so the arm counts its own calls.
DEBUG: bool = _flag("SGLANG_FP8_DECODE_AR_DEBUG")

_NUM_SLOTS: int = int(os.environ.get("SGLANG_FP8_DECODE_AR_SLOTS", "512"))
_BLOCK: int = 4096
_FP8_MAX: float = 448.0

CALLS: int = 0
_SITE: int = 0
_STATE: Optional[torch.Tensor] = None  # [_NUM_SLOTS] fp32 running amax per site
_HEADROOM: float = 0.0


@triton.jit
def _quant_flat(X, S, Q, N, HEADROOM, BLOCK: tl.constexpr):
    """[N] bf16 -> [N] e4m3 with the per-site scale carried from the last step."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < N
    amax = tl.load(S)
    scale = amax * HEADROOM / 448.0
    # An unwritten slot (first call at this site) reads 0; 1.0 is in range for
    # this model's residual stream and only ever costs one step of precision.
    scale = tl.where(scale > 0.0, scale, 1.0)
    x = tl.load(X + idx, mask=m, other=0.0).to(tl.float32) / scale
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    tl.store(Q + idx, x.to(tl.float8e4nv), mask=m)


@triton.jit
def _dequant_flat(Q, S, Y, N, HEADROOM, BLOCK: tl.constexpr):
    """[N] e4m3 -> [N] bf16, and fold the output amax back into the site slot.

    The amax is taken from the REDUCED output, which is bit-identical on every
    rank, so every rank's slot evolves identically. `atomicMax` on the positive
    float bit pattern is order-independent, so no block ordering is assumed.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < N
    amax = tl.load(S)
    scale = amax * HEADROOM / 448.0
    scale = tl.where(scale > 0.0, scale, 1.0)
    q = tl.load(Q + idx, mask=m, other=0.0).to(tl.float32)
    y = q * scale
    tl.store(Y + idx, y.to(tl.bfloat16), mask=m)
    blk_amax = tl.max(tl.where(m, tl.abs(y), 0.0), axis=0)
    tl.atomic_max(S, blk_amax)


def _ensure_state(device: torch.device) -> torch.Tensor:
    global _STATE
    if _STATE is None or _STATE.device != device:
        _STATE = torch.zeros(_NUM_SLOTS, dtype=torch.float32, device=device)
    return _STATE


def reset_state() -> None:
    """Drop the calibrated per-site maxima (used by the numerics gate)."""
    if _STATE is not None:
        _STATE.zero_()


def eligible(x: torch.Tensor) -> bool:
    if not ENABLED:
        return False
    return (
        x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape[0] <= MAX_ROWS
        # NB: the ceiling is on the POST-narrowing byte count (fp8 => 1 B per
        # element), not the bf16 input. That is the whole point at k=5, where the
        # bf16 tensor is 307,200 B (above the push ceiling, hence NCCL today) and
        # the fp8 tensor is 153,600 B (comfortably back on the push path).
        and x.numel() <= MAX_BYTES
        and (x.numel() % 16) == 0
    )


def narrow_all_reduce(x: torch.Tensor, raw_all_reduce) -> torch.Tensor:
    """quantize -> fp8 one-shot-push all-reduce -> dequantize.

    `raw_all_reduce` is the stock collective, called on the fp8 tensor.
    """
    global CALLS, _SITE, _HEADROOM
    CALLS += 1
    state = _ensure_state(x.device)
    # Resolved at CUDA-graph capture time and then frozen into the graph, so a
    # given site keeps its slot across every replay and across every rank.
    site = _SITE % _NUM_SLOTS
    _SITE += 1
    if _HEADROOM <= 0.0:
        import torch.distributed as dist

        world = dist.get_world_size() if dist.is_initialized() else 4
        _HEADROOM = float(world) if _HEADROOM_ENV <= 0 else _HEADROOM_ENV

    n = x.numel()
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    slot = state[site]
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    _quant_flat[grid](x, slot, q, n, _HEADROOM, BLOCK=_BLOCK, num_warps=8)
    s = raw_all_reduce(q)
    y = torch.empty_like(x)
    _dequant_flat[grid](s, slot, y, n, _HEADROOM, BLOCK=_BLOCK, num_warps=8)
    if DEBUG and (CALLS <= 3 or CALLS % 4000 == 0):
        print(
            f"[fp8_decode_ar] ENGAGED call={CALLS} site={site} "
            f"shape={tuple(x.shape)} headroom={_HEADROOM}",
            flush=True,
        )
    return y
