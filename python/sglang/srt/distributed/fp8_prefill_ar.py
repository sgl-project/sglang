"""fp8-e4m3 narrow-dtype all-reduce for the PREFILL residual collectives.

WHY THIS EXISTS
DSV4.1-Flash emits ~81 bf16 tensor-parallel all-reduces per extend forward at
[T, 5120], ~47 MB per call, at 38.8 GB/s busbw on a node with NO NVLink. That
collective is pure wire time: rank spread is 0.8% (so none of it is barrier
wait), per-call fixed cost is 5.2 us (0.29%), and busbw is flat from 2.95 MB to
188 MB (no large-message cliff a custom kernel could fix). NCCL is already at
its optimum here -- `NCCL_P2P_LEVEL=SYS` is set in production and is itself
worth 1.50x.

The only remaining degree of freedom is therefore WIRE WIDTH. `dist.all_reduce`
on `torch.float8_e4m3fn` is supported natively by this stack (torch 2.13.0+cu130,
NCCL 2.29.7) -- no custom collective, no new communicator -- and halves the wire
time exactly: 1.824 ms -> 0.924 ms (0.506x) at the production prefill shape.

Note this is a DIFFERENT mechanism from the quantized all-GATHER scheme that was
closed earlier. That one's wire ratio is n/(2r), which at n=4, r=2 is 1.00x --
no gain, which is why it was a pessimization. Narrowing the reduce DTYPE has
wire ratio 1/r = 0.50x, unconditionally in n.

TWO INTERCEPTION POINTS, TWO INDEPENDENT SWITCHES
  AR#1  `SGLANG_FP8_PREFILL_AR`   -- the attention output collective, emitted
        inside `wo_b` (RowParallelLinear, reduce_results=True). Reached through
        an OPT-IN per-instance marker `_fp8_prefill_ar` set only on the DSV4
        attention `wo_b`, so no other RowParallelLinear in the model is touched.
        40 calls per extend forward.
  AR#2  `SGLANG_FP8_PREFILL_AR2`  -- the post-experts MoE combine, emitted by
        `DeepseekV2MoE.forward_normal` / `.forward_normal_dual_stream`
        (DSV4.1's MoE INHERITS DeepseekV2MoE). 40 calls per extend forward.

        *** READ THIS BEFORE EXTENDING COVERAGE AGAIN. *** An earlier attempt
        put the AR#2 hook in `layers/moe/fused_moe_triton/layer.py`'s
        `if self.reduce_results and (moe_tp_size > 1 or moe_ep_size > 1)` branch.
        On this build (flashinfer_mxfp4 MoE runner, ep-size 8 -> moe_tp_size 1,
        a2a backend "none") that branch NEVER EXECUTES: the live combine
        all-reduce is the one in `deepseek_v2.py`. The hook was therefore dead
        code and the arm silently covered only half the surface -- witnessed as
        `_quant_kernel` firing 40x instead of ~80x per extend forward. Verify any
        new site with a live kernel-count census, never by reading the code.

Both switches default OFF; with neither set every entry point here is a straight
call to `tensor_model_parallel_all_reduce` and the served numerics are
bit-identical to stock.

THE SCHEME (per-token COMMON scale)
The scale must be COMMON across ranks: sum(q_i * s_i) != s * sum(q_i) unless the
s_i agree, so a per-rank scale silently computes the wrong sum. Hence:

    per-token amax  ->  all_reduce(MAX) on the T-element amax vector (18 KB)
                    ->  quantize to e4m3  ->  all_reduce(SUM) in fp8
                    ->  dequantize to bf16

The three elementwise steps are FUSED TRITON KERNELS, one pass each, not eager
torch. That is not a style preference: eager costs ~331 us/call here and drops
the win from 43.6% to 28.4%.

Finer scale granularity buys NOTHING and costs a lot: per-128-group reads
5.038% rel-Fro against per-token's 5.042%, because the error is set by e4m3's
3-bit mantissa plus the sum headroom, not by scale resolution. Per-token is the
right granularity.

HEADROOM
`scale = amax_common * HEADROOM / 448`, so each rank's quantized magnitude is
bounded by 448/HEADROOM. At HEADROOM = world_size the sum over ranks is bounded
by 448 and the fp8 accumulator CANNOT overflow, by construction. The default is
therefore the world size, which is provably overflow-free.

DECODE IS DELIBERATELY UNTOUCHED. The decode all-reduce is 240 KiB and
latency-bound, not bandwidth-bound -- halving its bytes is worth ~0, and it
does not even take this code path (decode's collective is the custom
`sglang::all_reduce_1shot_push_kernel`, not NCCL). Every entry point here is
additionally gated on the forward mode being EXTEND, with a token-count floor as
a second, independent guard.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.distributed import (
    get_tp_group,
    tensor_model_parallel_all_reduce,
)

# --- configuration (all read once, at import) --------------------------------


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _env_int_set(name: str) -> frozenset:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    return frozenset(int(t) for t in raw.replace(" ", ",").split(",") if t)


#: AR#1 master switch (attention `wo_b`). OFF by default.
ENABLED: bool = _env_flag("SGLANG_FP8_PREFILL_AR")

#: AR#2 master switch (post-experts MoE combine). OFF by default, and
#: INDEPENDENT of AR#1 so either half can be shipped or reverted on its own.
ENABLED2: bool = _env_flag("SGLANG_FP8_PREFILL_AR2")

#: True if either half is armed. The attention site publishes the forward mode
#: for AR#2's benefit, so it must run when AR#2 alone is on.
ANY_ENABLED: bool = ENABLED or ENABLED2

#: Second, independent guard against ever catching a decode collective. Decode
#: runs at most `cuda-graph-max-bs-decode` x (gamma+1) = 192 rows; prefill runs
#: up to the 8192 chunked-prefill size. 1024 sits several x clear of both.
MIN_TOKENS: int = int(os.environ.get("SGLANG_FP8_PREFILL_AR_MIN_TOKENS", "1024"))

#: See HEADROOM in the module docstring. 0 means "use world size" (overflow-free).
_HEADROOM_ENV: float = float(os.environ.get("SGLANG_FP8_PREFILL_AR_HEADROOM", "0"))

#: Per-site numerics fallback for AR#2: a comma-separated list of layer ids to
#: leave on the stock bf16 collective. Measured at full coverage the accumulated
#: 80-site drift is 2.156% against the model's own accepted 2.31%, so this is
#: NOT needed and defaults to empty; it exists so coverage can be trimmed
#: without a code change if a future checkpoint is less tolerant.
EXCLUDE2: frozenset = _env_int_set("SGLANG_FP8_PREFILL_AR2_EXCLUDE")

#: Engagement witnesses. This campaign has repeatedly been fooled by a rig that
#: looked healthy while the arm never fired, so each half counts its own calls.
DEBUG: bool = _env_flag("SGLANG_FP8_PREFILL_AR_DEBUG")
DEBUG2: bool = _env_flag("SGLANG_FP8_PREFILL_AR2_DEBUG")

#: How many leading calls each witness prints. 3 is enough to show the arm fired
#: at the production shape; raise it to count calls PER FORWARD, which is the
#: only way to prove COVERAGE rather than mere engagement -- patch 12 passed an
#: "it fired" check while silently covering half the sites.
DEBUG_CALLS: int = int(os.environ.get("SGLANG_FP8_PREFILL_AR_DEBUG_CALLS", "3"))
CALLS: int = 0
CALLS2: int = 0

_FP8_MAX: float = 448.0  # e4m3 max finite magnitude
_BLOCK: int = 1024


# --- forward-mode propagation ------------------------------------------------
# The MoE combine site has no `forward_batch` in scope and sglang's ForwardContext
# carries only `attn_backend`, so the real `forward_mode.is_extend()` is published
# here by the attention site (which does have it) and read by the MoE site. Both
# run in the same layer of the same forward, attention strictly first, and a
# forward runs synchronously on a single Python thread per worker -- the same
# assumption sglang's own `forward_context._current` global already makes.
#
# The LAYER id, by contrast, is NOT published from attention: AR#2 reads
# `self.layer_id` off the MoE module itself, which is exact. An earlier design
# hooked the generic collective funnel and had to infer the layer from the last
# attention call, which mislabels the first call of every forward with a stale
# id. Reading it at the site removes that trap entirely.

_IS_EXTEND: bool = False


def set_extend(is_extend: bool) -> None:
    global _IS_EXTEND
    _IS_EXTEND = bool(is_extend)


def get_extend() -> bool:
    return _IS_EXTEND


# --- fused kernels -----------------------------------------------------------


@triton.jit
def _row_amax_kernel(X, S, H, stride_x, BLOCK: tl.constexpr):
    """Per-row (per-token) absolute max of a [T, H] tensor -> [T] fp32."""
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], tl.float32)
    for off in range(0, H, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < H
        x = tl.load(X + row * stride_x + idx, mask=m, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(S + row, tl.max(acc, axis=0))


@triton.jit
def _quant_kernel(X, S, Q, H, stride_x, stride_q, HEADROOM, BLOCK: tl.constexpr):
    """[T, H] bf16 -> [T, H] e4m3, using the COMMON per-row scale in S."""
    row = tl.program_id(0)
    amax = tl.load(S + row)
    scale = amax * HEADROOM / 448.0
    # An all-zero row has amax 0; keep the scale finite and let the row stay 0.
    scale = tl.where(scale > 0.0, scale, 1.0)
    inv = 1.0 / scale
    for off in range(0, H, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < H
        x = tl.load(X + row * stride_x + idx, mask=m, other=0.0).to(tl.float32)
        q = x * inv
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(Q + row * stride_q + idx, q.to(tl.float8e4nv), mask=m)


@triton.jit
def _dequant_kernel(Q, S, Y, H, stride_q, stride_y, HEADROOM, BLOCK: tl.constexpr):
    """[T, H] e4m3 -> [T, H] bf16, undoing the COMMON per-row scale in S."""
    row = tl.program_id(0)
    amax = tl.load(S + row)
    scale = amax * HEADROOM / 448.0
    scale = tl.where(scale > 0.0, scale, 1.0)
    for off in range(0, H, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < H
        q = tl.load(Q + row * stride_q + idx, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + row * stride_y + idx, (q * scale).to(tl.bfloat16), mask=m)


# --- the collective ----------------------------------------------------------


def _pg():
    return get_tp_group().device_group


def fp8_all_reduce(x: torch.Tensor, group=None) -> torch.Tensor:
    """All-reduce `x` ([T, H], bf16) over the TP group with an fp8-e4m3 wire dtype.

    Returns a NEW bf16 tensor; `x` is not modified.
    """
    group = _pg() if group is None else group
    world = dist.get_world_size(group=group)
    headroom = float(world) if _HEADROOM_ENV <= 0 else _HEADROOM_ENV

    T, H = x.shape
    grid = (T,)

    amax = torch.empty(T, dtype=torch.float32, device=x.device)
    _row_amax_kernel[grid](x, amax, H, x.stride(0), BLOCK=_BLOCK, num_warps=8)

    # The scale MUST agree across ranks or the reduction is arithmetically wrong.
    dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)

    q = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x.device)
    _quant_kernel[grid](
        x, amax, q, H, x.stride(0), q.stride(0), headroom,
        BLOCK=_BLOCK, num_warps=8,
    )

    dist.all_reduce(q, op=dist.ReduceOp.SUM, group=group)

    y = torch.empty_like(x)
    _dequant_kernel[grid](
        q, amax, y, H, q.stride(0), y.stride(0), headroom,
        BLOCK=_BLOCK, num_warps=8,
    )
    return y


def _shape_eligible(x: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[0] >= MIN_TOKENS
    )


def eligible(x: torch.Tensor, is_extend: Optional[bool] = None) -> bool:
    if not ENABLED:
        return False
    if not (_IS_EXTEND if is_extend is None else is_extend):
        return False  # decode / target-verify: leave the stock path alone
    return _shape_eligible(x)


def maybe_fp8_all_reduce(
    x: torch.Tensor, is_extend: Optional[bool] = None
) -> torch.Tensor:
    """AR#1 drop-in for `tensor_model_parallel_all_reduce` at the attention site."""
    if eligible(x, is_extend):
        global CALLS
        CALLS += 1
        y = fp8_all_reduce(x)
        if DEBUG and (CALLS <= DEBUG_CALLS or CALLS % 4000 == 0):
            print(
                f"[fp8_prefill_ar] ENGAGED call={CALLS} shape={tuple(x.shape)}",
                flush=True,
            )
        return y
    return tensor_model_parallel_all_reduce(x)


def eligible2(
    x: torch.Tensor,
    layer_id: Optional[int] = None,
    is_extend: Optional[bool] = None,
) -> bool:
    if not ENABLED2:
        return False
    if not (_IS_EXTEND if is_extend is None else is_extend):
        return False  # decode / target-verify: leave the stock path alone
    if layer_id is not None and layer_id in EXCLUDE2:
        return False
    return _shape_eligible(x)


def maybe_fp8_all_reduce_ar2(
    x: torch.Tensor,
    layer_id: Optional[int] = None,
    is_extend: Optional[bool] = None,
) -> torch.Tensor:
    """AR#2 drop-in for `tensor_model_parallel_all_reduce` at the MoE combine.

    A pass-through to the stock collective unless `SGLANG_FP8_PREFILL_AR2` is
    set AND this is an extend forward AND the tensor clears the token floor.
    """
    if eligible2(x, layer_id, is_extend):
        global CALLS2
        CALLS2 += 1
        y = fp8_all_reduce(x)
        if DEBUG2 and (CALLS2 <= DEBUG_CALLS or CALLS2 % 4000 == 0):
            print(
                f"[fp8_prefill_ar2] ENGAGED call={CALLS2} layer={layer_id} "
                f"shape={tuple(x.shape)}",
                flush=True,
            )
        return y
    return tensor_model_parallel_all_reduce(x)
