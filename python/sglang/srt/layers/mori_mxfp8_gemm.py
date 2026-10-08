# SPDX-License-Identifier: Apache-2.0
"""mori's mxfp8 GEMM in place of the native mxfp8 linear, on ROCm gfx950.

Separate from `mori_gemm_ar`, which fuses the GEMM with `wo_b`'s all-reduce.
This is the same GEMM with nothing fused onto it, and it exists because most of
the win is the multiply rather than the overlap. bf16 in and bf16 out, the whole
pipeline including quantisation, at `N=5120 K=2048 M=16384` on MI355X:

    bf16 (fake_quant + hipBLASLt)   281.3 us
    sglang mxfp8 (tl.dot_scaled)    274.9
    mori mxfp8                      196.0     -30.3%

**The two are different optimisations with different reach, and their gains do
not add.** At `wo_b` the fused path already runs this GEMM, so this one only
catches the calls fusing declined. At `wq_b` it is the whole story: a
`ColumnParallelLinear` has no all-reduce, so there is nothing to fuse and the
fused op is structurally inapplicable. Per-rank at TP4:

    wq_b   N=8192 K=1280   ColumnParallel -- no collective
    wo_b   N=5120 K=2048   RowParallel

The ordering falls out for free and needs no coordination: the model calls
`fused_wo_b` first and only reaches the linear when it declines, so this hook
sees exactly the remainder.

Hooked at `fp8_hip._apply_native`, which is the one place every
`mxfp8_native_ready` layer passes through, and where the input has already been
normalised to `(x, input_scale, input_on_fp8_grid)`.

Both operand forms the hook can hand over are served: a bf16 activation, and
the fp8-plus-row-major-scale one a fused producer emits. The second needs the
scale converting, which is not free, and is still a win -- see `_a_operands`.

**Two mori kernels come through here, not one.** Above 32 tokens it is the
GEMM above, gated on `_MIN_GRID`; at or below 32, and only for an already-fp8
activation, it is a skinny GEMM built for decode. They are different kernels
with different operand layouts and different reasons for being faster, and the
split between them is `_GEMV_MAX_M`, which documents why it is the operand form
rather than M that decides.

Returning None means "use the normal path" and is not a failure: an unsupported
shape, a bf16 activation at decode, or an M in the band neither kernel wins all
land there.
"""

from __future__ import annotations

import logging
import os

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.mori_mxfp8_common import (
    mori_weight,
    mxfp8_ready,
    mxfp8_shape,
    quantize_packed,
)

logger = logging.getLogger(__name__)

#: Workgroups mori's GEMM grid needs before it is worth using at all.
#:
#: **Not a floor on M.** mori's tile is 256 rows by 256 columns, so its grid is
#: `ceildiv(M, 256) * (N / 256)` and what starves it is the product, not either
#: factor. A floor on M alone is therefore only ever right for the N it was
#: measured at -- and this was measured at N=8192 and N=5120, then applied to
#: layers a quarter that wide.
#:
#: Scored over the 77 points this gate actually decides -- 12 shapes from the
#: checkpoint, M above the GEMV's 32 tokens, and a shape `supports_gemm`
#: accepts -- by the percentage each threshold gets wrong, losses served plus
#: wins declined, cold:
#:
#:     gate            served & slower   forfeited   total
#:     grid >= 48             2.9%         0.0%      2.9%
#:     grid >= 64 (this)      2.9%         0.0%      2.9%
#:     grid >= 80             0.3%         7.0%      7.3%
#:     grid >= 128            0.3%        60.7%     60.9%
#:
#: **64, where this said 80.** The move is a measurement fix, not a retune: the
#: benchmark had been cooling only mori's weight, while SGLang's `hipblaslt_bf16`
#: route reads a dequantised bf16 copy that stayed in cache. With both measured
#: cold SGLang is slower on those rows, and mori starts winning at a smaller
#: grid than it appeared to. 48 and 64 score identically because no measured
#: point falls between them, so this is the edge of the evidence rather than a
#: fitted optimum.
#:
#: A floor on M alone is a different matter and is wrong by an order of
#: magnitude. `M >= 1280`, read off wq_b and wo_b, served wkv at M=2048 for
#: +100% and wq_a for +38%.
_MIN_GRID = 64

#: The tile this grid is counted in. Must track mori's `MXFP8_BLOCK_M` and
#: `DEFAULT_BLOCK_N`; both are 256 and neither is a knob a caller turns.
_TILE = 256



def _grid_worth_it(m: int, n: int) -> bool:
    """Whether mori's GEMM grid is big enough on a 256-CU part. See `_MIN_GRID`."""
    return -(-m // _TILE) * (n // _TILE) >= _MIN_GRID

#: At or below this M mori has a second kernel, a skinny GEMM built for decode,
#: and it is served on one condition: **the activation must already be fp8**.
#:
#: The kernels alone, cold, against `mxfp8_gemv` on the same operands:
#:
#:     M     wq_b 8192x1280   wo_b 5120x2048
#:      1        -7.1%            -7.7%
#:      8        -4.4%            -8.4%
#:     32        +1.4%           -10.7%
#:
#: But sglang's GEMV *quantises a bf16 activation inside the kernel*, and mori's
#: takes fp8, so a bf16 caller pays a separate `mxfp8_e4m3_quantize` launch.
#: That pass is 2.0-2.3us on every shape -- near enough all of it launch, since
#: it moves at most 128KB -- and on `wq_b` and `wo_b` that is more than the whole
#: kernel win:
#:
#:     M     bf16 in, whole pipeline    wq_b      wo_b
#:      1                               +21.4%    +21.7%
#:      8                               +33.8%    +26.4%
#:     32                               +19.7%     +5.0%
#:
#: So the gate is the operand form, not M. When the activation arrives fp8 with
#: its row-major `[M, K/32]` ue8m0 scale -- what a fused producer emits -- mori
#: needs *no* conversion at all, unlike the GEMM path above, which has to run
#: `preshuffle_a_scale` on it.
#:
#: **This is not a general fact about bf16, and it was stated as one here.**
#: sglang's fusion quantises the same activation once per workgroup, so its cost
#: grows with N and K where mori's separate pass does not. Measured across all
#: twelve of the checkpoint's shapes, mori wins the bf16 pipeline wherever
#: sglang's in-kernel quantise exceeds one launch -- `wkv` by 6%, `wq_a` by 5%,
#: `wq_b` at TP1 by 16%. It stays declined anyway: the layers this hook is
#: enabled on are `wq_b` and `wo_b`, which are the two worst rows of that table,
#: and a gate would have to predict sglang's redundant cost from the shape across
#: a boundary thinner than run-to-run noise. See mori's README, "Why a bf16
#: activation is not simply worse".
_GEMV_MAX_M = 32

_ops: dict[tuple[int, int], object] = {}
_gemv_ops: dict[tuple[int, int], object] = {}
_disabled = False
_warned_reject = False

#: Two of these run per layer (wq_b and wo_b), so 40 layers is 80 calls a
#: forward; this logs roughly every ten forwards.
_SHAPE_EVERY = 800
_SHAPE_LOG = os.environ.get("SGLANG_OPT_MORI_MXFP8_GEMM_SHAPE_LOG") == "1"
_shape_hist: dict = {}
_shape_calls = 0


#: The env read, done once. Not `.get()` per call: this runs on every linear of
#: every layer, including decode's, where it does nothing but decline. Reading
#: and parsing os.environ each time measured 1.4us against 0.29 -- 0.39ms per
#: decode step across 7 shapes and 40 layers, which is 8-11% of ITL spent by a
#: path that is not doing anything.
_enabled: bool | None = None


def mori_mxfp8_available() -> bool:
    """Static gate, cheap enough to call per layer -- see `_enabled`."""
    global _enabled
    if _enabled is None:
        _enabled = envs.SGLANG_OPT_MORI_MXFP8_GEMM.get()
    return _enabled and not _disabled


def _op_for(n: int, k: int):
    """The op for this shape, or False if mori cannot express it.

    One per (N, K) and no per-M cache behind it: `c_m` is a runtime argument, so
    one compile serves every token count. That matters here more than anywhere
    -- a server's prefill batch size changes constantly, and a per-M cache would
    pay a multi-second compile on nearly every batch.
    """
    hit = _ops.get((n, k))
    if hit is not None:
        return hit
    from mori.ops.gemm_ar import Mxfp8GemmOp, supports_gemm

    if not supports_gemm(n, k):
        _ops[(n, k)] = False
        return False
    _ops[(n, k)] = Mxfp8GemmOp(n=n, k=k)
    return _ops[(n, k)]


def _gemv_op_for(n: int, k: int):
    """The skinny-GEMM op for this shape, or False if mori cannot express it."""
    hit = _gemv_ops.get((n, k))
    if hit is not None:
        return hit
    from mori.ops.gemm_ar import Mxfp8GemvOp, supports_gemv

    if not supports_gemv(n, k):
        _gemv_ops[(n, k)] = False
        return False
    _gemv_ops[(n, k)] = Mxfp8GemvOp(n=n, k=k)
    return _gemv_ops[(n, k)]


def _try_gemv(layer, x_2d, input_scale, m, n, k):
    """The skinny GEMM, or None to fall through. See `_GEMV_MAX_M`.

    `input_scale` being None means a bf16 activation, which this declines --
    not because the kernel cannot take one, but because quantising it costs more
    than the kernel saves.
    """
    if m > _GEMV_MAX_M or input_scale is None:
        return None
    op = _gemv_op_for(n, k)
    if op is False:
        return None
    weight, _ = mori_weight(layer)
    # The GEMV takes both scales exactly as the checkpoint stores them: the
    # weight's [N/32, K/32] and the activation's row-major [M, K/32]. No
    # conversion, which is the whole reason it is worth serving here.
    return op(
        x_2d,
        weight,
        input_scale.view(torch.uint8).contiguous(),
        layer.weight_scale_mx_e8m0.data.contiguous().view(torch.uint8),
    )


def _a_operands(op, x, input_scale, m):
    """A's fp8 values and mori's packed A scale, plus the padded M.

    Two input forms reach here and both are served.

    A **bf16** activation is quantised straight into mori's layout, which costs
    nothing extra: the packed store stays inside the 64 bytes one quantiser
    program already owns.

    An activation a fused producer already quantised -- what `wq_b` gets from
    `rmsnorm_fake_quant_fp8` at large M -- arrives as fp8 plus a *row-major*
    ``[M, K/32]`` scale, which mori cannot index, so it needs
    `preshuffle_a_scale`. That is a real extra pass, 22-24us at wq_b's shape,
    and it is still worth paying: against the native route, which takes this
    form with no conversion at all, mori is -38.1% at M=2048 and -22.2% at
    M=16384 *including* the conversion.

    Declining it was tempting and would have been wrong -- `wq_b` is the layer
    this path exists for, being column-parallel with nothing to fuse, and at the
    M where it matters its operand is always this form.
    """
    from mori.ops.gemm_ar import preshuffle_a_scale

    m_pad = op.padded_m(m)
    if input_scale is None:
        x_in = x if m_pad == m else op.pad_rows(x, m_pad)
        return quantize_packed(x_in), m_pad

    # Already fp8. Pad the values with zeros -- a padded row contributes zero
    # whatever its scale says -- and the scale with anything the layout accepts.
    if m_pad != m:
        x = op.pad_rows(x, m_pad)
        input_scale = torch.nn.functional.pad(
            input_scale, (0, 0, 0, m_pad - m)
        )
    return (x, preshuffle_a_scale(input_scale.to(torch.int32))), m_pad


def mori_mxfp8_linear(
    layer,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: torch.Tensor | None,
    input_on_fp8_grid: bool,
) -> torch.Tensor | None:
    """The native mxfp8 linear, through mori. ``None`` means "use the normal path".

    ``input_on_fp8_grid`` needs no special handling: it says the bf16 input is
    already rounded onto the fp8 grid, which makes the quantisation below
    lossless rather than a second rounding.
    """
    global _disabled, _warned_reject

    if not mori_mxfp8_available():
        return None

    # A bf16 activation on a decode-sized batch is the one case worth rejecting
    # before anything else is read: neither kernel serves it, and it is most of
    # the calls this hook ever sees -- 40 layers times several linears, every
    # step. Everything below runs on prefill batches only, which are few.
    m = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1]
    if m <= _GEMV_MAX_M and input_scale is None:
        if _SHAPE_LOG:
            _record_shape(*mxfp8_shape(layer), m, False)
        return None

    x_2d = x.view(-1, x.shape[-1])
    n, k = mxfp8_shape(layer)
    if k != x_2d.shape[1]:
        if _SHAPE_LOG:
            _record_shape(n, k, m, False)
        return None

    try:
        if not mxfp8_ready(layer):
            return None
        if m > _GEMV_MAX_M and not _grid_worth_it(m, n):
            if _SHAPE_LOG:
                _record_shape(n, k, m, False)
            return None
        if m <= _GEMV_MAX_M:
            out = _try_gemv(layer, x_2d, input_scale, m, n, k)
            if _SHAPE_LOG:
                _record_shape(n, k, m, out is not None)
            if out is None:
                return None
            if bias is not None:
                out = out + bias
            return out.view(*x.shape[:-1], n)
        op = _op_for(n, k)
        if op is False:
            if _SHAPE_LOG:
                _record_shape(n, k, m, False)
            return None
        operands = _a_operands(op, x_2d, input_scale, m)
        if operands is None:
            if _SHAPE_LOG:
                _record_shape(n, k, m, False)
            return None
        (q_input, a_scale), m_pad = operands
        weight, b_scale = mori_weight(layer)
        out = op(q_input, weight, a_scale, b_scale)[:m]
        if _SHAPE_LOG:
            _record_shape(n, k, m, True)
    except ValueError as err:
        # A shape or contract rejection is about *this call*. Disabling the
        # process on one would be a standing hazard: a server's M changes with
        # every batch, so one unlucky shape would switch the path off for good.
        if not _warned_reject:
            _warned_reject = True
            logger.warning(
                "mori mxfp8 GEMM declined a call and fell back for it; further "
                "declines are silent: %s",
                err,
            )
        return None
    except Exception as err:  # noqa: BLE001 - anything else is not per-call
        _disabled = True
        logger.warning(
            "mori mxfp8 GEMM failed and is disabled for this process; falling "
            "back to the native linear: %s",
            err,
        )
        return None

    if bias is not None:
        out = out + bias
    if envs.SGLANG_DEBUG_MORI_MXFP8_GEMM.get():
        _cross_check(layer, x_2d, out, m, n, m_pad)
    return out.view(*x.shape[:-1], n)


def _cross_check(layer, x_2d, out, m, n, m_pad):
    """Log relL2 against the route this replaced, on the same inputs."""
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
    )

    ref = mxfp8_native_blockscaled_linear(
        x_2d,
        layer.weight.view(torch.uint8),
        layer.weight_scale_mx_e8m0,
        weight_bf16=layer.weight_bf16,
    )
    rel = (
        torch.linalg.vector_norm(out.float() - ref.float())
        / torch.linalg.vector_norm(ref.float())
    ).item()
    logger.info(
        "mori mxfp8 GEMM check: N=%d K=%d M=%d(pad %d) relL2=%.3e",
        n,
        layer.weight_scale_mx_e8m0.shape[1] * 32,
        m,
        m_pad,
        rel,
    )


def _record_shape(n: int, k: int, m: int, served: bool) -> None:
    """Log what M each shape is handed, as a periodic histogram.

    A path that declined every call looks exactly like one that ran and was not
    worth it: the model stays correct, the profile stays plausible, and a
    benchmark reports the baseline under the optimisation's name. This is what
    tells the two apart. Set SGLANG_OPT_MORI_MXFP8_GEMM_SHAPE_LOG=1.
    """
    global _shape_calls
    key = (n, k, m, bool(served))
    _shape_hist[key] = _shape_hist.get(key, 0) + 1
    _shape_calls += 1
    if _shape_calls >= _SHAPE_EVERY:
        _shape_calls = 0
        items = sorted(_shape_hist.items(), key=lambda kv: -kv[1])
        logger.info(
            "mori mxfp8 shapes: %s",
            " ".join(
                f"{n}x{k}/M={m}{'+' if ok else '-'}x{c}"
                for (n, k, m, ok), c in items
            ),
        )
        _shape_hist.clear()
