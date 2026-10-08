# SPDX-License-Identifier: Apache-2.0
"""Fused GEMM + all-reduce for DeepSeek-V4.1-Flash ``wo_b`` on ROCm gfx950, via mori cco.

``wo_b`` is a ``RowParallelLinear``, so today it runs a GEMM and then an NCCL
all-reduce of the ``[M, hidden]`` result. The two can overlap: mori's fused
kernel writes each destination's row band straight into a symmetric window and,
as soon as a band's tiles are done, hands it to the SDMA copy engines; the
reduce and all-gather then run as their own kernels.

**Three separate things are on offer here and they do not compose the way the
V4-Pro numbers suggest**, so the thresholds below are measured against what
V4.1-Flash actually runs (``mxfp8_native_blockscaled_linear`` + all-reduce)
rather than inherited. At TP4 N=5120 K=2048 on an idle MI355X:

* **The GEMM.** wo_b's tuning table picks ``hipblaslt_bf16`` above M=8192.
  mori's mxfp8 GEMM does the whole bf16-in/bf16-out pipeline at M=16384 in
  196.0us against 281.3, **-30.3%** -- before any fusion.
* **The fp8 all-gather wire.** -15.6% on its own, without fusing at all, at a
  cost in accuracy: relL2 2.37e-03 -> 2.31e-02.
* **The fusion**, which is bounded by how much GEMM there is to hide the
  transfer behind: 178us of GEMM against 1414us of communication, so the
  ceiling is 11-15%. Making the GEMM faster lowered it.

Whole-layer, against today's path, once mori pinned its launcher dispatch:

    M       bf16 wire    fp8 wire
     5120      -5.8%      -20.7%
     8192     -19.4%      -34.1%
    16384     -16.2%      -32.9%

An earlier revision of this file read those as +11.0% / -0.5% at M=5120 and
concluded the bf16 wire was not worth enabling. That was a property of the
measurement, not of the wire: the fused path was spending 181.6us per call in
FlyDSL's per-dispatch bookkeeping and was host-bound, with the GPU idle between
phases. Both wires pay now.

**Check that mori was built with `BUILD_CCO_SDMA=ON` before believing any
measurement of this path.** With it off every put silently does nothing: the
all-reduce returns mostly the local slice, the model still answers, and the
fused path looks *faster* than it is because it is not moving any data. Measured
that way it reads -6.8% instead of -2.3%, and the fp8 pull -- the only leg that
does not go through SDMA -- looks like the slowest of the three instead of the
fastest. Perplexity is what catches it: 862511 against 3.26 on the same text.

Prerequisites, all of which this module checks rather than assumes:

* a mori built with ``BUILD_CCO_SDMA=ON``. Setting ``BUILD_CCO_SDMA=1`` in the
  environment only rebuilds the *device* bitcode; if the host library was built
  without it there are no SDMA queues, every put silently does nothing, and the
  all-reduce quietly produces zeros. Point ``PYTHONPATH`` at a mori built with
  the flag on.
* ``MORI_ENABLE_SDMA=1`` at process start.
* a layer sglang's native mxfp8 route already prepared
  (``layer.mxfp8_native_ready``) -- see ``_fused_weight``.

The kernel itself is ``mori.ops.gemm_ar.GemmAllReduceOp``, which owns the
symmetric window, the per-M compile cache and the row padding. What stays here
is the part that is sglang's: the TP communicator, how the window is sized, the
operand conversions, and whether fusing is worth it at this shape.

The fused epilogue pushes a whole ``BLOCK_M`` row band to one destination, so a
destination's row slice has to be a whole number of bands: M must be a multiple
of ``tp_size * 256``. That granule is mxfp8's, not a choice -- see ``_BLOCK_M``.
Ragged M is zero-padded up to it, which costs GEMM rows and buys the overlap --
see ``_MIN_PAD_FILL``. Decode and small batches still fall back to the ordinary
path, and that is expected rather than a failure.
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

#: Smallest padded M worth fusing on a **bf16** wire.
#:
#: Re-derived after mori pinned its launcher dispatch. The previous 16384 came
#: from a measurement where the fused path paid 181.6us of FlyDSL bookkeeping in
#: Python per call and was host-bound; "the bf16 wire is not worth it" was true
#: of that layer and is not true of this one. Against what V4.1-Flash runs today
#: at TP4 N=5120 K=2048, by m_pad, two runs:
#:
#:     m_pad  1024   +16.4%      m_pad  8192   -12.2% .. -19.4%
#:     m_pad  2048    +1.8%      m_pad  9216   -17.5% .. -20.8%
#:     m_pad  4096    -0.9%      m_pad 13312    -5.6% ..  -8.5%
#:     m_pad  5120   +17.0% .. -5.8%          m_pad 16384  -16.2%
#:
#: 8192 rather than 5120, and that costs a real -5.8% at M=5120 exactly. m_pad
#: 5120 is not one answer: it wins at fill 1.000 and loses at 0.918 (+8.4%) and
#: 0.820 (+17.0%), because the fused cost is fixed by the padded size while the
#: baseline follows the true M. No fill threshold separates those from the
#: *winning* low-fill points above (0.879 at m_pad 8192 wins -17.6%), so the
#: floor is what has to do it.
_MIN_FUSED_M = 8192
#: The same for the **fp8** wire, and it is a much lower bar because that wire
#: runs 15-25% under the bf16 one -- enough to clear the baseline's own
#: per-bucket swing. Everything from m_pad 2048 up wins:
#:
#:     m_pad  2048   -8.8%       m_pad  5120   -2.3% .. -20.7%
#:     m_pad  3072  -12.7%       m_pad  8192  -28.1% .. -32.7%
#:     m_pad  4096  -15.4%       m_pad 16384  -32.9%
#:
#: Only m_pad 1024 loses (+10.3%): one row band per destination, so there is
#: nothing for the scatter to overlap with.
_MIN_FUSED_M_FP8_GATHER = 2048
#: Padding buys the overlap and costs GEMM rows, so a ragged M has to be full
#: enough that the gain at the padded size still clears it.
#:
#: This binds on the fp8 wire and not on the bf16 one. At m_pad >= 8192 the
#: granule (tp_size * 256) puts the worst possible fill at 0.875, above this
#: either way; at the fp8 floor of 2048 a chunk can be half empty. 0.80 is
#: below every fill measured -- the lowest, 0.820 at M=4200, still wins -2.3%
#: on the fp8 wire -- and guards the range below that, which is not measured.
_MIN_PAD_FILL = 0.80
_state: _FusedWoB | None = None
_disabled = False
_warned_layout = False
_warned_reject = False

#: Log what M this layer is actually handed, as a periodic histogram.
#:
#: M is the token count of one forward, not of one request, so whether the fused
#: path engages depends on how the scheduler batches -- which is not something to
#: infer from the client's concurrency. Set
#: SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG=1 to find out.
_SHAPE_LOG = os.environ.get("SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG") == "1"
_shape_hist: dict = {}
_shape_calls = 0
#: 61 wo_b calls make one forward, so this logs roughly every ten of them.
_SHAPE_EVERY = 610


def _record_shape(m, eligible, world_size):
    global _shape_calls
    key = (m, _padded_m(m, world_size), bool(eligible))
    _shape_hist[key] = _shape_hist.get(key, 0) + 1
    _shape_calls += 1
    if _shape_calls >= _SHAPE_EVERY:
        _shape_calls = 0
        items = sorted(_shape_hist.items(), key=lambda kv: -kv[1])
        logger.info(
            "wo_b shapes: %s",
            " ".join(f"M={m}/pad{p}{'+' if e else '-'}x{c}" for (m, p, e), c in items),
        )
        _shape_hist.clear()


#: mori's mxfp8 kernel packs a lane's four M tiles into one scale dword and
#: picks the byte with the MFMA's opsel, which is four tiles only at BLOCK_M 256
#: -- so a destination's row band is 256 rows here, not blockscale's 128, and M
#: pads to a multiple of ``tp_size * 256``.
_BLOCK_M = 256


def _padded_m(m: int, world_size: int) -> int:
    from mori.ops.gemm_ar import padded_m

    return padded_m(m, world_size, _BLOCK_M)


_logged_config = False


def _shuffled_once():
    global _logged_config
    if _logged_config:
        return True
    _logged_config = True
    return False


def _fused_weight(layer):
    """The op's B operand and B scale, or None if this layer cannot be fused.

    The conversion itself is shared with the standalone GEMM path; what is local
    here is the decision to decline, and warning once when the reason is that
    sglang never prepared the layer.
    """
    global _warned_layout
    if not mxfp8_ready(layer):
        if not _warned_layout:
            _warned_layout = True
            logger.warning(
                "mori fused wo_b: this layer is not mxfp8_native_ready (weight "
                "shape %s), which the fused kernel requires; not fusing.",
                tuple(layer.weight.shape),
            )
        return None
    return mori_weight(layer)


class _FusedWoB:
    """The TP communicator and mori's op, one per rank.

    The window, the per-M compile cache and the row padding all live in
    ``GemmAllReduceOp``; this holds the cco ``Communicator`` built from sglang's
    TP group, which is the part mori cannot construct for us.
    """

    def __init__(self, m_max: int, n: int, k: int):
        import torch.distributed as dist
        from mori.cco import Communicator, UniqueId
        from mori.ops.gemm_ar import GemmAllReduceOp
        from sglang.srt.distributed import get_tp_group

        tp = get_tp_group()
        self.rank = tp.rank_in_group
        self.world_size = tp.world_size
        self.n, self.k, self.m_max = n, k, m_max

        payload = [bytes(Communicator.get_unique_id()) if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=tp.ranks[0], group=tp.cpu_group)
        uid = UniqueId.from_bytes(payload[0])

        # The window is VMM memory outside torch's allocator, so the run has to
        # leave room for it (lower --mem-fraction-static). Sized from m_max, and
        # it cannot grow afterwards.
        gather_dtype = (
            "fp8" if envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER.get() else "bf16"
        )
        window_bytes = GemmAllReduceOp.window_bytes_for(
            self.world_size,
            m_max=m_max,
            n=n,
            block_m=_BLOCK_M,
            gather_dtype=gather_dtype,
        )
        self._comm_ctx = Communicator.init(
            self.world_size,
            self.rank,
            uid,
            per_rank_vmm=4 * window_bytes + (512 << 20),
        )
        self.comm = self._comm_ctx.__enter__()
        try:
            # Which way the fp8 gather moves. The LSA pull widens on the way in
            # instead of in a second kernel, and it wins in both places: 951us
            # against SDMA's 1011 on the layer, 1041.2ms of GPU busy against
            # 1052.1 in the server.
            self.op = GemmAllReduceOp(
                self.comm,
                n=n,
                k=k,
                m_max=m_max,
                quant="mxfp8",
                gather_dtype=gather_dtype,
                gather_transport=os.environ.get(
                    "SGLANG_OPT_FUSED_WO_B_AR_GATHER_TRANSPORT", "lsa"
                ),
            )
            # Prove the collective moves bytes before serving a single token.
            # A mori built without BUILD_CCO_SDMA=ON -- the default, and what the
            # CI image ships -- compiles the puts out: every kernel still
            # launches, nothing moves, the all-reduce returns mostly the local
            # slice, and the model still answers fluently while being wrong.
            # It also measures *faster* that way, which is how a whole
            # end-to-end campaign came out at -6.8% instead of -2.3%. This costs
            # one collective at m_max, on kernels the first real call compiles
            # anyway.
            self.op.self_test()
        except Exception:
            # The communicator holds a VMM reservation the whole process pays
            # for. Half-constructing this object and leaving it open makes the
            # *fallback* path fail too, which is how a simple attribute error
            # here once took the server down.
            self._comm_ctx.__exit__(None, None, None)
            raise
        logger.info(
            "mori fused wo_b: window %.0f MiB, tp=%d, M<=%d, N=%d, K=%d, gather=%s",
            self.op.window_bytes / 2**20,
            self.world_size,
            m_max,
            n,
            k,
            gather_dtype,
        )

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        return self.op.pad_rows(x, m_pad)

    def run(self, q_input, x_scale_raw, weight, weight_scale) -> torch.Tensor:
        return self.op(q_input, weight, x_scale_raw, weight_scale)


def _window_m_max(m_pad: int, world_size: int) -> int:
    """Rows the symmetric window is sized for.

    Taken from the chunked-prefill limit rather than the first request seen, so
    a short prompt arriving first cannot fix a window too small for a full chunk
    later -- the window cannot grow once allocated.
    """
    # `get_global_server_args()` is retired; the scheduling namespace carries
    # the value in effect, which is what the other layers read.
    from sglang.srt.runtime_context import get_schedule

    limit = get_schedule().chunked_prefill_size
    if limit is not None and limit > 0:
        return max(m_pad, _padded_m(limit, world_size))
    return m_pad


def _eligible(m: int, n: int, k: int, world_size: int) -> bool:
    """Whether fusing is both expressible and worth it at this shape.

    mori's ``supports`` answers the first; the thresholds here answer the
    second, and stay on this side because they are measured against what the
    model runs today rather than being a property of the kernel.
    """
    from mori.ops.gemm_ar import supports

    if not supports(m, n, k, world_size, quant="mxfp8"):
        return False
    floor = (
        _MIN_FUSED_M_FP8_GATHER
        if envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER.get()
        else _MIN_FUSED_M
    )
    m_pad = _padded_m(m, world_size)
    return m_pad >= floor and m >= _MIN_PAD_FILL * m_pad


def fused_wo_b_available() -> bool:
    """Static gate, cheap enough to call per layer."""
    return not _disabled and envs.SGLANG_OPT_FUSED_WO_B_AR.get()


def fused_wo_b(layer, x: torch.Tensor) -> torch.Tensor | None:
    """wo_b's `GEMM + all-reduce`, fused. ``None`` means "use the normal path".

    ``x`` is the bf16 activation this rank holds, ``[M, K]``. The result is the
    all-reduced ``[M, N]``, as a **view of the symmetric window** -- the next
    layer's wo_b overwrites it, which is safe because the caller consumes it in
    the residual add of the same layer. ``SGLANG_DEBUG_FUSED_WO_B_AR`` forces a copy
    and cross-checks against the unfused path.
    """
    global _state, _disabled

    if not fused_wo_b_available():
        return None

    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Fp8GridActivation
    from sglang.srt.distributed import get_tp_group

    # wo_a hands wo_b either a plain bf16 activation or an Fp8GridActivation --
    # a bf16 tensor already rounded onto wo_b's fp8 grid, so quantising it below
    # is lossless rather than a second rounding. Either way what the GEMM needs
    # is the bf16 tensor.
    if isinstance(x, Fp8GridActivation):
        x = x.x

    m, k = x.shape
    if not hasattr(layer, "weight_scale_mx_e8m0"):
        return None
    n, w_k = mxfp8_shape(layer)
    if w_k != k:
        return None
    world_size = get_tp_group().world_size
    eligible = _eligible(m, n, k, world_size)
    if _SHAPE_LOG:
        _record_shape(m, eligible, world_size)
    if not eligible:
        return None

    m_pad = _padded_m(m, world_size)
    try:
        if _state is None:
            _state = _FusedWoB(
                m_max=_window_m_max(m_pad, world_size),
                n=n,
                k=k,
            )
        if m_pad > _state.m_max or n != _state.n or k != _state.k:
            return None
        prepared = _fused_weight(layer)
        if prepared is None:
            return None
        weight, b_scale = prepared
        x_in = x if m_pad == m else _state.pad_rows(x, m_pad)
        # The quantiser writes mori's packed scale layout itself, so there is no
        # conversion pass after it. Zero-padded rows quantise to zero values
        # (their scale is tiny but finite), and rows are independent in a GEMM,
        # so the padding contributes nothing to any real row.
        q_input, x_scale = quantize_packed(x_in)
        out = _state.run(q_input, x_scale, weight, b_scale)
        out = out[:m]
    except ValueError as err:
        # A shape or contract rejection is about *this call*, not about the
        # path. Disabling the process on one would be a standing hazard: the op
        # raises ValueError for an M it cannot serve, and a server's M changes
        # with every batch, so one unlucky shape used to switch the whole
        # optimisation off for good.
        global _warned_reject
        if not _warned_reject:
            _warned_reject = True
            logger.warning(
                "mori fused wo_b declined a call and fell back for it; further "
                "declines are silent: %s",
                err,
            )
        return None
    except Exception as err:  # noqa: BLE001 - anything else is not per-call
        _disabled = True
        logger.warning(
            "mori fused wo_b failed and is disabled for this process; "
            "falling back to the split path: %s",
            err,
        )
        return None

    if envs.SGLANG_DEBUG_FUSED_WO_B_AR.get():
        if not _shuffled_once():
            o = _state.op
            logger.info(
                "mori fused wo_b config: rank=%d/%d m_max=%d n=%d k=%d queues=%d "
                "gather=%s/%s m=%d m_pad=%d",
                o.rank,
                o.world_size,
                o.m_max,
                o.n,
                o.k,
                o.sdma_queues,
                o.gather_dtype,
                o.gather_transport,
                m,
                m_pad,
            )
        out = out.clone()
        # Same inputs, immediately again: if the two disagree the server context
        # is racing the collective; if they agree the op is deterministic here
        # and merely disagrees with the reference.
        again = _state.run(q_input, x_scale, weight, b_scale)[:m].clone()
        self_rel = ((out.float() - again.float()).norm() / out.float().norm()).item()
        ref, _ = layer(x)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        logger.info(
            "mori fused wo_b check: M=%d relL2=%.3e self_rel=%.3e", m, rel, self_rel
        )
    return out
