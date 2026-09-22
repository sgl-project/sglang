# SPDX-License-Identifier: Apache-2.0
"""Fused GEMM + all-reduce for DSV4 ``wo_b`` on ROCm gfx950, via mori cco.

``wo_b`` is a ``RowParallelLinear``, so today it runs a block-scale fp8 GEMM and
then an NCCL all-reduce of the ``[M, hidden]`` result. The two can overlap:
mori's fused kernel writes each destination's row band straight into a symmetric
window and, as soon as a band's tiles are done, hands it to the SDMA copy
engines; the reduce and all-gather then run as their own kernels.

At ``[16384, 7168]`` K=2048 on 8x MI355X the pair costs 1419.5us and the fused
kernel 1146.2, i.e. **-19.4%**. Essentially all of it is the overlap: the same
pipeline unfused is 1465.9us, and the GEMM alone is 369.4. In-server, over one
20000-token prefill captured with the same warm-up and flush on both sides, GPU
busy goes **1096.0ms -> 1070.3, -25.7ms / -2.3%**, and the fp8 gather takes it
to **1041.2, -5.0%**. The request's own wall time agrees: 1.1969s -> 1.1728 ->
1.1385. The layer-level win is much larger than the end-to-end one because
`wo_b` is about 12% of the profile.

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

The kernel itself is ``mori.ops.gemm_ar.GemmAllReduceOp``, which owns the
symmetric window, the per-M compile cache and the row padding. What stays here
is the part that is sglang's: the TP communicator, how the window is sized, and
whether fusing is worth it at this shape.

The fused epilogue pushes a whole ``BLOCK_M`` row band to one destination, so a
destination's row slice has to be a whole number of bands: M must be a multiple
of ``tp_size * 128``. Ragged M is zero-padded up to that, which costs GEMM rows
and buys the overlap -- see ``_MIN_PAD_FILL``. Decode and small batches still
fall back to the ordinary path, and that is expected rather than a failure.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: Smallest padded M worth fusing. Measured at ``[*, 7168] K=2048`` on 8 ranks
#: against the aiter GEMM + NCCL pair the model runs today: 1024 -2.6%, 2048
#: -7.8%, 4096 -12.3%, 8192 -15.7%, 16384 -19.4%. At 1024 a destination gets a
#: single row band, so there is nothing to overlap and the fused kernel is 1.2%
#: *slower* than the same pipeline unfused.
_MIN_FUSED_M = 4096
#: The fp8 gather used to need a higher floor: with the SDMA transport its two
#: conversion kernels were a fixed cost against a transfer that shrinks with M,
#: so it was 2.3% *slower* at M=4096 and only paid from 8192 up. The LSA pull
#: removed that -- it widens on the way in, so there is no second kernel and no
#: re-read of the landed fp8 -- and fp8 now wins wherever fusing does at all.
#: Measured on the fused path at ``[*, 7168] K=2048``, fp8/lsa against bf16:
#: M=4096 -7.3% (335.8 -> 311.3us), M=8192 -13.8% (593.7 -> 512.0).
#:
#: Keeping 8192 is not merely conservative, it is a loss: a chunk between the
#: two floors takes neither wire and falls back to NCCL entirely.
_MIN_FUSED_M_FP8_GATHER = _MIN_FUSED_M
#: Padding buys the overlap and costs GEMM rows, so the fill ratio has to clear
#: the gain at the padded size. 0.88 is break-even at ``_MIN_FUSED_M`` (12%
#: more rows against a 12.3% gain) and increasingly safe above it. Measured:
#: M=3616 padded to 4096 is 342.1us against 371.4 for the unfused pair.
_MIN_PAD_FILL = 0.88
_state: Optional[_FusedWoB] = None
_disabled = False
_warned_layout = False
_warned_reject = False

#: Log what M this layer is actually handed, as a periodic histogram.
#:
#: M is the token count of one forward, not of one request, so whether the fused
#: path engages depends on how the scheduler batches -- which is not something to
#: infer from the client's concurrency. Set
#: SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG=1 to find out.
_shape_hist: dict = {}
_shape_calls = 0
#: 61 wo_b calls make one forward, so this logs roughly every ten of them.
_SHAPE_EVERY = 610


def _record_shape(m, eligible, world_size):
    global _shape_calls
    from mori.ops.gemm_ar import padded_m

    key = (m, padded_m(m, world_size), bool(eligible))
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


_logged_config = False


def _shuffled_once():
    global _logged_config
    if _logged_config:
        return True
    _logged_config = True
    return False


def _fused_weight(layer):
    """The weight to hand the op, or None if this layer cannot be fused.

    The op requires B preshuffled. sglang does that at load time, but only when
    the aiter block-fp8 linear is what consumes the weight and the tuned triton
    GEMM does not cover the shape, recording it as ``layer.aiter_bpreshuffled``.
    Handing the row-major one to the op does not fail -- it returns an
    uncorrelated result -- so check rather than assume.
    """
    global _warned_layout
    if getattr(layer, "aiter_bpreshuffled", False):
        return layer.weight
    if not _warned_layout:
        _warned_layout = True
        logger.warning(
            "mori fused wo_b: this layer's weight is not aiter-preshuffled "
            "(shape %s), which the fused kernel requires; not fusing.",
            tuple(layer.weight.shape),
        )
    return None


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
            self.world_size, m_max=m_max, n=n, gather_dtype=gather_dtype
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
                gather_dtype=gather_dtype,
                gather_transport=envs.SGLANG_OPT_FUSED_WO_B_AR_GATHER_TRANSPORT.get(),
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
    from mori.ops.gemm_ar import padded_m

    from sglang.srt.runtime_context import get_schedule

    # The schedule bag, not the ServerArgs record: the record answers with what
    # the operator typed, and chunked_prefill_size is one of the fields
    # resolution fills in when it was left unset.
    limit = get_schedule().chunked_prefill_size
    if limit is not None and limit > 0:
        return max(m_pad, padded_m(limit, world_size))
    return m_pad


def _eligible(m: int, n: int, k: int, world_size: int) -> bool:
    """Whether fusing is both expressible and worth it at this shape.

    mori's ``supports`` answers the first; the thresholds here answer the
    second, and stay on this side because they are measured against what the
    model runs today rather than being a property of the kernel.
    """
    from mori.ops.gemm_ar import padded_m, supports

    if not supports(m, n, k, world_size):
        return False
    floor = (
        _MIN_FUSED_M_FP8_GATHER
        if envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER.get()
        else _MIN_FUSED_M
    )
    m_pad = padded_m(m, world_size)
    return m_pad >= floor and m >= _MIN_PAD_FILL * m_pad


def fused_wo_b_available() -> bool:
    """Static gate, cheap enough to call per layer."""
    return not _disabled and envs.SGLANG_OPT_FUSED_WO_B_AR.get()


def fused_wo_b(layer, x: torch.Tensor) -> Optional[torch.Tensor]:
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

    import aiter
    from mori.ops.gemm_ar import padded_m

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.quantization.fp8_utils import aiter_per1x128_quant

    m, k = x.shape
    n = layer.weight.shape[0]
    world_size = get_tp_group().world_size
    eligible = _eligible(m, n, k, world_size)
    if envs.SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG.get():
        _record_shape(m, eligible, world_size)
    if not eligible:
        return None

    m_pad = padded_m(m, world_size)
    try:
        if _state is None:
            _state = _FusedWoB(
                m_max=_window_m_max(m_pad, world_size),
                n=n,
                k=k,
            )
        if m_pad > _state.m_max or n != _state.n or k != _state.k:
            return None
        x_in = x if m_pad == m else _state.pad_rows(x, m_pad)
        # Same quantisation the unfused path does. transpose_scale=True makes
        # the quantiser write the group scale in physical [K/128, M] order,
        # which is exactly what the kernel indexes.
        q_input, x_scale = aiter_per1x128_quant(
            x_in, quant_dtype=aiter.dtypes.fp8, transpose_scale=True
        )
        weight = _fused_weight(layer)
        if weight is None:
            return None
        # Flat, explicitly. transpose_scale=True writes the group scale in
        # K/128-major order but keeps the (M, K/128) shape, so the tensor's shape
        # does not describe its contents and the op refuses to guess -- see
        # _flatten_a_scale. Passing the 2-D tensor transposes it and returns an
        # answer that is wrong by relL2 0.36, which is perplexity 862511 against
        # the unfused path's 3.26.
        out = _state.run(q_input, x_scale.reshape(-1), weight, layer.weight_scale_inv)
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
        again = _state.run(
            q_input, x_scale.reshape(-1), weight, layer.weight_scale_inv
        )[:m].clone()
        self_rel = ((out.float() - again.float()).norm() / out.float().norm()).item()
        ref, _ = layer(x)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        logger.info(
            "mori fused wo_b check: M=%d relL2=%.3e self_rel=%.3e", m, rel, self_rel
        )
    return out
