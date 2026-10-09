# SPDX-License-Identifier: Apache-2.0
"""Fused GEMM + all-reduce for DeepSeek-V4.1-Flash ``wo_b`` on ROCm gfx950, via mori cco.

``wo_b`` is a ``RowParallelLinear``: an mxfp8 GEMM and then an all-reduce of the
``[M, hidden]`` result. mori's fused kernel writes each destination's row band
into a symmetric window and hands it to the SDMA copy engines as soon as that
band's tiles are done, so the transfer runs inside the GEMM; the reduce and
all-gather still run as their own kernels.

A mori built with ``BUILD_CCO_SDMA=OFF`` compiles the puts out, and then the
all-reduce returns mostly the local slice while the model still answers -- and
measures *faster*, having moved no bytes. ``GemmAllReduceOp.self_test()`` is
what catches that; the flag has been ON by default since ROCm/mori#710.

Needs ``MORI_ENABLE_SDMA=1`` at process start and a layer one of the two gfx950
mxfp8 routes prepared -- see ``mxfp8_route``.

The fused epilogue pushes a whole ``BLOCK_M`` row band to one destination, so M
must be a multiple of ``tp_size * 256``. That granule is mxfp8's, not a choice
-- see ``_BLOCK_M``. Ragged M is zero-padded up to it, which costs GEMM rows and
buys the overlap -- see ``_MIN_PAD_FILL``. Decode and small batches fall back to
the ordinary path, and that is expected rather than a failure.

``mori.ops.gemm_ar.GemmAllReduceOp`` owns the symmetric window, the per-M
compile cache and the row padding. What stays here is sglang's: the TP
communicator, how the window is sized, the operand conversions, and whether
fusing is worth it at this shape.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# ---- operand conversion ----------------------------------------------

#: Rows one packed A-scale group spans, and the rows one quantiser program owns.
#: They are the same 64 by construction, which is what lets the quantiser write
#: mori's layout without any extra traffic -- see `_mxfp8_quant_packed_kernel`.
QUANT_BLOCK_M = 64


def mxfp8_shape(layer) -> tuple[int, int]:
    """The layer's logical ``(N, K)``, read off the ue8m0 scale.

    **Not** ``layer.weight.shape``: the native route rebinds the weight to its
    shuffled ``[N/16, K/128, 2048]``, and reading that first axis as N
    disqualifies every layer without a word.
    """
    sn, sk = layer.weight_scale_mx_e8m0.shape
    return sn * 32, sk * 32


def mxfp8_route(layer) -> str | None:
    """Which gfx950 mxfp8 route prepared this layer, or None if neither did.

    Both leave the ``[N/32, K/32]`` exponent bytes on
    ``layer.weight_scale_mx_e8m0`` and differ only in the weight: ``native``
    rebinds it to the shuffled ``[N/16, K/128, 2048]``, ``aiter`` leaves plain
    ``[N, K]``. Reading one as the other returns an uncorrelated result rather
    than failing, so `mori_weight` branches on this instead of guessing.
    """
    if not hasattr(layer, "weight_scale_mx_e8m0"):
        return None
    if getattr(layer, "mxfp8_native_ready", False):
        return "native"
    if getattr(layer, "mxfp8_aiter_ready", False):
        return "aiter"
    return None


def mxfp8_ready(layer) -> bool:
    """Whether either gfx950 mxfp8 route already prepared this layer."""
    return mxfp8_route(layer) is not None


def mori_weight(layer):
    """The layer's B operand and B scale in mori's layouts, cached on the layer.

    sglang and mori differ only in how a lane's two 64-wide K sub-blocks sit:
    sglang interleaves them inside its 32 bytes, mori keeps them as two 16-byte
    blocks. Cached because the permutation copies the whole weight.
    """
    cached = getattr(layer, "_mori_b", None)
    if cached is not None:
        return cached
    n, k = mxfp8_shape(layer)
    w = layer.weight.data.contiguous().view(torch.uint8)
    if mxfp8_route(layer) == "aiter":
        # That route leaves the weight in its checkpoint [N, K] order, which is
        # what preshuffle_b is defined on, so mori emits its own layout from the
        # source rather than permuting sglang's.
        from mori.ops.gemm_ar import preshuffle_b

        b = preshuffle_b(w.reshape(n, k).view(layer.weight.dtype))
    else:
        # [N/16, K/128, 64 lanes, 2 sub-blocks, 16B] -> the two sub-blocks split out
        b = (
            w.reshape(n // 16, k // 128, 64, 2, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(n, k)
            .view(layer.weight.dtype)
        )
    b_scale = (
        layer.weight_scale_mx_e8m0.data.contiguous()
        .view(torch.uint8)
        .t()
        .contiguous()
        .to(torch.int32)
        .reshape(-1)
    )
    layer._mori_b = (b, b_scale)
    return layer._mori_b


@triton.jit
def _mxfp8_quant_packed_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    BLOCK_M: tl.constexpr,
):
    """sglang's ``_mxfp8_quant_kernel`` writing mori's scale layout directly.

    A variant rather than a stride argument on the original, because the layout
    is not expressible as strides: mori wants element ``(m, kb)`` at
    ``kb*M + (m//64)*64 + (m%16)*4 + (m%64)//16``, which permutes *within* each
    64-row group so a lane's four M tiles land in one dword.

    Free here: the destination stays inside the 64 bytes this program already
    owns, so only the store's order changes.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    sb = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    descale = tl.exp2(sb - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)
    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    dst = pid_b * M + (offs_m // 64) * 64 + (offs_m % 16) * 4 + (offs_m % 64) // 16
    tl.store(s_ptr + dst, sb.to(tl.uint8), mask=m_mask)


def quantize_packed(x: torch.Tensor):
    """bf16 ``[M, K]`` -> (fp8 e4m3 values, mori's packed ue8m0 A scale).

    ``M`` must be a multiple of `QUANT_BLOCK_M`; pad the bf16 input first, so
    the scale is built over the padded M and its packed layout needs no repair.
    """
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import MXFP8_VALUE_DTYPE

    m, k = x.shape
    xq = torch.empty((m, k), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scale = torch.empty((k // 32) * m, dtype=torch.uint8, device=x.device)
    _mxfp8_quant_packed_kernel[(triton.cdiv(m, QUANT_BLOCK_M), k // 32)](
        x,
        xq,
        scale,
        m,
        k,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        BLOCK_M=QUANT_BLOCK_M,
    )
    return xq, scale.view(torch.int32)


#: Smallest padded M worth fusing on a **bf16** wire, measured against what
#: V4.1-Flash runs today. 5120 is not one answer: the fused cost is fixed by the
#: padded size while the baseline follows the true M, so what separates the
#: winning shapes from the losing ones is a floor rather than a fill ratio.
_MIN_FUSED_M = 8192
#: The same for the **fp8** wire, a lower bar because that wire runs 15-25%
#: under the bf16 one. Only m_pad 1024 loses: one row band per destination
#: leaves the scatter nothing to overlap with.
_MIN_FUSED_M_FP8_GATHER = 2048
#: Padding buys the overlap and costs GEMM rows, so a ragged M has to be full
#: enough that the gain at the padded size still clears it. Binds on the fp8
#: wire only: at m_pad >= 8192 the granule already puts the worst fill at 0.875.
_MIN_PAD_FILL = 0.80
_state: _FusedWoB | None = None
_disabled = False
_warned_layout = False
_warned_reject = False
_mori_ok: bool | None = None

#: M is the token count of one forward, not of one request, so whether the
#: fused path engages depends on how the scheduler batches it.
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
    """The op's B operand and B scale, or None if this layer cannot be fused."""
    global _warned_layout
    if not mxfp8_ready(layer):
        if not _warned_layout:
            _warned_layout = True
            logger.warning(
                "mori fused wo_b: neither gfx950 mxfp8 route prepared this "
                "layer (weight shape %s), which the fused kernel requires; "
                "not fusing.",
                tuple(layer.weight.shape),
            )
        return None
    return mori_weight(layer)


class _FusedWoB:
    """The TP communicator and mori's op, one per rank.

    Holds the cco ``Communicator`` built from sglang's TP group, which is the
    part mori cannot construct for us.
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
            # Prove the collective moves bytes before serving a token. A mori
            # built with BUILD_CCO_SDMA=OFF compiles the puts out: the kernels
            # still launch, the all-reduce returns mostly the local slice, and
            # the model answers fluently while being wrong -- and measures
            # *faster*, having moved nothing.
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
    second, which is not a property of the kernel.
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


def _mori_has_gemm_ar() -> bool:
    """Whether the pinned mori carries ``ops.gemm_ar``, memoised.

    Checked rather than assumed because the import sits on the per-call path
    below the switch: a mori predating the op would raise from inside a forward
    rather than fall back.
    """
    global _mori_ok, _disabled
    if _mori_ok is None:
        try:
            import mori.ops.gemm_ar  # noqa: F401

            _mori_ok = True
        except ImportError as err:
            _mori_ok = False
            _disabled = True
            logger.warning(
                "mori fused wo_b: this mori has no ops.gemm_ar, so the fused "
                "path is off for this process: %s",
                err,
            )
    return _mori_ok


def fused_wo_b_available() -> bool:
    """Static gate, cheap enough to call per layer."""
    return (
        not _disabled
        and envs.SGLANG_OPT_FUSED_WO_B_AR.get()
        and _mori_has_gemm_ar()
    )


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
    # wo_a's MXFP8 epilogue (SGLANG_HIP_WO_A_MXFP8) instead hands over fp8 plus
    # a ue8m0 scale, with no bf16 left to quantise, and the op does its own
    # quantisation. Only decode reaches that form, which _eligible declines on
    # M anyway, so declining here costs no fusing.
    if not isinstance(x, torch.Tensor):
        return None

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
