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

import torch
import triton
import triton.language as tl

# Module scope, not inside the kernel: Triton resolves a called `@triton.jit`
# function when it compiles the caller's AST, and a local import does not
# reliably reach it.
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_quant
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
    """``fp8_grid_quant`` writing mori's scale layout directly.

    A variant rather than a stride argument on the original, because the layout
    is not expressible as strides: mori wants element ``(m, kb)`` at
    ``kb*M + (m//64)*64 + (m%16)*4 + (m%64)//16``, which permutes *within* each
    64-row group so a lane's four M tiles land in one dword.

    Free here: the destination stays inside the 64 bytes this program already
    owns, so only the store's order changes.

    The scale rule is imported rather than restated. This used to carry its own
    copy of it -- ``ceil(log2(amax / 448))`` -- which is not the same function:
    ``fp8_grid_quant`` reads the exponent off the IEEE bits and is therefore
    exact when ``amax / 448`` is a power of two, where an approximate ``log2``
    can land just above the integer and ``ceil`` then picks one exponent too
    high. Both rules agreeing is exactly what ``fused_wo_b``'s claim that
    re-quantising an ``Fp8GridActivation`` is lossless rests on, so there must
    be one of them.
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
    xq, sb = fp8_grid_quant(x)
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
_warned_window = False
_mori_ok: bool | None = None
_switch_on: bool | None = None

_shape_hist: dict = {}
_shape_calls = 0
#: 61 wo_b calls make one forward, so this logs roughly every ten of them.
_SHAPE_EVERY = 610


def _record_shape(m, m_pad, eligible):
    global _shape_calls
    key = (m, m_pad, bool(eligible))
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


def prepare_wo_b_weight(layer) -> None:
    """Build this layer's fused B operand now, at weight-processing time.

    ``mori_weight`` re-lays the weight out, which is a full second copy of it
    -- the original stays, because decode and every declined call still read
    it. Doing that lazily on the first fused call allocated it *while serving*,
    after the memory profiler had already sized the KV cache around its
    absence: 61 layers of it, about 1.3 GiB at TP2. Preparing here puts it in
    front of the profiler instead, and turns a mid-serving OOM into one at
    startup.

    Best effort on purpose. A failure here leaves ``_mori_b`` unset, the per
    call path falls back to building it, and that path is identical on every
    rank -- so this is not a place where ranks can disagree.
    """
    if not fused_wo_b_available() or not _reduces_over_tp_group(layer):
        return
    if not mxfp8_ready(layer) or not hasattr(layer, "weight_scale_mx_e8m0"):
        return
    mori_weight(layer)


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
                gather_transport=envs.SGLANG_OPT_FUSED_WO_B_AR_GATHER_TRANSPORT.get(),
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

    def close(self) -> None:
        """Release the communicator's VMM reservation.

        Used when a peer failed to construct: this rank's op is sound but
        useless, and the reservation is a per-process cost for the lifetime of
        the server whether or not anything calls it.
        """
        self.op = None
        self._comm_ctx.__exit__(None, None, None)


def _construct_collectively(m_max: int, n: int, k: int) -> bool:
    """Build the op on every rank, or disable the path on every rank.

    The decision has to be collective. mori's phases end in device-side
    cross-rank barriers, so a rank that builds and uses the op while a peer
    fell back to the split path spins on the GPU forever: no timeout, no
    error, a hung server. That is what a rank-local ``_disabled`` produced.

    Rank-local failures are realistic -- an OOM in ``Communicator.init``'s VMM
    reservation, or in ``self_test``'s buffers -- so this agrees on the outcome
    over the CPU group before any rank is allowed to call the op, and the ranks
    that *succeeded* tear their communicator down again when a peer did not.

    Construction is attempted once per process. A failure disables the path for
    good rather than being retried: every retry is another broadcast, another
    multi-hundred-MiB VMM reservation and another teardown, 61 times a forward.
    """
    global _state, _disabled

    import torch.distributed as dist

    from sglang.srt.distributed import get_tp_group

    tp = get_tp_group()
    state = None
    err = None
    try:
        state = _FusedWoB(m_max=m_max, n=n, k=k)
    except Exception as exc:  # noqa: BLE001 - reported below, then disabled
        err = exc

    ok = torch.tensor([0 if err is not None else 1], dtype=torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=tp.cpu_group)
    if int(ok.item()) == 1:
        _state = state
        return True

    _disabled = True
    if state is not None:
        state.close()
    if err is not None:
        logger.warning(
            "mori fused wo_b could not be constructed on rank %d; the fused "
            "path is off for this process on every rank: %s",
            tp.rank_in_group,
            err,
        )
    else:
        logger.warning(
            "mori fused wo_b could not be constructed on a peer rank; the "
            "fused path is off for this process on every rank"
        )
    return False


def _window_m_max(m_pad: int, world_size: int) -> int:
    """Rows the symmetric window is sized for.

    Taken from the deployment's prefill ceiling rather than the first request
    seen, because the window cannot grow once allocated: a short prompt
    arriving first would otherwise fix a window too small for everything after
    it, and every later call would silently fall back for the life of the
    process.

    ``max_prefill_buffer_tokens`` is what the other buffer-sizing callers here
    use, and it answers in the cases a bare ``chunked_prefill_size`` does not
    -- chunked prefill disabled, and PP dynamic chunking, which probes above
    the chunk size. Its own zero case falls through to ``max_prefill_tokens``,
    the same way ``disaggregation/common/conn.py`` does it.
    """
    from sglang.srt.runtime_context import get_schedule, max_prefill_buffer_tokens

    limit = max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens
    if limit:
        return max(m_pad, _padded_m(int(limit), world_size))
    # Nothing declares a ceiling. Sizing from this call is then the only option
    # left, and `_warn_once_over_window` reports it if a later M outgrows it.
    return m_pad


def _warn_once_over_window(m_pad: int, m_max: int) -> None:
    """Report the first M the window cannot hold.

    Declining is correct -- the window cannot grow -- but it is also permanent
    for every M at least this large, so it should not be silent. One line,
    because it then repeats on most prefills.
    """
    global _warned_window
    if _warned_window:
        return
    _warned_window = True
    logger.warning(
        "mori fused wo_b: M=%d exceeds the window's %d rows, falling back for "
        "this and any larger M; size it with --chunked-prefill-size or "
        "--max-prefill-tokens. Further declines are silent.",
        m_pad,
        m_max,
    )


def _eligible(m: int, m_pad: int, n: int, k: int, world_size: int) -> bool:
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
    return m_pad >= floor and m >= _MIN_PAD_FILL * m_pad


def _reduces_over_tp_group(layer) -> bool:
    """Whether this layer's own all-reduce is the one the op would perform.

    Fusing *performs* the all-reduce, so it is a substitute only where the
    split path would have done the same one. The op always reduces over the
    full TP group, while ``wo_b`` is declared ``parallel_group="attn_tp"`` --
    the same ranks only when attention is not data-parallel. Under
    ``--attn-dp-size 8`` the attn-TP group is one rank, the split path reduces
    nothing, and fusing would sum rows belonging to eight unrelated DP ranks.

    The conditions are ``RowParallelLinear.forward``'s own, read off the layer
    rather than restated at the call site. That is where the previous version
    got it wrong: the caller gated ``defer_all_reduce`` and let everything
    making *that* false through to the fused path, including the cases where
    the layer reduces over another group or does not reduce at all.

    Static -- true or false for the life of the process -- so weight prep can
    ask it too. ``_substitutes_layer_all_reduce`` adds the per-forward flag.

    Not checked: ``quantize_communications`` would make the split path's reduce
    numerically different, but it is rejected at startup on anything but NPU.
    ``skip_all_reduce`` is the caller's own argument, and it passes False
    whenever it reaches here.
    """
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.utils import get_group_rank_size

    _, tp_size = get_group_rank_size(layer.tp_group)
    if not (layer.reduce_results and tp_size > 1):
        return False
    # Both of these reduce over the attn-TP group instead.
    if layer.use_decode_attn_tp or layer.use_dp_attention_reduce:
        return False
    # Equal size would do while attn_tp is a subgroup of tp, but comparing the
    # ranks says what is meant and does not rest on that staying true.
    return tuple(layer.tp_group.ranks) == tuple(get_tp_group().ranks)


def _substitutes_layer_all_reduce(layer) -> bool:
    """``_reduces_over_tp_group`` plus the flags that vary per forward."""
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce

    # The mHC post folds the reduce in; the split path skips it here.
    if should_skip_mlp_all_reduce():
        return False
    return _reduces_over_tp_group(layer)


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
    """Static gate, called for every layer of every forward.

    The switch is memoised because it cannot change after startup and this sits
    on the per-layer path of decode and of non-ROCm runs alike, where it was an
    env lookup per call to return False 61 times a forward. ``_disabled`` stays
    dynamic -- it is what a collective construction failure sets -- and is
    checked first so the off path is one bool.
    """
    global _switch_on
    if _disabled:
        return False
    if _switch_on is None:
        _switch_on = envs.SGLANG_OPT_FUSED_WO_B_AR.get()
    return _switch_on and _mori_has_gemm_ar()


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

    if not _substitutes_layer_all_reduce(layer):
        return None

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
    # Computed once and threaded through: it was recomputed in `_eligible`, in
    # `_record_shape` and again below, and it is a call into mori.
    m_pad = _padded_m(m, world_size)
    eligible = _eligible(m, m_pad, n, k, world_size)
    # M is the token count of one forward, not of one request, so whether the
    # fused path engages depends on how the scheduler batches it. Read per call
    # rather than at import, so `envs....override()` in a test is honoured.
    if envs.SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG.get():
        _record_shape(m, m_pad, eligible)
    if not eligible:
        return None

    if _state is None and not _construct_collectively(
        _window_m_max(m_pad, world_size), n, k
    ):
        return None

    # Every reason to decline a call is a check here rather than an exception
    # caught below, and every one of them reads the same on all ranks -- the
    # shape is identical across a TP group, and `_substitutes_layer_all_reduce`
    # has already excluded the configurations where it is not. That matters
    # more than it looks: declining on one rank and fusing on another hangs the
    # group on mori's device-side barriers.
    if m_pad > _state.m_max:
        _warn_once_over_window(m_pad, _state.m_max)
        return None
    if n != _state.n or k != _state.k:
        return None
    prepared = _fused_weight(layer)
    if prepared is None:
        return None
    weight, b_scale = prepared

    # Past this point there is no fallback, deliberately. An OOM in pad_rows or
    # quantize_packed is rank-local, and `run` is collective: returning None
    # from either would put this rank on the split path while its peers sit in
    # mori's barriers, which is a silent hang rather than a visible failure. A
    # process that dies is the better outcome and is what the caller can see.
    x_in = x if m_pad == m else _state.pad_rows(x, m_pad)
    # The quantiser writes mori's packed scale layout itself, so there is no
    # conversion pass after it. Zero-padded rows quantise to zero values (their
    # scale is tiny but finite), and rows are independent in a GEMM, so the
    # padding contributes nothing to any real row.
    q_input, x_scale = quantize_packed(x_in)
    out = _state.run(q_input, x_scale, weight, b_scale)
    out = out[:m]

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
