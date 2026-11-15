"""BF16 GEMM + attention-TP all-reduce for Kimi-K3's o_proj on ROCm gfx950, via mori.

K3 keeps attention in BF16. o_proj is a ``RowParallelLinear(reduce_results=False)``
whose attention-TP sum the stage boundary completes. mori's
``GemmAllReduceOp(quant="bf16")`` runs the GEMM with its epilogue handing each
destination's row band to the SDMA engines as soon as that band is done, then
reduces and all-gathers in the same op, so its output is already the complete
sum.

The handoff to the boundary: the attention stage declares a ``sum_impl``
(``attn_tp_sum``) when this path is on. A fused o_proj records the tensor it
returned; ``attn_tp_sum`` passes exactly that tensor through and all-reduces
anything else. Whether a call fuses depends only on M, which every attention-TP
rank shares, so the ranks always agree.

The fused output aliases the op's symmetric window and the next call on the
same op overwrites it. K3 consumes it before that: the FFN read folds it into
the residual, and the next layer's attention read materialises a fresh prefix
before its o_proj runs. ``SGLANG_MORI_OPROJ_AR_CLONE=1`` copies it anyway.

Under SP-MoE (an EP a2a backend with attention TP) a MoE layer reads its FFN
input on this rank's attention-TP slice of the rows, so the boundary completes
the attention sum with a reduce-scatter, not an all-reduce. There the op runs
reduce-scatter only (``GemmAllReduceOp.reduce_scatter``, no all-gather at all),
and ``reduce_scatter_add`` -- a scattering ReadoutFusion K3 offers on those
layers -- hands the boundary this rank's reduced slice plus its residual.

Opt in with ``SGLANG_MORI_OPROJ_AR=1`` and ``MORI_ENABLE_SDMA=1``. Prefill only:
M pads to a multiple of ``attn_tp_size * 256`` and smaller or mostly-padding M
falls back to the ordinary path.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_BLOCK_M = 256
# Fused cost follows the padded M, the baseline the true M; at ~-15..-20% per
# call the padding must stay under that.
_MIN_PAD_FILL = 0.85

_enabled: Optional[bool] = None
_disabled = False
_state: Optional[_State] = None
_prepared = False
# (data_ptr, numel) of the last fused output not yet handed to the boundary.
_reduced: Optional[Tuple[int, int]] = None
# (data_ptr, numel, slice) of the last reduce-scatter-only o_proj output: the
# full-shaped placeholder o_proj returned, and this rank's reduced rows.
_scattered: Optional[Tuple[int, int, torch.Tensor]] = None


def enabled() -> bool:
    """Static opt-in, read once."""
    global _enabled
    if _enabled is None:
        _enabled = (
            is_hip()
            and envs.SGLANG_MORI_OPROJ_AR.get()
            and get_parallel().attn_tp_size > 1
            and get_parallel().attn_dp_size == 1
        )
    return _enabled and not _disabled


def _min_m() -> int:
    return envs.SGLANG_MORI_OPROJ_AR_MIN_M.get()


def attn_tp_sum(partial: torch.Tensor, forward_batch=None) -> torch.Tensor:
    """The attention stage's declared ATTN_TP sum: a fused o_proj output is
    already complete; anything else takes the ordinary all-reduce."""
    global _reduced
    if _reduced is not None and _reduced == (partial.data_ptr(), partial.numel()):
        _reduced = None
        return partial
    if _reduced is not None and envs.SGLANG_MORI_OPROJ_AR_DEBUG.get():
        logger.warning(
            "mori o_proj gemm+ar: sum of M=%d is not the fused output %s",
            partial.shape[0],
            _reduced,
        )
    return get_parallel().attn_tp_group.all_reduce(partial)


def reduce_scatter_add(hidden_states, residual, forward_batch=None):
    """ReadoutFusion run for SP-MoE layers: complete o_proj's attention-TP sum
    onto this rank's slice with the pending residual add. Takes only the
    tensor a reduce-scatter-only o_proj returned (whose slice it already
    holds); anything else is declined, before any collective, so the boundary
    reduce-scatters it as usual."""
    global _scattered
    if _scattered is None or _scattered[:2] != (
        hidden_states.data_ptr(),
        hidden_states.numel(),
    ):
        return None
    reduced = _scattered[2]
    _scattered = None
    if residual is None:
        # A write layer's read leaves the residual empty; the slice alone is
        # the stream. Copied out of the window, which the next o_proj reuses.
        return reduced.clone()
    if residual.shape[0] != reduced.shape[0]:
        parallel = get_parallel()
        residual = residual.tensor_split(parallel.attn_tp_size)[parallel.attn_tp_rank]
    return residual + reduced


def sum_impl():
    """The SumImplementation K3's attention stage declares, or None when off."""
    if not enabled():
        return None
    from sglang.srt.layers.layer_boundary.output import SumImplementation

    return SumImplementation(attn_tp_sum)


class _State:
    """One cco communicator over attention TP and one op per o_proj (N, K)."""

    def __init__(self, shapes, m_max: int):
        import torch.distributed as dist
        from mori.cco import Communicator, UniqueId
        from mori.ops.gemm_ar import GemmAllReduceOp

        group = get_parallel().attn_tp_group
        self.rank, self.world_size = group.rank_in_group, group.world_size
        try:
            payload = [bytes(Communicator.get_unique_id()) if self.rank == 0 else None]
        except Exception:
            payload = [None]
            dist.broadcast_object_list(
                payload, src=group.ranks[0], group=group.cpu_group
            )
            raise
        dist.broadcast_object_list(payload, src=group.ranks[0], group=group.cpu_group)
        if payload[0] is None:
            raise RuntimeError("rank 0 could not make a mori unique id")
        uid = UniqueId.from_bytes(payload[0])

        window = sum(
            GemmAllReduceOp.window_bytes_for(
                self.world_size, m_max=m_max, n=n, k=k, quant="bf16"
            )
            for n, k in shapes
        )
        self.comm = Communicator.init(
            self.world_size, self.rank, uid, per_rank_vmm=4 * window + (512 << 20)
        )
        self.ops: Dict[Tuple[int, int], GemmAllReduceOp] = {}
        try:
            for n, k in shapes:
                self.ops[(n, k)] = GemmAllReduceOp(
                    self.comm, n=n, k=k, m_max=m_max, quant="bf16"
                )
        except Exception:
            self.close()
            raise
        self.m_max = next(iter(self.ops.values())).m_max
        logger.info(
            "mori o_proj gemm+ar: window %.0f MiB, attn_tp=%d, M<=%d, shapes=%s",
            window / 2**20,
            self.world_size,
            self.m_max,
            sorted(self.ops),
        )

    def compile_and_test(self) -> None:
        # Every padded M a fused call can take, so no FlyDSL compile lands in
        # a forward. _compiled is mori-private; there is no public compile-only
        # entry point.
        granule = self.world_size * _BLOCK_M
        for op in self.ops.values():
            for m_pad in range(granule, op.m_max + 1, granule):
                if m_pad >= _min_m():
                    op._compiled(m_pad, m_pad)
        for op in self.ops.values():
            op.self_test()

    def close(self) -> None:
        for op in self.ops.values():
            op.close()
        self.ops = {}
        self.comm.destroy()


def _agreed(ok: bool) -> bool:
    import torch.distributed as dist

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(
        flag, op=dist.ReduceOp.MIN, group=get_parallel().attn_tp_group.cpu_group
    )
    return bool(flag.item())


def _layer_shape(o_proj) -> Optional[Tuple[int, int]]:
    """(N, K) of a plain BF16 o_proj over the attention-TP group, else None."""
    w = getattr(o_proj, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
        return None
    if getattr(o_proj, "bias", None) is not None:
        return None
    if tuple(o_proj.tp_group.ranks) != tuple(get_parallel().attn_tp_group.ranks):
        return None
    return tuple(w.shape)


def _prefill_ceiling() -> int:
    from sglang.srt.runtime_context import get_schedule, max_prefill_buffer_tokens

    return int(max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens or 0)


def prepare(attn_modules) -> None:
    """Build the window and wrap each eligible o_proj, collectively, at load.

    ``attn_modules`` are ``(attention module, reduce_scatter_only)`` pairs for
    the o_proj whose sum the boundary completes; ``reduce_scatter_only`` marks
    an SP-MoE layer. Every rank sees the same modules, so the ranks agree on
    whether to be here.
    """
    global _state, _disabled, _prepared
    if not enabled() or _prepared:
        return
    _prepared = True
    layers = []
    for attn, rs_only in attn_modules:
        if getattr(attn, "all_reduce_fusion", False):
            continue
        shape = _layer_shape(attn.o_proj)
        if shape is not None:
            layers.append((attn.o_proj, shape, rs_only))
    shapes = sorted({s for _, s, _ in layers})
    m_max = _prefill_ceiling()
    if not shapes or not m_max:
        logger.warning("mori o_proj gemm+ar: nothing to fuse (m_max=%d)", m_max)
        _disabled = True
        return

    state, err = None, None
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        state = _State(shapes, m_max)
    except Exception as exc:  # noqa: BLE001 - agreed below
        err = exc
    if _agreed(err is None):
        try:
            state.compile_and_test()
        except Exception as exc:  # noqa: BLE001 - agreed below
            err = exc
    if not _agreed(err is None):
        _disabled = True
        if state is not None:
            state.close()
        logger.warning("mori o_proj gemm+ar off on every rank: %s", err)
        return
    _state = state
    for o_proj, shape, rs_only in layers:
        o_proj.quant_method = _FusedApply(
            o_proj.quant_method, state.ops[shape], rs_only
        )
    logger.info(
        "mori o_proj gemm+ar: fusing %d o_proj layers, %d of them reduce-scatter only",
        len(layers),
        sum(rs for _, _, rs in layers),
    )


class _FusedApply:
    """o_proj's quant method with the GEMM fused to the attention-TP sum.

    Hooked at ``apply`` rather than ``forward``: K3's MLA wraps o_proj.forward
    with its output gate (``x * sigmoid(gate)`` before the GEMM), and a forward
    hook installed later would skip it. ``apply`` is the GEMM alone, below
    every such wrapper. RowParallelLinear's caller-owned output and SP paths
    do not go through ``apply``, so they keep the ordinary GEMM.
    """

    def __init__(self, inner, op, rs_only: bool):
        self.inner = inner
        self.op = op
        self.rs_only = rs_only
        self._rs_in: Optional[torch.Tensor] = None
        self.clone = envs.SGLANG_MORI_OPROJ_AR_CLONE.get()
        self.debug = envs.SGLANG_MORI_OPROJ_AR_DEBUG.get()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def apply(self, layer, x, bias=None):
        global _reduced, _scattered
        if _reduced is not None or _scattered is not None:
            # The boundary never took the previous fused output: it would have
            # summed (or reduce-scattered) an already-reduced tensor.
            raise RuntimeError("mori o_proj gemm+ar: a fused output was not consumed")
        if self.rs_only:
            if bias is None and x.dim() == 2 and not _capturing():
                hit = self._fused_rs(layer.weight, x)
                if hit is not None:
                    full, reduced = hit
                    _count(x.shape[0])
                    if self.debug:
                        self._check_rs(layer, x, reduced)
                    _scattered = (full.data_ptr(), full.numel(), reduced)
                    return full
            return self.inner.apply(layer, x, bias=bias)
        out = None
        if bias is None and x.dim() == 2 and not _capturing():
            out = _fused(self.op, layer.weight, x)
        if out is None:
            return self.inner.apply(layer, x, bias=bias)
        _count(out.shape[0])
        if self.clone or self.debug:
            out = out.clone()
        if self.debug:
            ref = get_parallel().attn_tp_group.all_reduce(self.inner.apply(layer, x))
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            logger.info("mori o_proj gemm+ar check: M=%d relL2=%.3e", out.shape[0], rel)
        _reduced = (out.data_ptr(), out.numel())
        return out

    def _fused_rs(self, weight, x):
        """(full-shaped placeholder, this rank's reduced rows), or None.

        The boundary's reduce-scatter gives rank r ``x.tensor_split(world)[r]``;
        mori gives rank r the r-th band of the padded rows. Each rank's rows go
        to the start of its band, zero-padded, so the two splits agree.
        """
        op = self.op
        m, k = x.shape
        if m < _min_m() or k != op.k or x.dtype != torch.bfloat16:
            return None
        if get_forward().sp_active:
            return None
        ws = op.world_size
        per, extra = divmod(m, ws)
        rows = per + (1 if extra else 0)
        m_pad = op.padded_m(rows * ws)
        if m_pad > op.m_max or m < _MIN_PAD_FILL * m_pad:
            return None
        band = m_pad // ws
        if m_pad == m:
            x_in = x
        else:
            if self._rs_in is None or self._rs_in.shape[0] < m_pad:
                self._rs_in = torch.empty((op.m_max, k), dtype=x.dtype, device=x.device)
            x_in = self._rs_in[:m_pad]
            v = x_in.view(ws, band, k)
            if extra:
                v[:extra, :rows].copy_(x[: extra * rows].view(extra, rows, k))
                v[extra:, :per].copy_(x[extra * rows :].view(ws - extra, per, k))
                v[extra:, per:].zero_()
                v[:extra, rows:].zero_()
            else:
                v[:, :per].copy_(x.view(ws, per, k))
                v[:, per:].zero_()
        reduced = op.reduce_scatter(x_in, weight, logical_m=m_pad)
        my_rows = per + (1 if op.rank < extra else 0)
        full = reduced._base
        if full is None:
            raise RuntimeError("reduce_scatter did not return a view of the window")
        return full[:m], reduced[:my_rows]

    def _check_rs(self, layer, x, reduced):
        parallel = get_parallel()
        ref = self.inner.apply(layer, x)
        parallel.attn_tp_group.all_reduce(ref)
        ref = ref.tensor_split(parallel.attn_tp_size)[parallel.attn_tp_rank]
        rel = ((reduced.float() - ref.float()).norm() / ref.float().norm()).item()
        logger.info("mori o_proj gemm+rs check: M=%d relL2=%.3e", x.shape[0], rel)


_fused_calls = 0
_fused_rows = 0


def _count(m: int) -> None:
    """Log how much prefill took the fused path, every 1000 fused calls."""
    global _fused_calls, _fused_rows
    _fused_calls += 1
    _fused_rows += m
    if _fused_calls % 1000 == 0 and get_parallel().attn_tp_group.rank_in_group == 0:
        logger.info(
            "mori o_proj gemm+ar: %d fused calls, mean M %.0f",
            _fused_calls,
            _fused_rows / _fused_calls,
        )


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


def _fused(op, weight, x) -> Optional[torch.Tensor]:
    """The reduced [M, N], or None to take the ordinary path. Every decline
    depends only on M and the static shape, which all ranks share."""
    m, k = x.shape
    if m < _min_m() or k != op.k or x.dtype != torch.bfloat16:
        return None
    if get_forward().sp_active:
        return None
    m_pad = op.padded_m(m)
    if m_pad > op.m_max or m < _MIN_PAD_FILL * m_pad:
        return None
    x_in = x if m_pad == m else op.pad_rows(x, m_pad)
    return op(x_in, weight, logical_m=m)[:m]
