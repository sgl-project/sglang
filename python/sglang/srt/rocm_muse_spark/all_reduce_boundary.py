"""The attention O projection's TP all-reduce fused into the attention boundary (gfx1250,
``SGLANG_ROCM_MUSE_AR_BOUNDARY=1``).

``muse_spark/all_reduce_norm_residual_hip.cuh`` runs a row-owned 2shot_push on the carv2 push plane and the
boundary of ``FusedPostNormResidualPreNorm`` on the reduced row: weight-free post-norm, per-channel gated
residual, weighted pre-norm. It takes the widths where the plain all-reduce runs the flat 2shot_push, so its
branch is bitwise that all-reduce's; the norms round where the Triton boundary kernel does.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_gfx1250_supported

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        FusedPostNormResidualPreNorm,
    )

# one 256-thread block holds a row, 4 bf16 vectors per thread
_MAX_HIDDEN = 256 * 4 * 8

# vec: per-vector exchange; unrolled: from kTwoShotPushUnrolledMinBytes the plain unrolled 2shot_push's
# exchange; factored: segment sums travel with the reduced rows, elementwise boundary on every rank
_KERNEL_MODES = {"vec": 0, "unrolled": 1, "factored": 2}
_MODE = _KERNEL_MODES[os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_KERNEL", "factored")]
# factored only: batches of at least this many rows run the vec kernel instead (0: never). With FlyDSL in the
# process the factored kernel stalls ~100 ms on rare calls at 64 rows; vec does not.
_VEC_FROM_ROWS = int(os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_VEC_FROM_ROWS", "0"))
# batches below this many rows keep the plain all-reduce (TP8 decode: no e2e gain at 8 rows, +1.9% at 16)
_MIN_ROWS = int(os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_MIN_ROWS", "0"))
# batches above this many rows keep the plain all-reduce (0: no limit). TP8 decode with FlyDSL's flagless_le8m3
# routing: its flagless two-shot from 64 rows (1 MiB) beats the fused kernels' carv2 exchange.
_MAX_ROWS = int(os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_MAX_ROWS", "0"))
# rows an own push plane is sized for; larger batches keep the plain all-reduce
_OWN_PUSH_MAX_ROWS = 512
# batches of at most this many rows run the boundary on FlyDSL's role-split one-shot instead
# (ar_boundary_flydsl_oneshot.py; 0: never); needs sglang_meta_common's FlyDSL all-reduce helpers
_FLYDSL_ONESHOT_MAX_ROWS = int(
    os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_FLYDSL_ONESHOT_MAX_ROWS", "0")
)
# MoE batches of at most this many rows unpool only this rank's hidden / TP output columns and gather them inside
# the FFN boundary kernel (ar_boundary_flydsl_gather.py; 0: never); needs sglang_meta_common's FlyDSL all-reduce
# helpers and aiter's FlyDSL gfx1250 bf16 GEMM
_GATHER_FFN_MAX_ROWS = int(
    os.environ.get("SGLANG_ROCM_MUSE_GATHER_FFN_BOUNDARY_MAX_ROWS", "0")
)
# decode batches of exactly these many rows keep the stack's attention and MoE tails (comma-separated; empty:
# none). The default leaves 4 and 8 to the AMD monokernel's gfx1250 Muse Spark build, faster there; target verify
# forwards are not affected
_DECODE_SKIP_ROWS = frozenset(
    int(r)
    for r in os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_DECODE_SKIP_ROWS", "4,8").split(",")
    if r.strip()
)


def _mode_for(num_rows: int) -> int:
    if _MODE == 2 and 0 < _VEC_FROM_ROWS <= num_rows:
        return 0
    return _MODE


@cache_once
def _jit_module(world_size: int, mode: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl(), mode)
    return load_jit(
        "muse_all_reduce_norm_residual_hip",
        *args,
        cuda_files=["muse_spark/all_reduce_norm_residual_hip.cuh"],
        cuda_wrappers=[("run", f"MuseAllReduceNormResidualKernel<{args}>::run")],
    )


def ar_boundary_enabled() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_ROCM_MUSE_AR_BOUNDARY.get()


def decode_rows_skipped(forward_batch, num_rows: int) -> bool:
    return num_rows in _DECODE_SKIP_ROWS and forward_batch.forward_mode.is_decode()


def _ensure_comm(hidden: int) -> None:
    """The kernel's push plane, set once on the TP group: its carv2 custom all-reduce when it has one, else a
    push-only carv2 instance of the kernel's own, made collectively (every TP rank builds the same layers)
    and sized for ``_OWN_PUSH_MAX_ROWS`` rows, without ring planes, so it takes little from the KV cache.
    The plain all-reduces keep whatever backend the group has; graph capture enters an own instance's
    context through ``GroupCoordinator.graph_capture``."""
    from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
        two_shot_push_slot_bytes,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    tp = get_parallel().tp_group
    if tp.world_size < 2 or hasattr(tp, "ar_boundary_comm"):
        return
    ca_comm = tp.ca_comm
    if isinstance(ca_comm, CustomAllReduceV2) and not ca_comm.disabled:
        tp.ar_boundary_comm = ca_comm
        return
    rows, ws = _OWN_PUSH_MAX_ROWS, tp.world_size
    push_size = max(
        two_shot_push_slot_bytes(rows * hidden * 2, ws),
        -(-rows // ws) * max(_slot_row_bytes(hidden, ws, mode) for mode in (0, 2)),
    )
    comm = CustomAllReduceV2(
        tp.cpu_group,
        tp.device,
        max_pull_size=0,
        max_push_size=push_size,
        ring_simple_buffer_size=0,
        hier_ring_simple_buffer_size=0,
    )
    tp.ar_boundary_comm = None if comm.disabled else comm


def _ensure_one_shot(boundary: FusedPostNormResidualPreNorm) -> None:
    """The FlyDSL one-shot boundary kernel's scratch and compiled launchers, set once on the TP group
    (collectively); none without sglang_meta_common."""
    tp = get_parallel().tp_group
    if hasattr(tp, "ar_boundary_one_shot"):
        return
    tp.ar_boundary_one_shot = None
    # one lane per 256-column piece when a wave folds its row's sums
    if (
        _FLYDSL_ONESHOT_MAX_ROWS <= 0
        or tp.world_size not in (2, 4, 8)
        or boundary._hidden_size != 8 * 32 * 32
    ):
        return
    try:
        import sglang_meta_common.ops.comm.flydsl_ar  # noqa: F401
    except ImportError:
        return
    from sglang.srt.rocm_muse_spark.ar_boundary_flydsl_oneshot import (
        FlyDSLArBoundaryOneShot,
    )

    tp.ar_boundary_one_shot = FlyDSLArBoundaryOneShot(
        tp,
        boundary._hidden_size,
        _FLYDSL_ONESHOT_MAX_ROWS,
        boundary._post_norm.eps,
        boundary._pre_norm.eps,
    )
    tp.ar_boundary_one_shot.warm()


def gather_ffn_boundary(hidden: int, post_eps: float, pre_eps: float):
    """The TP group's FlyDSL gather + FFN boundary kernel (FlyDSLGatherFfnBoundary), or None when off; built
    collectively on the first MoE layer's init and shared by every layer."""
    tp = get_parallel().tp_group
    if hasattr(tp, "gather_ffn_boundary"):
        return tp.gather_ffn_boundary
    tp.gather_ffn_boundary = None
    if (
        _GATHER_FFN_MAX_ROWS <= 0
        or tp.world_size not in (2, 4, 8)
        or hidden != 8 * 32 * 32
    ):
        return None
    try:
        import aiter.ops.flydsl.gemm_a16w16_gfx1250  # noqa: F401
        import sglang_meta_common.ops.comm.flydsl_ar  # noqa: F401
    except ImportError:
        return None
    from sglang.srt.rocm_muse_spark.ar_boundary_flydsl_gather import (
        FlyDSLGatherFfnBoundary,
    )

    tp.gather_ffn_boundary = FlyDSLGatherFfnBoundary(
        tp, hidden, _GATHER_FFN_MAX_ROWS, post_eps, pre_eps, post_weight=True
    )
    tp.gather_ffn_boundary.warm()
    return tp.gather_ffn_boundary


def _factored_layout(hidden: int, world_size: int) -> tuple[int, int] | None:
    """(segments, pieces) per row of the factored kernel, or None where it falls back to the
    per-vector one: a segment is a warp of 32 lanes times kTwoShotPushReduceUnroll 16-byte
    vectors, a piece 32 lanes times 2, and a lane per piece gathers the row's sums."""
    reduce_unroll = 8 if world_size <= 4 else 4
    seg_elems = 32 * reduce_unroll * 8
    if hidden % seg_elems:
        return None
    segs = hidden // seg_elems
    pieces = segs * reduce_unroll // 2
    return (segs, pieces) if pieces <= 32 else None


def _slot_row_bytes(hidden: int, world_size: int, mode: int) -> int:
    """Push slot bytes per owned row: the partial and the reduced copy (+ the factored kernel's segment sums)."""
    layout = _factored_layout(hidden, world_size) if mode == 2 else None
    sums = 16 * layout[0] * layout[1] if layout else 0
    return 2 * hidden * 2 + sums


def _boundary_form_supported(boundary: FusedPostNormResidualPreNorm) -> bool:
    """The attention boundary's form: weight-free post-norm, per-channel gate, weighted pre-norm."""
    hidden = boundary._hidden_size
    return (
        not boundary._post_norm_is_input_scale
        and not boundary._post_norm.has_weight
        and boundary._use_residual_gate
        and boundary._residual_add.gate.numel() == hidden
        and boundary._pre_norm.has_weight
        and not boundary._pre_norm_is_input_scale
        and hidden % 8 == 0
        and hidden <= _MAX_HIDDEN
    )


class AllReduceAttnBoundary:
    """``boundary`` applied to this rank's O projection partial, with the TP all-reduce in the same kernel.

    ``applicable`` must hold for the batch, and the caller must skip its own all-reduce.
    """

    def __init__(self, boundary: FusedPostNormResidualPreNorm):
        self._boundary = boundary
        self._form_supported = is_gfx1250_supported() and _boundary_form_supported(
            boundary
        )
        self._applies: dict[tuple[int, bool], bool] = {}
        if self._form_supported:
            _ensure_comm(boundary._hidden_size)
            _ensure_one_shot(boundary)

    def _comm(self):
        return getattr(get_parallel().tp_group, "ar_boundary_comm", None)

    def _one_shot_for(self, num_rows: int):
        one_shot = getattr(get_parallel().tp_group, "ar_boundary_one_shot", None)
        return one_shot if one_shot is not None and one_shot.supports(num_rows) else None

    def _decide(self, comm, num_rows: int) -> bool:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            choose_algo,
        )

        if not 2 <= comm.world_size <= 8 or num_rows < _MIN_ROWS:
            return False
        if 0 < _MAX_ROWS < num_rows:
            return False
        row_bytes = self._boundary._hidden_size * 2
        nbytes = num_rows * row_bytes
        # 1shot widths keep the plain push all-reduce
        if not comm.uses_two_shot_push(nbytes):
            return False
        # hier_2shot_push also reports TWO_SHOT_PUSH; its pair + cross stages beat the flat exchange here
        if (
            comm.forced_algo is None
            and choose_algo(
                comm.config, nbytes, can_use_graph=comm._can_use_graph()
            ).use_hier
        ):
            return False
        rows_per_rank = -(-num_rows // comm.world_size)
        slot_row = _slot_row_bytes(
            self._boundary._hidden_size, comm.world_size, _mode_for(num_rows)
        )
        return rows_per_rank * slot_row <= comm._max_push_size

    def applicable(self, stream: torch.Tensor) -> bool:
        if not self._form_supported:
            return False
        if stream.dtype != torch.float32 or not stream.is_contiguous():
            return False
        num_rows = stream.shape[0]
        if num_rows == 0:
            return False
        if self._one_shot_for(num_rows) is not None:
            return True
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        comm = self._comm()
        if not isinstance(comm, CustomAllReduceV2) or comm.disabled:
            return False
        # capture and eager read different heuristics of the comm's config
        key = (num_rows, comm._can_use_graph())
        applies = self._applies.get(key)
        if applies is None:
            applies = self._applies[key] = self._decide(comm, num_rows)
        return applies

    def __call__(
        self,
        stream: torch.Tensor,
        partial: torch.Tensor,
        router_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(pre-norm output in bf16, new fp32 stream, the pre-norm output in fp32 when ``router_dtype`` is
        fp32) of the boundary on ``all_reduce(partial)``."""
        assert router_dtype in (None, torch.float32), router_dtype
        boundary = self._boundary
        boundary._ensure_cached_weights(torch.bfloat16)
        one_shot = self._one_shot_for(partial.shape[0])
        if one_shot is not None:
            return one_shot(
                stream,
                partial.contiguous(),
                boundary._cc_pre_weight,
                boundary._cc_alpha,
                boundary._cc_beta,
                with_router=router_dtype is not None,
            )
        comm = self._comm()
        output = torch.empty_like(partial)
        new_stream = torch.empty_like(stream)
        router_out = None if router_dtype is None else torch.empty_like(stream)
        _jit_module(comm.world_size, _mode_for(partial.shape[0])).run(
            comm.obj,
            partial.contiguous(),
            stream,
            boundary._cc_pre_weight,
            boundary._cc_alpha,
            boundary._cc_beta,
            output,
            new_stream,
            router_out,
            float(boundary._post_norm.eps),
            float(boundary._pre_norm.eps),
        )
        return output, new_stream, router_out
