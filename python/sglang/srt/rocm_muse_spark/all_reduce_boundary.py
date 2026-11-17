"""The attention O projection's TP all-reduce fused into the attention boundary (gfx1250, ``SGLANG_ROCM_MUSE_AR_BOUNDARY``).

``muse_spark/all_reduce_norm_residual_hip.cuh`` runs a row-owned 2shot_push on the carv2 push plane and the
boundary of ``FusedPostNormResidualPreNorm`` on the reduced row: weight-free post-norm, per-channel gated
residual, weighted pre-norm. It takes the widths where the plain all-reduce runs the flat 2shot_push, so its
branch is bitwise that all-reduce's; the norms round where the Triton boundary kernel does.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)
_LOG_DECISIONS = os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_LOG", "0") == "1"
# vec: per-vector exchange; unrolled: from kTwoShotPushUnrolledMinBytes the plain unrolled 2shot_push's
# exchange; factored: segment sums travel with the reduced rows, elementwise boundary on every rank
_KERNEL_MODES = {"vec": 0, "unrolled": 1, "factored": 2}
_MODE = _KERNEL_MODES[os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_KERNEL", "vec")]
_logged: set = set()


def _log_once(key, msg: str) -> None:
    if _LOG_DECISIONS and key not in _logged:
        _logged.add(key)
        logger.warning("AR boundary: %s", msg)


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl(), _MODE)
    _log_once("jit", f"kernel mode={_MODE}")
    return load_jit(
        "muse_all_reduce_norm_residual_hip",
        *args,
        cuda_files=["muse_spark/all_reduce_norm_residual_hip.cuh"],
        cuda_wrappers=[("run", f"MuseAllReduceNormResidualKernel<{args}>::run")],
    )


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

    def _comm(self):
        return get_parallel().tp_group.ca_comm

    def _decide(self, comm, num_rows: int) -> bool:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            choose_algo,
        )

        if not 2 <= comm.world_size <= 8:
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
        # a slot holds the partial and the reduced copy of each rank's rows (+ segment sums)
        rows_per_rank = -(-num_rows // comm.world_size)
        return rows_per_rank * self._slot_row_bytes(comm.world_size) <= comm._max_push_size

    def _slot_row_bytes(self, world_size: int) -> int:
        hidden = self._boundary._hidden_size
        layout = _factored_layout(hidden, world_size) if _MODE == 2 else None
        sums = 16 * layout[0] * layout[1] if layout else 0
        return 2 * hidden * 2 + sums

    def applicable(self, stream: torch.Tensor) -> bool:
        if not self._form_supported:
            _log_once("form", "off: boundary form or arch not supported")
            return False
        if stream.dtype != torch.float32 or not stream.is_contiguous():
            _log_once(("stream", stream.dtype), f"off: stream {stream.dtype}")
            return False
        num_rows = stream.shape[0]
        if num_rows == 0:
            return False
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        comm = self._comm()
        if not isinstance(comm, CustomAllReduceV2) or comm.disabled:
            _log_once("comm", f"off: tp ca_comm is {type(comm).__name__}")
            return False
        # capture and eager read different heuristics of the comm's config
        key = (num_rows, comm._can_use_graph())
        applies = self._applies.get(key)
        if applies is None:
            applies = self._applies[key] = self._decide(comm, num_rows)
            _log_once(
                key,
                f"rows={num_rows} graph={key[1]} {'FUSED' if applies else 'plain'}",
            )
        return applies

    def __call__(
        self, stream: torch.Tensor, partial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(pre-norm output in bf16, new fp32 stream) of the boundary on ``all_reduce(partial)``."""
        boundary = self._boundary
        boundary._ensure_cached_weights(torch.bfloat16)
        comm = self._comm()
        output = torch.empty_like(partial)
        new_stream = torch.empty_like(stream)
        _jit_module(comm.world_size).run(
            comm.obj,
            partial.contiguous(),
            stream,
            boundary._cc_pre_weight,
            boundary._cc_alpha,
            boundary._cc_beta,
            output,
            new_stream,
            float(boundary._post_norm.eps),
            float(boundary._pre_norm.eps),
        )
        return output, new_stream
