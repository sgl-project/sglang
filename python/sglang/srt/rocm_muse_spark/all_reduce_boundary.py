"""The attention O projection's TP all-reduce fused into the attention boundary (gfx1250, ``SGLANG_ROCM_MUSE_AR_BOUNDARY``).

``muse_spark/all_reduce_norm_residual_hip.cuh`` runs a row-owned 2shot_push on the carv2 push plane and the
boundary of ``FusedPostNormResidualPreNorm`` on the reduced row: weight-free post-norm, per-channel gated
residual, weighted pre-norm. It takes the widths where the plain all-reduce runs the flat 2shot_push, so its
branch is bitwise that all-reduce's; the norms round where the Triton boundary kernel does.
"""

from __future__ import annotations

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


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    return load_jit(
        "muse_all_reduce_norm_residual_hip",
        *args,
        cuda_files=["muse_spark/all_reduce_norm_residual_hip.cuh"],
        cuda_wrappers=[("run", f"MuseAllReduceNormResidualKernel<{args}>::run")],
    )


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
        # a slot holds the partial and the reduced copy of each rank's rows
        rows_per_rank = -(-num_rows // comm.world_size)
        return 2 * rows_per_rank * row_bytes <= comm._max_push_size

    def applicable(self, stream: torch.Tensor) -> bool:
        if not self._form_supported:
            return False
        if stream.dtype != torch.float32 or not stream.is_contiguous():
            return False
        num_rows = stream.shape[0]
        if num_rows == 0:
            return False
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
