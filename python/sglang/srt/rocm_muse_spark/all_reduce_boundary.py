"""The attention O projection's TP all-reduce fused into the attention boundary (gfx1250,
``--enable-aiter-allreduce-fusion`` or ``SGLANG_ROCM_MUSE_AR_BOUNDARY``).

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
# rows an own push plane is sized for; larger batches keep the plain all-reduce
_OWN_PUSH_MAX_ROWS = 512


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


def ar_boundary_modes() -> tuple[bool, bool]:
    """(decode, prefill): whether those batches take the kernel. ``SGLANG_ROCM_MUSE_AR_BOUNDARY`` turns on
    both, as does ``--enable-aiter-allreduce-fusion`` less its ``--disable-aiter-allreduce-fusion-in-*``."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec

    if envs.SGLANG_ROCM_MUSE_AR_BOUNDARY.get():
        return True, True
    comm = get_exec().comm
    if not comm.enable_aiter_allreduce_fusion:
        return False, False
    return (
        not comm.disable_aiter_allreduce_fusion_in_decode,
        not comm.disable_aiter_allreduce_fusion_in_prefill,
    )


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

    def __init__(
        self, boundary: FusedPostNormResidualPreNorm, modes: tuple[bool, bool]
    ):
        self._boundary = boundary
        self._decode, self._prefill = modes
        self._form_supported = is_gfx1250_supported() and _boundary_form_supported(
            boundary
        )
        self._applies: dict[tuple[int, bool], bool] = {}
        if self._form_supported:
            _ensure_comm(boundary._hidden_size)

    def _comm(self):
        return getattr(get_parallel().tp_group, "ar_boundary_comm", None)

    def _decide(self, comm, num_rows: int) -> bool:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            choose_algo,
        )

        if not 2 <= comm.world_size <= 8 or num_rows < _MIN_ROWS:
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

    def applicable(self, stream: torch.Tensor, is_extend: bool) -> bool:
        if not (self._prefill if is_extend else self._decode):
            return False
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
