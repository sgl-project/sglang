"""Push all-reduce fused with the attention norm-residual boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import (
    _cluster_launch_config,
    norm_residual_launch_config,
)
from sglang.srt.runtime_context import get_forward, get_parallel

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.layernorm.muse_spark_norm_residual import (
        FusedPostNormResidualPreNorm,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

_VEC = 8
_MAX_WORLD = 8
# Measured on SM107 and GB300 at TP4 / TP8: 1-shot is faster while num_tokens * world_size <= 128.
_ONE_SHOT_MAX_TOKENS_X_WORLD = 128
# (compute capability, world size) -> (min tokens, max tokens, two-shot, cluster size) launches that beat the
# defaults in the serving decode chain (o_proj GEMM -> boundary -> pre-expert GEMM), per architecture.
_LAUNCH_OVERRIDES = {
    # GB300 TP4, M 48-64.
    ((10, 3), 4): ((48, 64, True, 1),),
    # SM107 TP4, M 48; M 40 and 56 not measured.
    ((10, 7), 4): ((48, 48, False, 4),),
}


@cache_once
def _supports_clusters() -> bool:
    return torch.cuda.get_device_capability()[0] >= 10


def _launch_override_for(
    capability: Tuple[int, int], num_tokens: int, world_size: int
) -> Optional[Tuple[bool, int]]:
    """(two-shot, cluster size) of the override covering ``num_tokens``, if any."""
    overrides = _LAUNCH_OVERRIDES.get((capability, world_size), ())
    for min_tokens, max_tokens, two_shot, cluster_size in overrides:
        if min_tokens <= num_tokens <= max_tokens:
            return two_shot, cluster_size
    return None


@cache_once
def _launch_override(num_tokens: int, world_size: int) -> Optional[Tuple[bool, int]]:
    return _launch_override_for(
        torch.cuda.get_device_capability(), num_tokens, world_size
    )


def _uses_two_shot(num_tokens: int, world_size: int) -> bool:
    override = _launch_override(num_tokens, world_size)
    if override is not None:
        return override[0]
    return num_tokens * world_size > _ONE_SHOT_MAX_TOKENS_X_WORLD


def _launch_config(
    hidden_size: int, num_tokens: int, world_size: int
) -> Tuple[int, int, bool]:
    """(cluster size, threads, prefetch) of the boundary's norm."""
    override = _launch_override(num_tokens, world_size)
    if override is None:
        return norm_residual_launch_config(
            hidden_size=hidden_size, num_tokens=num_tokens
        )
    cluster_size = override[1]
    return cluster_size, *_cluster_launch_config(hidden_size, num_tokens, cluster_size)


def _uses_multicast(comm: CustomAllReduceV2) -> bool:
    # Measured on SM107 and GB300: multicast stores win across two nodes (TP8) and are neutral at TP4.
    return comm.has_multicast and comm.world_size >= 8


def _slot_bytes(num_tokens: int, hidden_size: int, world_size: int) -> int:
    nbytes = num_tokens * hidden_size * 2
    return (
        2 * nbytes // world_size if _uses_two_shot(num_tokens, world_size) else nbytes
    )


@cache_once
def _jit_module(
    hidden_size: int,
    world_size: int,
    num_threads: int,
    cluster_size: int,
    two_shot: bool,
    multicast: bool,
    prefetch_weights: bool,
) -> Module:
    args = make_cpp_args(
        hidden_size,
        world_size,
        num_threads,
        cluster_size,
        two_shot,
        multicast,
        is_arch_support_pdl(),
        prefetch_weights,
    )
    return load_jit(
        "muse_spark_all_reduce_norm_residual",
        *args,
        cuda_files=["muse_spark/all_reduce_norm_residual.cuh"],
        cuda_wrappers=[
            ("run_std", f"MuseSparkAllReduceNormResidual<{args}, false, false>::run"),
            (
                "run_std_router",
                f"MuseSparkAllReduceNormResidual<{args}, false, true>::run",
            ),
            ("run_scale", f"MuseSparkAllReduceNormResidual<{args}, true, false>::run"),
            (
                "run_scale_router",
                f"MuseSparkAllReduceNormResidual<{args}, true, true>::run",
            ),
        ],
    )


def all_reduce_norm_residual_supported(
    comm: Optional[CustomAllReduceV2], num_tokens: int, hidden_size: int
) -> bool:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or not 2 <= comm.world_size <= _MAX_WORLD
        or num_tokens <= 0
        or hidden_size % (comm.world_size * _VEC)
        or _slot_bytes(num_tokens, hidden_size, comm.world_size) > comm.max_push_size
    ):
        return False
    if not _supports_clusters():
        return False
    try:
        cluster_size, _, _ = _launch_config(hidden_size, num_tokens, comm.world_size)
    except RuntimeError:
        return False
    # Correctness: each block of the cluster kernel needs its own epoch counter; the row kernel
    # caps its grid at the counter count and loops over rows.
    return cluster_size == 1 or num_tokens * cluster_size <= comm.config.num_push_blocks


def all_reduce_norm_residual(
    comm: CustomAllReduceV2,
    partial: torch.Tensor,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool = False,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    want_router: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(output, new_stream, router_out)`` of the boundary applied to the all-reduced ``partial``.

    ``partial`` is this rank's bf16 [M, H] row-parallel output; the outputs are bitwise those of the push
    all-reduce (1shot_push / 2shot_push) followed by ``fused_norm_residual`` at the same cluster size.
    """
    num_tokens, hidden_size = stream.shape
    cluster_size, num_threads, prefetch_weights = _launch_config(
        hidden_size, num_tokens, comm.world_size
    )
    output = torch.empty_like(partial)
    new_stream = torch.empty_like(stream)
    router_out = torch.empty_like(stream) if want_router else None

    module = _jit_module(
        hidden_size,
        comm.world_size,
        num_threads,
        cluster_size,
        _uses_two_shot(num_tokens, comm.world_size),
        _uses_multicast(comm),
        prefetch_weights,
    )
    fn = getattr(
        module,
        ("run_scale" if post_norm_is_input_scale else "run_std")
        + ("_router" if want_router else ""),
    )
    fn(
        comm.obj,
        partial.contiguous(),
        stream,
        output,
        new_stream,
        router_out,
        post_weight,
        pre_weight,
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
    )
    return output, new_stream, router_out


class FusedAllReducePostNormResidualPreNorm:
    """``boundary`` applied to this rank's row-parallel partial: the attention TP all-reduce runs in the boundary kernel.

    Called like ``FusedPostNormResidualPreNorm``; ``applicable`` must hold, and the caller must skip its own all-reduce.
    """

    def __init__(
        self,
        boundary: FusedPostNormResidualPreNorm,
        hidden_size: int,
        branch_dtype: torch.dtype,
    ):
        self._boundary = boundary
        self._hidden_size = hidden_size
        self._bf16_branch = branch_dtype == torch.bfloat16

    def applicable(self, num_tokens: int) -> bool:
        return (
            self._bf16_branch
            and not get_forward().sp_active
            and self._boundary.jit_selected()
            and all_reduce_norm_residual_supported(
                comm=get_parallel().attn_tp_group.ca_comm,
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
            )
        )

    def __call__(
        self,
        stream: torch.Tensor,
        branch: torch.Tensor,
        branch_dtype: torch.dtype = torch.bfloat16,
        router_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert router_dtype in (None, torch.float32)
        return all_reduce_norm_residual(
            get_parallel().attn_tp_group.ca_comm,
            branch,
            stream,
            **self._boundary.jit_kernel_kwargs(branch_dtype),
            want_router=router_dtype is not None,
        )
