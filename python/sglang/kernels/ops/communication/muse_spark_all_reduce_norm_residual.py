"""Push all-reduce fused with the attention norm-residual boundary (FlashInfer's kernel on CustomAllReduceV2's plane)."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import (
    _cluster_launch_config,
    norm_residual_launch_config,
)
from sglang.srt.runtime_context import get_forward, get_parallel

if TYPE_CHECKING:
    from sglang.kernels.ops.layernorm.muse_spark_norm_residual import (
        FusedPostNormResidualPreNorm,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

_VEC = 8
_MAX_WORLD = 8
# 1-shot while num_tokens * world_size <= 128 (measured on SM107 and GB300, TP4 / TP8). o_proj pushes the
# partial when its GEMM can; otherwise the kernel publishes it.
_ONE_SHOT_MAX_TOKENS_X_WORLD = 128
# (compute capability, world size) -> (min tokens, max tokens, two-shot, cluster size) launches that beat the
# defaults in the serving decode chain (o_proj GEMM -> boundary -> pre-expert GEMM), per architecture.
_LAUNCH_OVERRIDES = {
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


class _LaunchPlan(NamedTuple):
    # 1-shot exchanges every rank's whole partial, pushed by o_proj or published by the kernel.
    one_shot: bool
    slot_bytes: int  # of each rank's push slot the call needs
    cluster_size: int
    threads: int


@cache_once
def _launch_plan(num_tokens: int, hidden_size: int, world_size: int) -> _LaunchPlan:
    """This device's launch policy for one shape; raises RuntimeError if the norm cannot launch it."""
    override = _launch_override_for(
        torch.cuda.get_device_capability(), num_tokens, world_size
    )
    if override is None:
        one_shot = num_tokens * world_size <= _ONE_SHOT_MAX_TOKENS_X_WORLD
        cluster_size, threads, _ = norm_residual_launch_config(
            hidden_size=hidden_size, num_tokens=num_tokens
        )
    else:
        one_shot, cluster_size = not override[0], override[1]
        threads, _ = _cluster_launch_config(hidden_size, num_tokens, cluster_size)
    nbytes = num_tokens * hidden_size * 2
    slot_bytes = nbytes if one_shot else 2 * nbytes // world_size
    return _LaunchPlan(one_shot, slot_bytes, cluster_size, threads)


def _uses_multicast(comm: CustomAllReduceV2) -> bool:
    # Measured on SM107 and GB300: multicast stores win across two nodes (TP8) and are neutral at TP4.
    return comm.has_multicast and comm.world_size >= 8


@cache_once
def _workspace(comm: CustomAllReduceV2, hidden_size: int):
    """FlashInfer's all-reduce dual norm on ``comm``'s push plane; collective on first use."""
    from flashinfer.comm.mnnvl_cutedsl_ag import PushPlane
    from flashinfer.comm.mnnvl_cutedsl_ar_dual_norm import (
        MNNVLCuteDSLAllReduceDualNormWorkspace,
    )

    lease = comm._lease
    slots, slot_bytes = lease.data.push_slots, lease.capacity.push_slot_bytes
    return MNNVLCuteDSLAllReduceDualNormWorkspace(
        comm.world_size,
        comm.rank,
        comm.world_size * slot_bytes // (hidden_size * 2),
        hidden_size,
        group=lease.group,
        push_plane=PushPlane(
            buffer=slots[comm.rank],
            multicast_address=lease.data.push_mc,
            epoch=lease.control.epoch,
            half_bytes=comm.world_size * slot_bytes,
            atom_bytes=4,  # the plane's LamportTrait word
            peer_addresses=torch.tensor(
                [t.data_ptr() for t in slots], dtype=torch.int64, device=comm.device
            ),
        ),
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
        or not comm.has_multicast  # FlashInfer's kernel needs multicast
        or not 2 <= comm.world_size <= _MAX_WORLD
        or num_tokens <= 0
        or hidden_size % (comm.world_size * _VEC)
        or not _supports_clusters()
    ):
        return False
    try:
        plan = _launch_plan(num_tokens, hidden_size, comm.world_size)
    except RuntimeError:
        return False
    # Every cluster must be co-resident: a cluster polls the same cluster on every peer.
    return plan.slot_bytes <= comm._max_push_size and (
        plan.cluster_size == 1
        or num_tokens * plan.cluster_size <= comm.config.num_push_blocks
    )


def o_proj_push(comm: CustomAllReduceV2, x: torch.Tensor, weight: torch.Tensor) -> bool:
    """``x @ weight.T`` stored into every rank's 1-shot slot of ``comm``'s push plane; False, with nothing
    launched, when the served GEMM has no peer-store epilogue (the same answer on every rank)."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_peer_store

    lease = comm._lease
    multicast = _uses_multicast(comm)
    return bf16_gemm_peer_store(
        x.contiguous(),
        weight,
        peer_ptrs=None if multicast else [t.data_ptr() for t in lease.data.push_slots],
        multicast_ptr=lease.data.push_mc if multicast else None,
        epoch_ptr=lease.control.epoch.data_ptr() + 4,  # EpochCounter phase word
        slot_bytes=lease.capacity.push_slot_bytes,
        rank=comm.rank,
        world_size=comm.world_size,
        layout="broadcast",
    )


class FusedAllReducePostNormResidualPreNorm:
    """``boundary`` applied to this rank's row-parallel partial: the attention TP all-reduce runs in the boundary kernel.

    Called like ``FusedPostNormResidualPreNorm``; ``applicable`` must hold, and the caller must skip its own all-reduce.
    The outputs match the push all-reduce (1shot_push / 2shot_push) followed by ``fused_norm_residual``
    at the same cluster size, within BF16 rounding.
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

    def push_applicable(self, num_tokens: int) -> bool:
        """Whether o_proj may push into the boundary's slots (``o_proj_push``); needs ``applicable``."""
        world_size = get_parallel().attn_tp_group.ca_comm.world_size
        return _launch_plan(num_tokens, self._hidden_size, world_size).one_shot

    def __call__(
        self,
        stream: torch.Tensor,
        branch: Optional[torch.Tensor],
        branch_dtype: torch.dtype = torch.bfloat16,
        router_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``branch`` is this rank's bf16 [M, H] row-parallel partial, or None when ``o_proj_push`` stored it."""
        from flashinfer.comm.mnnvl_cutedsl_ar_dual_norm import (
            LLAllReduceDualNormTuning,
            all_reduce_gated_residual_next_rmsnorm,
        )

        assert router_dtype in (None, torch.float32)
        comm = get_parallel().attn_tp_group.ca_comm
        plan = _launch_plan(stream.shape[0], self._hidden_size, comm.world_size)
        kwargs = self._boundary.jit_kernel_kwargs(branch_dtype)
        new_stream, output, router_out = all_reduce_gated_residual_next_rmsnorm(
            None if branch is None else branch.contiguous(),
            stream,
            _workspace(comm, self._hidden_size),
            post_eps=kwargs["post_eps"],
            next_eps=kwargs["pre_eps"],
            alpha=kwargs["alpha"],
            beta=kwargs["beta"],
            post_weight=kwargs["post_weight"],
            post_weight_is_input_scale=kwargs["post_norm_is_input_scale"],
            next_weight=kwargs["pre_weight"],
            want_router=router_dtype is not None,
            use_oneshot=plan.one_shot,
            tuning=LLAllReduceDualNormTuning(plan.cluster_size, plan.threads),
        )
        return output, new_stream, router_out
