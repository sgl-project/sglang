"""TP symmetric-workspace push slots for the fused Muse Spark MoE finalize + all-reduce.

The kernel is ``FusedPostExpertNormWeightedSumTRTLLM`` in
``sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce``.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        WorkspaceLease,
    )


class FinalizePushWorkspace:
    """Push slots of the TP workspace; dependency-ordered launches only.

    Storage comes from ``lease`` and is resolved on first use, after the workspace
    is allocated; ``policy`` (the TP custom all-reduce) picks 1-shot vs 2-shot.
    ``device_group`` is the TP device group the unfused finalize paths all-reduce on.
    """

    def __init__(
        self,
        lease: WorkspaceLease,
        policy: CustomAllReduceV2,
        hidden_dim=4096,
        *,
        device_group: dist.ProcessGroup,
        multicast=False,
    ):
        self.lease = lease
        self.policy = policy
        self.group = device_group
        self.rank = lease.rank
        self.world_size = lease.world_size
        self.device = lease.device
        self.sm_count = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.hidden_dim = hidden_dim
        self.wants_multicast = multicast
        config = (self.controller_count, self.sm_count, hidden_dim, multicast)
        configs = [None] * self.world_size
        dist.all_gather_object(configs, config, group=lease.group)
        if any(peer != config for peer in configs):
            raise ValueError(
                "Finalize AR workspace configuration must match across ranks"
            )
        if hidden_dim <= 0 or hidden_dim % 256:
            raise ValueError("Finalize push requires H divisible by 256")
        if self.slot_bytes < hidden_dim * 2:
            raise ValueError("Finalize push workspace holds less than one token")
        self.fallback_logged = False
        self.fusion_logged = False

    @property
    def slot_bytes(self) -> int:
        """The slot this finalize may fill (its lease's), which bounds the token counts."""
        return self.lease.request.push_slot_bytes

    @cached_property
    def _slot_stride_bytes(self) -> int:
        # the kernel addresses peers' slots by the allocated stride, which may be larger
        return self.lease.capacity.push_slot_bytes

    @cached_property
    def peers(self):
        return self.lease.data.push_slots

    @cached_property
    def counter(self) -> torch.Tensor:
        return self.lease.control.epoch

    @cached_property
    def multicast(self) -> bool:
        return self.wants_multicast and self.lease.has_multicast

    @cached_property
    def multicast_ptr(self) -> Optional[int]:
        return self.lease.data.push_mc

    @cached_property
    def one_shot_max_tokens(self) -> int:
        return min(384, self.slot_bytes // (self.hidden_dim * 2))

    @cached_property
    def two_shot_max_tokens(self) -> int:
        # A 2shot_push slot holds a partial half and a reduced half of an M x H / world slice.
        # 2-shot also needs every rank to own whole 8-column vectors of H.
        if self.hidden_dim % (8 * self.world_size):
            return 0
        return self.slot_bytes * self.world_size // (4 * self.hidden_dim)

    @cached_property
    def max_tokens(self) -> int:
        return max(self.one_shot_max_tokens, self.two_shot_max_tokens)

    def uses_one_shot_push(self, num_tokens: int) -> bool:
        return self.policy.uses_one_shot_push(num_tokens * self.hidden_dim * 2)

    def uses_two_shot_push(self, num_tokens: int) -> bool:
        """Fused 2-shot exactly where the standalone AR would pick 2shot_push."""
        return num_tokens <= self.two_shot_max_tokens and (
            self.policy.uses_two_shot_push(num_tokens * self.hidden_dim * 2)
        )

    def can_fuse(self, tp_group, num_tokens: int) -> bool:
        """Fuse where the standalone AR runs a push algorithm, respecting explicit overrides:
        2shot_push, or 1shot_push up to 384 tokens. No standalone AR thresholds or buffer
        allocations are changed.
        """
        if (
            torch.compiler.is_compiling()
            or self.policy is not tp_group.ca_comm
            or self.policy.disabled
            or num_tokens <= 0
        ):
            return False
        one_shot = num_tokens <= self.one_shot_max_tokens and self.uses_one_shot_push(
            num_tokens
        )
        if not (one_shot or self.uses_two_shot_push(num_tokens)):
            return False
        output = (
            self.peers[self.rank]
            .view(torch.bfloat16)
            .view(-1)[: num_tokens * self.hidden_dim]
            .view(num_tokens, self.hidden_dim)
        )
        return (
            not tp_group._can_use_flashinfer_allreduce(output)
            and tp_group._resolve_outplace_all_reduce_method(output) == "ca"
        )

    @property
    def controller_count(self) -> int:
        # The persistent grid's width: one co-resident CTA per SM. The phase is one
        # shared word, so the grid is not bound to any counter array.
        return self.sm_count

    def validate(self, output: torch.Tensor, cluster_size: int = 1) -> None:
        if (
            output.device != self.device
            or output.dtype != torch.bfloat16
            or output.ndim != 2
            or output.shape[1] != self.hidden_dim
            or output.shape[0]
            > (
                self.two_shot_max_tokens
                if self.uses_two_shot_push(output.shape[0])
                else self.one_shot_max_tokens
            )
            or not output.is_contiguous()
            or output.data_ptr() % 16
        ):
            raise ValueError("Unsupported output for finalize push workspace")
        if cluster_size not in (1, 2, 4, 8, 16) or cluster_size > self.sm_count:
            raise ValueError("Unsupported cluster size")

        if output.shape[0] == 0:
            raise ValueError("Finalize push requires M>0")
        if cluster_size > 1 and output.shape[0] * cluster_size > self.controller_count:
            raise ValueError(
                "Single-token clusters require M * C <= controller_count; use C=1 otherwise"
            )

    def arguments(self) -> tuple:
        import cutlass
        from cutlass.cute.runtime import make_ptr

        return (
            tuple(make_ptr(cutlass.Uint32, p.data_ptr()) for p in self.peers),
            make_ptr(cutlass.Uint32, self.counter.data_ptr()),
            cutlass.Int32(self.rank),
            cutlass.Int32(self._slot_stride_bytes // 4),
            make_ptr(cutlass.Uint32, self.peers[self.rank].data_ptr()),
            make_ptr(
                cutlass.Uint32, self.multicast_ptr or self.peers[self.rank].data_ptr()
            ),
        )
