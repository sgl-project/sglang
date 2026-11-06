"""Borrowed CustomAllReduceV2 push buffers for the fused Muse Spark MoE finalize + all-reduce.

The kernel is ``FusedPostExpertNormWeightedSumTRTLLM`` in
``sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce``.
"""

import torch
import torch.distributed as dist


class FinalizePushWorkspace:
    """Borrow TP push buffers before capture; dependency-ordered launches only."""

    def __init__(
        self,
        communicator,
        device_group: dist.ProcessGroup,
        hidden_dim=4096,
        *,
        multicast=False,
    ):
        self.communicator = communicator  # Keep the allocation owner alive.
        self.group = device_group
        self.rank = communicator.rank
        self.world_size = communicator.world_size
        self.device = communicator.device
        self.sm_count = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.hidden_dim = hidden_dim
        self.slot_bytes = communicator.max_push_size
        self.peers = communicator._push_workspaces
        self.counter = communicator._push_counter
        self.multicast = multicast
        self.multicast_ptr = communicator._push_multicast_ptr
        config = (
            self.controller_count,
            self.slot_bytes,
            self.sm_count,
            hidden_dim,
            multicast,
        )
        configs = [None] * self.world_size
        dist.all_gather_object(configs, config, group=communicator.group)
        if any(peer != config for peer in configs):
            raise ValueError(
                "Finalize AR workspace configuration must match across ranks"
            )
        if multicast and not communicator.has_multicast:
            raise ValueError(
                "Finalize multicast requires a multicast-capable push workspace"
            )
        if not 0 < self.controller_count <= self.sm_count:
            raise ValueError("Unsupported finalize AR controller budget")
        if hidden_dim <= 0 or hidden_dim % 256 or self.slot_bytes < hidden_dim * 2:
            raise ValueError(
                "Finalize push requires H divisible by 256 and sufficient buffer capacity"
            )
        self.one_shot_max_tokens = min(384, self.slot_bytes // (hidden_dim * 2))
        # A 2shot_push slot holds a partial half and a reduced half of an M x H / world slice.
        # 2-shot also needs every rank to own whole 8-column vectors of H.
        self.two_shot_max_tokens = (
            self.slot_bytes * self.world_size // (4 * hidden_dim)
            if hidden_dim % (8 * self.world_size) == 0
            else 0
        )
        self.max_tokens = max(self.one_shot_max_tokens, self.two_shot_max_tokens)
        self.fallback_logged = False
        self.fusion_logged = False

    def uses_one_shot_push(self, num_tokens: int) -> bool:
        return self.communicator.uses_one_shot_push(num_tokens * self.hidden_dim * 2)

    def uses_two_shot_push(self, num_tokens: int) -> bool:
        """Fused 2-shot exactly where the standalone AR would pick 2shot_push."""
        return num_tokens <= self.two_shot_max_tokens and (
            self.communicator.uses_two_shot_push(num_tokens * self.hidden_dim * 2)
        )

    def can_fuse(self, tp_group, num_tokens: int) -> bool:
        """Fuse where the standalone AR runs a push algorithm, respecting explicit overrides:
        2shot_push, or 1shot_push up to 384 tokens. No standalone AR thresholds or buffer
        allocations are changed.
        """
        if (
            torch.compiler.is_compiling()
            or self.communicator is not tp_group.ca_comm
            or self.communicator.disabled
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
        return self.counter.numel()

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
            cutlass.Int32(self.slot_bytes // 4),
            make_ptr(cutlass.Uint32, self.peers[self.rank].data_ptr()),
            make_ptr(
                cutlass.Uint32, self.multicast_ptr or self.peers[self.rank].data_ptr()
            ),
        )
