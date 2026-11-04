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
        self.max_tokens = min(384, self.slot_bytes // (hidden_dim * 2))
        self.fallback_logged = False
        self.fusion_logged = False

    def uses_one_shot_push(self, num_tokens: int) -> bool:
        return self.communicator.uses_one_shot_push(num_tokens * self.hidden_dim * 2)

    def can_fuse(self, tp_group, num_tokens: int) -> bool:
        """Use custom AR storage while respecting explicit algorithm overrides.

        Fusion has a different crossover from a standalone AR launch. Allow
        push through 384 tokens when its allocated slot fits, even if ordinary
        CUDA-graph AR would select pull above its 288-token crossover (H=4096),
        but not where the standalone AR would pick 2shot_push.
        No standalone AR thresholds or buffer allocations are changed.
        """
        if (
            torch.compiler.is_compiling()
            or self.communicator is not tp_group.ca_comm
            or self.communicator.disabled
            or not 0 < num_tokens <= self.max_tokens
        ):
            return False
        if self.world_size > 4 and not self.uses_one_shot_push(num_tokens):
            return False
        if self.communicator.uses_two_shot_push(num_tokens * self.hidden_dim * 2):
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
            and (
                self.communicator.override_algo is None
                or self.uses_one_shot_push(num_tokens)
            )
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
            or output.shape[0] > self.max_tokens
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
