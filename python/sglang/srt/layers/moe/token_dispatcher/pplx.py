from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_dp_size

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput

try:
    from pplx_kernels import AllToAll, nvshmem_init

    use_pplx = True
except ImportError:
    use_pplx = False

class PPLXDispatchOutput(NamedTuple):
    """PPLX dispatch output.
    Contains expert-grouped hidden states after all-to-all dispatch.
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    expert_num_tokens: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.PPLX


assert isinstance(PPLXDispatchOutput, DispatchOutput)


class PPLXCombineInput(NamedTuple):
    """PPLX Combine Input.
    Contains expert outputs after MoE computation, ready for a2a combine.
    """

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.PPLX


assert isinstance(PPLXCombineInput, CombineInput)


class PPLXDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        experts_per_token: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
        block_size: int = 128,
        device_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()

        if not use_pplx:
            raise ImportError(
                "PPLX is not installed. Please install PPLX package from "
                "https://github.com/perplexityai/pplx-kernels."
            )

        # group: used for PPLX registration (should support CPU for internal sync)
        # device_group: used for NVSHMEM and CUDA operations (NCCL backend)
        self.group = group
        self.device_group = device_group if device_group is not None else group
        
        self.max_num_tokens = envs.SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.hidden_dim = hidden_dim
        self.hidden_dim_bytes = hidden_dim_bytes
        self.hidden_dim_scale_bytes = hidden_dim_scale_bytes
        # Get internode from environment variable (True for multi-node, False for single-node)
        self.internode = envs.SGLANG_PPLX_INTERNODE.get()
        # Get dp_size from distributed config (same pattern as get_attention_dp_size)
        self.dp_size = get_attention_dp_size()
        self.block_size = block_size

        # Use device_group for rank/world_size since that's for CUDA operations
        self.rank = dist.get_rank(self.device_group)
        self.world_size = dist.get_world_size(self.device_group)
        self.num_dp = self.world_size // self.dp_size
        
        self.num_local_experts = num_experts // self.world_size

        # Initialize NVSHMEM before creating AllToAll
        from cuda.core.experimental import Device
        dev = Device(torch.cuda.current_device())
        dev.set_current()
        
        nvshmem_init(
            global_rank=self.rank,
            local_rank=torch.cuda.current_device(),
            world_size=self.world_size,
            device=dev,
        )

        # Get or register the process group name for PPLX to resolve
        # PPLX needs a group that supports CPU operations for internal sync
        from torch.distributed.distributed_c10d import (
            _get_process_group_name,
            _register_process_group,
        )
        
        try:
            self.group_name = _get_process_group_name(self.group)
        except Exception:
            self.group_name = "pplx_default"
            _register_process_group(self.group_name, self.group)

        if self.internode:
            self.ata = AllToAll.internode(
                max_num_tokens=self.max_num_tokens,
                num_experts=self.num_experts,
                experts_per_token=self.experts_per_token,
                rank=self.rank,
                world_size=self.world_size,
                dp_size=self.dp_size,
                hidden_dim=self.hidden_dim,
                hidden_dim_bytes=self.hidden_dim_bytes,
                hidden_dim_scale_bytes=self.hidden_dim_scale_bytes,
            )
        else:
            self.ata = AllToAll.intranode(
                max_num_tokens=self.max_num_tokens,
                num_experts=self.num_experts,
                experts_per_token=self.experts_per_token,
                rank=self.rank,
                world_size=self.world_size,
                dp_size=self.dp_size,
                hidden_dim=self.hidden_dim,
                hidden_dim_bytes=self.hidden_dim_bytes,
                hidden_dim_scale_bytes=self.hidden_dim_scale_bytes,
                group_name=self.group_name,
            )
        
        self._dispatch_indices: Optional[torch.Tensor] = None
        self._dispatch_weights: Optional[torch.Tensor] = None
        self._bound_m: Optional[torch.Tensor] = None
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> PPLXDispatchOutput:
        """Dispatch tokens to experts using pplx a2a.

        Args:
            hidden_states: Input hidden states, shape (num_tokens, hidden_dim)
            topk_output: TopK routing output containing topk_ids and topk_weights

        Returns:
            PPLXDispatchOutput with expert-grouped hidden states after a2a dispatch.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype

        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        # allocate output buffers
        # shape: (num_local_experts, max_tokens_per_expert, hidden_dim)
        max_tokens_per_expert = self.max_num_tokens * self.num_dp
        expert_x = torch.empty(
            (self.num_local_experts, max_tokens_per_expert, self.hidden_dim),
            dtype=dtype,
            device=device,
        )
        expert_num_tokens = torch.empty(
            self.num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        expert_x_scale: Optional[torch.Tensor] = None
        if self.hidden_dim_scale_bytes > 0:
            # fp8 quantization - create scale tensors
            scale_dim = (self.hidden_dim + self.block_size - 1) // self.block_size
            expert_x_scale = torch.empty(
                (
                    self.num_local_experts,
                    max_tokens_per_expert,
                    scale_dim,
                ),
                dtype=torch.float32,
                device=device,
            )

        indices_uint32 = topk_ids.to(torch.uint32)
        bound_m = torch.tensor(
            [num_tokens],
            dtype=torch.uint32,
            device=device
        )

        self.ata.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=hidden_states,
            dp_x_scale=None, # TODO: handle fp8 scales if needed
            indices=indices_uint32,
            bound_m=bound_m,
        )

        self._dispatch_indices = indices_uint32
        self._dispatch_weights = topk_weights
        self._bound_m = bound_m

        return PPLXDispatchOutput(
            hidden_states=expert_x,
            hidden_states_scale=expert_x_scale,
            expert_num_tokens=expert_num_tokens,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
    def combine(
        self,
        combine_input: PPLXCombineInput,
    ) -> torch.Tensor:
        """Combine expert outputs using pplx a2a.

        Args:
            combine_input: PPLXCombineInput containing expert outputs and routing info

        Returns:
            torch.Tensor: Combined hidden states after a2a combine.
        """
        hidden_states = combine_input.hidden_states
        topk_ids = combine_input.topk_ids
        topk_weights = combine_input.topk_weights

        device = hidden_states.device
        dtype = hidden_states.dtype

        out_tokens = torch.empty(
            (self.max_num_tokens, self.hidden_dim),
            dtype=dtype,
            device=device,
        )

        indices_uint32 = topk_ids.to(torch.uint32)

        self.ata.combine(
            out_tokens=out_tokens,
            indices=indices_uint32,
            weights=topk_weights,
            expert_y=hidden_states,
            bound_m=self._bound_m,
        )

        num_tokens = int(self._bound_m[0].item())

        self._dispatch_indices = None
        self._dispatch_weights = None
        self._bound_m = None

        return out_tokens[:num_tokens]

    def destroy(self) -> None:
        if self.ata is not None:
            self.ata.destroy()
            self.ata = None
