from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.utils import get_int_env_var, is_flashinfer_available, round_up


logger = logging.getLogger(__name__)


class FlashinferDispatchOutput(NamedTuple):
    """Flashinfer EP dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: TopKOutput
    # Provide an output tensor to fused_moe so it writes directly to our buffer
    moe_output: Optional[torch.Tensor] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(FlashinferDispatchOutput, DispatchOutput)

if is_flashinfer_available():
    from flashinfer.comm.mnnvl import CommBackend

    class TorchDistributedCommBackend(CommBackend):
        """
        Use torch distributed instead of MPI to set up flashinfer MNNVL workspaces during initialization
        """

        def __init__(self, group: dist.ProcessGroup):
            self._group = group

        def Get_rank(self) -> int:
            return self._group.rank()

        def Get_size(self) -> int:
            return self._group.size()

        def allgather(self, data: int):
            gathered = [None] * self.Get_size()
            dist.all_gather_object(gathered, data, group=self._group)
            return gathered

        def Split(self, color: int, key: int):
            # No need to split, we already use the proper group
            return self

        def barrier(self):
            dist.barrier(group=self._group)


class FlashinferDispatcher(BaseDispatcher):
    """Main dispatcher class for Flashinfer A2A backend."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int = None,
        num_local_experts: int = None,  # Unused
        hidden_size: int = None,
        params_dtype: torch.dtype = None,  # Unused
    ):
        try:
            from flashinfer.comm.mapping import Mapping
            from flashinfer.comm.mnnvl import MnnvlConfig
            from flashinfer.comm.trtllm_moe_alltoall import (
                MoeAlltoAll,
                moe_a2a_get_workspace_size_per_rank,
            )
        except ImportError:
            raise ImportError(
                "Flashinfer is not installed or does not support A2A. "
                "Please install the appropriate version of Flashinfer."
            )
        self.group = group
        self.router_topk = router_topk
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # TODO: Can this be a server arg and shared with deepep/mooncakeep?
        self.max_num_tokens = (
            get_int_env_var("SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 1024)
            * self.group.size()
        )

        # Calculate workspace size
        total_dispatch_payload_size_per_token = (
            hidden_size // 2  # nvfp4 hidden states
            + hidden_size // 16  # fp8 scaling factors
            + self.router_topk * 4  # int32 topks ids
            + self.router_topk * 4  # float32 topk weights
        )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states
        self.workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.group.size(),
            max_num_tokens=self.max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )
        # Pad to required alignment for mnnvl memory
        self.workspace_size = round_up(self.workspace_size, 1 << 29)
        self.mapping = Mapping(
            rank=self.group.rank(),
            tp_size=self.group.size(),
            moe_ep_size=self.group.size(),
            world_size=self.group.size(),
            gpus_per_node=torch.cuda.device_count(),
            pp_size=1,
            cp_size=1,
        )
        self.moe_a2a = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=self.max_num_tokens,
            top_k=self.router_topk,
            num_experts=self.num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=MnnvlConfig(
                comm_backend=TorchDistributedCommBackend(self.group)
            ),
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        from flashinfer import fp4_quantize, nvfp4_block_scale_interleave

        global_scale = self.quant_config.get("input_global_scale", None)
        assert (
            global_scale is not None
        ), "input_global_scale is not set, use SGLANG_MOE_NVFP4_DISPATCH=1"

        output_dtype = hidden_states.dtype
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        # Handle case where there are no tokens on this DP worker
        # moe_a2a.dispatch requires at least one token
        self.has_dummy_token = False
        if hidden_states.shape[0] == 0:
            self.has_dummy_token = True
            hidden_states = torch.zeros(
                (1, hidden_states.shape[1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            topk_ids = torch.full(
                (1, topk_ids.shape[1]), -1, dtype=topk_ids.dtype, device=topk_ids.device
            )
            topk_weights = torch.zeros(
                (1, topk_weights.shape[1]),
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )

        x, x_sf = fp4_quantize(hidden_states, global_scale, is_sf_swizzled_layout=False)

        payloads = [x, x_sf, topk_ids, topk_weights]
        expert_id_payload_index = 2
        self.runtime_max_tokens_per_rank = (
            max(get_dp_global_num_tokens())
            if get_dp_global_num_tokens() is not None
            else x.shape[0]
        )
        recv_tensors = self.moe_a2a.dispatch(
            topk_ids,
            payloads,
            self.runtime_max_tokens_per_rank,
            invalid_token_expert_id=-1,  # Caution: Cutlass MoE uses num_slots as invalid token expert id
            expert_id_payload_index=expert_id_payload_index,
        )
        x_recv, x_sf_recv, topk_ids_recv, topk_weights_recv = recv_tensors
        x = x_recv.view(-1, x_recv.shape[-1])
        x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])
        # TODO: fuse interleave into cutlass moe
        x_sf = nvfp4_block_scale_interleave(x_sf)
        topk_ids = topk_ids_recv.view(-1, topk_ids_recv.shape[-1])
        topk_weights = topk_weights_recv.view(-1, topk_weights_recv.shape[-1])

        # Provide an output tensor to fused_moe so it writes directly to our buffer
        moe_output = None
        if get_moe_runner_backend().is_flashinfer_cutlass():
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                self.runtime_max_tokens_per_rank, self.hidden_size, output_dtype
            ).view(-1, self.hidden_size)
        return FlashinferDispatchOutput(
            x,
            x_sf,
            StandardTopKOutput(topk_weights, topk_ids, topk_output.router_logits),
            moe_output,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        hidden_states = combine_input.hidden_states
        output_hidden_size = hidden_states.shape[-1]
        hidden_states = self.moe_a2a.combine(
            hidden_states.view(
                self.group.size(), self.runtime_max_tokens_per_rank, output_hidden_size
            ),
            self.runtime_max_tokens_per_rank,
            payload_in_workspace=get_moe_runner_backend().is_flashinfer_cutlass(),
        )

        # Remove dummy token if it was added in dispatch
        if self.has_dummy_token:
            hidden_states = torch.empty(
                0,
                output_hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        del self.runtime_max_tokens_per_rank
        del self.has_dummy_token
        return hidden_states
