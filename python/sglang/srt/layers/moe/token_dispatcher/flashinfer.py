from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.token_dispatcher import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
    TorchDistributedCommBackend,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_int_env_var

try:
    from flashinfer import fp4_quantize, nvfp4_block_scale_interleave
    from flashinfer.comm import MoeAlltoAll, moe_a2a_get_workspace_size_per_rank
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig

    use_flashinfer = True
except ImportError:
    use_flashinfer = False

logger = logging.getLogger(__name__)

MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()


class FlashinferDispatchOutput(NamedTuple):
    """Flashinfer EP dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: StandardTopKOutput
    # Provide an output tensor to fused_moe so it writes directly to our buffer
    moe_output: Optional[torch.Tensor] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.FLASHINFER


assert isinstance(FlashinferDispatchOutput, DispatchOutput)


class FlashinferCombineInput(NamedTuple):
    """Flashinfer combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.FLASHINFER


assert isinstance(FlashinferCombineInput, CombineInput)


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
        super().__init__()
        if not use_flashinfer:
            raise ImportError(
                "Flashinfer is not installed or does not support A2A. "
                "Please install the appropriate version of Flashinfer."
            )

        self.ep_size = group.size()
        self.ep_rank = group.rank()
        self.router_topk = router_topk
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts

        # TODO: Can other moe runners use payload_in_workspace too?
        self.payload_in_workspace = get_moe_runner_backend().is_flashinfer_cutlass()

        # TODO: Can this be a server arg and shared with deepep/mooncakeep?
        self.max_num_tokens = (
            get_int_env_var("SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 1024)
            * self.ep_size
        )

        # Calculate workspace size. For eagle mode, use the larger workspace size since nextn layer will be unquantized.
        speculative_algo = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )
        if MOE_NVFP4_DISPATCH and not speculative_algo.is_eagle():
            total_dispatch_payload_size_per_token = (
                hidden_size // 2  # nvfp4 hidden states
                + hidden_size // 16  # fp8 scaling factors
                + self.router_topk * 4  # int32 topks ids
                + self.router_topk * 4  # float32 topk weights
            )
        else:
            total_dispatch_payload_size_per_token = (
                hidden_size * 2  # bf16 hidden states
                + self.router_topk * 4  # int32 topks ids
                + self.router_topk * 4  # float32 topk weights
            )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states
        self.workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.ep_size,
            max_num_tokens=self.max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )

        self.mapping = Mapping(
            rank=self.ep_rank,
            tp_size=self.ep_size,
            moe_ep_size=self.ep_size,
            world_size=self.ep_size,
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
            mnnvl_config=MnnvlConfig(comm_backend=TorchDistributedCommBackend(group)),
        )

        # Preallocate dummy tensors (to overcome numLocalTokens > 0 restriction)
        self.dummy_x = torch.empty(
            (1, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        # -1 will be ignored by flashinfer cutlass moe
        self.dummy_topk_ids = torch.full(
            (1, self.router_topk), -1, dtype=torch.int32, device="cuda"
        )
        # Hack for dispatch with dummy token - will route the dummy token to this rank so it doesn't require any transfer.
        self.dummy_topk_ids_current_rank = torch.full(
            (1, self.router_topk),
            self.ep_rank * self.num_local_experts,
            dtype=torch.int32,
            device="cuda",
        )
        self.dummy_topk_weights = torch.zeros(
            (1, self.router_topk), dtype=torch.float32, device="cuda"
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> FlashinferDispatchOutput:
        output_dtype = hidden_states.dtype
        x = hidden_states
        x_sf = None
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        # Handle case where there are no tokens on this DP worker
        # moe_a2a.dispatch requires at least one token
        self.has_dummy_token = False
        if x.shape[0] == 0:
            logger.warning("No tokens on this DP worker, using dummy token")
            self.has_dummy_token = True
            x = self.dummy_x
            topk_ids = self.dummy_topk_ids
            topk_weights = self.dummy_topk_weights

        global_scale = self.quant_config.get("input_global_scale", None)
        if global_scale is not None:
            if x.shape[0] > 0:
                x, x_sf = fp4_quantize(x, global_scale, is_sf_swizzled_layout=False)
            else:
                x = torch.zeros(
                    0, self.hidden_size // 2, dtype=torch.uint8, device=x.device
                )
                x_sf = torch.zeros(
                    0, self.hidden_size // 16, dtype=torch.uint8, device=x.device
                )

        payloads = []
        payloads.append(x)
        if x_sf is not None:
            payloads.append(x_sf)
            expert_id_payload_index = 2
        else:
            expert_id_payload_index = 1
        payloads.append(topk_ids)
        payloads.append(topk_weights)

        self.runtime_max_tokens_per_rank = (
            max(get_dp_global_num_tokens())
            if get_dp_global_num_tokens() is not None
            else x.shape[0]
        )
        recv_tensors = self.moe_a2a.dispatch(
            self.dummy_topk_ids_current_rank if self.has_dummy_token else topk_ids,
            payloads,
            self.runtime_max_tokens_per_rank,
            expert_id_payload_index=expert_id_payload_index,
        )
        if x_sf is not None:
            x_recv, x_sf_recv, topk_ids_recv, topk_weights_recv = recv_tensors
            x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])
            # TODO: fuse interleave into cutlass moe
            x_sf = nvfp4_block_scale_interleave(x_sf)
        else:
            x_recv, topk_ids_recv, topk_weights_recv = recv_tensors
        x = x_recv.view(-1, x_recv.shape[-1])
        topk_ids = topk_ids_recv.view(-1, topk_ids_recv.shape[-1])
        topk_weights = topk_weights_recv.view(-1, topk_weights_recv.shape[-1])

        # Provide an output tensor to fused_moe so it writes directly to our buffer
        moe_output = None
        if self.payload_in_workspace:
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                self.runtime_max_tokens_per_rank, self.hidden_size, output_dtype
            ).view(-1, self.hidden_size)
        return FlashinferDispatchOutput(
            x,
            x_sf,
            StandardTopKOutput(topk_weights, topk_ids, topk_output.router_logits),
            moe_output,
        )

    def combine(self, combine_input: FlashinferCombineInput) -> torch.Tensor:
        hidden_states = combine_input.hidden_states
        output_hidden_size = hidden_states.shape[-1]
        hidden_states = self.moe_a2a.combine(
            hidden_states.view(
                self.ep_size, self.runtime_max_tokens_per_rank, output_hidden_size
            ),
            self.runtime_max_tokens_per_rank,
            payload_in_workspace=self.payload_in_workspace,
        )

        # Remove dummy token if it was added in dispatch
        if self.has_dummy_token:
            hidden_states = hidden_states[1:, :]

        del self.runtime_max_tokens_per_rank
        del self.has_dummy_token
        return hidden_states
