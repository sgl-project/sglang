from __future__ import annotations

from typing import Any, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting
from sglang.srt.hardware_backend.npu.moe.init_routing import NPUMoEInitRouting_v2
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import AscendOpsConfig
from sglang.srt.runtime_context import get_parallel


class AscendOpsDispatchOutput(NamedTuple):
    """Metadata passed from Ascend Ops dispatch to the MoE runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int
    prefill_mode: str
    is_shortcut: bool = False

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_OPS


class AscendOpsCombineInput(NamedTuple):
    """Expert output consumed by the matching Ascend Ops combine path."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_OPS


class AscendOpsDispatcher(BaseDispatcher):
    """Select one of the four Ascend Ops dispatch/combine execution paths.

    The methods below form the integration boundary for the future NPU
    operators.  Keeping path selection here makes the runner independent of
    whether a batch is prefill or decode and keeps the output contract stable.
    """

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        ascend_ops_config: Optional[AscendOpsConfig] = None,
    ) -> None:
        super().__init__()
        self.ascend_ops_config = ascend_ops_config or AscendOpsConfig()
        # Prefill communication is scoped to TP; decode routing is scoped to EP.
        self.tp_group = get_parallel().tp_group.device_group
        self.ep_group = get_parallel().moe_ep_group.device_group
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.top_k = moe_runner_config.top_k
        self.hidden_size = moe_runner_config.hidden_size
        self.quant_config: Optional[dict] = None
        self._dispatch_output: Optional[AscendOpsDispatchOutput] = None
        self._dispatch_path: Optional[str] = None
        self._dispatch_group = None
        self._alltoall_state: Optional[dict[str, Any]] = None
        self._init_routing = NPUMoEInitRouting_v2(quant_mode=-1)
        self._finalize_routing = NPUFinalizeRouting(drop_pad_mode=2)

    def set_quant_config(self, quant_config: dict) -> None:
        """Store quantization settings for the future routing kernels."""

        self.quant_config = quant_config

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        if get_is_extend_in_batch():
            if self.ascend_ops_config.prefill == "allgather":
                return self._dispatch_prefill_allgather(
                    hidden_states, topk_weights, topk_ids
                )
            return self._dispatch_prefill_alltoallv(
                hidden_states, topk_weights, topk_ids
            )

        if self.ascend_ops_config.decode == "low_latency":
            return self._dispatch_decode_low_latency(
                hidden_states, topk_weights, topk_ids
            )
        return self._dispatch_decode_fused_op(hidden_states, topk_weights, topk_ids)

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if combine_input.format != CombineInputFormat.ASCEND_OPS:
            raise TypeError(
                "AscendOpsDispatcher.combine expects an AscendOpsCombineInput, "
                f"got {combine_input.format.value}"
            )
        if self._dispatch_output is None or self._dispatch_path is None:
            raise RuntimeError("Ascend Ops combine called before dispatch")

        path = self._dispatch_path
        if path == "prefill_allgather":
            return self._combine_prefill_allgather(combine_input)
        if path == "prefill_alltoallv":
            return self._combine_prefill_alltoallv(combine_input)
        if path == "decode_low_latency":
            return self._combine_decode_low_latency(combine_input)
        if path == "decode_fused_op":
            return self._combine_decode_fused_op(combine_input)
        raise RuntimeError(f"unknown Ascend Ops dispatch path: {path}")

    # Dispatch path boundaries -------------------------------------------------

    def _dispatch_prefill_allgather(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "prefill_allgather"
        self._dispatch_group = self.tp_group
        raise NotImplementedError("Ascend Ops prefill allgather dispatch is pending")

    def _dispatch_prefill_alltoallv(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "prefill_alltoallv"
        self._dispatch_group = self.tp_group
        return self._dispatch_alltoallv(hidden_states, topk_weights, topk_ids)

    def _dispatch_decode_low_latency(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "decode_low_latency"
        self._dispatch_group = self.ep_group
        raise NotImplementedError("Ascend Ops decode low-latency dispatch is pending")

    def _dispatch_decode_fused_op(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "decode_fused_op"
        self._dispatch_group = self.ep_group
        raise NotImplementedError("Ascend Ops decode fused-op dispatch is pending")

    # Combine path boundaries --------------------------------------------------

    def _combine_prefill_allgather(self, combine_input):
        raise NotImplementedError("Ascend Ops prefill allgather combine is pending")

    def _combine_prefill_alltoallv(self, combine_input):
        return self._combine_alltoallv(combine_input.hidden_states)

    def _combine_decode_low_latency(self, combine_input):
        raise NotImplementedError("Ascend Ops decode low-latency combine is pending")

    def _combine_decode_fused_op(self, combine_input):
        raise NotImplementedError("Ascend Ops decode fused-op combine is pending")

    # Prefill alltoallv --------------------------------------------------------

    def _routing_quant_mode(self) -> int:
        dtype = (self.quant_config or {}).get("dispatcher_output_dtype", "bf16")
        dtype = getattr(dtype, "value", dtype)

        dtype = {
            "mxfp8": "mx_fp8_e4m3",
            "mxfp4": "mx_fp4_e2m1",
        }.get(dtype, dtype)
        modes = {
            "bf16": -1,
            "int8": 1,
            "mx_fp8_e4m3": 3,
            "mx_fp4_e2m1": 9,
        }
        if dtype not in modes:
            raise NotImplementedError(
                "Ascend Ops alltoallv currently supports dispatcher output "
                f"dtypes {tuple(modes)}, got {dtype!r}"
            )
        return modes[dtype]

    def _alltoall_counts(
        self, topk_ids: torch.Tensor, group
    ) -> tuple[list[int], list[int], torch.Tensor]:
        if self.num_experts is None or self.num_local_experts is None:
            raise ValueError("Ascend Ops alltoallv requires expert counts")
        ep_size = dist.get_world_size(group=group)
        if self.num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by EP size={ep_size}"
            )
        local_experts = self.num_experts // ep_size
        if self.num_local_experts != local_experts:
            raise ValueError(
                "Ascend Ops alltoallv requires num_local_experts to match "
                f"num_experts / ep_size ({self.num_local_experts} != {local_experts})"
            )

        local_counts = torch.histc(
            topk_ids.reshape(-1).to(torch.float32),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)
        gathered_counts = torch.empty(
            ep_size * self.num_experts,
            dtype=local_counts.dtype,
            device=local_counts.device,
        )
        if ep_size == 1:
            gathered_counts.copy_(local_counts)
        else:
            dist.all_gather_into_tensor(gathered_counts, local_counts, group=group)
        gathered_counts = gathered_counts.view(ep_size, self.num_experts)
        rank = dist.get_rank(group=group)
        local_start = rank * local_experts
        local_end = local_start + local_experts
        recv_counts = gathered_counts[:, local_start:local_end]
        input_splits = (
            local_counts.view(ep_size, local_experts).sum(dim=1).cpu().tolist()
        )
        output_splits = recv_counts.sum(dim=1).cpu().tolist()
        return input_splits, output_splits, recv_counts

    def _alltoall_single(
        self,
        hidden_states: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group,
    ) -> torch.Tensor:
        output = hidden_states.new_empty(
            (sum(output_splits),) + tuple(hidden_states.shape[1:])
        )
        dist.all_to_all_single(
            output,
            hidden_states.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return output

    def _dispatch_alltoallv(self, hidden_states, topk_weights, topk_ids):
        topk_ids = topk_ids.to(torch.int32)
        group = self._dispatch_group
        input_splits, output_splits, recv_counts = self._alltoall_counts(
            topk_ids, group
        )

        quant_mode = self._routing_quant_mode()
        permuted, expanded_row_idx, _, hidden_states_scale = self._init_routing_tokens(
            hidden_states, topk_ids, self.num_experts, quant_mode
        )

        scale_for_comm = hidden_states_scale
        if quant_mode not in (-1, 1) and scale_for_comm is not None:
            scale_for_comm = scale_for_comm.view(torch.uint8)
        received_scale = (
            self._alltoall_single(scale_for_comm, output_splits, input_splits, group)
            if scale_for_comm is not None
            else None
        )

        received = self._alltoall_single(permuted, output_splits, input_splits, group)
        local_counts = recv_counts.reshape(-1)
        ep_size = recv_counts.shape[0]
        local_ids = torch.arange(
            self.num_local_experts, dtype=torch.int32, device=received.device
        ).repeat(ep_size, 1)
        local_ids = torch.repeat_interleave(local_ids.reshape(-1), local_counts)
        local_ids = local_ids.view(-1, 1)
        if quant_mode not in (-1, 1) and received_scale is not None:
            received_scale = received_scale.reshape(received_scale.shape[0], -1, 2)
        if self.num_local_experts > 1 and hidden_states.shape[0] > 0:
            local_permuted, local_expanded_row_idx, _, local_scale = (
                self._init_routing_tokens(
                    received,
                    local_ids,
                    self.num_local_experts,
                    quant_mode,
                    scale=received_scale,
                    reuse_scale=quant_mode != -1,
                )
            )
        else:
            # With one local expert the alltoall result is already in expert
            # order; the reference strategy skips the second routing pass.
            local_permuted = received
            local_expanded_row_idx = torch.arange(
                received.shape[0], dtype=torch.int32, device=received.device
            )
            local_scale = received_scale

        local_expert_tokens = recv_counts.sum(dim=0).to(torch.int64)
        if quant_mode != -1 and self.num_local_experts > 1:
            local_permuted = local_permuted.view(self._routing_quant_dtype(quant_mode))
        local_topk_ids = local_ids
        local_topk_weights = torch.ones(
            (local_ids.shape[0], 1), dtype=topk_weights.dtype, device=topk_weights.device
        )
        self._alltoall_state = {
            "group": group,
            "input_splits": input_splits,
            "output_splits": output_splits,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "expanded_row_idx": expanded_row_idx.clone(),
            "local_expanded_row_idx": local_expanded_row_idx.clone(),
            "local_topk_ids": local_topk_ids,
            "local_topk_weights": local_topk_weights,
            "hidden_shape": hidden_states.shape,
        }
        self._dispatch_output = AscendOpsDispatchOutput(
            hidden_states=local_permuted,
            hidden_states_scale=local_scale,
            topk_weights=local_topk_weights,
            topk_ids=local_topk_ids,
            expanded_row_idx=local_expanded_row_idx,
            expert_tokens=local_expert_tokens,
            group_list_type=1,
            prefill_mode="alltoallv",
        )
        return self._dispatch_output

    def _init_routing_tokens(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_mode: int,
        scale: Optional[torch.Tensor] = None,
        reuse_scale: bool = False,
    ):
        if quant_mode == -1:
            return NPUMoEInitRouting_v2(quant_mode=-1)._init_routing(
                hidden_states, topk_ids, num_experts, topk_ids.shape[1]
            )

        import torch_npu

        quant_dtype = self._routing_quant_dtype(quant_mode)
        if reuse_scale:
            # This is the second routing pass in the reference alltoall
            # strategy.  The scale is supplied by the first pass after the
            # communication and quant_mode is intentionally omitted.
            if scale is None:
                raise ValueError("quantized local routing requires a scale")
            return torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                scale=scale,
                expert_num=num_experts,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                x_dtype=quant_dtype,
            )

        return torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            quant_mode=quant_mode,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            active_expert_range=[0, num_experts],
        )

    @staticmethod
    def _routing_quant_dtype(quant_mode: int) -> torch.dtype:
        return {
            1: torch.int8,
            3: torch.float8_e4m3fn,
            9: torch.float4_e2m1fn_x2,
        }[quant_mode]

    @staticmethod
    def _synchronize_npu(tensor: torch.Tensor) -> None:
        """Wait for NPU communication/routing writes before consuming metadata."""
        if tensor.device.type == "npu":
            torch.npu.synchronize(tensor.device)

    def _combine_alltoallv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self._alltoall_state
        if state is None:
            raise RuntimeError("Ascend Ops alltoallv combine called before dispatch")

        if self.num_local_experts > 1 and hidden_states.shape[0] > 0:
            local_output = self._finalize_routing_no_export(
                hidden_states,
                None,
                state["local_expanded_row_idx"],
            )
        else:
            local_output = hidden_states
        source_order = self._alltoall_single(
            local_output,
            state["input_splits"],
            state["output_splits"],
            state["group"],
        )
        output = self._finalize_routing_no_export(
            source_order,
            state["topk_weights"],
            state["expanded_row_idx"],
        ).view(state["hidden_shape"])

        self._alltoall_state = None
        self._dispatch_output = None
        return output

    @staticmethod
    def _finalize_routing_no_export(
        hidden_states: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        expanded_row_idx: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0 and expanded_row_idx.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(
                "Ascend Ops finalize metadata mismatch: "
                f"hidden_states has {hidden_states.shape[0]} rows but "
                f"expanded_row_idx has {expanded_row_idx.shape[0]} rows"
            )
        hidden_states = hidden_states.contiguous()
        expanded_row_idx = expanded_row_idx.to(torch.int32).contiguous()
        if topk_weights is not None:
            topk_weights = topk_weights.contiguous()
        return torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )
