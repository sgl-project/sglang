"""Ascend Ops MoE communication dispatcher.

Expert regrouping is handled by the Ascend runner hooks. This module owns the
AllGather, AllToAllV, and MC2 communication paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe.routing import (
    INT8_QUANT_MODE,
    MXFP8_QUANT_MODE,
    ROUTING_QUANT_MODES,
    normalize_mxfp_scale,
    npu_moe_finalize_routing,
    npu_moe_init_routing,
)
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import AscendOpsConfig, DispatcherOutputDtype
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu

if is_npu():
    import torch_npu

# Routing and MC2 use different quant-mode enums.
_MC2_QUANT_MODE = {
    DispatcherOutputDtype.BF16: 0,
    DispatcherOutputDtype.INT8: 2,
    DispatcherOutputDtype.MXFP8: 4,
}

_QUANT_MODES = {
    "routing": ROUTING_QUANT_MODES,
    "mc2": _MC2_QUANT_MODE,
}

# MC2 expects an ATen ScalarType integer for ``y_dtype``.
_TORCH_DTYPE_TO_ATEN = {
    torch.int8: 1,
    torch.int32: 3,
    torch.float16: 5,
    torch.float32: 6,
    torch.bfloat16: 15,
    torch.bits8: 21,
    torch.float8_e5m2: 23,
    torch.float8_e4m3fn: 24,
}


class AscendOpsDispatchOutput(NamedTuple):
    """Dispatcher output shared by the three Ascend execution paths."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    expert_tokens: Optional[torch.Tensor]
    group_list_type: int
    exec_mode: str
    forward_state: AscendOpsForwardState
    quant_mode: int = -1
    recv_counts: Optional[torch.Tensor] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_OPS


class AscendOpsCombineInput(NamedTuple):
    """Expert output consumed by the matching Ascend Ops combine path."""

    hidden_states: torch.Tensor
    forward_state: AscendOpsForwardState

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_OPS


@dataclass
class _MC2State:
    """Per-forward metadata required by MC2 combine."""

    num_tokens: int
    assist_info_for_combine: torch.Tensor
    ep_send_counts: torch.Tensor
    x_active_mask: Optional[torch.Tensor]
    expand_scales: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class _AllGatherState:
    group: Any
    row_sizes: list[int]
    input_dtype: torch.dtype
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None
    expanded_row_idx: Optional[torch.Tensor] = None
    expert_tokens: Optional[torch.Tensor] = None
    num_routes: Optional[int] = None


@dataclass
class _AllToAllVState:
    group: Any
    input_splits: list[int]
    output_splits: list[int]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    hidden_shape: torch.Size
    local_expanded_row_idx: Optional[torch.Tensor] = None


AscendOpsForwardState = _MC2State | _AllGatherState | _AllToAllVState


class AscendOpsDispatcher(BaseDispatcher):
    """Select and execute an Ascend dispatch/combine path."""

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        moe_ascend_ops_config: Optional[AscendOpsConfig] = None,
    ) -> None:
        super().__init__()
        self.moe_ascend_ops_config = moe_ascend_ops_config or AscendOpsConfig()
        parallel = get_parallel()
        self.tp_group = parallel.tp_group.device_group
        self.ep_coordinator = parallel.moe_ep_group
        self.ep_group = self.ep_coordinator.device_group
        self.ep_size = self.ep_coordinator.world_size
        self.ep_rank = self.ep_coordinator.rank_in_group
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.top_k = moe_runner_config.top_k
        self.hidden_size = moe_runner_config.hidden_size
        self.quant_config: Optional[dict] = None
        self._ep_comm_name: Optional[str] = None
        self._pad_buffers: dict[tuple, tuple] = {}
        if self.moe_ascend_ops_config.prefill == "allgather":
            if not (
                parallel.tp_size == parallel.moe_ep_size
                and self.num_local_experts * self.ep_size == self.num_experts
                and not moe_runner_config.num_fused_shared_experts
            ):
                raise ValueError(
                    "NPU AllGather requires TP=EP, contiguous expert placement "
                    "and no fused shared experts"
                )

    def set_quant_config(self, quant_config: dict) -> None:
        """Store quantization settings used by the routing kernels."""

        self.quant_config = quant_config

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        if get_is_extend_in_batch():
            if self.moe_ascend_ops_config.prefill == "allgather":
                return self._dispatch_allgather(
                    hidden_states, topk_weights, topk_ids, self.tp_group
                )
            return self._dispatch_alltoallv(
                hidden_states, topk_weights, topk_ids, self.tp_group
            )

        return self._dispatch_low_latency(hidden_states, topk_weights, topk_ids)

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if combine_input.format != CombineInputFormat.ASCEND_OPS:
            raise TypeError(
                "AscendOpsDispatcher.combine expects an AscendOpsCombineInput, "
                f"got {combine_input.format.value}"
            )
        state = combine_input.forward_state
        if isinstance(state, _AllGatherState):
            return self._combine_allgather(combine_input.hidden_states, state)
        if isinstance(state, _AllToAllVState):
            return self._combine_alltoallv(combine_input.hidden_states, state)
        if isinstance(state, _MC2State):
            return self._combine_mc2(combine_input.hidden_states, state)
        raise TypeError(f"unknown Ascend Ops forward state: {type(state)!r}")

    # Decode (MC2 low latency)

    def _dispatch_low_latency(self, hidden_states, topk_weights, topk_ids):
        num_tokens = hidden_states.shape[0]
        pad_bs = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        if pad_bs and num_tokens > pad_bs:
            raise ValueError(
                f"ascend_ops decode got {num_tokens} tokens on this rank but "
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK limits the "
                f"padded batch to {pad_bs}; "
                "raise it or lower the decode batch size."
            )
        return self._dispatch_mc2(
            hidden_states,
            topk_ids,
            topk_weights,
            padded_bs=pad_bs or num_tokens,
            dtype_mode="low_latency",
        )

    def _dispatch_mc2(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        padded_bs: int,
        dtype_mode: str,
    ) -> AscendOpsDispatchOutput:
        """Dispatch one decode batch through MC2."""
        num_tokens = hidden_states.shape[0]
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.float()
        quant_mode = self._quant_mode("mc2", dtype_mode)

        x_active_mask = None
        if padded_bs != num_tokens:
            hidden_states, topk_ids, topk_weights, x_active_mask = self._pad_batch(
                hidden_states, topk_ids, topk_weights, num_tokens, padded_bs
            )

        mxfp8_args = (
            {"y_dtype": _TORCH_DTYPE_TO_ATEN[torch.float8_e4m3fn]}
            if quant_mode == _MC2_QUANT_MODE[DispatcherOutputDtype.MXFP8]
            else {}
        )

        (
            expand_x,
            dynamic_scales,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            _tp_recv_counts,
            _expand_scales,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            hidden_states,
            topk_ids,
            self._get_ep_comm_name(),
            self.ep_size,
            self.ep_rank,
            self.num_experts,
            x_active_mask=x_active_mask,
            quant_mode=quant_mode,
            global_bs=0,
            expert_token_nums_type=1,
            comm_alg=self.moe_ascend_ops_config.ll_dispatch_alg,
            **mxfp8_args,
        )

        if quant_mode == _MC2_QUANT_MODE[DispatcherOutputDtype.MXFP8]:
            if hidden_states.shape[1] % 64 != 0:
                raise ValueError(
                    "MC2 MXFP8 decode requires hidden_size to be a multiple "
                    f"of 64, got {hidden_states.shape[1]}"
                )
            # Convert the flat e8m0 scale to the runner's pair-split layout.
            dynamic_scales = dynamic_scales.reshape(
                dynamic_scales.shape[0], hidden_states.shape[1] // 64, 2
            )

        expand_scales = None

        state = _MC2State(
            num_tokens=num_tokens,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_recv_counts,
            x_active_mask=x_active_mask,
            expand_scales=expand_scales,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        return AscendOpsDispatchOutput(
            hidden_states=expand_x,
            hidden_states_scale=(
                dynamic_scales
                if quant_mode != _MC2_QUANT_MODE[DispatcherOutputDtype.BF16]
                else None
            ),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_tokens=expert_token_nums,
            group_list_type=1,
            exec_mode="mc2",
            forward_state=state,
        )

    def _pad_batch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        padded_bs: int,
    ):
        """Copy a possibly shorter batch into reusable padded buffers."""
        key = (padded_bs, hidden_states.shape[1], topk_ids.shape[1])
        bufs = self._pad_buffers.get(key)
        if bufs is None:
            bufs = (
                hidden_states.new_zeros((padded_bs, hidden_states.shape[1])),
                topk_ids.new_zeros((padded_bs, topk_ids.shape[1])),
                topk_weights.new_zeros((padded_bs, topk_weights.shape[1])),
                torch.zeros(padded_bs, dtype=torch.bool, device=hidden_states.device),
                0,
            )
        padded_x, padded_ids, padded_weights, mask, filled = bufs

        padded_x[:num_tokens] = hidden_states
        padded_ids[:num_tokens] = topk_ids
        padded_weights[:num_tokens] = topk_weights

        mask[:num_tokens] = True
        if filled > num_tokens:
            padded_x[num_tokens:filled] = 0
            padded_ids[num_tokens:filled] = 0
            padded_weights[num_tokens:filled] = 0
            mask[num_tokens:filled] = False

        self._pad_buffers[key] = (
            padded_x,
            padded_ids,
            padded_weights,
            mask,
            num_tokens,
        )
        return padded_x, padded_ids, padded_weights, mask

    def _combine_mc2(
        self, hidden_states: torch.Tensor, state: _MC2State
    ) -> torch.Tensor:
        """Run ``npu_moe_distribute_combine_v2`` for the batch dispatched last."""
        combined = torch_npu.npu_moe_distribute_combine_v2(
            hidden_states,
            state.topk_ids,
            state.assist_info_for_combine,
            state.ep_send_counts,
            state.topk_weights,
            self._get_ep_comm_name(),
            self.ep_size,
            self.ep_rank,
            self.num_experts,
            x_active_mask=state.x_active_mask,
            expand_scales=state.expand_scales,
            global_bs=0,
            comm_quant_mode=0,
            comm_alg=self.moe_ascend_ops_config.ll_combine_alg,
        )

        return combined[: state.num_tokens]

    def _quant_mode(self, op: str, dtype_mode: Optional[str] = None) -> int:
        """Quant mode op ``"routing"`` or ``"mc2"`` uses for the wire dtype."""
        table = _QUANT_MODES.get(op)
        if table is None:
            raise ValueError(
                f"unknown Ascend Ops quant-mode op {op!r}; "
                f"expected one of {tuple(_QUANT_MODES)}"
            )
        dtype = self._resolve_dispatch_dtype(dtype_mode)
        try:
            return table[dtype]
        except KeyError:
            raise NotImplementedError(
                f"ascend_ops {op} dispatch does not support dispatcher output "
                f"dtype {dtype.value}; supported: "
                f"{tuple(d.value for d in table)}"
            ) from None

    def _resolve_dispatch_dtype(
        self, dtype_mode: Optional[str] = None
    ) -> DispatcherOutputDtype:
        """Resolve a mode-specific dtype, falling back to the shared dtype."""
        quant_config = self.quant_config or {}
        raw = None
        if dtype_mode is not None:
            raw = quant_config.get(f"{dtype_mode}_dispatcher_output_dtype")
        if raw is None:
            raw = quant_config.get("dispatcher_output_dtype")
        if raw is None:
            return DispatcherOutputDtype.BF16
        return DispatcherOutputDtype(raw)

    def _get_ep_comm_name(self) -> str:
        if self._ep_comm_name is None:
            backend = self.ep_group._get_backend(torch.device("npu"))
            if hasattr(backend, "get_hccl_comm_name"):
                name = backend.get_hccl_comm_name(self.ep_rank)
            else:
                name = self.ep_group.get_hccl_comm_name(self.ep_rank)
            self._ep_comm_name = name
        return self._ep_comm_name

    # Prefill: AllGather

    @staticmethod
    def _allgather_row_sizes(local_rows: int, group, device) -> list[int]:
        """Exchange the actual token rows contributed by every TP rank."""

        world_size = dist.get_world_size(group)
        local_rows_tensor = torch.tensor([local_rows], dtype=torch.int64, device=device)
        gathered_rows = [torch.empty_like(local_rows_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_rows, local_rows_tensor, group=group)
        return [int(rows.item()) for rows in gathered_rows]

    def _dispatch_allgather(self, hidden_states, topk_weights, topk_ids, group):
        """Gather the full batch on every TP rank."""
        input_dtype = hidden_states.dtype
        quant_mode = self._quant_mode("routing")
        if quant_mode not in (-1, INT8_QUANT_MODE, MXFP8_QUANT_MODE):
            raise NotImplementedError(
                "NPU AllGather supports BF16, INT8 and MXFP8 routing"
            )
        # CP and DP can split a batch unevenly (for example with interleaved or
        # zigzag schedules).  Deriving row counts from DP metadata assumes an
        # equal split and becomes incorrect as soon as CP is enabled.  Exchange
        # the actual local row count so the subsequent padded all-gather and
        # reduce-scatter use the exact layout produced by the attention path.
        row_sizes = self._allgather_row_sizes(
            hidden_states.shape[0], group, hidden_states.device
        )
        world_size = len(row_sizes)
        max_rows = max(row_sizes)
        if max_rows == 0:
            raise RuntimeError("NPU prefill requires at least one global token")

        equal_rows = min(row_sizes) == max_rows

        def gather_rows(tensor):
            if equal_rows:
                padded = tensor.contiguous()
            else:
                padded = tensor.new_zeros((max_rows, *tensor.shape[1:]))
                padded[: tensor.shape[0]].copy_(tensor)
            gathered = tensor.new_empty((world_size * max_rows, *tensor.shape[1:]))
            dist.all_gather_into_tensor(gathered, padded, group=group)
            if equal_rows:
                return gathered
            return torch.cat(
                [
                    gathered[rank * max_rows : rank * max_rows + rows]
                    for rank, rows in enumerate(row_sizes)
                ],
                dim=0,
            )

        scales = None
        if quant_mode == MXFP8_QUANT_MODE:
            if hidden_states.shape[0] > 0:
                hidden_states, scales = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                scales = normalize_mxfp_scale(scales).view(torch.uint8)
            else:
                hidden_states = hidden_states.new_empty(
                    (0, self.hidden_size), dtype=torch.float8_e4m3fn
                )
                scales = hidden_states.new_empty(
                    (0, (self.hidden_size + 63) // 64, 2), dtype=torch.uint8
                )

        state = _AllGatherState(
            group=group,
            row_sizes=row_sizes,
            input_dtype=input_dtype,
        )
        return AscendOpsDispatchOutput(
            hidden_states=gather_rows(hidden_states),
            hidden_states_scale=gather_rows(scales) if scales is not None else None,
            topk_weights=gather_rows(topk_weights.float()),
            topk_ids=gather_rows(topk_ids.to(torch.int32)),
            expert_tokens=None,
            group_list_type=1,
            exec_mode="allgather",
            quant_mode=quant_mode,
            forward_state=state,
        )

    def _combine_allgather(
        self, contributions: torch.Tensor, state: _AllGatherState
    ) -> torch.Tensor:
        """ReduceScatter contributions back to the local shard."""
        sizes = state.row_sizes
        group = state.group
        max_rows = max(sizes)
        if min(sizes) == max_rows:
            padded = contributions
        else:
            padded = contributions.new_zeros((len(sizes) * max_rows, self.hidden_size))
            offset = 0
            for rank, rows in enumerate(sizes):
                padded[rank * max_rows : rank * max_rows + rows].copy_(
                    contributions[offset : offset + rows]
                )
                offset += rows
        output = contributions.new_empty((max_rows, self.hidden_size))
        dist.reduce_scatter_tensor(output, padded, group=group)
        output = output[: sizes[dist.get_rank(group)]].to(state.input_dtype)
        return output

    # Prefill: AllToAllV

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

    def _dispatch_alltoallv(self, hidden_states, topk_weights, topk_ids, group):
        """Send each token to its expert-owning rank."""
        topk_ids = topk_ids.to(torch.int32)
        input_splits, output_splits, recv_counts = self._alltoall_counts(
            topk_ids, group
        )

        quant_mode = self._quant_mode("routing")
        routed = npu_moe_init_routing(
            hidden_states, topk_ids, self.num_experts, quant_mode=quant_mode
        )

        scale_for_comm = routed.scale
        if quant_mode not in (-1, INT8_QUANT_MODE) and scale_for_comm is not None:
            scale_for_comm = scale_for_comm.view(torch.uint8)
        received_scale = (
            self._alltoall_single(scale_for_comm, output_splits, input_splits, group)
            if scale_for_comm is not None
            else None
        )
        received = self._alltoall_single(
            routed.hidden_states, output_splits, input_splits, group
        )

        state = _AllToAllVState(
            group=group,
            input_splits=input_splits,
            output_splits=output_splits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expanded_row_idx=routed.expanded_row_idx.clone(),
            hidden_shape=hidden_states.shape,
        )
        return AscendOpsDispatchOutput(
            hidden_states=received,
            hidden_states_scale=received_scale,
            topk_weights=None,
            topk_ids=None,
            expert_tokens=None,
            group_list_type=1,
            exec_mode="alltoallv",
            quant_mode=quant_mode,
            recv_counts=recv_counts,
            forward_state=state,
        )

    def _combine_alltoallv(
        self, local_output: torch.Tensor, state: _AllToAllVState
    ) -> torch.Tensor:
        """Reverse AllToAllV and apply source-side router weights."""
        source_order = self._alltoall_single(
            local_output,
            state.input_splits,
            state.output_splits,
            state.group,
        )
        output = npu_moe_finalize_routing(
            source_order,
            state.expanded_row_idx,
            topk_weights=state.topk_weights,
        ).view(state.hidden_shape)
        return output
