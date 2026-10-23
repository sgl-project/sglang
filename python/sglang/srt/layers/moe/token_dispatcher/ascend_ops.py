"""Ascend Ops MoE token dispatcher.

Prefill and decode use different communication paths:

* prefill: TP-scoped ``allgather`` or ``alltoallv``, built on
  ``npu_moe_init_routing_v2`` plus ``all_to_all_single``.
* decode ``low_latency``: the MC2 MoE distribute ops
  (``npu_moe_distribute_dispatch_v2`` / ``npu_moe_distribute_combine_v2``),
  which perform the EP all-to-all (with optional int8/MX quantisation) around
  the expert GEMMs.
* decode ``fused_op``: a single ``dispatch_ffn_combine`` kernel (not
  implemented yet).

``torch_npu`` is imported lazily inside the path methods because this module is
imported on every platform (``token_dispatcher/__init__`` and the triton MoE
layer import it unconditionally).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
)
from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    NPUMoEInitRouting_v2,
    _normalize_mxfp_scale,
)
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

# ``quant_mode`` of the MC2 distribute ops, matching the enum used by
# ``npu_moe_distribute_dispatch_setup``: 0 unquant, 1 static, 2 per-token
# dynamic int8, 3 per-group dynamic fp8, 4 MX (e8m0 block scales).
_MC2_QUANT_MODE = {
    DispatcherOutputDtype.BF16: 0,
    DispatcherOutputDtype.INT8: 2,
    DispatcherOutputDtype.MXFP8: 4,
}


class AscendOpsDispatchOutput(NamedTuple):
    """Metadata passed from Ascend Ops dispatch to the MoE runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: Optional[torch.Tensor]
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


@dataclass
class _MC2State:
    """Per-forward state the MC2 combine needs from the matching dispatch.

    The top-k ids and weights are not repeated here: the dispatch output already
    carries the (possibly padded) copies and both are cleared together.
    """

    num_tokens: int
    assist_info_for_combine: torch.Tensor
    ep_send_counts: torch.Tensor
    x_active_mask: Optional[torch.Tensor]
    expand_scales: Optional[torch.Tensor]


class AscendOpsDispatcher(BaseDispatcher):
    """Select one of the Ascend Ops dispatch/combine execution paths.

    The methods below form the integration boundary for the NPU operators.
    Keeping path selection here makes the runner independent of whether a
    batch is prefill or decode and keeps the output contract stable.
    """

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        ascend_ops_config: Optional[AscendOpsConfig] = None,
    ) -> None:
        super().__init__()
        self.ascend_ops_config = ascend_ops_config or AscendOpsConfig()
        # Prefill communication is scoped to TP; decode routing is scoped to EP.
        parallel = get_parallel()
        self.tp_group = parallel.tp_group.device_group
        self.attn_tp_size = parallel.attn_tp_size
        self.ep_coordinator = parallel.moe_ep_group
        self.ep_group = self.ep_coordinator.device_group
        self.ep_size = self.ep_coordinator.world_size
        self.ep_rank = self.ep_coordinator.rank_in_group
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.top_k = moe_runner_config.top_k
        self.hidden_size = moe_runner_config.hidden_size
        self.quant_config: Optional[dict] = None
        self._dispatch_output: Optional[AscendOpsDispatchOutput] = None
        self._dispatch_path: Optional[str] = None
        self._dispatch_group = None
        self._ep_comm_name: Optional[str] = None
        self._mc2_state: Optional[_MC2State] = None
        self._pad_buffers: dict[tuple, tuple] = {}
        self._alltoall_state: Optional[dict[str, Any]] = None
        self._allgather_state: Optional[dict[str, Any]] = None
        if self.ascend_ops_config.prefill == "allgather":
            if not (
                parallel.tp_size == parallel.moe_ep_size
                and parallel.attn_cp_size == 1
                and self.num_local_experts * self.ep_size == self.num_experts
                and not moe_runner_config.num_fused_shared_experts
            ):
                raise ValueError(
                    "NPU AllGather requires TP=EP, Attention CP=1, "
                    "contiguous expert placement and no fused shared experts"
                )
        self._init_routing = NPUMoEInitRouting_v2(quant_mode=-1)
        self._finalize_routing = NPUFinalizeRouting(drop_pad_mode=2)

    def set_quant_config(self, quant_config: dict) -> None:
        """Store quantization settings used by the routing kernels."""

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
        return self._dispatch_allgather(hidden_states, topk_weights, topk_ids)

    def _dispatch_prefill_alltoallv(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "prefill_alltoallv"
        self._dispatch_group = self.tp_group
        return self._dispatch_alltoallv(hidden_states, topk_weights, topk_ids)

    def _dispatch_decode_low_latency(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "decode_low_latency"
        self._dispatch_group = self.ep_group
        num_tokens = hidden_states.shape[0]
        # Without DP attention every EP rank steps with the same BS, so no
        # padding is needed (0). Set it when ranks can disagree.
        pad_bs = envs.SGLANG_ASCEND_OPS_DECODE_PAD_BS.get()
        if pad_bs and num_tokens > pad_bs:
            raise ValueError(
                f"ascend_ops decode got {num_tokens} tokens on this rank but "
                f"SGLANG_ASCEND_OPS_DECODE_PAD_BS pads the batch to {pad_bs}; "
                "raise it or lower the decode batch size."
            )
        return self._mc2_dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            padded_bs=pad_bs or num_tokens,
            dtype_mode="low_latency",
            prefill_mode="",
        )

    def _dispatch_decode_fused_op(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "decode_fused_op"
        self._dispatch_group = self.ep_group
        raise NotImplementedError("Ascend Ops decode fused-op dispatch is pending")

    # Combine path boundaries --------------------------------------------------

    def _combine_prefill_allgather(self, combine_input):
        return self._combine_allgather(combine_input.hidden_states)

    def _combine_prefill_alltoallv(self, combine_input):
        return self._combine_alltoallv(combine_input.hidden_states)

    def _combine_decode_low_latency(self, combine_input):
        return self._mc2_combine(combine_input)

    def _combine_decode_fused_op(self, combine_input):
        raise NotImplementedError("Ascend Ops decode fused-op combine is pending")

    # MC2 dispatch/combine -----------------------------------------------------

    def _mc2_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        padded_bs: int,
        dtype_mode: str,
        prefill_mode: str,
    ) -> AscendOpsDispatchOutput:
        """Run ``npu_moe_distribute_dispatch_v2`` for one batch.

        ``padded_bs`` is the BS every rank in the EP group uses for this call;
        the trailing rows are dropped after combine via ``x_active_mask``.
        """
        import torch_npu

        num_tokens = hidden_states.shape[0]
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.float()
        quant_mode = self._quant_mode(dtype_mode)

        x_active_mask = None
        if padded_bs != num_tokens:
            hidden_states, topk_ids, topk_weights, x_active_mask = self._pad_batch(
                hidden_states, topk_ids, topk_weights, num_tokens, padded_bs
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
            # Per-expert counts, matching ``group_list_type=1`` of the runner.
            expert_token_nums_type=1,
            comm_alg=self.ascend_ops_config.ll_dispatch_alg,
        )

        if quant_mode == 4 and hidden_states.shape[1] % 64 == 0:
            # The op returns a flat ``(A, h // 32)`` e8m0 scale; the runner's
            # MX kernels take the pair-split ``(A, h // 64, 2)`` form.
            dynamic_scales = dynamic_scales.reshape(
                dynamic_scales.shape[0], hidden_states.shape[1] // 64, 2
            )

        # Only meaningful when top-k weights were handed to dispatch, which we
        # never do. Some torch_npu builds return an empty (0,) tensor here, but
        # A3 returns an uninitialised ``(a,)`` one, so never forward it.
        expand_scales = None

        self._mc2_state = _MC2State(
            num_tokens=num_tokens,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_recv_counts,
            x_active_mask=x_active_mask,
            expand_scales=expand_scales,
        )
        self._dispatch_output = AscendOpsDispatchOutput(
            hidden_states=expand_x,
            # The op returns an empty scale tensor in the non-quantised case;
            # the runner expects None there.
            hidden_states_scale=dynamic_scales if quant_mode != 0 else None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expanded_row_idx=None,
            expert_tokens=expert_token_nums,
            group_list_type=1,
            prefill_mode=prefill_mode,
        )
        return self._dispatch_output

    def _pad_batch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        padded_bs: int,
    ):
        """Copy the batch into buffers reused across steps.

        ``padded_bs`` differs from ``num_tokens`` whenever EP ranks disagree on
        the batch size (DP attention), so the buffers are allocated once per
        shape and afterwards only the rows that changed are touched: the active
        prefix is overwritten, and the tail is cleared only when the previous
        call filled more rows than this one.
        """
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

        # Inactive rows must be the trailing ones and must stay zeroed.
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

    def _mc2_combine(self, combine_input: AscendOpsCombineInput) -> torch.Tensor:
        """Run ``npu_moe_distribute_combine_v2`` for the batch dispatched last."""
        import torch_npu

        state = self._mc2_state
        output = self._dispatch_output
        if state is None or output is None:
            raise RuntimeError("Ascend Ops combine called before dispatch")

        # The combine op applies the top-k weights itself.
        combined = torch_npu.npu_moe_distribute_combine_v2(
            combine_input.hidden_states,
            output.topk_ids,
            state.assist_info_for_combine,
            state.ep_send_counts,
            output.topk_weights,
            self._get_ep_comm_name(),
            self.ep_size,
            self.ep_rank,
            self.num_experts,
            x_active_mask=state.x_active_mask,
            expand_scales=state.expand_scales,
            global_bs=0,
            comm_quant_mode=0,
            comm_alg=self.ascend_ops_config.ll_combine_alg,
        )

        self._mc2_state = None
        self._dispatch_output = None
        return combined[: state.num_tokens]

    def _quant_mode(self, dtype_mode: str) -> int:
        dtype = self._resolve_dispatch_dtype(dtype_mode)
        try:
            return _MC2_QUANT_MODE[dtype]
        except KeyError:
            raise NotImplementedError(
                "ascend_ops MC2 dispatch does not support dispatcher output "
                f"dtype {dtype.value}; supported: bf16, int8, mxfp8"
            ) from None

    def _resolve_dispatch_dtype(self, dtype_mode: str) -> DispatcherOutputDtype:
        """Resolve the dispatch wire dtype for ``dtype_mode`` (normal/low_latency)."""
        quant_config = self.quant_config or {}
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

    # Prefill allgather --------------------------------------------------------

    def _dispatch_allgather(self, hidden_states, topk_weights, topk_ids):
        if self._dispatch_output is not None:
            raise RuntimeError("NPU dispatch called before the previous combine")
        group = self._dispatch_group
        input_dtype = hidden_states.dtype
        quant_mode = self._routing_quant_mode()
        if quant_mode not in (-1, 1, MXFP8_QUANT_MODE):
            raise NotImplementedError(
                "NPU AllGather supports BF16, INT8 and MXFP8 routing"
            )
        world_size = dist.get_world_size(group)
        # The MoE boundary already splits each DP shard across attention TP.
        # Mirror tensor_split using CPU counts, including scheduler padding.
        sizes = get_dp_global_num_tokens()
        if sizes is None:
            raise RuntimeError("NPU AllGather requires scheduler DP token counts")
        row_sizes = []
        for rows in sizes:
            base, extra = divmod(rows, self.attn_tp_size)
            row_sizes.extend(
                base + (rank < extra) for rank in range(self.attn_tp_size)
            )
        if (
            len(row_sizes) != world_size
            or row_sizes[dist.get_rank(group)] != hidden_states.shape[0]
        ):
            raise ValueError("NPU AllGather input rows disagree with DP token counts")
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
            import torch_npu

            # Quantize local rows once, then gather MXFP8 payload and scales.
            if hidden_states.shape[0] > 0:
                hidden_states, scales = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                scales = _normalize_mxfp_scale(scales).view(torch.uint8)
            else:
                hidden_states = hidden_states.new_empty(
                    (0, self.hidden_size), dtype=torch.float8_e4m3fn
                )
                scales = hidden_states.new_empty(
                    (0, (self.hidden_size + 63) // 64, 2), dtype=torch.uint8
                )

        hidden_states = gather_rows(hidden_states)
        if scales is not None:
            scales = gather_rows(scales)
        topk_weights = gather_rows(topk_weights.float())
        topk_ids = gather_rows(topk_ids.to(torch.int32))
        local_ids = topk_ids - self.ep_rank * self.num_local_experts
        is_local = (local_ids >= 0) & (local_ids < self.num_local_experts)
        topk_weights = topk_weights.masked_fill(~is_local, 0)
        # Operator constraint: npu_moe_init_routing_v2 needs a non-negative
        # out-of-range expert ID for dropped routes. Do not use -1: it can
        # corrupt valid expert counts even when the routing weight is zero.
        # num_local_experts is just beyond the valid local expert ID range.
        local_ids = local_ids.masked_fill(~is_local, self.num_local_experts)
        if scales is not None:
            # Already MXFP8: reorder payload and scales without requantizing.
            x, row_idx, expert_tokens, scales = torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                local_ids,
                active_num=local_ids.numel(),
                expert_num=self.num_local_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                row_idx_type=0,
                active_expert_range=[0, self.num_local_experts],
                quant_mode=-1,
                scale=scales,
                x_dtype=torch.float8_e4m3fn,
            )
            scales = _normalize_mxfp_scale(scales)
        else:
            x, row_idx, expert_tokens, scales = self._init_routing_tokens(
                hidden_states, local_ids, self.num_local_experts, quant_mode
            )
        # Grouped GEMM and activation use device counts, so keep the capacity.
        # If any route is dropped, the final capacity row is unused. Combine
        # zeros unused rows on device before this sentinel is read by finalize.
        row_idx = row_idx.masked_fill(~is_local.reshape(-1), local_ids.numel() - 1)
        self._allgather_state = {
            "group": group,
            "row_sizes": row_sizes,
            "input_dtype": input_dtype,
        }
        self._dispatch_output = AscendOpsDispatchOutput(
            hidden_states=x,
            hidden_states_scale=scales,
            topk_weights=topk_weights,
            topk_ids=local_ids,
            expanded_row_idx=row_idx,
            expert_tokens=expert_tokens,
            group_list_type=1,
            prefill_mode="allgather",
        )
        return self._dispatch_output

    def _combine_allgather(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self._allgather_state
        if state is None:
            raise RuntimeError("NPU AllGather combine called before dispatch")
        dispatched = self._dispatch_output
        sizes = state["row_sizes"]
        group = state["group"]
        # Operator constraint: finalize mode 2 requires the full
        # [global_tokens * top_k, hidden] shape, not just the valid expert rows.
        expert_output = torch.zeros(
            (dispatched.topk_ids.numel(), self.hidden_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        expert_output[: hidden_states.shape[0]].copy_(hidden_states)
        # Keep the valid count on device. Zero unused capacity before finalize:
        # dropped routes must not read NaN even when their weight is zero.
        unused_rows = torch.arange(
            expert_output.shape[0], device=expert_output.device
        ) >= dispatched.expert_tokens.sum()
        expert_output.masked_fill_(unused_rows[:, None], 0)
        # Keep weighted expert sums in FP32 across ranks. Rounding each
        # rank's partial output to BF16 before ReduceScatter loses precision.
        contributions = self._finalize_routing._finalize_routing(
            expert_output,
            topk_weights=dispatched.topk_weights,
            expanded_row_idx=dispatched.expanded_row_idx,
            topk_ids=dispatched.topk_ids,
        )
        # Equal-sized chunks already have the layout required by ReduceScatter.
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
        output = output[: sizes[dist.get_rank(group)]].to(state["input_dtype"])
        self._dispatch_output = None
        self._allgather_state = None
        return output

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
            (local_ids.shape[0], 1),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
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
        if (
            hidden_states.shape[0] > 0
            and expanded_row_idx.shape[0] != hidden_states.shape[0]
        ):
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
