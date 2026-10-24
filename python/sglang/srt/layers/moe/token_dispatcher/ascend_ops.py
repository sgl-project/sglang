"""Ascend Ops MoE token dispatcher.

Communication only: the expert-order regrouping and the local weighted
unpermute live in the runner's ``ascend_ops -> ascend`` permute hooks
(``moe_runner/ascend.py``), via the shared entry points in
``hardware_backend/npu/moe/routing.py``.

Paths (``--ascend-ops-config``):

* prefill ``allgather``: TP all-gather; every rank computes its local
  experts over the full batch.
* prefill ``alltoallv``: token all-to-all over the TP group; the source-side
  routing pass produces the wire layout.
* decode ``low_latency``: MC2 distribute ops
  (``npu_moe_distribute_dispatch_v2`` / ``npu_moe_distribute_combine_v2``)
  around the expert GEMMs.
* decode ``fused_op``: single fused kernel (not implemented yet).
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
from sglang.srt.hardware_backend.npu.moe.routing import (
    INT8_QUANT_MODE,
    MXFP8_QUANT_MODE,
    ROUTING_QUANT_MODES,
    npu_moe_init_routing,
    npu_moe_finalize_routing,
    normalize_mxfp_scale,
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
from sglang.srt.utils import is_npu

if is_npu():
    import torch_npu

# Quant mode per consumer op.  The routing op's modes are shared with every
# other Ascend MoE path (see routing.py); the MC2 distribute ops define
# their own enum (``npu_moe_distribute_dispatch_setup``: 0 unquant, 2
# per-token int8, 4 MX), so the same dtype maps to a different number.
_MC2_QUANT_MODE = {
    DispatcherOutputDtype.BF16: 0,
    DispatcherOutputDtype.INT8: 2,
    DispatcherOutputDtype.MXFP8: 4,
}

_QUANT_MODES = {
    "routing": ROUTING_QUANT_MODES,
    "mc2": _MC2_QUANT_MODE,
}

# ``y_dtype`` of the MC2 ops is an ATen ``ScalarType`` value (op schema:
# ``int?``), not a ``torch.dtype``, and torch exposes no public dtype -> value
# conversion, so the enum is spelled out here.
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
    """What the Ascend Ops dispatcher hands to the MoE runner.

    Field meaning by ``exec_mode``:

    - ``"mc2"`` (decode): ``hidden_states`` / ``hidden_states_scale`` are the
      MC2 dispatch output (already grouped per expert), ``topk_weights`` /
      ``topk_ids`` the (possibly padded) originals the MC2 combine needs.
    - ``"alltoallv"`` (prefill): ``hidden_states`` is the raw a2a payload in
      wire order, ``hidden_states_scale`` its scale, ``recv_counts`` the
      per-source/per-expert receive counts the pre-permute derives the local
      route ids from.
    - ``"allgather"`` (prefill): ``hidden_states`` / ``hidden_states_scale``
      are the gathered payload (bf16, or MXFP8 pre-quantized with its scale)
      and ``topk_ids`` / ``topk_weights`` the gathered router output; the
      local-expert filter and the regrouping run in the pre-permute.

    ``quant_mode`` is the routing quant mode (see
    ``hardware_backend/npu/moe/routing.py``); the MC2 path leaves it at -1.
    ``forward_state`` carries communication metadata through the runner to
    the matching combine call and is scoped to one forward.
    """

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
    """Per-forward state the MC2 combine needs from the matching dispatch.

    The padded top-k ids and weights travel with this state so combine does not
    depend on dispatcher-global forward state.
    """

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
class _AllToAllState:
    group: Any
    input_splits: list[int]
    output_splits: list[int]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    hidden_shape: torch.Size
    local_expanded_row_idx: Optional[torch.Tensor] = None


AscendOpsForwardState = _MC2State | _AllGatherState | _AllToAllState


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
        self._ep_comm_name: Optional[str] = None
        self._pad_buffers: dict[tuple, tuple] = {}
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

    def set_quant_config(self, quant_config: dict) -> None:
        """Store quantization settings used by the routing kernels."""

        self.quant_config = quant_config

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        if get_is_extend_in_batch():
            if self.ascend_ops_config.prefill == "allgather":
                return self._dispatch_allgather(
                    hidden_states, topk_weights, topk_ids, self.tp_group
                )
            return self._dispatch_alltoallv(
                hidden_states, topk_weights, topk_ids, self.tp_group
            )

        return self._dispatch_low_latency(
            hidden_states, topk_weights, topk_ids
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if combine_input.format != CombineInputFormat.ASCEND_OPS:
            raise TypeError(
                "AscendOpsDispatcher.combine expects an AscendOpsCombineInput, "
                f"got {combine_input.format.value}"
            )
        state = combine_input.forward_state
        if isinstance(state, _AllGatherState):
            return self._combine_allgather(combine_input.hidden_states, state)
        if isinstance(state, _AllToAllState):
            return self._combine_alltoallv(combine_input.hidden_states, state)
        if isinstance(state, _MC2State):
            return self._combine_mc2(combine_input.hidden_states, state)
        raise TypeError(f"unknown Ascend Ops forward state: {type(state)!r}")

    # Decode (MC2 low latency) -------------------------------------------------

    def _dispatch_low_latency(self, hidden_states, topk_weights, topk_ids):
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
        """Run ``npu_moe_distribute_dispatch_v2`` for one batch.

        ``padded_bs`` is the BS every rank in the EP group uses for this call;
        the trailing rows are dropped after combine via ``x_active_mask``.
        """
        num_tokens = hidden_states.shape[0]
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.float()
        quant_mode = self._quant_mode("mc2", dtype_mode)

        x_active_mask = None
        if padded_bs != num_tokens:
            hidden_states, topk_ids, topk_weights, x_active_mask = self._pad_batch(
                hidden_states, topk_ids, topk_weights, num_tokens, padded_bs
            )

        # Every quantised mode defaults the payload to int8 unless ``y_dtype`` is
        # given, which is invalid next to the e8m0 block scale MX emits.
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
            # Per-expert counts, matching ``group_list_type=1`` of the runner.
            expert_token_nums_type=1,
            comm_alg=self.ascend_ops_config.ll_dispatch_alg,
            **mxfp8_args,
        )

        if quant_mode == _MC2_QUANT_MODE[DispatcherOutputDtype.MXFP8]:
            if hidden_states.shape[1] % 64 != 0:
                raise ValueError(
                    "MC2 MXFP8 decode requires hidden_size to be a multiple "
                    f"of 64, got {hidden_states.shape[1]}"
                )
            # The op returns a flat ``(A, h // 32)`` e8m0 scale; the runner's
            # MX kernels take the pair-split ``(A, h // 64, 2)`` form.
            dynamic_scales = dynamic_scales.reshape(
                dynamic_scales.shape[0], hidden_states.shape[1] // 64, 2
            )

        # Only meaningful when top-k weights were handed to dispatch, which we
        # never do. Some torch_npu builds return an empty (0,) tensor here, but
        # A3 returns an uninitialised ``(a,)`` one, so never forward it.
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
            # The op returns an empty scale tensor in the unquantised case;
            # the runner expects None there.
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

    def _combine_mc2(
        self, hidden_states: torch.Tensor, state: _MC2State
    ) -> torch.Tensor:
        """Run ``npu_moe_distribute_combine_v2`` for the batch dispatched last."""
        # The combine op applies the top-k weights itself.
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
            comm_alg=self.ascend_ops_config.ll_combine_alg,
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
        """Resolve the dispatch wire dtype.

        ``dtype_mode`` (e.g. ``"low_latency"``) first consults the
        mode-specific ``<mode>_dispatcher_output_dtype`` key before falling
        back to the shared ``dispatcher_output_dtype``; ``None`` reads only
        the shared key.
        """
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

    # Prefill allgather --------------------------------------------------------

    def _dispatch_allgather(self, hidden_states, topk_weights, topk_ids, group):
        """Gather the full batch on every TP rank."""
        input_dtype = hidden_states.dtype
        quant_mode = self._quant_mode("routing")
        if quant_mode not in (-1, INT8_QUANT_MODE, MXFP8_QUANT_MODE):
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
            # Quantize local rows once, then gather MXFP8 payload and scales.
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
        """ReduceScatter the per-rank contributions back to the local shard."""
        sizes = state.row_sizes
        group = state.group
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
        output = output[: sizes[dist.get_rank(group)]].to(state.input_dtype)
        return output

    # Prefill alltoallv --------------------------------------------------------

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
        """Send each token to the rank owning its expert."""
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

        state = _AllToAllState(
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
        self, local_output: torch.Tensor, state: _AllToAllState
    ) -> torch.Tensor:
        """Reverse the a2a and apply the router weights at the source.

        ``local_output`` arrives from the runner's post-permute already mapped
        back to wire order (the post-permute undoes the local expert
        regrouping with ``local_expanded_row_idx``).
        """
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
