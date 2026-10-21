"""Ascend Ops MoE token dispatcher.

Prefill and decode use different communication paths:

* prefill: TP-scoped ``allgather`` or ``alltoallv`` (implemented separately).
* decode ``low_latency``: the torch_npu MoE distribute v2 ops
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
from typing import NamedTuple, Optional

import torch

from sglang.srt.environ import envs
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

# ``quant_mode`` of the DS v2 distribute ops, matching the enum used by
# ``npu_moe_distribute_dispatch_setup``: 0 unquant, 1 static, 2 per-token
# dynamic int8, 3 per-group dynamic fp8, 4 MX (e8m0 block scales).
_DS_QUANT_MODE = {
    DispatcherOutputDtype.BF16: 0,
    DispatcherOutputDtype.INT8: 2,
    DispatcherOutputDtype.MXFP8: 4,
}

# ``comm_alg`` recommended by the Ascend DS v2 owners: dispatch uses the
# full-mesh algorithm, combine keeps the op's default.
_DISPATCH_COMM_ALG = "fullmesh_v1"
_COMBINE_COMM_ALG = ""


def _debug_rank0() -> bool:
    """True on global rank 0 while SGLANG_ASCEND_OPS_DEBUG is enabled."""
    if not envs.SGLANG_ASCEND_OPS_DEBUG.get():
        return False
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_rank() == 0


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
class _DSv2State:
    """Per-forward state the DS v2 combine needs from the matching dispatch."""

    num_tokens: int
    expert_ids: torch.Tensor
    expert_scales: torch.Tensor
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
        self.ep_coordinator = parallel.moe_ep_group
        self.ep_group = self.ep_coordinator.device_group
        self.ep_size = self.ep_coordinator.world_size
        self.ep_rank = self.ep_coordinator.rank_in_group
        self.num_experts = moe_runner_config.num_experts
        self.top_k = moe_runner_config.top_k
        self.hidden_size = moe_runner_config.hidden_size
        self._dispatch_output: Optional[AscendOpsDispatchOutput] = None
        self._dispatch_path: Optional[str] = None
        self._dispatch_group = None
        self._ep_comm_name: Optional[str] = None
        self._ds_state: Optional[_DSv2State] = None

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
        raise NotImplementedError("Ascend Ops prefill allgather dispatch is pending")

    def _dispatch_prefill_alltoallv(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "prefill_alltoallv"
        self._dispatch_group = self.tp_group
        raise NotImplementedError("Ascend Ops prefill alltoallv dispatch is pending")

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
        return self._ds_dispatch(
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
        raise NotImplementedError("Ascend Ops prefill allgather combine is pending")

    def _combine_prefill_alltoallv(self, combine_input):
        raise NotImplementedError("Ascend Ops prefill alltoallv combine is pending")

    def _combine_decode_low_latency(self, combine_input):
        return self._ds_combine(combine_input)

    def _combine_decode_fused_op(self, combine_input):
        raise NotImplementedError("Ascend Ops decode fused-op combine is pending")

    # DS v2 dispatch/combine ---------------------------------------------------

    def _ds_dispatch(
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
            padded_hidden_states = hidden_states.new_zeros(
                (padded_bs, hidden_states.shape[1])
            )
            padded_hidden_states[:num_tokens] = hidden_states
            hidden_states = padded_hidden_states

            padded_ids = topk_ids.new_zeros((padded_bs, topk_ids.shape[1]))
            padded_ids[:num_tokens] = topk_ids
            topk_ids = padded_ids

            padded_weights = topk_weights.new_zeros(
                (padded_bs, topk_weights.shape[1])
            )
            padded_weights[:num_tokens] = topk_weights
            topk_weights = padded_weights

            # Inactive rows must be the trailing ones.
            x_active_mask = torch.zeros(
                padded_bs, dtype=torch.bool, device=hidden_states.device
            )
            x_active_mask[:num_tokens] = True

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
            comm_alg=_DISPATCH_COMM_ALG,
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

        if _debug_rank0():
            print(
                f"[ascend_ops dispatch] path={prefill_mode or self._dispatch_path} "
                f"tokens={num_tokens} padded={padded_bs} qmode={quant_mode} "
                f"expand_x={tuple(expand_x.shape)}:{expand_x.dtype} "
                f"absmax={expand_x.abs().max().item():.4f} "
                f"etoks_sum={int(expert_token_nums.sum())} "
                f"etoks={expert_token_nums[:8].tolist()} "
                f"scales={tuple(dynamic_scales.shape)} "
                f"[{dynamic_scales.min().item():.4g}, "
                f"{dynamic_scales.max().item():.4g}]",
                flush=True,
            )

        self._ds_state = _DSv2State(
            num_tokens=num_tokens,
            expert_ids=topk_ids,
            expert_scales=topk_weights,
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

    def _ds_combine(self, combine_input: AscendOpsCombineInput) -> torch.Tensor:
        """Run ``npu_moe_distribute_combine_v2`` for the batch dispatched last."""
        import torch_npu

        state = self._ds_state
        if state is None:
            raise RuntimeError("Ascend Ops combine called before dispatch")

        # The combine op applies the top-k weights itself.
        combined = torch_npu.npu_moe_distribute_combine_v2(
            combine_input.hidden_states,
            state.expert_ids,
            state.assist_info_for_combine,
            state.ep_send_counts,
            state.expert_scales,
            self._get_ep_comm_name(),
            self.ep_size,
            self.ep_rank,
            self.num_experts,
            x_active_mask=state.x_active_mask,
            expand_scales=state.expand_scales,
            global_bs=0,
            comm_quant_mode=0,
            comm_alg=_COMBINE_COMM_ALG,
        )
        if _debug_rank0():
            expert_x = combine_input.hidden_states.float()
            print(
                f"[ascend_ops combine] expert_x={tuple(expert_x.shape)} "
                f"absmax={expert_x.abs().max().item():.4f} "
                f"nan={int(torch.isnan(expert_x).sum())} "
                f"wsum={float(state.expert_scales.sum()):.4f} "
                f"out_absmax={combined.float().abs().max().item():.4f}",
                flush=True,
            )

        self._ds_state = None
        self._dispatch_output = None
        return combined[: state.num_tokens]

    def _quant_mode(self, dtype_mode: str) -> int:
        dtype = self._resolve_dispatch_dtype(dtype_mode)
        try:
            return _DS_QUANT_MODE[dtype]
        except KeyError:
            raise NotImplementedError(
                "ascend_ops DS v2 dispatch does not support dispatcher output "
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
