from __future__ import annotations

import torch
from torch import nn

from sglang.srt.distributed import moe_tensor_model_parallel_all_reduce
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig


class MuseSpark1xInt4Experts(FusedMoE):
    """Routed experts using AutoGPTQ INT4 or per-layer BF16 weights.

    The dense equivalents of the packed weights are ``w13 [2*I, H]`` and
    ``w2 [H, I]``.

    With ``moe_tp_size > 1`` the intermediate dim is sharded instead of the
    expert set: every rank holds every expert but only ``I / tp`` of each one,
    so each rank does an equal share of the work no matter how the router
    spreads a batch. See ``forward`` for the reduction that costs.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        expert_hidden_size: int,
        intermediate_size: int,
        top_k: int,
        layer_id: int,
        quant_config: QuantizationConfig,
        prefix: str,
    ) -> None:
        super().__init__(
            num_experts=num_experts,
            hidden_size=expert_hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            top_k=top_k,
            params_dtype=torch.get_default_dtype(),
            reduce_results=False,
            quant_config=quant_config,
            prefix=prefix,
            activation="silu",
            inplace=False,
        )
        from sglang.srt.layers.quantization.gptq.gptq import GPTQMoEMethod
        from sglang.srt.layers.quantization.gptq.schemes.gptq_cpu import (
            GPTQIntelAMXMoEScheme,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        if not (
            isinstance(self.quant_method, UnquantizedFusedMoEMethod)
            or (
                isinstance(self.quant_method, GPTQMoEMethod)
                and isinstance(self.scheme, GPTQIntelAMXMoEScheme)
            )
        ):
            raise ValueError(
                f"Muse Spark experts require GPTQ INT4 or BF16 MoE, got "
                f"quant_method={type(self.quant_method).__name__}, "
                f"scheme={type(self.scheme).__name__}"
            )
        self.local_expert_offset = self.moe_ep_rank * self.num_local_experts
        self.expert_hidden_size = expert_hidden_size
        self.intermediate_size = intermediate_size
        self.nvfp4_dynamic_per_token = False

    def _fused_experts(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        local_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One grouped MoE call over the flattened (token, slot) pairs.

        ``fused_experts_cpu`` folds the top-k weighted sum into its epilogue,
        which would put the reduction before Muse's post-expert norm. Passing
        topk=1 with unit weights makes that epilogue an identity, so this
        returns the per-(token, expert) outputs the norm needs -- the CPU
        equivalent of the FP4 path's ``do_finalize=False``.
        """
        rows = token_ids.numel()
        x = hidden_states.index_select(0, token_ids).contiguous()
        ids = local_ids.to(torch.int32).view(rows, 1)
        weights = torch.ones((rows, 1), dtype=torch.float32, device=x.device)
        dispatch_output = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        return self.run_moe_core(dispatch_output).hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: nn.Module | None = None,
        *,
        packed_topk_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.get("pre_quantized") is not None:
            raise ValueError("CPU experts do not accept pre-quantized input")
        output = torch.zeros_like(hidden_states)
        # Padded rows carry topk_id -1, which falls outside this rank's experts.
        local_ids = topk_ids - self.local_expert_offset
        routed = ((local_ids >= 0) & (local_ids < self.num_local_experts)).nonzero(
            as_tuple=False
        )
        if routed.numel() == 0:
            return output
        token_ids = routed[:, 0]
        slots = routed[:, 1]
        local_ids = local_ids[token_ids, slots]

        expert_output = self._fused_experts(hidden_states, token_ids, local_ids)
        if self.moe_tp_size > 1:
            # Each rank holds a slice of the intermediate dim, so expert_output
            # is a partial sum. The post-expert norm is per-(token, expert) and
            # nonlinear, so the reduction has to happen here rather than on the
            # block's output: one [rows, H] all-reduce instead of one [T, H].
            # Every rank then holds the complete result, so the caller must not
            # all-reduce again.
            expert_output = moe_tensor_model_parallel_all_reduce(expert_output)
        if post_expert_norm is not None:
            expert_output = post_expert_norm(expert_output)
        expert_output = expert_output * topk_scores[token_ids, slots].unsqueeze(-1)
        output.index_add_(0, token_ids, expert_output.to(output.dtype))
        return output


__all__ = [
    "MuseSpark1xInt4Experts",
]
