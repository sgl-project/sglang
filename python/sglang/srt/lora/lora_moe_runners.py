# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""LoRA-aware MoE runners that integrate LoRA deltas into the MoE computation.

The key insight is that LoRA deltas must be added at specific points:
1. After gate_up projection, BEFORE activation (halfway through)
2. After down projection, BEFORE final reduction (at the end)

This differs from computing LoRA independently and adding at the very end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerCore,
    TritonRunnerInput,
    TritonRunnerOutput,
)
from sglang.srt.utils import is_cuda, is_hip

_is_hip = is_hip()
_is_cuda = is_cuda()

if _is_cuda or _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul


@dataclass
class LoRAInfo:
    """LoRA weights and dispatch info for MoE computation."""

    # LoRA weights: [num_loras, num_experts, dim1, dim2]
    gate_up_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, max_rank, hidden_dim]
    gate_up_lora_b_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, gate_up_dim, max_rank]
    down_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, max_rank, intermediate_dim]
    down_lora_b_weights: torch.Tensor  # [num_loras, num_experts, hidden_dim, max_rank]

    # Dispatch info (sorted by expert)
    token_ids: torch.Tensor  # [num_dispatched] - original token indices
    expert_ids: torch.Tensor  # [num_dispatched] - expert IDs
    lora_ids: torch.Tensor  # [num_dispatched] - LoRA adapter IDs

    # LoRA config per adapter
    lora_ranks: torch.Tensor  # [num_loras]
    lora_scalings: torch.Tensor  # [num_loras]
    adapter_enabled: torch.Tensor  # [num_loras] - which adapters are enabled

    num_experts: int


class TritonRunnerCoreWithLoRA(TritonRunnerCore):
    """
    LoRA-aware wrapper around TritonRunnerCore.

    Integrates LoRA deltas at the correct points in the MoE forward pass:
    1. Base gate_up projection + LoRA gate_up delta -> activation
    2. Base down projection + LoRA down delta -> final reduction

    This follows the vLLM/HF approach where LoRA is fused into the computation
    rather than computed independently.
    """

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        lora_info: Optional[LoRAInfo] = None,
    ) -> TritonRunnerOutput:
        """
        Run MoE with integrated LoRA computation.

        This method extends TritonRunnerCore.run() by inserting LoRA delta
        computations at the correct points in the MoE forward pass.

        Args:
            runner_input: Standard Triton runner input
            quant_info: Quantization info for base weights
            running_state: Running state dict
            lora_info: Optional LoRA weights and dispatch info

        Returns:
            TritonRunnerOutput with combined base + LoRA output
        """
        # If no LoRA, use base implementation
        if lora_info is None:
            return super().run(runner_input, quant_info, running_state)

        # Extract common variables
        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        sorted_token_ids = runner_input.sorted_token_ids
        expert_ids = runner_input.expert_ids
        num_tokens_post_padded = runner_input.num_tokens_post_padded

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        a13_scale = quant_info.a13_scale
        a2_scale = quant_info.a2_scale
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        w13_zp = quant_info.w13_zp
        w2_zp = quant_info.w2_zp
        block_shape = quant_info.block_shape
        per_channel_quant = quant_info.per_channel_quant
        use_fp8_w8a8 = quant_info.use_fp8_w8a8
        use_int8_w8a8 = quant_info.use_int8_w8a8
        use_int8_w8a16 = quant_info.use_int8_w8a16
        use_int4_w4a16 = quant_info.use_int4_w4a16

        activation = self.config.activation
        no_combine = self.config.no_combine
        inplace = self.config.inplace
        gemm1_alpha = self.config.gemm1_alpha
        gemm1_limit = self.config.gemm1_clamp_limit
        routed_scaling_factor = self.config.routed_scaling_factor
        apply_router_weight_on_input = self.config.apply_router_weight_on_input

        assert self.config.is_gated, "Only gated MoEs are supported for Triton runner"

        M = hidden_states.shape[0]
        E, N, _ = w13.shape
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        # Import functions needed for MoE computation
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            invoke_fused_moe_kernel,
            moe_sum_reduce_torch_compile,
            moe_sum_reduce_triton,
            swiglu_with_alpha_and_limit,
        )

        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        sorted_token_ids = runner_input.sorted_token_ids
        expert_ids = runner_input.expert_ids
        num_tokens_post_padded = runner_input.num_tokens_post_padded

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        a13_scale = quant_info.a13_scale
        a2_scale = quant_info.a2_scale
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        w13_zp = quant_info.w13_zp
        w2_zp = quant_info.w2_zp
        block_shape = quant_info.block_shape
        per_channel_quant = quant_info.per_channel_quant
        use_fp8_w8a8 = quant_info.use_fp8_w8a8
        use_int8_w8a8 = quant_info.use_int8_w8a8
        use_int8_w8a16 = quant_info.use_int8_w8a16
        use_int4_w4a16 = quant_info.use_int4_w4a16

        activation = self.config.activation
        no_combine = self.config.no_combine
        inplace = self.config.inplace
        gemm1_alpha = self.config.gemm1_alpha
        gemm1_limit = self.config.gemm1_clamp_limit
        routed_scaling_factor = self.config.routed_scaling_factor
        apply_router_weight_on_input = self.config.apply_router_weight_on_input

        assert self.config.is_gated, "Only gated MoEs are supported for Triton runner"

        M = hidden_states.shape[0]
        E, N, _ = w13.shape
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        # ============================================================
        # Stage 1: Gate/Up projection (base)
        # ============================================================
        intermediate_cache1 = torch.empty(
            (M, topk_ids.shape[1], N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # output shape: [M, top_k, N] and in original token order since we do not pass in c_sorted=True. If we
        # want to get the output in sorted order by expert, we can pass in c_sorted=True.
        # TODO: determine whether we should pass in c_sorted=True. That will make it less readable and different from base run method.
        # but we won't have to scatter the lora delta to the correct positions before adding them to this base output.
        invoke_fused_moe_kernel(
            hidden_states,
            w13,
            b13,
            intermediate_cache1,
            a13_scale,
            w13_scale,
            w13_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            running_state["config"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

        # ============================================================
        # Stage 1.5: Add LoRA gate_up delta BEFORE activation
        # ============================================================
        self._add_lora_gate_up_delta(
            hidden_states=hidden_states,
            intermediate_cache=intermediate_cache1,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            lora_info=lora_info,
        )

        # ============================================================
        # Stage 2: Activation (SiLU or GELU)
        # ============================================================
        intermediate_cache2 = torch.empty(
            (M * topk_ids.shape[1], N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if activation == "silu":
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = swiglu_with_alpha_and_limit(
                    intermediate_cache1.view(-1, N),
                    gemm1_alpha,
                    gemm1_limit,
                )
            elif _is_cuda or _is_hip:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                raise ValueError(f"Unsupported platform for activation: {activation=}")
        elif activation == "gelu":
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                raise ValueError(f"Unsupported platform for activation: {activation=}")
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        # ============================================================
        # Stage 3: Down projection (base)
        # ============================================================
        intermediate_cache3 = torch.empty(
            (M, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if no_combine:
            assert not inplace
            out_hidden_states = torch.empty(
                (M, topk_ids.shape[1], w2.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif inplace:
            out_hidden_states = hidden_states
        else:
            out_hidden_states = torch.empty_like(hidden_states)

        # output shape: [M, hidden_dim] and in original token order since we do not pass in c_sorted=True. If we
        # want to get the output in sorted order by expert, we can pass in c_sorted=True. We use  no_combine=False because the next
        # LoRA computation requires input to be [M, hidden_dim]
        # TODO: determine whether we should pass in c_sorted=True. That will make it less readable and different from base run method.
        # but we won't have to scatter the lora delta to the correct positions before adding them to this base output.
        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states.unsqueeze(0)
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            running_state["config"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

        # ============================================================
        # Stage 3.5: Add LoRA down delta BEFORE final reduction
        # ============================================================
        # intermediate_cache2 is in the original token order and token-major order.
        self._add_lora_down_delta(
            intermediate_input=intermediate_cache2,
            intermediate_cache=intermediate_cache3,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            apply_router_weight_on_input=apply_router_weight_on_input,
            lora_info=lora_info,
        )

        # we still need to combine the output

        # ============================================================
        # Stage 4: Final reduction (sum across top_k)
        # ============================================================
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states,
                ).squeeze(dim=1)
            else:
                if M <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
        elif _is_hip:
            from vllm import _custom_ops as vllm_ops

            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )
        else:
            from vllm import _custom_ops as vllm_ops

            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )

        return TritonRunnerOutput(
            hidden_states=out_hidden_states,
        )

    def _add_lora_gate_up_delta(
        self,
        hidden_states: torch.Tensor,  # [M, hidden_dim]
        intermediate_cache: torch.Tensor,  # [M, top_k, gate_up_dim]
        topk_ids: torch.Tensor,  # [M, top_k]
        topk_weights: torch.Tensor,  # [M, top_k]
        lora_info: LoRAInfo,
    ) -> None:
        """
        Add LoRA gate_up delta to intermediate_cache in-place.

        For each (token, expert) pair, computes:
            delta = scaling * B @ (A @ hidden_states[token])
        and adds it to intermediate_cache[token, k] where k is the top_k index.
        """
        from sglang.srt.lora.triton_ops import fused_moe_lora

        M, top_k, gate_up_dim = intermediate_cache.shape
        num_dispatched = lora_info.token_ids.shape[0]

        # Compute LoRA delta where intermediate_cache needs to be [M, top_k, gate_up_dim] in original token order.
        # Output shape: [M, top_k, gate_up_dim]
        # Hidden_states shape: [M, hidden_dim] (handles token duplication internally)

        num_tokens_post_padded_formatted = torch.tensor(
            [num_dispatched], dtype=torch.int32, device=hidden_states.device
        )
        actual_max_lora_rank = int(lora_info.lora_ranks.max().item())

        # Skip LoRA computation if no LoRA adapters have non-zero rank
        if actual_max_lora_rank == 0:
            return

        lora_a_stacked = [lora_info.gate_up_lora_a_weights]
        lora_b_stacked = [lora_info.gate_up_lora_b_weights]

        fused_moe_lora(
            output=intermediate_cache,
            qcurr_hidden_states=hidden_states,
            lora_a_stacked=lora_a_stacked,
            lora_b_stacked=lora_b_stacked,
            topk_weights=topk_weights,  # Use actual routing weights
            sorted_token_ids=lora_info.token_ids.unsqueeze(0),
            expert_ids=lora_info.expert_ids.unsqueeze(0),
            num_tokens_post_padded=num_tokens_post_padded_formatted,
            max_lora_rank=actual_max_lora_rank,
            top_k_num=top_k,
            lora_ids=lora_info.lora_ids,
            adapter_enabled=lora_info.adapter_enabled,
            shrink_block_size_m=64,
            shrink_block_size_n=64,
            shrink_block_size_k=64,
            shrink_group_size_m=8,
            shrink_num_warps=4,
            shrink_num_stages=2,
            shrink_split_k=1,
            expand_block_size_m=64,
            expand_block_size_n=64,
            expand_block_size_k=64,
            expand_group_size_m=8,
            expand_num_warps=4,
            expand_num_stages=2,
            expand_split_k=1,
        )

    def _add_lora_down_delta(
        self,
        intermediate_input: torch.Tensor,  # [M * top_k, intermediate_dim]
        intermediate_cache: torch.Tensor,  # [M, top_k, hidden_dim]
        topk_ids: torch.Tensor,  # [M, top_k]
        topk_weights: torch.Tensor,  # [M, top_k]
        apply_router_weight_on_input: bool,
        lora_info: LoRAInfo,
    ) -> None:
        """
        Add LoRA down delta to intermediate_cache in-place.

        For each (token, expert) pair, computes:
            delta = scaling * B @ (A @ intermediate_input[dispatched_idx])
        and adds it to intermediate_cache[token, k].
        """
        from sglang.srt.lora.triton_ops import fused_moe_lora

        M, top_k, hidden_dim = intermediate_cache.shape

        # intermediate_input is the input from the previous stage and is in original token order.

        # Data format adaptation for vLLM kernel
        num_dispatched_down = lora_info.token_ids.shape[0]
        num_tokens_post_padded_formatted = torch.tensor(
            [num_dispatched_down], dtype=torch.int32, device=intermediate_input.device
        )
        actual_max_lora_rank = int(lora_info.lora_ranks.max().item())

        # Skip LoRA computation if no LoRA adapters have non-zero rank
        if actual_max_lora_rank == 0:
            return

        # Handle multi-LoRA: stack weights for all loaded LoRAs
        # lora_info.down_lora_a_weights shape: [num_loras, num_experts, max_rank, intermediate_dim]
        # Note: LoRA scaling factors (lora_info.lora_scalings) are already applied to weights during loading
        max_loras = len(lora_info.lora_ranks)

        # Validate weight dimensions match expectations
        assert (
            lora_info.down_lora_a_weights.shape[0] == max_loras
        ), f"Expected {max_loras} LoRAs, got {lora_info.down_lora_a_weights.shape[0]}"
        assert (
            lora_info.adapter_enabled.shape[0] >= max_loras
        ), f"adapter_enabled too small: {lora_info.adapter_enabled.shape[0]} < {max_loras}"

        lora_a_stacked = [lora_info.down_lora_a_weights]
        lora_b_stacked = [lora_info.down_lora_b_weights]

        fused_moe_lora(
            output=intermediate_cache,
            qcurr_hidden_states=intermediate_input,
            lora_a_stacked=lora_a_stacked,
            lora_b_stacked=lora_b_stacked,
            topk_weights=topk_weights,  # Use the routing weights passed to this function
            sorted_token_ids=lora_info.token_ids.unsqueeze(0),
            expert_ids=lora_info.expert_ids.unsqueeze(0),
            num_tokens_post_padded=num_tokens_post_padded_formatted,
            max_lora_rank=actual_max_lora_rank,
            top_k_num=top_k,
            lora_ids=lora_info.lora_ids,
            adapter_enabled=lora_info.adapter_enabled,
            shrink_block_size_m=64,
            shrink_block_size_n=64,
            shrink_block_size_k=64,
            shrink_group_size_m=8,
            shrink_num_warps=4,
            shrink_num_stages=2,
            shrink_split_k=1,
            expand_block_size_m=64,
            expand_block_size_n=64,
            expand_block_size_k=64,
            expand_group_size_m=8,
            expand_num_warps=4,
            expand_num_stages=2,
            expand_split_k=1,
        )
