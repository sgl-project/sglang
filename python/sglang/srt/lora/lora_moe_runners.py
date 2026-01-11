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
        self._add_lora_down_delta(
            intermediate_input=intermediate_cache2,
            intermediate_cache=intermediate_cache3,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            apply_router_weight_on_input=apply_router_weight_on_input,
            lora_info=lora_info,
        )

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
        lora_info: LoRAInfo,
    ) -> None:
        """
        Add LoRA gate_up delta to intermediate_cache in-place.

        For each (token, expert) pair, computes:
            delta = scaling * B @ (A @ hidden_states[token])
        and adds it to intermediate_cache[token, k] where k is the top_k index.
        """
        from sglang.srt.lora.triton_ops.per_expert_lora_moe import (
            per_expert_lora_forward,
        )

        M, top_k, gate_up_dim = intermediate_cache.shape
        num_dispatched = lora_info.token_ids.shape[0]

        # Compute LoRA delta for each (token, expert) pair
        # Output shape: [num_dispatched, gate_up_dim]
        lora_delta = torch.zeros(
            (num_dispatched, gate_up_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        _, lora_delta = per_expert_lora_forward(
            hidden_states=hidden_states,
            lora_a_weights=lora_info.gate_up_lora_a_weights,
            lora_b_weights=lora_info.gate_up_lora_b_weights,
            token_ids=lora_info.token_ids,
            expert_ids=lora_info.expert_ids,
            lora_ids=lora_info.lora_ids,
            lora_ranks=lora_info.lora_ranks,
            lora_scalings=lora_info.lora_scalings,
            num_experts=lora_info.num_experts,
            base_output=lora_delta,
            is_down_proj=False,
        )

        # Add delta to intermediate_cache at the right positions
        # We need to map from dispatched indices back to (token, top_k_idx) pairs
        self._scatter_add_to_topk_cache(
            lora_delta=lora_delta,
            intermediate_cache=intermediate_cache,
            token_ids=lora_info.token_ids,
            expert_ids=lora_info.expert_ids,
            topk_ids=topk_ids,
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
        from sglang.srt.lora.triton_ops.per_expert_lora_moe import (
            per_expert_lora_forward,
        )

        M, top_k, hidden_dim = intermediate_cache.shape

        # Build indices to gather from intermediate_input
        # For each dispatched (token, expert) pair, find which top_k slot it corresponds to
        lora_intermediate_input = self._gather_dispatched_inputs(
            intermediate_input=intermediate_input,
            token_ids=lora_info.token_ids,
            expert_ids=lora_info.expert_ids,
            topk_ids=topk_ids,
            M=M,
            top_k=top_k,
        )

        # Compute LoRA delta for down projection
        num_dispatched = lora_info.token_ids.shape[0]
        lora_delta = torch.zeros(
            (num_dispatched, hidden_dim),
            dtype=intermediate_input.dtype,
            device=intermediate_input.device,
        )

        # IMPORTANT: For down_proj, the input (lora_intermediate_input) is already
        # gathered and indexed by dispatched position (0, 1, ..., num_dispatched-1),
        # not by original token position. So we pass identity indices for token_ids
        # to make the kernel read from the correct positions.
        dispatched_indices = torch.arange(
            num_dispatched,
            device=lora_info.token_ids.device,
            dtype=lora_info.token_ids.dtype,
        )

        _, lora_delta = per_expert_lora_forward(
            hidden_states=lora_intermediate_input,
            lora_a_weights=lora_info.down_lora_a_weights,
            lora_b_weights=lora_info.down_lora_b_weights,
            token_ids=dispatched_indices,  # Use identity indices, not original token_ids
            expert_ids=lora_info.expert_ids,
            lora_ids=lora_info.lora_ids,
            lora_ranks=lora_info.lora_ranks,
            lora_scalings=lora_info.lora_scalings,
            num_experts=lora_info.num_experts,
            base_output=lora_delta,
            is_down_proj=True,
        )

        # Apply router weights if not already applied to input
        # This matches the base MoE behavior
        if not apply_router_weight_on_input:
            # Get router weights for each dispatched pair
            router_weights = self._gather_router_weights(
                topk_weights=topk_weights,
                token_ids=lora_info.token_ids,
                expert_ids=lora_info.expert_ids,
                topk_ids=topk_ids,
            )
            lora_delta = lora_delta * router_weights.unsqueeze(-1)

        # Add delta to intermediate_cache
        self._scatter_add_to_topk_cache(
            lora_delta=lora_delta,
            intermediate_cache=intermediate_cache,
            token_ids=lora_info.token_ids,
            expert_ids=lora_info.expert_ids,
            topk_ids=topk_ids,
        )

    def _scatter_add_to_topk_cache(
        self,
        lora_delta: torch.Tensor,  # [num_dispatched, dim]
        intermediate_cache: torch.Tensor,  # [M, top_k, dim]
        token_ids: torch.Tensor,  # [num_dispatched]
        expert_ids: torch.Tensor,  # [num_dispatched]
        topk_ids: torch.Tensor,  # [M, top_k]
    ) -> None:
        """
        Scatter-add lora_delta to intermediate_cache based on dispatch info.

        For each dispatched index d:
            - token_id = token_ids[d]
            - expert_id = expert_ids[d]
            - Find k such that topk_ids[token_id, k] == expert_id
            - intermediate_cache[token_id, k] += lora_delta[d]
        """
        M, top_k, dim = intermediate_cache.shape

        # Find the top_k index for each dispatched pair
        # topk_ids[token_ids] gives [num_dispatched, top_k]
        # We need to find which column matches expert_ids
        expanded_topk = topk_ids[token_ids]  # [num_dispatched, top_k]
        expert_mask = expanded_topk == expert_ids.unsqueeze(
            1
        )  # [num_dispatched, top_k]

        # Get the k index for each dispatched pair
        k_indices = expert_mask.int().argmax(dim=1)  # [num_dispatched]

        # Compute flat indices into intermediate_cache viewed as [M * top_k, dim]
        flat_indices = token_ids * top_k + k_indices  # [num_dispatched]

        # Reshape cache for scatter_add
        cache_flat = intermediate_cache.view(M * top_k, dim)

        # Scatter add
        cache_flat.scatter_add_(
            0,
            flat_indices.unsqueeze(-1).expand(-1, dim),
            lora_delta.to(cache_flat.dtype),
        )

    def _gather_dispatched_inputs(
        self,
        intermediate_input: torch.Tensor,  # [M * top_k, dim]
        token_ids: torch.Tensor,  # [num_dispatched]
        expert_ids: torch.Tensor,  # [num_dispatched]
        topk_ids: torch.Tensor,  # [M, top_k]
        M: int,
        top_k: int,
    ) -> torch.Tensor:
        """
        Gather intermediate inputs for dispatched (token, expert) pairs.

        Returns tensor of shape [num_dispatched, dim].
        """
        # Find which top_k slot each dispatched pair corresponds to
        expanded_topk = topk_ids[token_ids]  # [num_dispatched, top_k]
        expert_mask = expanded_topk == expert_ids.unsqueeze(1)
        k_indices = expert_mask.int().argmax(dim=1)  # [num_dispatched]

        # Compute flat indices
        flat_indices = token_ids * top_k + k_indices

        # Gather
        return intermediate_input[flat_indices]

    def _gather_router_weights(
        self,
        topk_weights: torch.Tensor,  # [M, top_k]
        token_ids: torch.Tensor,  # [num_dispatched]
        expert_ids: torch.Tensor,  # [num_dispatched]
        topk_ids: torch.Tensor,  # [M, top_k]
    ) -> torch.Tensor:
        """
        Gather router weights for dispatched (token, expert) pairs.

        Returns tensor of shape [num_dispatched].
        """
        # Find which top_k slot each dispatched pair corresponds to
        expanded_topk = topk_ids[token_ids]  # [num_dispatched, top_k]
        expert_mask = expanded_topk == expert_ids.unsqueeze(1)
        k_indices = expert_mask.int().argmax(dim=1)  # [num_dispatched]

        # Gather weights
        expanded_weights = topk_weights[token_ids]  # [num_dispatched, top_k]
        return expanded_weights[
            torch.arange(len(token_ids), device=topk_weights.device), k_indices
        ]
