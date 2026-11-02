# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# Copyright 2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
"""FlashInfer's FP4 routed MoE split into routing metadata and the expert body.

The routing metadata only needs the packed top-k ids, so it can be built on the router stream while the
main stream quantizes activations; the main stream then launches the expert body alone. Output matches
trtllm_fp4_block_scale_routed_moe(do_finalize=False) with the default (untuned) tile and config.

Adapted from FlashInfer 0.7.2: the runner and arguments in ``RoutedMoeSplit.run_body`` from
``trtllm_fp4_block_scale_moe_op`` (flashinfer/fused_moe/core.py), and ``_default_tile`` from
``selectDefaultTileN`` (csrc/trtllm_fused_moe_kernel_launcher.cu). flashinfer-ai/flashinfer#8050 makes
the body-only call public as ``trtllm_fp4_block_scale_routed_moe(..., routing_metadata=...)``; switch
``run_body`` to it once SGLang requires a FlashInfer release that has it.
"""

from __future__ import annotations

from typing import Optional

import torch
from flashinfer.fused_moe import core
from flashinfer.fused_moe.core import (
    MoeRunnerInputs,
    populate_trtllm_moe_routing_metadata_,
    trtllm_moe_allocate_routing_metadata_multi_tile,
)
from flashinfer.fused_moe.da_moe import DABody
from flashinfer.tllm_enums import (
    Fp8QuantizationType,
    RoutingInputMode,
    WeightLayout,
    deduce_trtllm_gen_tensor_dtype,
)

# FlashInfer's NvFP4 x NvFP4 tile candidates; sync with selectDefaultTileN.
_FP4_TILES = (8, 16, 32, 64, 128, 192, 256)
# Measured range: decode up to c = 128 with 8 draft tokens. FlashInfer's own bound is
# trtllm_moe_max_da_multi_tile_tokens(num_experts), 2048 at 256 experts.
_MAX_SPLIT_TOKENS = 1024
# FlashInfer 0.7.2 builds no tile-192 FP4 kernel for SM107, while _FP4_TILES (GB300's list) has one;
# cap the split at tile 128 so it never asks for a tile the build lacks. FlashInfer main ships
# SM107 tile 192 (flashinfer-ai/flashinfer#8050 tests it); drop the cap with the switch to that API.
_MAX_SPLIT_TILE = 128


def _default_tile(*, num_tokens: int, top_k: int, num_local_experts: int) -> int:
    """The tile FlashInfer picks without a tuned tactic: one below the tile covering tokens per expert."""
    per_expert = num_tokens * top_k / num_local_experts
    i = next(
        (k for k, t in enumerate(_FP4_TILES) if t >= per_expert), len(_FP4_TILES) - 1
    )
    return _FP4_TILES[max(i - 1, 0)]


# Smallest token count from which the split was measured to beat the fused routing, per
# (compute capability, TP size); unlisted configurations keep the fused routing. On GB300 the router
# stream gates the experts below M = 512, where the split only adds overhead.
_SPLIT_MIN_TOKENS: dict[tuple[tuple[int, int], int], int] = {
    ((10, 3), 4): 512,
    ((10, 3), 8): 512,
    ((10, 7), 4): 8,
    ((10, 7), 8): 8,
}


def split_enabled(
    *, num_tokens: int, tp_size: int, capability: tuple[int, int]
) -> bool:
    min_tokens = _SPLIT_MIN_TOKENS.get((tuple(capability), tp_size))
    return min_tokens is not None and num_tokens >= min_tokens


def split_supported(*, num_tokens: int, top_k: int, num_local_experts: int) -> bool:
    return (
        num_tokens <= _MAX_SPLIT_TOKENS
        and _default_tile(
            num_tokens=num_tokens, top_k=top_k, num_local_experts=num_local_experts
        )
        <= _MAX_SPLIT_TILE
    )


class RoutedMoeSplit:
    """Graph-stable routing metadata for one token count, shared by all layers.

    Layers run in order on both streams: the router stream forks from main after the previous
    layer's body and its deferred finalize, so a populate never overwrites metadata still being read.
    """

    def __init__(
        self,
        packed_topk_ids: torch.Tensor,
        *,
        num_experts: int,
        local_expert_offset: int,
        num_local_experts: int,
    ):
        num_tokens, top_k = packed_topk_ids.shape
        self.tile_n = _default_tile(
            num_tokens=num_tokens, top_k=top_k, num_local_experts=num_local_experts
        )
        self._metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
            packed_topk_ids,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
            tile_ns=(self.tile_n,),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
        self._slot = self._metadata.slots[0]
        self._body = DABody(tactic=-1, tile_n=self.tile_n)

    def populate(self, packed_topk_ids: torch.Tensor) -> None:
        populate_trtllm_moe_routing_metadata_(self._metadata, packed_topk_ids)

    def run_body(
        self,
        *,
        topk_ids: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        per_token_scale: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: float,
        routing_method_type: int,
        do_finalize: bool,
        activation_type: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Takes trtllm_fp4_block_scale_routed_moe's arguments with packed ``topk_ids``; returns
        (gemm2_out, expert_weights, expanded_idx_to_permuted_idx), as do_finalize=False does."""
        if do_finalize:
            raise ValueError(
                "the split returns per-expert outputs only (do_finalize=False)"
            )
        module = core.get_trtllm_moe_sm100_module()
        num_tokens = topk_ids.shape[0]
        hidden_size = hidden_states.shape[-1] * 2
        device = hidden_states.device
        # Mirrors the runner trtllm_fp4_block_scale_moe_op builds for the same call.
        runner = module.MoERunner(
            module.moe_op,
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=deduce_trtllm_gen_tensor_dtype(
                hidden_states, hidden_states_scale
            ),
            dtype_weights=deduce_trtllm_gen_tensor_dtype(
                gemm1_weights, gemm1_weights_scale
            ),
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,
            hidden_size=hidden_size,
            hidden_size_output=hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
            use_per_token_scaling=per_token_scale is not None,
            num_experts=num_experts,
            num_fused_shared_experts=0,
        )
        inputs = MoeRunnerInputs(
            output=torch.empty(num_tokens, 0, dtype=torch.bfloat16, device=device),
            routing_logits=None,
            topk_ids=topk_ids,
            expert_weights=self._slot.expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_lora_delta=None,
            per_token_scale=per_token_scale,
        ).to_list()
        kwargs = dict(
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
            num_experts=num_experts,
            routing_bias=routing_bias,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=gemm1_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=gemm2_bias,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=output2_scale_scalar,
            n_group=n_group,
            topk_group=topk_group,
            local_expert_offset=local_expert_offset,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method_type,
            # Same PDL choice as the ordinary routed call: FlashInfer keeps it off for this
            # pipeline on SM107 (fused_moe.core._device_support_moe_pdl).
            enable_pdl=core.device_support_pdl(device)
            and torch.cuda.get_device_capability(device) != (10, 7),
            do_finalize=do_finalize,
            activation_type=activation_type,
            num_fused_shared_experts=0,
            norm_topk_prob=True,
            routing_replay_out=None,
            valid_hidden_size=None,
            valid_intermediate_size=None,
        )
        body_runner = module.DABodyRunner(runner)
        # Allocated per call, as the ordinary routed call does: under graph capture the buffers
        # come from the graph pool shared by every captured batch size.
        workspace = body_runner.prepare_body(inputs, self._body, self._slot, **kwargs)
        body_runner.forward_from_metadata(
            inputs, self._body, self._slot, workspace, **kwargs
        )
        # Body workspace field 4 is the FC2 output; sync with FlashInfer's FP4 DA body ABI.
        return (
            workspace[4],
            self._slot.expert_weights,
            self._slot.expanded_idx_to_permuted_idx,
        )


_splits: dict[tuple, RoutedMoeSplit] = {}


def get_routed_moe_split(
    packed_topk_ids: torch.Tensor,
    *,
    num_experts: int,
    local_expert_offset: int,
    num_local_experts: int,
) -> RoutedMoeSplit:
    key = (
        tuple(packed_topk_ids.shape),
        packed_topk_ids.device,
        num_experts,
        local_expert_offset,
        num_local_experts,
    )
    split = _splits.get(key)
    if split is None:
        split = _splits[key] = RoutedMoeSplit(
            packed_topk_ids,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
        )
    return split
