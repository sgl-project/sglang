# SPDX-License-Identifier: Apache-2.0
"""gfx950 routes of minimax_m3_nvfp4: the lm_head GEMM on aiter's tuned bf16 GEMM table,
and the NVFP4 routed experts requantized to MXFP4 at load for aiter's fused MoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.runtime_context import get_exec, get_platform

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunner, MoeRunnerConfig


class AiterTunedLmHeadMethod(UnquantizedEmbeddingMethod):
    """Unquantized lm_head whose logits GEMM goes through aiter's tuned bf16 GEMM
    table (tgemm, ROCm); shapes without a row run F.linear."""

    def __init__(self):
        super().__init__()
        # bound here: the module-level tgemm import follows get_bool_env_var, which reads
        # SGLANG_USE_AITER more narrowly than the envs check that selects this method
        from aiter.tuned_gemm import tgemm

        self._tgemm = tgemm

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = layer.weight.dtype
        return self._tgemm.mm(x.to(dtype), layer.weight, bias, otype=dtype)


def uses_aiter_tuned_lm_head(layer: torch.nn.Module) -> bool:
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    if not isinstance(layer, ParallelLMHead):
        return False
    # the bf16 lm_head rows ship in the gfx950 aiter table registered by the M3 model
    # override; an fp32 lm_head and the deterministic / on-policy modes keep
    # LogitsProcessor's own GEMM
    platform = get_platform()
    if not (platform.is_hip and platform.is_gfx95 and envs.SGLANG_USE_AITER.get()):
        return False
    exec_ = get_exec()
    return (
        not exec_.features.enable_fp32_lm_head
        and not exec_.deterministic.enable_deterministic_inference
        and exec_.deterministic.rl_on_policy_target is None
    )


def create_requantized_runner(moe_runner_config: MoeRunnerConfig) -> MoeRunner:
    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
    from sglang.srt.layers.moe.moe_runner.aiter import aiter_swiglu_oai_limit

    if (
        moe_runner_config.gemm1_clamp_limit is not None
        and aiter_swiglu_oai_limit(moe_runner_config) is None
    ):
        raise NotImplementedError(
            "aiter implements SwiGLU-OAI only at alpha 1.702; this "
            f"checkpoint uses {moe_runner_config.gemm1_alpha}"
        )
    return MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)


def process_requantized_weights(layer: torch.nn.Module) -> None:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    for prefix in ("w13", "w2"):
        _requantize_experts(layer, prefix)

    # aiter reads shuffled E8M0 scales and (16,16)-preshuffled tiles, or corrupts silently
    for prefix in ("w13", "w2"):
        scale = getattr(layer, f"{prefix}_weight_scale")
        num_experts, rows, _ = scale.shape
        scale.data = e8m0_shuffle(scale.data.view(num_experts * rows, -1)).view(
            num_experts, rows, -1
        )
        weight = getattr(layer, f"{prefix}_weight")
        weight.data = shuffle_weight(weight.data.contiguous(), (16, 16))
        weight.is_shuffled = True


def _requantize_experts(layer: torch.nn.Module, prefix: str):
    """Requantize one fused NVFP4 expert tensor to MXFP4, an expert at a time.

    M3.1 stores an encode global scale, so its reciprocal is what `dequantize_nvfp4`
    multiplies by.
    """
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

    packed = getattr(layer, f"{prefix}_weight_packed").data
    block_scale = getattr(layer, f"{prefix}_weight_scale").data
    global_scale = getattr(layer, f"{prefix}_weight_global_scale").data.float()

    num_experts = packed.shape[0]
    stacked_weights = stacked_scales = None
    for expert in range(num_experts):
        if prefix == "w13":
            # global_scale[expert] is [gate, up]; expand each over its half of the rows
            half = packed[expert].shape[0] // 2
            decode_scale = torch.cat(
                [
                    global_scale[expert, 0].reciprocal().repeat(half),
                    global_scale[expert, 1].reciprocal().repeat(half),
                ]
            ).view(-1, 1)
        else:
            decode_scale = global_scale[expert].reciprocal().view(1, 1)
        expert_weight = dequantize_nvfp4(
            packed[expert],
            block_scale[expert],
            decode_scale,
            out_dtype=torch.float32,
        )
        requantized_weight, requantized_scale = dynamic_mxfp4_quant(
            expert_weight.to(torch.bfloat16)
        )
        if stacked_weights is None:
            stacked_weights = torch.empty(
                (num_experts, *requantized_weight.shape),
                dtype=requantized_weight.dtype,
                device=requantized_weight.device,
            )
            stacked_scales = torch.empty(
                (num_experts, *requantized_scale.shape),
                dtype=requantized_scale.dtype,
                device=requantized_scale.device,
            )
        stacked_weights[expert] = requantized_weight
        stacked_scales[expert] = requantized_scale

    setattr(
        layer,
        f"{prefix}_weight",
        torch.nn.Parameter(stacked_weights, requires_grad=False),
    )
    setattr(
        layer,
        f"{prefix}_weight_scale",
        torch.nn.Parameter(stacked_scales, requires_grad=False),
    )
    # drop the NVFP4 source: the largest tensors in the model, live twice until here
    setattr(layer, f"{prefix}_weight_packed", None)


def apply_requantized_experts(scheme, layer: torch.nn.Module, dispatch_output):
    from aiter.ops.flydsl.moe_common import GateMode

    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterMoeQuantInfo,
        AiterQuantType,
        aiter_swiglu_oai_limit,
    )

    w13 = layer.w13_weight.view(torch.float4_e2m1fn_x2)
    w2 = layer.w2_weight.view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = w2.is_shuffled = True

    quant_info = AiterMoeQuantInfo(
        # the loader keeps gate and up as separate row ranges; aiter defaults to INTERLEAVE
        fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
        w13_weight=w13,
        w2_weight=w2,
        quant_type=AiterQuantType.PER_1X32,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        expert_mask=layer.dispatcher.expert_mask_gpu,
        swiglu_limit=aiter_swiglu_oai_limit(scheme.moe_runner_config) or 0.0,
    )
    return scheme.runner.run(dispatch_output, quant_info)
