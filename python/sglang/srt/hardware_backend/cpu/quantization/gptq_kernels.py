# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.utils import logger

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig

__all__ = ["GPTQIntelAMXLinearKernel", "GPTQIntelAMXMoEKernel"]


class GPTQIntelAMXLinearKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _amx_process_weight_after_loading(
            layer, ["qweight", "qzeros", "scales"], None, "gptq"
        )
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias,
        )


def _slice_or_pad(
    t: torch.Tensor, dim: int, target: int, pad_value: float | int = 0
) -> torch.Tensor:
    cur = t.size(dim)
    if cur == target:
        return t
    if cur > target:
        idx = [slice(None)] * t.dim()
        idx[dim] = slice(0, target)
        return t[tuple(idx)].contiguous()

    pad_shape = list(t.shape)
    pad_shape[dim] = target - cur
    pad = torch.full(
        pad_shape,
        fill_value=pad_value,
        dtype=t.dtype,
        device=t.device,
    )
    return torch.cat([t, pad], dim=dim).contiguous()


def _replace_param(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    old = getattr(layer, name)
    new_param = torch.nn.Parameter(tensor, requires_grad=False)
    new_param.__dict__ = old.__dict__
    setattr(layer, name, new_param)


def _can_pack_gptq_int4_tensor(qweight: torch.Tensor, scales: torch.Tensor) -> bool:
    # qweight: [E, K/8, N], scales: [E, G, N]
    if qweight.dim() != 3 or scales.dim() != 3:
        return False
    k = qweight.size(1) * 8
    n = qweight.size(2)
    g = scales.size(1)
    if n % 32 != 0 or g <= 0:
        return False
    if k % g != 0:
        return False
    group_size = k // g
    block_k = 128 if group_size > 128 else group_size
    if block_k <= 0:
        return False
    return (k % block_k) == 0


def _normalize_moe_int4_for_amx(layer: torch.nn.Module, group_size: int) -> None:
    # Align GPTQ MoE tensors to AMX int4 pack assumptions and discard padded tails.
    w2_k_from_qweight = layer.w2_qweight.size(1) * 8
    w2_k_from_scales = layer.w2_scales.size(1) * group_size
    w2_k_from_qzeros = layer.w2_qzeros.size(1) * group_size
    target_w2_k = min(w2_k_from_qweight, w2_k_from_scales, w2_k_from_qzeros)
    if target_w2_k <= 0:
        return

    if target_w2_k % 8 != 0:
        target_w2_k = (target_w2_k // 8) * 8

    changed = False

    target_w2_k_div8 = target_w2_k // 8
    new_w2_qweight = _slice_or_pad(layer.w2_qweight.data, 1, target_w2_k_div8, 0)
    if new_w2_qweight.shape != layer.w2_qweight.shape:
        _replace_param(layer, "w2_qweight", new_w2_qweight)
        changed = True

    if hasattr(layer, "w2_g_idx"):
        new_w2_g_idx = _slice_or_pad(layer.w2_g_idx.data, 1, target_w2_k, 0)
        if new_w2_g_idx.shape != layer.w2_g_idx.shape:
            _replace_param(layer, "w2_g_idx", new_w2_g_idx)
            changed = True

    target_w13_n = target_w2_k * 2
    new_w13_qweight = _slice_or_pad(layer.w13_qweight.data, 2, target_w13_n, 0)
    if new_w13_qweight.shape != layer.w13_qweight.shape:
        _replace_param(layer, "w13_qweight", new_w13_qweight)
        changed = True

    new_w13_scales = _slice_or_pad(layer.w13_scales.data, 2, target_w13_n, 0)
    if new_w13_scales.shape != layer.w13_scales.shape:
        _replace_param(layer, "w13_scales", new_w13_scales)
        changed = True

    target_w13_n_div8 = target_w13_n // 8
    new_w13_qzeros = _slice_or_pad(layer.w13_qzeros.data, 2, target_w13_n_div8, 0)
    if new_w13_qzeros.shape != layer.w13_qzeros.shape:
        _replace_param(layer, "w13_qzeros", new_w13_qzeros)
        changed = True

    if changed:
        logger.warning(
            "Normalized GPTQ MoE shapes for AMX int4 pack: "
            "w13_qweight=%s w13_qzeros=%s w13_scales=%s "
            "w2_qweight=%s w2_qzeros=%s w2_scales=%s",
            tuple(layer.w13_qweight.shape),
            tuple(layer.w13_qzeros.shape),
            tuple(layer.w13_scales.shape),
            tuple(layer.w2_qweight.shape),
            tuple(layer.w2_qzeros.shape),
            tuple(layer.w2_scales.shape),
        )


class GPTQIntelAMXMoEKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.moe_runner_config: Optional[MoeRunnerConfig] = None

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.group_size and self.quant_config.group_size > 0:
            _normalize_moe_int4_for_amx(layer, self.quant_config.group_size)

        can_pack_w13 = _can_pack_gptq_int4_tensor(layer.w13_qweight, layer.w13_scales)
        can_pack_w2 = _can_pack_gptq_int4_tensor(layer.w2_qweight, layer.w2_scales)
        if not (can_pack_w13 and can_pack_w2):
            raise RuntimeError(
                "GPTQ CPU AMX MoE prepack shape mismatch after normalization: "
                f"w13_qweight={tuple(layer.w13_qweight.shape)} "
                f"w13_qzeros={tuple(layer.w13_qzeros.shape)} "
                f"w13_scales={tuple(layer.w13_scales.shape)} "
                f"w2_qweight={tuple(layer.w2_qweight.shape)} "
                f"w2_qzeros={tuple(layer.w2_qzeros.shape)} "
                f"w2_scales={tuple(layer.w2_scales.shape)}"
            )

        _amx_process_weight_after_loading(
            layer, ["w13_qweight", "w13_qzeros", "w13_scales"], None, "gptq"
        )
        _amx_process_weight_after_loading(
            layer, ["w2_qweight", "w2_qzeros", "w2_scales"], None, "gptq"
        )
        layer.use_intel_amx_backend = True

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation in ["silu", "relu2"], (
            "Only SiLU/ReLU2 activations are supported."
        )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        output = torch.ops.sgl_kernel.fused_experts_cpu(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            False,  # inplace See [Note] inplace should be False in fused_experts.
            CPUQuantMethod.INT4_W4A8,
            layer.w13_scales,  # w1_scale
            layer.w2_scales,  # w2_scale
            layer.w13_qzeros,
            layer.w2_qzeros,
            None,  # block_size
            None,  # w1 bias
            None,  # w3 bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
            self.moe_runner_config.activation,  # activation
        )
        return StandardCombineInput(hidden_states=output)
