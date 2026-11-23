# Copyright 2026 Antgroup. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Reblock logical TP shards using checkpoint-global FP8 scale coordinates."""

import torch

from sglang.srt.runtime_context import get_platform


def padded_bailing_fp8_intermediate_size(quant_config, intermediate_size, tp_size):
    if (
        get_platform().is_hip
        or quant_config is None
        or quant_config.get_name() != "fp8"
        or not getattr(quant_config, "is_checkpoint_fp8_serialized", False)
        or getattr(quant_config, "use_mxfp8", False)
        or getattr(quant_config, "is_fp4_experts", False)
        or getattr(quant_config, "weight_block_size", None) != [128, 128]
        or tp_size <= 1
    ):
        return None
    if intermediate_size % tp_size:
        raise ValueError("Bailing intermediate size must be divisible by its TP size")
    local_size = intermediate_size // tp_size
    if local_size % 128 == 0:
        return None
    return ((local_size + 127) // 128) * 128 * tp_size


class BailingFp8TPPadding:
    def __init__(
        self,
        owner,
        intermediate_size: int,
        tp_rank: int,
        tp_size: int,
        block_size: int,
        *,
        num_experts: int | None = None,
    ):
        if (
            tp_size <= 1
            or not 0 <= tp_rank < tp_size
            or intermediate_size % tp_size
            or block_size <= 0
        ):
            raise ValueError("Invalid Bailing FP8 TP padding geometry")
        self.owner = owner
        self.logical_size = intermediate_size // tp_size
        self.padded_size = (
            (self.logical_size + block_size - 1) // block_size * block_size
        )
        if self.logical_size == self.padded_size:
            raise ValueError("Bailing FP8 TP padding requires a partial block")
        self.intermediate_size = intermediate_size
        self.offset = tp_rank * self.logical_size
        self.block_size = block_size
        self.num_experts = num_experts
        self.scales = {}
        self.loaded_weights = set()
        # load() writes live weights before finalize() can see every tensor,
        # so the updater must reject online updates before any write.
        owner._derived_weight_cache_error = (
            "Online weight updates are not supported while Bailing FP8 TP "
            "reblocking is active: runtime FP8 weights and scales are rebuilt "
            "only from a complete checkpoint load. Restart to load new weights."
        )

    def _parameters(self, projection: str, expert_id: int | None):
        if projection not in ("gate_proj", "up_proj", "down_proj"):
            raise ValueError(f"Unknown Bailing FP8 projection {projection!r}")
        if self.num_experts is None:
            if expert_id is not None:
                raise ValueError("Shared Bailing FP8 weights have no expert index")
            layer = (
                self.owner.down_proj
                if projection == "down_proj"
                else self.owner.gate_up_proj
            )
            return layer.weight, layer.weight_scale_inv
        if expert_id is None or not 0 <= expert_id < self.num_experts:
            raise ValueError(f"Invalid Bailing FP8 expert index {expert_id}")
        if projection == "down_proj":
            return (
                self.owner.w2_weight[expert_id],
                self.owner.w2_weight_scale_inv[expert_id],
            )
        return (
            self.owner.w13_weight[expert_id],
            self.owner.w13_weight_scale_inv[expert_id],
        )

    def load(
        self,
        projection: str,
        value: torch.Tensor,
        *,
        is_scale: bool,
        expert_id: int | None = None,
    ) -> None:
        weight, _ = self._parameters(projection, expert_id)
        hidden_size = weight.shape[0] if projection == "down_proj" else weight.shape[1]
        full_shape = (
            (hidden_size, self.intermediate_size)
            if projection == "down_proj"
            else (self.intermediate_size, hidden_size)
        )
        expected_shape = (
            tuple((dim + self.block_size - 1) // self.block_size for dim in full_shape)
            if is_scale
            else full_shape
        )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Unexpected Bailing FP8 {projection} shape {tuple(value.shape)}; "
                f"expected {expected_shape}"
            )
        key = (expert_id, projection)
        if is_scale:
            if not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError(
                    "Bailing FP8 checkpoint scales must be finite and positive"
                )
            self.scales[key] = value.to(
                device=weight.device, dtype=torch.float32
            ).clone()
            return
        if value.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Bailing FP8 TP padding requires serialized E4M3 weights")
        dim = 1 if projection == "down_proj" else 0
        logical = value.narrow(dim, self.offset, self.logical_size)
        with torch.no_grad():
            if projection == "down_proj":
                weight.zero_()
                weight[:, : self.logical_size].copy_(logical)
            else:
                start = self.padded_size if projection == "up_proj" else 0
                target = weight[start : start + self.padded_size]
                target.zero_()
                target[: self.logical_size].copy_(logical)
        self.loaded_weights.add(key)

    def _dequantize(self, weight, scale, projection):
        rows = torch.arange(weight.shape[0], device=weight.device)
        cols = torch.arange(weight.shape[1], device=weight.device)
        if projection == "down_proj":
            cols = cols + self.offset
        else:
            rows = rows + self.offset
        factors = scale.index_select(0, rows // self.block_size).index_select(
            1, cols // self.block_size
        )
        return weight.float() * factors

    def _quantize(self, weight):
        rows, cols = weight.shape
        block = self.block_size
        blocks = weight.reshape(rows // block, block, cols // block, block)
        scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
        quantized = (
            (blocks / scale[:, None, :, None])
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        return quantized.reshape(rows, cols), scale

    @torch.no_grad()
    def finalize(self) -> None:
        experts = range(self.num_experts) if self.num_experts is not None else (None,)
        expected = {
            (expert, projection)
            for expert in experts
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        missing_weights = expected - self.loaded_weights
        missing_scales = expected - self.scales.keys()
        if missing_weights or missing_scales:
            raise ValueError(
                "Missing Bailing FP8 TP checkpoint tensors: "
                f"weights={sorted(missing_weights, key=str)}, "
                f"scales={sorted(missing_scales, key=str)}"
            )
        for expert in experts:
            gate_up, gate_up_scale = self._parameters("gate_proj", expert)
            down, down_scale = self._parameters("down_proj", expert)
            combined = torch.zeros_like(gate_up, dtype=torch.float32)
            for projection, start in (("gate_proj", 0), ("up_proj", self.padded_size)):
                original = gate_up[start : start + self.logical_size]
                combined[start : start + self.logical_size] = self._dequantize(
                    original, self.scales[(expert, projection)], projection
                )
            local_down = torch.zeros_like(down, dtype=torch.float32)
            local_down[:, : self.logical_size] = self._dequantize(
                down[:, : self.logical_size],
                self.scales[(expert, "down_proj")],
                "down_proj",
            )
            quantized, scale = self._quantize(combined)
            gate_up.copy_(quantized)
            gate_up_scale.copy_(scale)
            quantized, scale = self._quantize(local_down)
            down.copy_(quantized)
            down_scale.copy_(scale)
        self.scales.clear()
        self.loaded_weights.clear()
