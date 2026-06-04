# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.utils import per_tensor_dequantize

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.modelopt_quant import PerTokenNvFp4Config

logger = logging.getLogger(__name__)


class ModelOptPerTokenNvFp4FusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """Online per-token NVFP4 MoE method for BF16/FP16/FP8 checkpoints."""

    def __init__(self, quant_config: PerTokenNvFp4Config, layer_prefix: str):
        super().__init__(quant_config)
        self.layer_prefix = layer_prefix
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_prefix)
        self.layer_log_name = (
            f"layer {layer_match.group(1)} ({layer_prefix})"
            if layer_match is not None
            else layer_prefix
        )
        if not self.enable_flashinfer_trtllm_moe:
            raise ValueError(
                "--quantization per_token_nvfp4 supports only "
                "--moe-runner-backend flashinfer_trtllm or "
                "flashinfer_trtllm_routed."
            )

    @staticmethod
    def _quantize_weight_nvfp4(
        weight: torch.Tensor,
        weight_scale_2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from flashinfer import SfLayout, nvfp4_quantize

        if weight.ndim != 2:
            raise ValueError(
                "--quantization per_token_nvfp4 expects 2D expert weights, "
                f"got shape {tuple(weight.shape)}."
            )
        if weight.shape[-1] % 16 != 0:
            raise ValueError(
                "--quantization per_token_nvfp4 requires expert weight K to be "
                f"a multiple of 16, got shape {tuple(weight.shape)}."
            )

        if weight_scale_2 is None:
            # weight_scale_2 is the NVFP4 decode scale. FlashInfer consumes its
            # reciprocal as the global encode scale, matching 448 * 6 / amax.
            weight_amax = (
                weight.abs()
                .nan_to_num()
                .amax()
                .to(device=weight.device, dtype=torch.float32)
            )
            fp8_fp4_max = float(torch.finfo(torch.float8_e4m3fn).max) * 6.0
            weight_scale_2 = torch.where(
                weight_amax > 0,
                weight_amax / fp8_fp4_max,
                torch.ones_like(weight_amax),
            )
        else:
            weight_scale_2 = weight_scale_2.to(
                device=weight.device, dtype=torch.float32
            )
        fp4_weight, weight_sf = nvfp4_quantize(
            weight.contiguous(),
            1.0 / weight_scale_2,
            sfLayout=SfLayout.layout_linear,
            backend="cuda",
        )
        rows, cols = weight.shape
        weight_sf = weight_sf.view(torch.float8_e4m3fn).reshape(rows, cols // 16)
        return (
            fp4_weight.reshape(rows, cols // 2),
            weight_sf.contiguous(),
            weight_scale_2,
        )

    @staticmethod
    def _is_fp8_weight(weight: torch.Tensor) -> bool:
        fp8_dtypes = {
            dtype
            for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
            )
            if dtype is not None
        }
        return weight.dtype in fp8_dtypes

    @staticmethod
    def _is_fp8_weight_scale_name(weight_name: str) -> bool:
        return "weight_scale" in weight_name and "weight_scale_2" not in weight_name

    def _dequantize_fp8_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if getattr(self.quant_config, "use_mxfp8", False):
            raise ValueError(
                "--quantization per_token_nvfp4 does not support online "
                "requantization from MXFP8 expert checkpoints."
            )

        weight = weight.to(device).contiguous()
        weight_scale = weight_scale.to(device=device).contiguous()
        if weight_scale.dtype == torch.int32:
            weight_scale = inverse_transform_scale_ue8m0(
                weight_scale, mn=weight.shape[-2]
            )
        weight_scale = weight_scale.to(dtype=torch.float32).contiguous()

        if weight_scale.numel() == 1 or self.quant_config.weight_block_size is None:
            return (
                per_tensor_dequantize(weight, weight_scale)
                .to(torch.bfloat16)
                .contiguous()
            )

        return block_quant_dequant(
            weight,
            weight_scale,
            self.quant_config.weight_block_size,
            torch.bfloat16,
        ).contiguous()

    @staticmethod
    def _should_skip_loaded_expert(
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        expert_id: Optional[int],
    ) -> bool:
        if expert_id is None:
            return False
        if getattr(param, "_sglang_require_global_experts", False):
            return False
        # With EPLB or explicit expert placement, logical expert IDs can map to
        # one or more physical experts. Let the canonical MoE loader do that
        # mapping instead of pre-skipping from the trivial EP layout.
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

        if get_global_expert_location_metadata() is not None:
            return False
        return layer._map_global_expert_id_to_local_expert_id(expert_id) == -1

    @staticmethod
    def _scale_weight_name(weight_name: str) -> str:
        if "weight" in weight_name:
            return weight_name.replace("weight", "weight_scale")
        return f"{weight_name}.weight_scale"

    @staticmethod
    def _scale_2_weight_name(weight_name: str) -> str:
        if "weight" in weight_name:
            return weight_name.replace("weight", "weight_scale_2")
        return f"{weight_name}.weight_scale_2"

    def get_online_weight_loader(self, layer, original_weight_loader):
        pending_w13_weights = {}
        pending_w13_lock = threading.Lock()
        pending_fp8_weights = {}
        pending_fp8_weight_scales = {}
        pending_fp8_lock = threading.Lock()
        quantization_log_lock = threading.Lock()
        did_log_quantization = False

        def log_quantization_start() -> None:
            nonlocal did_log_quantization
            if did_log_quantization:
                return
            with quantization_log_lock:
                if did_log_quantization:
                    return
                logger.info(
                    "Running online NVFP4 quantization for MoE expert weights in %s.",
                    self.layer_log_name,
                )
                did_log_quantization = True

        def store_quantized_weight(
            param: torch.nn.Parameter,
            fp4_weight: torch.Tensor,
            weight_scale: torch.Tensor,
            weight_scale_2: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: Optional[int],
        ) -> None:
            original_weight_loader(
                param,
                fp4_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )

            scale_param = (
                layer.w13_weight_scale
                if shard_id in ("w1", "w3")
                else layer.w2_weight_scale
            )
            original_weight_loader(
                scale_param,
                weight_scale,
                weight_name=self._scale_weight_name(weight_name),
                shard_id=shard_id,
                expert_id=expert_id,
            )
            scale_2_param = (
                layer.w13_weight_scale_2
                if shard_id in ("w1", "w3")
                else layer.w2_weight_scale_2
            )
            original_weight_loader(
                scale_2_param,
                weight_scale_2,
                weight_name=self._scale_2_weight_name(weight_name),
                shard_id=shard_id,
                expert_id=expert_id,
            )

        def process_loaded_weight(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: Optional[int],
        ) -> None:
            log_quantization_start()
            if shard_id == "w2":
                loaded_weight = loaded_weight.to(param.device)
                fp4_weight, weight_scale, weight_scale_2 = self._quantize_weight_nvfp4(
                    loaded_weight
                )
                store_quantized_weight(
                    param,
                    fp4_weight,
                    weight_scale,
                    weight_scale_2,
                    weight_name,
                    shard_id,
                    expert_id,
                )
                return

            pending_key = expert_id
            current = (
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )
            with pending_w13_lock:
                pending = pending_w13_weights.pop(pending_key, None)
                if pending is None:
                    pending_w13_weights[pending_key] = current
                    return

            (
                pending_param,
                pending_weight,
                pending_name,
                pending_shard_id,
                pending_eid,
            ) = pending
            if pending_shard_id == shard_id:
                raise ValueError(
                    "--quantization per_token_nvfp4 expects paired w1/w3 expert "
                    f"weights, got two {shard_id} tensors for expert {expert_id}."
                )
            pending_weight = pending_weight.to(param.device)
            loaded_weight = loaded_weight.to(param.device)
            pending_rows = pending_weight.shape[0]
            loaded_rows = loaded_weight.shape[0]
            fp4_weight, weight_scale, weight_scale_2 = self._quantize_weight_nvfp4(
                torch.cat([pending_weight, loaded_weight], dim=0)
            )
            pending_fp4_weight, loaded_fp4_weight = fp4_weight.split(
                [pending_rows, loaded_rows], dim=0
            )
            pending_weight_scale, loaded_weight_scale = weight_scale.split(
                [pending_rows, loaded_rows], dim=0
            )
            store_quantized_weight(
                pending_param,
                pending_fp4_weight.contiguous(),
                pending_weight_scale.contiguous(),
                weight_scale_2,
                pending_name,
                pending_shard_id,
                pending_eid,
            )
            store_quantized_weight(
                param,
                loaded_fp4_weight.contiguous(),
                loaded_weight_scale.contiguous(),
                weight_scale_2,
                weight_name,
                shard_id,
                expert_id,
            )

        def process_fp8_weight(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: Optional[int],
        ) -> None:
            if not self._is_fp8_weight(loaded_weight):
                process_loaded_weight(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )
                return
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "--quantization per_token_nvfp4 received an FP8 expert "
                    "weight, but the checkpoint quantization config does not "
                    "declare serialized FP8 weights."
                )

            key = (expert_id, shard_id)
            with pending_fp8_lock:
                weight_scale = pending_fp8_weight_scales.pop(key, None)
                if weight_scale is None:
                    pending_fp8_weights[key] = (
                        param,
                        loaded_weight,
                        weight_name,
                        shard_id,
                        expert_id,
                    )
                    return

            log_quantization_start()
            loaded_weight = self._dequantize_fp8_weight(
                loaded_weight, weight_scale, param.device
            )
            process_loaded_weight(
                param, loaded_weight, weight_name, shard_id, expert_id
            )

        def process_fp8_weight_scale(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: Optional[int],
        ) -> None:
            key = (expert_id, shard_id)
            with pending_fp8_lock:
                pending = pending_fp8_weights.pop(key, None)
                if pending is None:
                    pending_fp8_weight_scales[key] = loaded_weight
                    return

            log_quantization_start()
            (
                pending_param,
                pending_weight,
                pending_name,
                pending_shard_id,
                pending_eid,
            ) = pending
            loaded_weight = self._dequantize_fp8_weight(
                pending_weight, loaded_weight, pending_param.device
            )
            process_loaded_weight(
                pending_param,
                loaded_weight,
                pending_name,
                pending_shard_id,
                pending_eid,
            )

        def online_per_token_nvfp4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: Optional[int],
        ):
            if shard_id not in ("w1", "w2", "w3"):
                original_weight_loader(
                    param,
                    loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                return
            if self._should_skip_loaded_expert(layer, param, expert_id):
                return

            if self._is_fp8_weight_scale_name(weight_name):
                process_fp8_weight_scale(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )
                return

            if "weight" in weight_name:
                process_fp8_weight(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )
                return

            original_weight_loader(
                param,
                loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )

        return online_per_token_nvfp4_weight_loader
