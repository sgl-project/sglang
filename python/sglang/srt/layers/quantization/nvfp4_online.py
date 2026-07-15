# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
    ModelOptQuantConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import (
    is_layer_skipped,
    per_tensor_dequantize,
)

logger = logging.getLogger(__name__)


class NvFp4OnlineConfig(ModelOptQuantConfig):
    """Config for `--quantization nvfp4_online`.

    This mode is a load-time conversion path, not a serialized NVFP4 checkpoint
    format. It reuses the ModelOpt NVFP4 MoE parameter layout and fills those
    parameters by converting BF16/FP16/FP8 expert tensors as they are loaded.
    Dense layers stay in the source checkpoint precision or quantization path.
    """

    # Marker consumed by the ModelOpt FP4 layout and the model loader. Serialized
    # NVFP4 checkpoints use ModelOptFp4Config instead.
    is_nvfp4_online = True
    is_checkpoint_nvfp4_serialized = False
    group_size = 16

    @staticmethod
    def _normalize_ignored_layers(
        ignored_layers: Optional[List[str]],
    ) -> List[str]:
        if not ignored_layers:
            return []
        normalized_ignored_layers = []
        for layer in ignored_layers:
            base = layer.removeprefix("model.")
            normalized_ignored_layers.append(base)
            normalized_ignored_layers.append(f"model.{base}")
        return list(dict.fromkeys(normalized_ignored_layers))

    def __init__(
        self,
        exclude_modules: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        weight_block_size: Optional[List[int]] = None,
        use_mxfp8: bool = False,
    ) -> None:
        source_ignored_layers = self._normalize_ignored_layers(exclude_modules)
        fp4_ignored_layers = list(source_ignored_layers)
        if ignored_layers_str := envs.SGLANG_FP4_IGNORED_LAYERS.get():
            fp4_ignored_layers.extend(
                layer.strip()
                for layer in ignored_layers_str.split(",")
                if layer.strip()
            )
        fp4_ignored_layers = self._normalize_ignored_layers(fp4_ignored_layers)
        super().__init__(
            kv_cache_quant_algo=None,
            exclude_modules=source_ignored_layers,
            packed_modules_mapping=packed_modules_mapping or {},
        )
        self.fp4_ignored_layers = fp4_ignored_layers
        # Weights use static NVFP4 scales, while FlashInfer computes activation
        # FP32 scales dynamically per token at runtime.
        self.use_per_token_activation = True
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_fp4_experts = False
        self.dequant_fp4_to_fp8 = False
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        self.use_mxfp8 = use_mxfp8

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4_online"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> NvFp4OnlineConfig:
        quant_method = str(config.get("quant_method", "")).lower()
        use_mxfp8 = "mxfp8" in quant_method
        is_checkpoint_fp8_serialized = "fp8" in quant_method or use_mxfp8
        ignored_layers = config.get("ignored_layers") or config.get(
            "modules_to_not_convert"
        )
        if isinstance(ignored_layers, str):
            ignored_layers = [ignored_layers]
        return cls(
            exclude_modules=ignored_layers,
            packed_modules_mapping=config.get("packed_modules_mapping"),
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=config.get("activation_scheme", "dynamic"),
            weight_block_size=config.get("weight_block_size"),
            use_mxfp8=use_mxfp8,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping
            ) or self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            if self.is_checkpoint_fp8_serialized:
                return Fp8LinearMethod(self)
            return UnquantizedLinearMethod()
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping
            ) or self.is_layer_excluded(prefix):
                return None
            if is_layer_skipped(
                prefix, self.fp4_ignored_layers, self.packed_modules_mapping
            ):
                if self.is_checkpoint_fp8_serialized:
                    return Fp8MoEMethod(self)
                return None
            return ModelOptNvFp4OnlineFusedMoEMethod(self, prefix)
        return None


class ModelOptNvFp4OnlineFusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """MoE method that converts source expert weights to NVFP4 during loading."""

    def __init__(self, quant_config: NvFp4OnlineConfig, layer_prefix: str):
        super().__init__(quant_config)
        self.layer_prefix = layer_prefix
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_prefix)
        self.layer_log_name = (
            f"layer {layer_match.group(1)} ({layer_prefix})"
            if layer_match is not None
            else layer_prefix
        )
        if not self.supports_nvfp4_online_moe:
            raise ValueError(
                "--quantization nvfp4_online supports flashinfer_trtllm, "
                "flashinfer_trtllm_routed, or flashinfer_cutedsl with "
                "FlashInfer A2A."
            )

    @staticmethod
    def _quantize_weight_nvfp4(
        weight: torch.Tensor,
        weight_scale_2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return packed NVFP4 weight, block scales, and per-tensor decode scale.

        The weight scale is static and per tensor. Callers pass an existing
        scale when multiple shards must share one global scale, for example the
        gated w1/w3 pair.
        """
        if weight.ndim != 2:
            raise ValueError(
                "--quantization nvfp4_online expects 2D expert weights, "
                f"got shape {tuple(weight.shape)}."
            )
        if not weight.is_floating_point():
            raise ValueError(
                "--quantization nvfp4_online expects floating-point source "
                f"expert weights, got dtype {weight.dtype}. Serialized packed "
                "FP4 weights must use --quantization modelopt_fp4."
            )
        if weight.shape[-1] % 16 != 0:
            raise ValueError(
                "--quantization nvfp4_online requires expert weight K to be "
                f"a multiple of 16, got shape {tuple(weight.shape)}."
            )

        from flashinfer import SfLayout, nvfp4_quantize

        if weight_scale_2 is None:
            # weight_scale_2 is the NVFP4 decode scale. FlashInfer consumes its
            # reciprocal as the global encode scale, matching 448 * 6 / amax.
            weight_amax = (
                weight.abs()
                .nan_to_num()
                .amax()
                .to(device=weight.device, dtype=torch.float32)
            )
            e4m3_max = (
                256.0
                if envs.FLASHINFER_NVFP4_4OVER6.get()
                and envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.get()
                else float(torch.finfo(torch.float8_e4m3fn).max)
            )
            fp8_fp4_max = e4m3_max * 6.0
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
            backend="cute-dsl",
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
        if self.quant_config.use_mxfp8:
            raise ValueError(
                "--quantization nvfp4_online does not support online "
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
            prefix, suffix = weight_name.rsplit("weight", 1)
            return f"{prefix}weight_scale{suffix}"
        return f"{weight_name}.weight_scale"

    @staticmethod
    def _scale_2_weight_name(weight_name: str) -> str:
        if "weight" in weight_name:
            prefix, suffix = weight_name.rsplit("weight", 1)
            return f"{prefix}weight_scale_2{suffix}"
        return f"{weight_name}.weight_scale_2"

    def get_online_weight_loader(self, layer, original_weight_loader):
        """Wrap the normal MoE loader with load-time NVFP4 conversion.

        The wrapper quantizes each eligible expert shard as soon as the loader
        sees enough source data, which avoids materializing and then converting
        the full checkpoint. FP8 checkpoints stream weight and scale tensors
        separately, so those pairs are staged until both sides have arrived.
        """
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
                    "--quantization nvfp4_online expects paired w1/w3 expert "
                    f"weights, got two {shard_id} tensors for expert {expert_id}."
                )
            pending_weight = pending_weight.to(param.device)
            loaded_weight = loaded_weight.to(param.device)
            pending_rows = pending_weight.shape[0]
            loaded_rows = loaded_weight.shape[0]
            # Quantize the gated pair together so w1/w3 share one amax-derived
            # per-tensor FP32 scale, matching the serialized NVFP4 convention.
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
                    "--quantization nvfp4_online received an FP8 expert "
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
            loaded_weight: torch.Tensor,
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

        def nvfp4_online_weight_loader(
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
                process_fp8_weight_scale(loaded_weight, shard_id, expert_id)
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

        return nvfp4_online_weight_loader
