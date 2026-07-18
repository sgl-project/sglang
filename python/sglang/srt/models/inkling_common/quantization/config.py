from __future__ import annotations

import json
import logging
import os
from typing import Any

import huggingface_hub
import torch
from huggingface_hub import snapshot_download

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)

logger = logging.getLogger(__name__)


def _get_raw_quant_config(
    model_config: ModelConfig,
) -> dict[str, Any] | None:
    """
    pared-down version of `model_loader.weight_utils.get_quant_config`

    Returns just the loaded quant config.
    """
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)
    if hf_quant_config is not None:
        return hf_quant_config

    model_name_or_path = model_config.model_path

    # A local path holds hf_quant_config.json directly; a remote HF repo id must
    # first resolve its JSON configs from the hub (mirrors weight_utils.get_quant_config).
    if os.path.isdir(model_name_or_path):
        hf_folder = model_name_or_path
    else:
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns="*.json",
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )

    quant_config_file = os.path.join(hf_folder, "hf_quant_config.json")
    if not os.path.exists(quant_config_file):
        return None
    with open(quant_config_file) as f:
        config = json.load(f)
        return config


def _map_exclude_modules(exclude_modules: list[str]) -> list[str]:
    """Map checkpoint exclude_modules names to SGLang model prefixes."""
    new_exclude_modules = set()

    for module in exclude_modules:
        if "audio" in module or "visual" in module:
            new_exclude_modules.add(module)
            continue
        module = module.removeprefix("model.")
        # Dense (non-MoE) MLP linears and the unembedding use different names in
        # the sglang module tree; translate so exclusion matches their prefixes.
        module = module.replace(".mlp.w13_dn", ".mlp.gate_up_proj").replace(
            ".mlp.w2_md", ".mlp.down_proj"
        )
        if module == "unembed":
            module = "lm_head"
        new_exclude_modules.add(module)

    return list(new_exclude_modules)


class InklingQuantizationConfigBase:
    exclude_modules: list[str]

    @classmethod
    def maybe_from_model_config(
        cls, model_config: ModelConfig
    ) -> InklingQuantizationConfigBase | None:
        raise NotImplementedError()

    @staticmethod
    def is_nvfp4(config: dict[str, Any]) -> bool:
        weight_quant_cfg = config["modelopt_quant_config"]["quant_cfg"][
            "*weight_quantizer"
        ]
        return tuple(weight_quant_cfg["num_bits"]) == (2, 1) and tuple(
            weight_quant_cfg["block_sizes"].get("scale_bits", [])
        ) == (4, 3)

    def exclude_layer(self, prefix: str) -> bool:
        if len(self.exclude_modules) == 0:
            return False
        return any(
            module in prefix
            or (
                prefix.startswith("language_model.")
                and module in prefix.removeprefix("language_model.")
            )
            for module in self.exclude_modules
        )


class InklingModelOptNvfp4Config(ModelOptFp4Config, InklingQuantizationConfigBase):
    moe_ep_size: int
    nvfp4_moe_backend: str

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str | None = None,
        group_size: int | None = None,
        exclude_modules: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
        # New Inkling args
        scales_2d: bool = False,
        moe_ep_size: int = 1,
        nvfp4_moe_backend: str = "trtllm-routed",
    ) -> None:
        # unfortunately parent types are completely incorrect
        super().__init__(
            is_checkpoint_nvfp4_serialized=is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo=kv_cache_quant_algo,  # type: ignore[reportArgumentType]
            group_size=group_size,  # type: ignore[reportArgumentType]
            exclude_modules=exclude_modules,  # type: ignore[reportArgumentType]
            packed_modules_mapping=packed_modules_mapping,
        )
        if group_size != 16:
            raise ValueError("Inkling only supports group size 16 for NVFP4")
        if scales_2d:
            self.dim1_group_size = group_size
        else:
            self.dim1_group_size = 1
        self.dim2_group_size = group_size
        self.moe_ep_size = moe_ep_size
        self.nvfp4_moe_backend = nvfp4_moe_backend

    @classmethod
    def get_name(cls) -> str:
        return "inkling_nvfp4"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        """Map layers to Inkling-compatible quant methods."""

        # hidden to avoid circular imports
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedFusedMoEMethod,
            UnquantizedLinearMethod,
        )

        if isinstance(layer, LinearBase):
            if self.exclude_layer(prefix):
                logger.debug(f"excluded linear layer for quantization: {prefix}")
                return UnquantizedLinearMethod()
            logger.debug(f"quantizing linear layer: {prefix}")
            return ModelOptFp4LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if self.exclude_layer(prefix):
                logger.debug(f"excluded fused MoE layer for quantization: {prefix}")
                from sglang.srt.models.inkling_common.util import (
                    shared_sink_uses_trtllm_bf16,
                )

                if layer.is_shared_fused_moe and shared_sink_uses_trtllm_bf16():
                    # bf16 shared-expert sink on flashinfer_trtllm_routed: arm the
                    # stock trtllm bf16 path (this instance flag drives the trtllm
                    # weight prep, the FLASHINFER_TRTLLM_ROUTED runner, and the
                    # forward branch in UnquantizedFusedMoEMethod). The loader's
                    # shared_w13 [up||gate] swap keys on the same predicate.
                    # Routed quant-EXCLUDED bf16 layers keep the triton runner.
                    return UnquantizedFusedMoEMethod(
                        use_triton_kernels=False,
                        use_flashinfer_trtllm_moe=True,
                    )
                return UnquantizedFusedMoEMethod(use_triton_kernels=False)
            logger.debug(f"quantizing fused MoE layer: {prefix}")
            # Upstream ModelOpt NVFP4 fused-MoE method. It de-interleaves the Inkling
            # interleaved-w13 layout when the layer sets inference_moe_w13_interleaved.
            return ModelOptNvFp4FusedMoEMethod(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InklingModelOptNvfp4Config:
        parent_config = ModelOptFp4Config.from_config(config)
        assert "quantization" in config, "quantization config is required"
        exclude_modules = _map_exclude_modules(parent_config.exclude_modules)
        quant_config = config["quantization"]
        scales_2d = quant_config.get("scales_2d", False)
        moe_ep_size = quant_config.get("moe_ep_size", 1)
        nvfp4_moe_backend = quant_config.get("nvfp4_moe_backend", "trtllm-routed")
        return cls(
            is_checkpoint_nvfp4_serialized=parent_config.is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo=parent_config.kv_cache_quant_algo,
            group_size=parent_config.group_size,
            exclude_modules=exclude_modules,
            packed_modules_mapping=parent_config.packed_modules_mapping,
            scales_2d=scales_2d,
            moe_ep_size=moe_ep_size,
            nvfp4_moe_backend=nvfp4_moe_backend,
        )

    @classmethod
    def maybe_from_model_config(
        cls, model_config: ModelConfig
    ) -> InklingModelOptNvfp4Config | None:
        from sglang.srt.distributed import (
            get_moe_expert_parallel_world_size,
        )

        raw_quant_config = _get_raw_quant_config(model_config)

        if raw_quant_config is None:
            return None

        quant_config = raw_quant_config
        assert isinstance(quant_config, dict), "quant_config must be a dict"
        if "quantization" in quant_config:
            # nested format
            quant_config = cls.get_from_keys(quant_config, ["quantization"])

        weight_quant_cfg = quant_config["modelopt_quant_config"]["quant_cfg"][
            "*weight_quantizer"
        ]

        if not cls.is_nvfp4(quant_config):
            return None

        scales_2d = weight_quant_cfg["block_sizes"].get("-2") is not None

        quant_config["scales_2d"] = scales_2d
        quant_config["moe_ep_size"] = get_moe_expert_parallel_world_size()
        quant_config["nvfp4_moe_backend"] = "trtllm-routed"

        # force parent class to fallback to nested format.
        return cls.from_config({"quantization": quant_config})


def get_quantization_config(
    model_config: ModelConfig,
) -> InklingQuantizationConfigBase | None:

    quant_config = _get_raw_quant_config(model_config)
    if quant_config is None:
        return None
    if "quantization" in quant_config:
        # nested format
        quant_config = QuantizationConfig.get_from_keys(quant_config, ["quantization"])

    if InklingQuantizationConfigBase.is_nvfp4(quant_config):
        return InklingModelOptNvfp4Config.maybe_from_model_config(model_config)
    return None
