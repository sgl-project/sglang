from typing import Any

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig


class Inkling2QuantConfig(QuantizationConfig):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if "mtp" in config:
            draft = Inkling2QuantConfig({"model": config["mtp"]})
            if draft.unembed_output_dtype != torch.bfloat16 or any(
                dtype != torch.bfloat16 for dtype in draft.tensor_dtypes.values()
            ):
                raise ValueError("Inkling2 MTP requires BF16 checkpoint precision")
        self.tensor_dtypes: dict[str, torch.dtype] = {}
        self.mxfp8_weights: set[str] = set()
        self.nvfp4_weights: set[str] = set()
        self.unembed_output_dtype = torch.bfloat16
        for group in config["model"]["parameter_groups"]:
            if unsupported := group.keys() - {
                "tensors",
                "output_dtype",
                "compute_dtype",
                "weight_transform",
                "weight_block_shape",
                "required_activation_quantization",
                "weight_scale_format",
                "fwd_output_dtype",
            }:
                raise ValueError(f"Unsupported Inkling2 precision: {unsupported}")
            compute = group["compute_dtype"]
            if compute not in ("float32", "bfloat16", "mxfp8", "nvfp4"):
                raise ValueError(f"Unsupported computation dtype: {compute}")
            expected_dtype = {
                "mxfp8": "float8_e4m3fn",
                "nvfp4": "uint8",
            }.get(compute, compute)
            if group["output_dtype"] != expected_dtype:
                raise ValueError(f"Unexpected storage dtype for {compute}")
            if "fwd_output_dtype" in group:
                if (
                    compute != "bfloat16"
                    or group["fwd_output_dtype"] != "float32"
                    or group["tensors"] != ["model.llm.unembed.weight"]
                ):
                    raise ValueError("Unsupported Inkling2 forward output precision")
                self.unembed_output_dtype = torch.float32
            if "weight_scale_format" in group and compute != "nvfp4":
                raise ValueError("Only NVFP4 weights use an outer scale")
            auxiliary_dtypes = {}
            if compute == "mxfp8":
                expected = dict(
                    format="mxfp8_e4m3",
                    block_size=32,
                    scale_dtype="ue8m0",
                    rounding="nearest",
                )
                if group["required_activation_quantization"] != expected:
                    raise ValueError("Unsupported MXFP8 activation precision")
                if group["weight_transform"] != "mxfp8_nearest":
                    raise ValueError("MXFP8 weights require mxfp8_nearest")
                if group["weight_block_shape"] not in ([1, 32], [32, 32]):
                    raise ValueError("Unsupported MXFP8 weight block shape")
                auxiliary_dtypes = {"_scale": torch.uint8}
            elif compute == "nvfp4":
                expected = dict(
                    format="nvfp4_e2m1",
                    block_size=16,
                    scale_dtype="float8_e4m3fn",
                    rounding="nearest",
                    amax_type="per_token",
                    scale_selection_rule="fouroversix_mae",
                )
                if group["required_activation_quantization"] != expected:
                    raise ValueError("Unsupported NVFP4 activation precision")
                if group["weight_transform"] != "nvfp4_nearest":
                    raise ValueError("NVFP4 weights require nvfp4_nearest")
                if group["weight_block_shape"] != [1, 16]:
                    raise ValueError("Unsupported NVFP4 weight block shape")
                if group.get("weight_scale_format") != "amax_denominator":
                    raise ValueError("NVFP4 weights require amax_denominator scales")
                auxiliary_dtypes = {
                    "_scale": torch.float8_e4m3fn,
                    "_amax": torch.float32,
                    "_outer_scale_denominator": torch.float32,
                }
            elif (
                group["required_activation_quantization"] is not None
                or group["weight_transform"] != "cast"
                or group["weight_block_shape"]
            ):
                raise ValueError("Dense precision must not request quantization")
            for name in group["tensors"]:
                if name in self.tensor_dtypes:
                    raise ValueError(f"Repeated precision entry: {name}")
                self.tensor_dtypes[name] = getattr(torch, expected_dtype)
                if compute in ("mxfp8", "nvfp4"):
                    if not name.endswith((".w13_weight", ".w2_weight")):
                        raise ValueError(
                            f"{compute.upper()} is only supported for expert weights: {name}"
                        )
                    weights = (
                        self.mxfp8_weights if compute == "mxfp8" else self.nvfp4_weights
                    )
                    weights.add(name)
                for suffix, dtype in auxiliary_dtypes.items():
                    scale_name = name + suffix
                    if scale_name in self.tensor_dtypes:
                        raise ValueError(f"Repeated precision entry: {scale_name}")
                    self.tensor_dtypes[scale_name] = dtype

    @classmethod
    def get_name(cls) -> str:
        return "inkling2"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Inkling2QuantConfig":
        return cls(config)

    def get_quant_method(self, layer: nn.Module, prefix: str):
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []

    def quantizes_input(self, checkpoint_name: str) -> bool:
        return (
            checkpoint_name in self.mxfp8_weights
            or checkpoint_name in self.nvfp4_weights
        )
