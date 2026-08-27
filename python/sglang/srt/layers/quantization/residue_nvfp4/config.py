"""Residue-aware ModelOpt NVFP4 quantization config.

A residue checkpoint IS a stock ModelOpt NVFP4 checkpoint plus
residue_kernel_metadata.json. The wrappers in this module cover both a plain
ModelOptFp4Config and the NVFP4 dense layers inside a
ModelOptMixedPrecisionConfig. They differ from the stock configs in exactly
one way: dense NVFP4 linear layers named by the metadata get the residue
linear method. Everything else -- exclusions, KV cache, MoE, other precision
families, and layers without residue metadata -- behaves like the parent.

Selection is checkpoint-driven: model loading wraps a supported ModelOpt
config when the metadata file is present and valid (see
maybe_wrap_residue_fp4_config), and `--quantization modelopt_fp4_residue`
selects it explicitly. A present-but-invalid metadata file fails the load; it
never silently degrades to plain NVFP4.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.residue_nvfp4.metadata import (
    ResidueMetadataError,
    ResidueModelSpec,
    load_residue_model_spec,
)


class ModelOptFp4ResidueConfig(ModelOptFp4Config):
    """ModelOptFp4Config + a validated per-layer residue spec."""

    def __init__(self, *args, residue_spec: ResidueModelSpec, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_checkpoint_nvfp4_serialized:
            raise ResidueMetadataError(
                "residue metadata requires a serialized NVFP4 checkpoint"
            )
        if self.is_awq:
            raise ResidueMetadataError(
                "residue metadata is not supported for NVFP4_AWQ checkpoints"
            )
        self.residue_spec = residue_spec

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp4_residue"

    @classmethod
    def get_min_capability(cls) -> int:
        # The residue kernels support datacenter Blackwell (SM100/SM103).
        return 100

    @classmethod
    def from_config(cls, config) -> ModelOptFp4ResidueConfig:
        # The metadata file lives next to the weights, not inside the HF
        # quantization config, so from_config alone cannot build this class.
        # Model loading attaches the spec via from_modelopt_config.
        raise ResidueMetadataError(
            "modelopt_fp4_residue cannot be built from the HF quantization "
            "config alone; it is selected by the residue_kernel_metadata.json "
            "next to the checkpoint (model loading wraps the modelopt_fp4 "
            "config automatically)."
        )

    @classmethod
    def from_modelopt_config(
        cls, base: ModelOptFp4Config, residue_spec: ResidueModelSpec
    ) -> ModelOptFp4ResidueConfig:
        """Wrap an already-parsed stock config with a validated residue spec."""
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        if not base.is_checkpoint_nvfp4_serialized:
            raise ResidueMetadataError(
                "residue metadata requires a serialized NVFP4 checkpoint"
            )
        if base.is_awq:
            raise ResidueMetadataError(
                "residue metadata is not supported for NVFP4_AWQ checkpoints"
            )
        self.residue_spec = residue_spec
        return self

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.residue_nvfp4.linear_method import (
            ModelOptFp4ResidueLinearMethod,
        )
        from sglang.srt.layers.quantization.utils import is_layer_skipped

        if isinstance(layer, LinearBase):
            if not (
                is_layer_skipped(
                    prefix, self.exclude_modules, self.packed_modules_mapping
                )
                or self.is_layer_excluded(prefix)
            ):
                # The method resolves the layer's residue spec by prefix; a
                # layer without one behaves exactly like the parent method.
                return ModelOptFp4ResidueLinearMethod(self, prefix)
        # Excluded linears, ParallelLMHead, attention KV cache, MoE: stock.
        return super().get_quant_method(layer, prefix)


class ModelOptMixedPrecisionResidueConfig(ModelOptMixedPrecisionConfig):
    """Mixed-precision ModelOpt config with residue on dense NVFP4 only.

    The outer mixed config remains responsible for resolving each layer's
    quantization algorithm. Only when the stock resolver chose the serialized
    NVFP4 dense-linear method do we substitute the residue-aware method.
    Consequently FP8/MXFP8/W4A16 layers, embeddings, KV cache, ParallelLMHead,
    and especially FusedMoE all remain on their stock implementations.
    """

    @classmethod
    def from_modelopt_config(
        cls,
        base: ModelOptMixedPrecisionConfig,
        residue_spec: ResidueModelSpec,
    ) -> ModelOptMixedPrecisionResidueConfig:
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        self.residue_spec = residue_spec
        self.nvfp4_config = ModelOptFp4ResidueConfig.from_modelopt_config(
            base.nvfp4_config, residue_spec
        )
        return self

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.residue_nvfp4.linear_method import (
            ModelOptFp4ResidueLinearMethod,
        )
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        method = super().get_quant_method(layer, prefix)
        if (
            isinstance(layer, LinearBase)
            and not isinstance(layer, ParallelLMHead)
            and type(method) is ModelOptFp4LinearMethod
            and self.residue_spec.spec_for(prefix) is not None
        ):
            return ModelOptFp4ResidueLinearMethod(self.nvfp4_config, prefix)
        return method


def maybe_wrap_residue_fp4_config(model_config, quant_config):
    """Wrap a ModelOpt config when the checkpoint carries residue metadata.

    Called from model loading with the resolved quant config. Returns the
    (possibly wrapped) config. Raises when the user explicitly requested
    modelopt_fp4_residue but the checkpoint has no metadata file, and when a
    metadata file exists but cannot be served -- never a silent fallback.
    """
    explicit = getattr(model_config, "quantization", None) == "modelopt_fp4_residue"

    supported_config = isinstance(
        quant_config, (ModelOptFp4Config, ModelOptMixedPrecisionConfig)
    )
    already_wrapped = isinstance(
        quant_config,
        (ModelOptFp4ResidueConfig, ModelOptMixedPrecisionResidueConfig),
    )
    if not supported_config or already_wrapped:
        if explicit:
            raise ResidueMetadataError(
                "--quantization modelopt_fp4_residue requires a plain ModelOpt "
                "NVFP4 "
                f"checkpoint; resolved config is {type(quant_config).__name__}"
            )
        return quant_config

    model_path = getattr(model_config, "model_path", None)
    if not model_path or not os.path.isdir(model_path):
        # Residue checkpoints are local exports; auto-detection only looks at
        # local directories.
        if explicit:
            raise ResidueMetadataError(
                "modelopt_fp4_residue requires a local checkpoint directory "
                f"containing residue_kernel_metadata.json (got {model_path!r})"
            )
        return quant_config

    spec: Optional[ResidueModelSpec] = load_residue_model_spec(model_path)
    if spec is None:
        if explicit:
            raise ResidueMetadataError(
                f"no residue_kernel_metadata.json in {model_path}; use "
                "--quantization modelopt_fp4 for a plain NVFP4 checkpoint"
            )
        return quant_config

    if isinstance(quant_config, ModelOptMixedPrecisionConfig):
        return ModelOptMixedPrecisionResidueConfig.from_modelopt_config(
            quant_config, spec
        )
    return ModelOptFp4ResidueConfig.from_modelopt_config(quant_config, spec)
