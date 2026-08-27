"""Metadata-driven dense NVFP4 residue support."""

from sglang.srt.layers.quantization.residue_nvfp4.metadata import (
    METADATA_FILENAME,
    ResidueLayerSpec,
    ResidueMetadataError,
    ResidueMode,
    ResidueModelSpec,
    find_residue_metadata,
    load_residue_model_spec,
    parse_residue_metadata,
)

__all__ = [
    "METADATA_FILENAME",
    "ModelOptFp4ResidueConfig",
    "ModelOptFp4ResidueLinearMethod",
    "ModelOptMixedPrecisionResidueConfig",
    "ResidueLayerSpec",
    "ResidueMetadataError",
    "ResidueMode",
    "ResidueModelSpec",
    "find_residue_metadata",
    "load_residue_model_spec",
    "maybe_wrap_residue_fp4_config",
    "parse_residue_metadata",
]


def __getattr__(name):
    # config/linear_method pull in the full quantization stack; load lazily
    # so the metadata contract stays importable on its own.
    if name in (
        "ModelOptFp4ResidueConfig",
        "ModelOptMixedPrecisionResidueConfig",
        "maybe_wrap_residue_fp4_config",
    ):
        from sglang.srt.layers.quantization.residue_nvfp4 import config

        return getattr(config, name)
    if name == "ModelOptFp4ResidueLinearMethod":
        from sglang.srt.layers.quantization.residue_nvfp4.linear_method import (
            ModelOptFp4ResidueLinearMethod,
        )

        return ModelOptFp4ResidueLinearMethod
    raise AttributeError(name)
