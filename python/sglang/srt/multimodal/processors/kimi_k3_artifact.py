"""Prompt-independent Kimi-K3 image preprocessing artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass(frozen=True)
class KimiK3PreprocessConfig:
    """The single source of truth for K3 artifact-producing choices."""

    patch_size: int
    merge_kernel_size: int
    in_patch_limit: int
    patch_limit_on_one_side: int
    fixed_output_tokens: Optional[int]
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    transparent_bg_config: Optional[dict]

    @classmethod
    def from_media_processor(cls, media_processor) -> KimiK3PreprocessConfig:
        config = media_processor.media_proc_cfg
        return cls(
            patch_size=int(config["patch_size"]),
            merge_kernel_size=int(config["merge_kernel_size"]),
            in_patch_limit=int(config["in_patch_limit"]),
            patch_limit_on_one_side=int(config["patch_limit_on_one_side"]),
            fixed_output_tokens=(
                None
                if config.get("fixed_output_tokens") is None
                else int(config["fixed_output_tokens"])
            ),
            image_mean=tuple(float(value) for value in config["image_mean"]),
            image_std=tuple(float(value) for value in config["image_std"]),
            transparent_bg_config=deepcopy(config.get("transparent_bg_config")),
        )


@dataclass(frozen=True)
class KimiK3ResizeConfig:
    num_tokens: int
    new_width: int
    new_height: int
    pad_width: int
    pad_height: int

    @classmethod
    def from_dict(cls, value: dict) -> KimiK3ResizeConfig:
        return cls(**{name: int(value[name]) for name in cls.__annotations__})

    def as_dict(self) -> dict[str, int]:
        return {
            "num_tokens": self.num_tokens,
            "new_width": self.new_width,
            "new_height": self.new_height,
            "pad_width": self.pad_width,
            "pad_height": self.pad_height,
        }


@dataclass(frozen=True)
class KimiK3DeferredConfig:
    backend: str
    feature_layout: str
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    transparent_bg_config: Optional[dict]

    def as_dict(self, resize_config: KimiK3ResizeConfig) -> dict:
        return {
            "backend": self.backend,
            "feature_layout": self.feature_layout,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "transparent_bg_config": self.transparent_bg_config,
            "resize_config": resize_config.as_dict(),
        }


@dataclass(frozen=True)
class KimiK3ImageArtifact:
    """One image's reusable metadata and, when already on CPU, its feature."""

    content_digest: str
    artifact_key: str
    feature_identity: str
    feature_hash: int
    original_size: tuple[int, int]
    resize_config: KimiK3ResizeConfig
    grid_thw: tuple[int, int, int]
    feature: Optional[torch.Tensor]
    deferred: Optional[KimiK3DeferredConfig] = None

    @property
    def has_feature(self) -> bool:
        return self.feature is not None

    @property
    def is_cpu_cacheable(self) -> bool:
        return self.feature is None or self.feature.device.type == "cpu"

    def cache_value(self) -> KimiK3ImageArtifact:
        """Never retain a CUDA tensor in the preprocess cache."""
        if self.feature is None or self.feature.device.type == "cpu":
            return self
        return replace(self, feature=None)

    def cache_size_items(self) -> tuple:
        """Return every owned value that contributes to the CPU cache budget."""
        deferred = None
        if self.deferred is not None:
            deferred = (
                self.deferred.backend,
                self.deferred.feature_layout,
                self.deferred.image_mean,
                self.deferred.image_std,
                self.deferred.transparent_bg_config,
            )
        return (
            self.content_digest,
            self.artifact_key,
            self.feature_hash,
            self.original_size,
            (
                self.resize_config.num_tokens,
                self.resize_config.new_width,
                self.resize_config.new_height,
                self.resize_config.pad_width,
                self.resize_config.pad_height,
            ),
            self.grid_thw,
            self.feature,
            deferred,
        )
