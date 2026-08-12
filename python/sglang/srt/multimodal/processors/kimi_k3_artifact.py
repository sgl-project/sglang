"""Prompt-independent Kimi-K3 image preprocessing artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from sglang.srt.multimodal.cache import MediaSnapshot


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


@dataclass(frozen=True)
class KimiK3MediaLookup:
    """Identity lookup result reusable by the later processor miss path."""

    artifact_key: str
    content_digest: str
    snapshot: Optional[MediaSnapshot]
    cached_artifact: Optional[KimiK3ImageArtifact]
