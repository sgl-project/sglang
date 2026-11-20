"""Prompt-independent MiniMax M3 image preprocessing artifacts."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MiniMaxM3ImageArtifact:
    content_digest: str
    artifact_key: str
    feature_hash: int
    grid_thw: tuple[int, int, int]
    feature: torch.Tensor

    @property
    def has_feature(self) -> bool:
        return True

    def cache_value(self):
        # This adapter produces CPU features only.
        return self

    def cache_size_items(self):
        return (
            self.content_digest,
            self.artifact_key,
            self.feature_hash,
            self.grid_thw,
            self.feature,
        )
