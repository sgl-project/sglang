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


class MiniMaxM3CachedImageProcessor:
    """Reuse grid metadata while the HF processor composes the current prompt.

    No pixels are concatenated here: each request's items reference the ordered
    artifacts directly. The wrapped processor supplies token expansion settings.
    """

    def __init__(self, processor, artifacts):
        self.processor = processor
        self.artifacts = artifacts

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def fetch_images(self, images):
        # These are request-local image placeholders, never URLs to reload.
        return images

    def __call__(self, images, **kwargs):
        if len(images) != len(self.artifacts):
            raise ValueError("Expected one MiniMax M3 artifact per image")
        return {
            "image_grid_thw": torch.tensor(
                [artifact.grid_thw for artifact in self.artifacts], dtype=torch.long
            )
        }
