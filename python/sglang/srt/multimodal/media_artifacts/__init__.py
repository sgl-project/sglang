"""Bridge multimodal processors and the shared preprocess cache.

A model processor turns raw media into model-specific ``MediaArtifact``
objects and later combines those artifacts with the current prompt. The
``MultimodalPreprocessCache`` stores cache-safe copies of the artifacts so a
later request can skip model preprocessing. With a trusted caller-provided
content hash, a hot hit can skip loading the media source as well.

This package defines that boundary:

* ``MediaArtifact`` is the common contract for model-specific cache items.
* ``MediaArtifactInput`` carries a decoded cache miss into a model processor.
* ``MediaArtifactCacheMixin`` owns lookup, single-flight miss handling, and
  result ordering around the model's ``prepare_artifact_batch`` method.

The package does not implement model preprocessing or a second cache. Model
logic stays in ``multimodal.processors``; storage and concurrency stay in
``multimodal.cache``.
"""

from sglang.srt.multimodal.media_artifacts.base import (
    MediaArtifact,
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)

__all__ = [
    "MediaArtifact",
    "MediaArtifactCacheMixin",
    "MediaArtifactInput",
]
