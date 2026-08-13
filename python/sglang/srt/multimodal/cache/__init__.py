"""Content-addressed caches used by multimodal preprocessing."""

from sglang.srt.multimodal.cache.identity import (
    CONTENT_HASH_PREFIX,
    MediaSnapshot,
    build_artifact_key,
    build_feature_hash,
    build_processor_fingerprint,
    media_preprocess_kwargs,
    parse_content_hash,
    snapshot_media,
)
from sglang.srt.multimodal.cache.preprocess_cache import (
    CacheLookup,
    CacheReservation,
    MultimodalPreprocessCache,
    estimate_cache_size_bytes,
)

__all__ = [
    "CONTENT_HASH_PREFIX",
    "CacheLookup",
    "CacheReservation",
    "MediaSnapshot",
    "MultimodalPreprocessCache",
    "build_artifact_key",
    "build_feature_hash",
    "build_processor_fingerprint",
    "estimate_cache_size_bytes",
    "media_preprocess_kwargs",
    "parse_content_hash",
    "snapshot_media",
]
