"""Content-addressed caches used by multimodal preprocessing."""

from sglang.srt.multimodal.cache.identity import (
    CONTENT_HASH_PREFIX,
    MediaSnapshot,
    PreprocessFingerprintProvider,
    build_artifact_key,
    build_processor_fingerprint,
    media_preprocess_kwargs,
    parse_content_hash,
    resolve_multimodal_item_hash,
    snapshot_media,
)
from sglang.srt.multimodal.cache.preprocess_cache import (
    CacheLookup,
    CacheMiss,
    CacheSizeProvider,
    MultimodalPreprocessCache,
    estimate_cache_size_bytes,
)

__all__ = [
    "CONTENT_HASH_PREFIX",
    "CacheSizeProvider",
    "CacheLookup",
    "CacheMiss",
    "MediaSnapshot",
    "MultimodalPreprocessCache",
    "PreprocessFingerprintProvider",
    "build_artifact_key",
    "build_processor_fingerprint",
    "estimate_cache_size_bytes",
    "media_preprocess_kwargs",
    "parse_content_hash",
    "resolve_multimodal_item_hash",
    "snapshot_media",
]
