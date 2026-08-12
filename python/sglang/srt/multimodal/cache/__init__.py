"""Content-addressed caches used by multimodal preprocessing."""

from sglang.srt.multimodal.cache.contracts import (
    EncoderMediaLookup,
    EncoderPreprocessArtifact,
    PreprocessCacheLookup,
)
from sglang.srt.multimodal.cache.identity import (
    CONTENT_HASH_PREFIX,
    MediaSnapshot,
    build_artifact_key,
    build_feature_hash,
    build_feature_identity,
    build_mm_global_cache_key,
    build_mm_radix_cache_namespace,
    build_processor_fingerprint,
    compact_feature_hash,
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
    "EncoderMediaLookup",
    "EncoderPreprocessArtifact",
    "MediaSnapshot",
    "MultimodalPreprocessCache",
    "PreprocessCacheLookup",
    "build_artifact_key",
    "build_feature_identity",
    "build_feature_hash",
    "build_mm_global_cache_key",
    "build_mm_radix_cache_namespace",
    "build_processor_fingerprint",
    "compact_feature_hash",
    "estimate_cache_size_bytes",
    "parse_content_hash",
    "snapshot_media",
]
