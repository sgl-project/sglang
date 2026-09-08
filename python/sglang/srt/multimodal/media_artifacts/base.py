"""Shared contracts and coordination for reusable multimodal artifacts.

A media artifact is the model-specific, prompt-independent state produced from
one media input. It keeps the metadata needed to rebuild a request (for example,
image size, token count, and encoder grid) and, when cacheable on CPU, the
processor feature itself. Prompt tokens and offsets are deliberately excluded.

The artifact connects the media preprocessor to request composition::

    raw media -> identity/cache lookup -> MediaArtifact
      cache miss: MediaArtifactInput -> prepare_artifact_batch()
      cache hit:  reuse the stored artifact
    MediaArtifact + current prompt -> MultimodalDataItem -> encoder/ViT

The current request uses the full artifact returned by preprocessing. The
preprocess cache stores ``artifact.cache_value()``, which may omit a CUDA feature
and retain only reusable metadata. Such a featureless artifact is usable only
when the downstream embedding cache already contains the encoded feature.

An artifact is therefore the logical preprocess-cache item. It is not the raw
media, a prompt-specific ``MultimodalDataItem``, or a ViT embedding-cache entry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.cache import (
    CacheLookup,
    CacheMiss,
    MediaSnapshot,
    build_artifact_key,
    media_preprocess_kwargs,
    parse_content_hash,
    snapshot_media,
)
from sglang.srt.utils import load_image


@runtime_checkable
class MediaArtifact(Protocol):
    """Common contract implemented by each model's preprocess artifact.

    ``content_digest`` identifies the media contents. ``artifact_key`` also
    includes every preprocessing choice that can change the artifact.
    ``feature_hash`` becomes ``MultimodalDataItem.hash`` and identifies the
    corresponding encoder embedding.
    """

    content_digest: str
    artifact_key: str
    feature_hash: int

    @property
    def has_feature(self) -> bool: ...

    def cache_value(self) -> MediaArtifact:
        """Return the cache-safe representation, possibly without a feature."""
        ...

    def cache_size_items(self) -> Sequence[Any]:
        """Return owned values counted against the preprocess-cache budget."""
        ...


@dataclass(frozen=True)
class MediaArtifactInput:
    """One decoded raw multimodal input that missed the preprocess cache.

    The shared cache layer has already validated its content digest, derived
    its artifact key, and claimed the cache miss. The model-specific artifact
    builder preprocesses ``media`` into a reusable ``MediaArtifact`` without
    loading or hashing the source again. This object is a transient handoff;
    it is not itself stored in the cache.
    """

    # hash of the media content
    content_digest: str
    # cache key containing the processor fingerprint, preprocess kwargs, media content
    artifact_key: str
    modality: Modality
    # the original media input that has been loaded and decoded (e.g., PIL.Image)
    media: Any


class MediaArtifactCacheMixin:
    """Turn media inputs into ordered artifacts, reusing cached work per item.

    The shared layer owns identity, cache lookup, single-flight, partial hits,
    and result ordering. A model adapter implements ``prepare_artifact_batch``
    and later composes the returned artifacts with the current prompt. Models
    can override snapshot/decode/key hooks for each modality without copying the
    cache algorithm. Every artifact-producing setting must be exposed through
    ``preprocess_fingerprint_payload``.
    """

    artifact_modality: Optional[Modality] = None
    artifact_option_defaults: Mapping[str, Any] = {"detail": "auto"}

    def artifact_preprocess_kwargs(
        self, source: Any, modality: Modality
    ) -> Mapping[str, Any]:
        """Return request options that can change this media's artifact.

        These options become part of the artifact key. Model adapters can
        override this hook when their request schema has additional knobs.
        """
        return media_preprocess_kwargs(source, defaults=self.artifact_option_defaults)

    def _resolve_artifact_modality(self, modality: Optional[Modality]) -> Modality:
        modality = modality or self.artifact_modality
        if modality is None:
            raise ValueError("A modality is required for artifact caching")
        return modality

    def _artifact_key(
        self,
        content_digest: str,
        source: Any,
        *,
        modality: Optional[Modality] = None,
    ) -> str:
        """Identify one artifact by media content and all preprocess choices."""
        if self.processor_fingerprint is None:
            raise RuntimeError("Artifact caching requires a processor fingerprint")
        modality = self._resolve_artifact_modality(modality)
        return build_artifact_key(
            content_digest,
            modality=modality.name.lower(),
            processor_fingerprint=self.processor_fingerprint,
            preprocess_kwargs=self.artifact_preprocess_kwargs(source, modality),
        )

    def decode_media_snapshot(self, snapshot: MediaSnapshot, modality: Modality) -> Any:
        """Decode an immutable snapshot for the model adapter.

        Image is the shared default. Future video/audio adapters must make their
        decode/sampling contract explicit by overriding this hook.
        """
        if modality != Modality.IMAGE:
            raise NotImplementedError(
                f"{modality.name.lower()} artifact decoding requires a model adapter"
            )
        data = snapshot.data
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        if isinstance(data, Image.Image):
            data.load()
            return data
        image, _ = load_image(data, self.gpu_image_decode)
        if isinstance(image, Image.Image):
            image.load()
        return image

    def snapshot_media_source(self, source: Any, modality: Modality) -> MediaSnapshot:
        """Capture immutable media content before decode and preprocessing.

        The shared implementation covers images. Video/audio adapters can add
        streaming or frame-sampling identities here without changing the cache
        coordinator.
        """
        if modality != Modality.IMAGE:
            raise NotImplementedError(
                f"{modality.name.lower()} artifact identity requires a model adapter"
            )
        return snapshot_media(source)

    def prepare_artifact_batch(
        self, entries: Sequence[MediaArtifactInput]
    ) -> list[MediaArtifact]:
        """Preprocess raw multimodal inputs that missed the preprocess cache.

        Each entry is one unique, decoded cache miss. Implementations must
        return one reusable artifact (the preprocess-cache item) per entry, in
        the same order, while preserving its content digest and artifact key.
        The shared layer uses the artifact for the current request and stores
        ``artifact.cache_value()`` for reuse.
        """
        raise NotImplementedError

    def artifact_usable(
        self, artifact: MediaArtifact, *, allow_featureless: bool
    ) -> bool:
        """Whether this request can use an artifact that may omit its feature.

        A metadata-only artifact is valid only after the scheduler has confirmed
        that the corresponding encoder embedding is already cached.
        """
        return artifact.has_feature or allow_featureless

    @staticmethod
    def validate_artifact(artifact: MediaArtifact, entry: MediaArtifactInput) -> None:
        """Enforce identity invariants shared by every model adapter."""
        if artifact.content_digest != entry.content_digest:
            raise ValueError("prepare_artifact_batch changed the media content digest")
        if artifact.artifact_key != entry.artifact_key:
            raise ValueError("prepare_artifact_batch changed the media artifact key")
        if (
            isinstance(artifact.feature_hash, bool)
            or not isinstance(artifact.feature_hash, int)
            or artifact.feature_hash < 0
        ):
            raise ValueError(
                "Media artifact feature_hash must be a non-negative integer"
            )

    async def _run_preprocess_and_build_artifact_batch(
        self, entries: Sequence[MediaArtifactInput]
    ) -> list[MediaArtifact]:
        """Run model preprocessing locally or on the processor worker pool, return the artifact"""
        if self.mm_processor_executor is None:
            return self.prepare_artifact_batch(entries)
        return await self.mm_processor_executor.run(
            self.prepare_artifact_batch, entries
        )

    def _get_cached_artifact(
        self,
        key: str,
        content_digest: str,
        modality: Modality,
        *,
        allow_featureless: bool,
    ) -> Optional[MediaArtifact]:
        """Return a compatible cached artifact without recording a cold miss.

        Identity mismatches are corrupt entries and are evicted. A featureless
        entry that this request cannot use is left for the miss path, which
        removes it temporarily and verifies the recomputed feature hash.
        """
        artifact = self.mm_preprocess_cache.get_if_present(
            key,
            lambda value: (
                isinstance(value, MediaArtifact)
                and value.artifact_key == key
                and value.content_digest == content_digest
            ),
            evict_on_reject=True,
        )
        if artifact is not None:
            self.validate_artifact(
                artifact,
                MediaArtifactInput(content_digest, key, modality, None),
            )
            if not self.artifact_usable(artifact, allow_featureless=allow_featureless):
                return None
        return artifact

    async def prepare_media_artifacts(
        self,
        media_data: Sequence[Any],
        *,
        content_hashes: Optional[Sequence[Optional[str]]] = None,
        featureless_hit_mask: Optional[Sequence[bool]] = None,
        modality: Optional[Modality] = None,
    ) -> list[MediaArtifact]:
        """Try resolving one preprocess-cache artifact for each processor input.

        Each media input is looked up independently, and results preserve the
        input order. A cache hit returns the stored artifact (the cache item).
        A miss snapshots and decodes the raw input, runs
        ``prepare_artifact_batch``, stores its cache-safe artifact, and returns
        the prepared artifact to the current request. Duplicate and concurrent
        misses share the same preprocessing work.

        This stage is prompt-independent. It does not create prompt tokens,
        offsets, or ``MultimodalDataItem`` objects; the model processor uses the
        returned artifacts to compose those request-specific values afterward.
        """
        modality = self._resolve_artifact_modality(modality)
        media_count = len(media_data)
        if content_hashes is None:
            content_hashes = [None] * media_count
        if len(content_hashes) != media_count:
            raise ValueError(
                f"mm_content_hashes has {len(content_hashes)} entries for "
                f"{media_count} {modality.name.lower()} items"
            )
        content_hashes = [parse_content_hash(value) for value in content_hashes]

        if featureless_hit_mask is None:
            featureless_hit_mask = [False] * media_count
        if len(featureless_hit_mask) != media_count:
            raise ValueError("featureless_hit_mask must align with media_data")

        # keep per-input state aligned for duplicates and partial hits
        artifacts: list[Optional[MediaArtifact]] = [None] * media_count
        snapshots: list[Optional[MediaSnapshot]] = [None] * media_count
        keys: list[Optional[str]] = [None] * media_count

        # 1. fast path: resolve trusted provided hash hits without reading media
        # e.g., an image could be submitted with a provided hash:
        # "image_url": {
        #    "url": "https://example.com/image.jpg",
        #    "content_hash": "sha256:<64-hex>"
        # }
        load_indices = []
        for index, (source, caller_hash, allow_featureless) in enumerate(
            zip(media_data, content_hashes, featureless_hit_mask)
        ):
            if self.trust_mm_content_hashes and caller_hash is not None:
                key = self._artifact_key(caller_hash, source, modality=modality)
                keys[index] = key
                artifact = self._get_cached_artifact(
                    key,
                    caller_hash,
                    modality,
                    allow_featureless=allow_featureless,
                )
                if artifact is not None:
                    artifacts[index] = artifact
                    continue
            load_indices.append(index)

        # 2. read cache: build artifact key from media snapshot then try reading cache
        snapshot_futures = {
            index: self.io_executor.submit(
                self.snapshot_media_source, media_data[index], modality
            )
            for index in load_indices
        }
        for index, future in snapshot_futures.items():
            snapshot = await asyncio.wrap_future(future)
            caller_hash = content_hashes[index]
            if caller_hash is not None and caller_hash != snapshot.content_digest:
                raise ValueError(
                    f"content hash mismatch for media_data[{index}]: "
                    f"expected {caller_hash}, got {snapshot.content_digest}"
                )
            snapshots[index] = snapshot
            key = self._artifact_key(
                snapshot.content_digest, media_data[index], modality=modality
            )
            keys[index] = key
            artifacts[index] = self._get_cached_artifact(
                key,
                snapshot.content_digest,
                modality,
                allow_featureless=featureless_hit_mask[index],
            )

        # 3. deduplicate misses before decode
        first_index_by_key: dict[str, int] = {}
        previous_metadata: dict[str, MediaArtifact] = {}
        for index in load_indices:
            if artifacts[index] is not None:
                continue
            key = keys[index]
            assert key is not None
            if key not in first_index_by_key:
                first_index_by_key[key] = index
                previous = self.mm_preprocess_cache.pop(key)
                if previous is not None:
                    previous_metadata[key] = previous

        unique_keys = list(first_index_by_key)

        # 4. submit one computation (preprocess) for each unique miss
        cache_results = self.mm_preprocess_cache.lookup_or_claim_many(
            unique_keys,
            predicate=lambda key, artifact: self.artifact_usable(
                artifact,
                allow_featureless=featureless_hit_mask[first_index_by_key[key]],
            ),
        )
        resolved_by_key: dict[str, MediaArtifact] = {}
        misses_to_compute: list[CacheMiss[str, MediaArtifact]] = []
        for key, result in zip(unique_keys, cache_results):
            if isinstance(result, CacheLookup):
                resolved_by_key[key] = result.value
            elif result.should_compute:
                misses_to_compute.append(result)

        if misses_to_compute:
            missed_task = self.mm_preprocess_cache.create_background_task(
                self._compute_cache_misses(
                    misses_to_compute,
                    first_index_by_key,
                    snapshots,
                    previous_metadata,
                    resolved_by_key,
                    modality,
                )
            )
            # shared work outlives cancellation of this request
            await asyncio.shield(missed_task)

        # 5. wait for misses already claimed by another request
        for key, result in zip(unique_keys, cache_results):
            if isinstance(result, CacheMiss) and not result.should_compute:
                resolved_by_key[key] = await self.mm_preprocess_cache.wait_for_miss(
                    result
                )

        # 6. restore the original processor-input order
        for index, artifact in enumerate(artifacts):
            if artifact is None:
                key = keys[index]
                assert key is not None
                artifacts[index] = resolved_by_key[key]
        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Artifact cache did not resolve every media item")
        return [artifact for artifact in artifacts if artifact is not None]

    async def _compute_cache_misses(
        self,
        misses_to_compute: Sequence[CacheMiss[str, MediaArtifact]],
        first_index_by_key: Mapping[str, int],
        snapshots: Sequence[Optional[MediaSnapshot]],
        previous_metadata: Mapping[str, MediaArtifact],
        resolved_by_key: dict[str, MediaArtifact],
        modality: Modality,
    ) -> None:
        """Decode and preprocess claimed misses, then wake concurrent waiters.

        The full artifact is returned to requests waiting on the miss. A
        possibly smaller ``artifact.cache_value()`` is retained in the bounded
        CPU cache. The two values differ when a CUDA feature must not be cached.
        """
        try:
            # 1. decode (load media) each unique miss
            missed_media = []
            for missed in misses_to_compute:
                index = first_index_by_key[missed.key]
                snapshot = snapshots[index]
                assert snapshot is not None
                media = await asyncio.wrap_future(
                    self.io_executor.submit(
                        self.decode_media_snapshot, snapshot, modality
                    )
                )
                missed_media.append(
                    MediaArtifactInput(
                        content_digest=snapshot.content_digest,
                        artifact_key=missed.key,
                        modality=modality,
                        media=media,
                    )
                )

            # 2. preprocess all decoded misses as one model batch
            missed_artifacts = await self._run_preprocess_and_build_artifact_batch(
                missed_media
            )
            if len(missed_artifacts) != len(misses_to_compute):
                raise ValueError(
                    "prepare_artifact_batch must return one artifact per cache miss"
                )
            for missed, entry, artifact in zip(
                misses_to_compute, missed_media, missed_artifacts
            ):
                self.validate_artifact(artifact, entry)
                previous = previous_metadata.get(missed.key)
                if (
                    previous is not None
                    and previous.feature_hash != artifact.feature_hash
                ):
                    raise ValueError(
                        "Cached media artifact feature hash changed for identical "
                        f"identity {missed.key}"
                    )
                cache_value = artifact.cache_value()
                self.validate_artifact(cache_value, entry)
                # 3. return full artifacts and retain cache-safe copies
                self.mm_preprocess_cache.complete_miss(
                    missed,
                    artifact,
                    cache_value=cache_value,
                )
                resolved_by_key[missed.key] = artifact
        except BaseException as error:
            for missed in misses_to_compute:
                if not missed.future.done():
                    self.mm_preprocess_cache.fail_miss(missed, error)
            raise
