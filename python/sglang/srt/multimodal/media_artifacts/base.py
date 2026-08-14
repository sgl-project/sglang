"""Shared contracts and coordination for reusable multimodal artifacts.

An artifact is the prompt-independent result of preprocessing one media item.
It is reused by the current request and is the logical item stored in the
multimodal preprocess cache.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Optional, Protocol, TypeVar, cast, runtime_checkable

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
    """A reusable per-media preprocess result and logical cache item."""

    content_digest: str
    artifact_key: str
    feature_hash: int

    @property
    def has_feature(self) -> bool: ...

    def cache_value(self) -> MediaArtifact: ...

    def cache_size_items(self) -> Sequence[Any]: ...


ArtifactT = TypeVar("ArtifactT", bound=MediaArtifact)


@dataclass(frozen=True)
class MediaArtifactInput:
    """One decoded raw multimodal input that missed the preprocess cache.

    The shared cache layer has already validated its content digest, derived
    its artifact key, and claimed the cache miss. The model-specific artifact
    builder can therefore preprocess ``media`` without loading or hashing it
    again.
    """

    content_digest: str
    artifact_key: str
    modality: Modality
    media: Any


class MediaArtifactCacheMixin(Generic[ArtifactT]):
    """Identity, single-flight, partial-hit, and cache lifecycle by modality.

    A model adapter only supplies the modality, batch artifact materialization,
    and request composition. A multi-modal model can call the same coordinator
    with different modalities and override snapshot/decode/key hooks without
    copying the cache algorithm. Processor implementations must expose every
    artifact-producing setting through ``preprocess_fingerprint_payload``.
    """

    artifact_modality: Optional[Modality] = None
    artifact_option_defaults: Mapping[str, Any] = {"detail": "auto"}

    def artifact_preprocess_kwargs(
        self, source: Any, modality: Modality
    ) -> Mapping[str, Any]:
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
                f"{modality.name.lower()} artifact decoding " "requires a model adapter"
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
                f"{modality.name.lower()} artifact identity " "requires a model adapter"
            )
        return snapshot_media(source)

    def prepare_artifact_batch(
        self, entries: Sequence[MediaArtifactInput]
    ) -> list[ArtifactT]:
        """Preprocess raw multimodal inputs that missed the preprocess cache.

        Each entry is one unique, decoded cache miss. Implementations must
        return one reusable artifact (the preprocess-cache item) per entry, in
        the same order, while preserving its content digest and artifact key.
        The shared layer uses the artifact for the current request and stores
        ``artifact.cache_value()`` for reuse.
        """
        raise NotImplementedError

    def artifact_usable(self, artifact: ArtifactT, *, allow_featureless: bool) -> bool:
        return artifact.has_feature or allow_featureless

    @staticmethod
    def validate_artifact(artifact: ArtifactT, entry: MediaArtifactInput) -> None:
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

    async def _run_artifact_batch(
        self, entries: Sequence[MediaArtifactInput]
    ) -> list[ArtifactT]:
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
    ) -> Optional[ArtifactT]:
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
    ) -> list[ArtifactT]:
        """Resolve identities and preprocess only per-media cache misses."""
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

        artifacts: list[Optional[ArtifactT]] = [None] * media_count
        snapshots: list[Optional[MediaSnapshot]] = [None] * media_count
        keys: list[Optional[str]] = [None] * media_count

        # Trusted callers may use a metadata hit without touching the source.
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

        # Deduplicate misses before decode and share their work across requests.
        first_index_by_key: dict[str, int] = {}
        previous_metadata: dict[str, ArtifactT] = {}
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
        cache_results = self.mm_preprocess_cache.lookup_or_claim_many(
            unique_keys,
            predicate=lambda key, artifact: self.artifact_usable(
                artifact,
                allow_featureless=featureless_hit_mask[first_index_by_key[key]],
            ),
        )
        resolved_by_key: dict[str, ArtifactT] = {}
        misses_to_compute: list[CacheMiss[str, ArtifactT]] = []
        for key, result in zip(unique_keys, cache_results):
            if isinstance(result, CacheLookup):
                resolved_by_key[key] = result.value
            elif result.should_compute:
                misses_to_compute.append(result)

        if misses_to_compute:
            miss_task = self.mm_preprocess_cache.create_background_task(
                self._compute_cache_misses(
                    misses_to_compute,
                    first_index_by_key,
                    snapshots,
                    previous_metadata,
                    resolved_by_key,
                    modality,
                )
            )
            # Shared work outlives cancellation of the request doing the work.
            await asyncio.shield(miss_task)

        for key, result in zip(unique_keys, cache_results):
            if isinstance(result, CacheMiss) and not result.should_compute:
                resolved_by_key[key] = await self.mm_preprocess_cache.wait_for_miss(
                    result
                )

        for index, artifact in enumerate(artifacts):
            if artifact is None:
                key = keys[index]
                assert key is not None
                artifacts[index] = resolved_by_key[key]
        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Artifact cache did not resolve every media item")
        return cast(list[ArtifactT], artifacts)

    async def _compute_cache_misses(
        self,
        misses_to_compute: Sequence[CacheMiss[str, ArtifactT]],
        first_index_by_key: Mapping[str, int],
        snapshots: Sequence[Optional[MediaSnapshot]],
        previous_metadata: Mapping[str, ArtifactT],
        resolved_by_key: dict[str, ArtifactT],
        modality: Modality,
    ) -> None:
        """Preprocess misses claimed by this request and publish their results."""
        try:
            miss_entries = []
            for miss in misses_to_compute:
                index = first_index_by_key[miss.key]
                snapshot = snapshots[index]
                assert snapshot is not None
                media = await asyncio.wrap_future(
                    self.io_executor.submit(
                        self.decode_media_snapshot, snapshot, modality
                    )
                )
                miss_entries.append(
                    MediaArtifactInput(
                        content_digest=snapshot.content_digest,
                        artifact_key=miss.key,
                        modality=modality,
                        media=media,
                    )
                )

            miss_artifacts = await self._run_artifact_batch(miss_entries)
            if len(miss_artifacts) != len(misses_to_compute):
                raise ValueError(
                    "prepare_artifact_batch must return one artifact per cache miss"
                )
            for miss, entry, artifact in zip(
                misses_to_compute, miss_entries, miss_artifacts
            ):
                self.validate_artifact(artifact, entry)
                previous = previous_metadata.get(miss.key)
                if (
                    previous is not None
                    and previous.feature_hash != artifact.feature_hash
                ):
                    raise ValueError(
                        "Cached media artifact feature hash changed for identical "
                        f"identity {miss.key}"
                    )
                cache_value = artifact.cache_value()
                self.validate_artifact(cache_value, entry)
                self.mm_preprocess_cache.complete_miss(
                    miss,
                    artifact,
                    cache_value=cache_value,
                )
                resolved_by_key[miss.key] = artifact
        except BaseException as error:
            for miss in misses_to_compute:
                if not miss.future.done():
                    self.mm_preprocess_cache.fail_miss(miss, error)
            raise
