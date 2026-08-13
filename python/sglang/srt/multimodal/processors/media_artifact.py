"""Reusable per-media artifact caching contract for multimodal processors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Optional, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.cache import (
    CacheLookup,
    CacheReservation,
    MediaSnapshot,
    PreprocessCacheLookup,
    build_artifact_key,
    media_preprocess_kwargs,
    parse_content_hash,
    snapshot_media,
)
from sglang.srt.utils import load_image


@runtime_checkable
class MediaArtifact(Protocol):
    """Minimum contract required by the shared artifact cache path."""

    content_digest: str
    artifact_key: str
    feature_identity: str
    feature_hash: int

    @property
    def has_feature(self) -> bool: ...

    def cache_value(self) -> MediaArtifact: ...


ArtifactT = TypeVar("ArtifactT", bound=MediaArtifact)


class MediaCacheRequest(Protocol):
    mm_content_hashes: Optional[Sequence[Optional[str]]]


@dataclass(frozen=True)
class MediaArtifactInput:
    """One decoded cache miss with its already-validated identity."""

    content_digest: str
    artifact_key: str
    modality: Modality
    media: Any


@dataclass(frozen=True)
class MediaArtifactLookup(Generic[ArtifactT]):
    """Identity and optional metadata resolved before processor dispatch."""

    content_digest: str
    artifact_key: str
    snapshot: Optional[MediaSnapshot]
    cached_artifact: Optional[ArtifactT]
    identity_source: str


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
    supports_early_mm_cache = True

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
        try:
            parse_content_hash(artifact.feature_identity)
        except ValueError as error:
            raise ValueError(
                "Media artifact feature_identity must be a SHA-256 digest"
            ) from error
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
        start = time.perf_counter()
        try:
            if self.mm_processor_executor is None:
                return self.prepare_artifact_batch(entries)
            return await self.mm_processor_executor.run(
                self.prepare_artifact_batch, entries
            )
        finally:
            self.observe_preprocess_phase("processor", time.perf_counter() - start)

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
                and self.artifact_usable(value, allow_featureless=allow_featureless)
            ),
        )
        if artifact is not None:
            self.validate_artifact(
                artifact,
                MediaArtifactInput(content_digest, key, modality, None),
            )
        return artifact

    def _normalize_content_hashes(
        self,
        content_hashes: Optional[Sequence[Optional[str]]],
        media_count: int,
        modality: Modality,
    ) -> list[Optional[str]]:
        if content_hashes is None:
            content_hashes = [None] * media_count
        if len(content_hashes) != media_count:
            raise ValueError(
                f"mm_content_hashes has {len(content_hashes)} entries for "
                f"{media_count} {modality.name.lower()} items"
            )
        return [parse_content_hash(value) for value in content_hashes]

    async def _lookup_media_artifacts(
        self,
        media_data: Sequence[Any],
        *,
        content_hashes: Optional[Sequence[Optional[str]]] = None,
        modality: Optional[Modality] = None,
    ) -> tuple[MediaArtifactLookup[ArtifactT], ...]:
        """Resolve strict identities and reusable metadata before dispatch."""
        modality = self._resolve_artifact_modality(modality)
        content_hashes = self._normalize_content_hashes(
            content_hashes, len(media_data), modality
        )
        lookups: list[Optional[MediaArtifactLookup[ArtifactT]]] = [None] * len(
            media_data
        )
        read_indices = []
        for index, (source, caller_hash) in enumerate(zip(media_data, content_hashes)):
            if self.trust_mm_content_hashes and caller_hash is not None:
                key = self._artifact_key(caller_hash, source, modality=modality)
                artifact = self._get_cached_artifact(
                    key, caller_hash, modality, allow_featureless=True
                )
                if artifact is not None:
                    lookups[index] = MediaArtifactLookup(
                        content_digest=caller_hash,
                        artifact_key=key,
                        snapshot=None,
                        cached_artifact=artifact,
                        identity_source="trusted",
                    )
                    continue
            read_indices.append(index)

        futures = {
            index: self.io_executor.submit(
                self.snapshot_media_source, media_data[index], modality
            )
            for index in read_indices
        }
        for index, future in futures.items():
            snapshot = await asyncio.wrap_future(future)
            caller_hash = content_hashes[index]
            if caller_hash is not None and caller_hash != snapshot.content_digest:
                raise ValueError(
                    f"content hash mismatch for media_data[{index}]: "
                    f"expected {caller_hash}, got {snapshot.content_digest}"
                )
            key = self._artifact_key(
                snapshot.content_digest, media_data[index], modality=modality
            )
            lookups[index] = MediaArtifactLookup(
                content_digest=snapshot.content_digest,
                artifact_key=key,
                snapshot=snapshot,
                cached_artifact=self._get_cached_artifact(
                    key,
                    snapshot.content_digest,
                    modality,
                    allow_featureless=True,
                ),
                identity_source="server_computed",
            )

        if any(lookup is None for lookup in lookups):
            raise RuntimeError("Artifact identity lookup did not resolve every item")
        return cast(tuple[MediaArtifactLookup[ArtifactT], ...], tuple(lookups))

    async def lookup_preprocess_cache(
        self, media_data: Sequence[Any], request_obj: MediaCacheRequest
    ) -> Optional[PreprocessCacheLookup]:
        """Expose generic per-media metadata to the scheduler lease path."""
        if (
            not self.mm_preprocess_cache.enabled
            or not media_data
            or any(self._is_preprocessed_input(item) for item in media_data)
        ):
            return None
        lookups = await self._lookup_media_artifacts(
            media_data,
            content_hashes=request_obj.mm_content_hashes,
        )
        return PreprocessCacheLookup(
            processor_state=lookups,
            feature_hashes=tuple(
                (
                    lookup.cached_artifact.feature_hash
                    if lookup.cached_artifact is not None
                    else None
                )
                for lookup in lookups
            ),
            feature_identities=tuple(
                (
                    lookup.cached_artifact.feature_identity
                    if lookup.cached_artifact is not None
                    else None
                )
                for lookup in lookups
            ),
            identity_sources=tuple(lookup.identity_source for lookup in lookups),
        )

    async def prepare_media_artifacts(
        self,
        media_data: Sequence[Any],
        *,
        content_hashes: Optional[Sequence[Optional[str]]] = None,
        featureless_hit_mask: Optional[Sequence[bool]] = None,
        modality: Optional[Modality] = None,
        media_lookups: Optional[Sequence[MediaArtifactLookup[ArtifactT]]] = None,
    ) -> list[ArtifactT]:
        """Resolve identities and preprocess only per-media cache misses."""
        modality = self._resolve_artifact_modality(modality)
        media_count = len(media_data)
        content_hashes = self._normalize_content_hashes(
            content_hashes, media_count, modality
        )

        if featureless_hit_mask is None:
            featureless_hit_mask = [False] * media_count
        if len(featureless_hit_mask) != media_count:
            raise ValueError("featureless_hit_mask must align with media_data")

        if media_lookups is None:
            media_lookups = await self._lookup_media_artifacts(
                media_data, content_hashes=content_hashes, modality=modality
            )
        if len(media_lookups) != media_count:
            raise ValueError("media_lookups must align with media_data")

        artifacts: list[Optional[ArtifactT]] = [None] * media_count
        snapshots = [lookup.snapshot for lookup in media_lookups]
        keys = [lookup.artifact_key for lookup in media_lookups]
        load_indices = []
        for index, (source, lookup, allow_featureless) in enumerate(
            zip(media_data, media_lookups, featureless_hit_mask)
        ):
            expected_key = self._artifact_key(
                lookup.content_digest, source, modality=modality
            )
            if lookup.artifact_key != expected_key:
                raise ValueError("Media preprocess options changed after cache lookup")
            if lookup.cached_artifact is not None and self.artifact_usable(
                lookup.cached_artifact, allow_featureless=allow_featureless
            ):
                self.validate_artifact(
                    lookup.cached_artifact,
                    MediaArtifactInput(
                        lookup.content_digest,
                        lookup.artifact_key,
                        modality,
                        None,
                    ),
                )
                artifacts[index] = lookup.cached_artifact
            else:
                load_indices.append(index)

        # A trusted metadata hit can skip the first read. If its embedding is
        # absent, verify the source before recomputing a feature.
        for index in load_indices:
            if snapshots[index] is not None:
                continue
            snapshot = await asyncio.wrap_future(
                self.io_executor.submit(
                    self.snapshot_media_source, media_data[index], modality
                )
            )
            if snapshot.content_digest != media_lookups[index].content_digest:
                raise ValueError(
                    f"trusted content hash mismatch for media_data[{index}]: "
                    f"expected {media_lookups[index].content_digest}, "
                    f"got {snapshot.content_digest}"
                )
            snapshots[index] = snapshot

        # Deduplicate misses before decode and reserve them across requests.
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
        reservations = self.mm_preprocess_cache.reserve_many(
            unique_keys,
            predicate=lambda key, artifact: self.artifact_usable(
                artifact,
                allow_featureless=featureless_hit_mask[first_index_by_key[key]],
            ),
        )
        resolved_by_key: dict[str, ArtifactT] = {}
        owners: list[CacheReservation[str, ArtifactT]] = []
        for key, reservation in zip(unique_keys, reservations):
            if isinstance(reservation, CacheLookup):
                resolved_by_key[key] = reservation.value
            elif reservation.owner:
                owners.append(reservation)

        if owners:
            owner_task = self.mm_preprocess_cache.create_background_task(
                self._fulfill_artifact_reservations(
                    owners,
                    first_index_by_key,
                    snapshots,
                    previous_metadata,
                    resolved_by_key,
                    modality,
                )
            )
            # Shared work outlives cancellation of the request that became owner.
            await asyncio.shield(owner_task)

        for key, reservation in zip(unique_keys, reservations):
            if isinstance(reservation, CacheReservation) and not reservation.owner:
                resolved_by_key[key] = await self.mm_preprocess_cache.wait(reservation)

        for index, artifact in enumerate(artifacts):
            if artifact is None:
                key = keys[index]
                assert key is not None
                artifacts[index] = resolved_by_key[key]
        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Artifact cache did not resolve every media item")
        return cast(list[ArtifactT], artifacts)

    async def _fulfill_artifact_reservations(
        self,
        owners: Sequence[CacheReservation[str, ArtifactT]],
        first_index_by_key: Mapping[str, int],
        snapshots: Sequence[Optional[MediaSnapshot]],
        previous_metadata: Mapping[str, ArtifactT],
        resolved_by_key: dict[str, ArtifactT],
        modality: Modality,
    ) -> None:
        try:
            owner_entries = []
            decode_start = time.perf_counter()
            try:
                for reservation in owners:
                    index = first_index_by_key[reservation.key]
                    snapshot = snapshots[index]
                    assert snapshot is not None
                    media = await asyncio.wrap_future(
                        self.io_executor.submit(
                            self.decode_media_snapshot, snapshot, modality
                        )
                    )
                    owner_entries.append(
                        MediaArtifactInput(
                            content_digest=snapshot.content_digest,
                            artifact_key=reservation.key,
                            modality=modality,
                            media=media,
                        )
                    )
            finally:
                self.observe_preprocess_phase(
                    "decode", time.perf_counter() - decode_start
                )

            owner_artifacts = await self._run_artifact_batch(owner_entries)
            if len(owner_artifacts) != len(owners):
                raise ValueError(
                    "prepare_artifact_batch must return one artifact per cache miss"
                )
            for reservation, entry, artifact in zip(
                owners, owner_entries, owner_artifacts
            ):
                self.validate_artifact(artifact, entry)
                previous = previous_metadata.get(reservation.key)
                if previous is not None and (
                    previous.feature_identity != artifact.feature_identity
                    or previous.feature_hash != artifact.feature_hash
                ):
                    raise ValueError(
                        "Cached media artifact feature identity or hash changed for "
                        "identical "
                        f"identity {reservation.key}"
                    )
                cache_value = artifact.cache_value()
                self.validate_artifact(cache_value, entry)
                self.mm_preprocess_cache.fulfill(
                    reservation,
                    artifact,
                    cache_value=cache_value,
                )
                resolved_by_key[reservation.key] = artifact
        except BaseException as error:
            for reservation in owners:
                if not reservation.future.done():
                    self.mm_preprocess_cache.fail(reservation, error)
            raise
