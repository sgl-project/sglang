"""One owner rank encodes each image span and broadcasts it to the ranks that
run the same prefill chunk; every agreement precedes the payload it guards."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache

logger = logging.getLogger(__name__)

SpanKey = Tuple[Optional[int], int]
SpanEncoder = Callable[[List[Any]], torch.Tensor | List[torch.Tensor]]
SpanSignature = Callable[[Any, int], Tuple[Any, ...]]

LOCAL_HIT = 0
OWNER_CACHE_BROADCAST = 1
OWNER_ENCODE_BROADCAST = 2

PHASE_PREPARE = "prepare"
PHASE_FEATURES = "features"
PHASE_FINALIZE = "finalize"


class MmOwnerProtocolError(RuntimeError):
    """Raised with identical text on every group member after a group-agreed failure."""


class ImageSpanRequest(msgspec.Struct, frozen=True):
    hash: Optional[int]
    span_len: int
    item: Any
    inside_chunk: bool
    duplicates: List[Any] = []


class ImageSpanKey(msgspec.Struct, frozen=True):
    hash: Optional[int]
    span_len: int
    geometry: Optional[Tuple[Any, ...]]


class RankManifest(msgspec.Struct, frozen=True):
    rank: int
    keys: List[ImageSpanKey]
    cached: List[bool]
    dtype: str
    width: int
    rids: List[str]
    error: Optional[str] = None


class OwnerPlan(msgspec.Struct, frozen=True):
    actions: List[int]
    owners: List[int]
    error: Optional[str] = None


class RankStatus(msgspec.Struct, frozen=True):
    rank: int
    error: Optional[str] = None


def select_owner_group(parallel) -> Optional[Any]:
    """The group whose members all execute the same requests; None keeps each
    rank encoding its own images."""
    replication = parallel.tp_size // parallel.attn_dp_size
    if replication <= 1:
        return None
    if parallel.attn_cp_size == 1:
        group = parallel.attn_tp_group
    elif parallel.attn_dp_size == 1 and parallel.attn_cp_size == parallel.tp_size:
        group = parallel.attn_cp_group
    else:
        return None
    return group if group.world_size == replication else None


def has_owner_span_work(
    mm_inputs: Sequence[Any],
    extend_prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
) -> bool:
    """Host-side mirror of the per-image scheduling path: does any raw
    single-span image overlap the chunk on every rank of the group."""
    for mm_input, prefix_len, extend_len in zip(
        mm_inputs, extend_prefix_lens, extend_seq_lens
    ):
        if mm_input is None or extend_len <= 0:
            continue
        items = [item for item in mm_input.mm_items if item is not None]
        if not items or any(
            item.precomputed_embeddings is not None or len(item.offsets) != 1
            for item in items
        ):
            continue
        for item in items:
            start, end = item.offsets[0]
            if end >= prefix_len and start < prefix_len + extend_len:
                return True
    return False


class MmOwnerSession(msgspec.Struct):
    group: Any
    device: Any
    dtype: Any
    width: int
    rids: List[str]
    signature: Any
    engaged: bool
    phase: str = PHASE_PREPARE
    in_collective: bool = False

    def resolve(
        self,
        requests: Sequence[ImageSpanRequest],
        cache: MultiModalStaticCache,
        encode: SpanEncoder,
    ) -> Dict[SpanKey, torch.Tensor]:
        if not self.engaged:
            raise RuntimeError(
                "owner protocol reached for a chunk whose host metadata has no image span"
            )
        # Owners allocate different amounts than receivers, so none of these
        # buffers may come out of a symmetric pool.
        saved_context = disable_symmetric_memory_context()
        try:
            return _resolve_owner_features(self, requests, cache, encode)
        finally:
            restore_symmetric_memory_context(saved_context)

    def features_ready(self) -> None:
        self._complete()
        self.phase = PHASE_FINALIZE

    @contextmanager
    def uncaptured(self) -> Iterator[None]:
        # A failure inside a collective leaves the group in an unknown state;
        # no later exchange may try to agree on it.
        self.in_collective = True
        yield
        self.in_collective = False

    @contextmanager
    def fence(self) -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            self._fail(exc)
            raise
        self._complete()

    def _fail(self, exc: BaseException) -> None:
        if (
            not self.engaged
            or self.in_collective
            or isinstance(exc, MmOwnerProtocolError)
        ):
            raise exc
        text = _describe(self, self.phase, exc)
        if self.phase == PHASE_PREPARE:
            try:
                _exchange_manifest(self, _manifest(self, [], [], error=text))
            except MmOwnerProtocolError as agreed:
                raise agreed from exc
        _exchange_status(self, text, exc)

    def _complete(self) -> None:
        if not self.engaged:
            return
        if self.phase == PHASE_PREPARE:
            raise RuntimeError(
                f"owner protocol {self.phase} completed without a manifest exchange"
            )
        error = None
        cause = None
        try:
            _synchronize(self.device)
        except Exception as exc:
            cause = exc
            error = _describe(self, self.phase, exc)
        _exchange_status(self, error, cause)


def _manifest(
    session: MmOwnerSession,
    keys: List[ImageSpanKey],
    cached: List[bool],
    error: Optional[str] = None,
) -> RankManifest:
    return RankManifest(
        rank=session.group.rank_in_group,
        keys=keys,
        cached=cached,
        dtype=str(session.dtype),
        width=session.width,
        rids=list(session.rids),
        error=error,
    )


def _exchange_manifest(session: MmOwnerSession, manifest: RankManifest) -> OwnerPlan:
    group = session.group
    with session.uncaptured():
        manifests = group.all_gather_object(manifest)
        plan = _plan_or_error(session, manifests) if group.rank_in_group == 0 else None
        plan = group.broadcast_object(plan, src=0)
    session.phase = PHASE_FEATURES
    if plan.error is not None:
        raise MmOwnerProtocolError(plan.error)
    return plan


def _plan_or_error(session: MmOwnerSession, manifests: List[RankManifest]) -> OwnerPlan:
    try:
        return _make_plan(manifests)
    except Exception as exc:
        return OwnerPlan(actions=[], owners=[], error=_describe(session, "plan", exc))


def _exchange_status(
    session: MmOwnerSession, error: Optional[str], cause: Optional[BaseException]
) -> None:
    with session.uncaptured():
        statuses = session.group.all_gather_object(
            RankStatus(rank=session.group.rank_in_group, error=error)
        )
    _raise_first_error(statuses, cause)


def _resolve_owner_features(
    session: MmOwnerSession,
    requests: Sequence[ImageSpanRequest],
    cache: MultiModalStaticCache,
    encode: SpanEncoder,
) -> Dict[SpanKey, torch.Tensor]:
    group = session.group
    features: Dict[SpanKey, torch.Tensor] = {}
    keys: List[ImageSpanKey] = []
    cached: List[bool] = []
    error = None
    try:
        keys, cached = _pin_local_cache(session, requests, cache, features)
    except Exception as exc:
        error = _describe(session, "manifest", exc)
    plan = _exchange_manifest(session, _manifest(session, keys, cached, error))

    if all(action == LOCAL_HIT for action in plan.actions):
        return features

    buffers: Dict[int, torch.Tensor] = {}
    error = None
    try:
        buffers = _prepare_transfers(session, requests, keys, plan, features, encode)
        _synchronize(session.device)
    except Exception as exc:
        error = _describe(session, "encode", exc)
    _exchange_status(session, error, None)

    with session.uncaptured():
        for index, (action, owner) in enumerate(zip(plan.actions, plan.owners)):
            if action != LOCAL_HIT:
                group.broadcast(buffers[index], src=owner)

    for index, key in enumerate(keys):
        if plan.actions[index] == LOCAL_HIT:
            continue
        span = buffers[index]
        features[(key.hash, key.span_len)] = span
        cache.set(key.hash, EmbeddingResult(embedding=span))
    return features


def _pin_local_cache(
    session: MmOwnerSession,
    requests: Sequence[ImageSpanRequest],
    cache: MultiModalStaticCache,
    features: Dict[SpanKey, torch.Tensor],
) -> Tuple[List[ImageSpanKey], List[bool]]:
    keys: List[ImageSpanKey] = []
    cached: List[bool] = []
    for request in requests:
        if request.hash is None:
            raise ValueError(
                f"image span of {request.span_len} tokens has no content hash"
            )
        geometry = session.signature(request.item, request.span_len)
        for duplicate in request.duplicates:
            other = session.signature(duplicate, request.span_len)
            if other != geometry:
                raise ValueError(
                    f"image hash {request.hash} ({request.span_len} tokens) occurs "
                    f"with different geometry: {geometry} vs {other}"
                )
        keys.append(
            ImageSpanKey(
                hash=request.hash, span_len=request.span_len, geometry=geometry
            )
        )
        span = _valid_cached_span(session, cache, request)
        if span is not None:
            features[(request.hash, request.span_len)] = span
        cached.append(span is not None)
    return keys, cached


def _valid_cached_span(
    session: MmOwnerSession,
    cache: MultiModalStaticCache,
    request: ImageSpanRequest,
) -> Optional[torch.Tensor]:
    entry = cache.get_single(request.hash)
    if entry is None:
        return None
    span = entry.embedding
    if (
        span.dim() == 2
        and span.shape[0] == request.span_len
        and span.shape[1] == session.width
        and span.dtype == session.dtype
        and span.device == session.device
    ):
        return span
    logger.warning(
        "Discarding cached multimodal embedding that cannot serve the current "
        "image span: cache_key=%s expected=(%d, %d, %s) cached=(%s, %s).",
        request.hash,
        request.span_len,
        session.width,
        session.dtype,
        tuple(span.shape),
        span.dtype,
    )
    cache.free(request.hash, None)
    return None


def _make_plan(manifests: List[RankManifest]) -> OwnerPlan:
    for manifest in manifests:
        if manifest.error is not None:
            return OwnerPlan(actions=[], owners=[], error=manifest.error)
    lead = manifests[0]
    for manifest in manifests[1:]:
        if (manifest.keys, manifest.dtype, manifest.width, manifest.rids) != (
            lead.keys,
            lead.dtype,
            lead.width,
            lead.rids,
        ):
            return OwnerPlan(
                actions=[],
                owners=[],
                error=(
                    "image manifest mismatch between group ranks 0 and "
                    f"{manifest.rank}: rids={lead.rids} vs {manifest.rids}, "
                    f"keys={lead.keys} vs {manifest.keys}, "
                    f"dtype={lead.dtype} vs {manifest.dtype}, "
                    f"width={lead.width} vs {manifest.width}"
                ),
            )
    replication = len(manifests)
    actions: List[int] = []
    owners: List[int] = []
    for index, key in enumerate(lead.keys):
        owner = key.hash % replication
        if all(manifest.cached[index] for manifest in manifests):
            action = LOCAL_HIT
        elif manifests[owner].cached[index]:
            action = OWNER_CACHE_BROADCAST
        else:
            action = OWNER_ENCODE_BROADCAST
        actions.append(action)
        owners.append(owner)
    return OwnerPlan(actions=actions, owners=owners)


def _prepare_transfers(
    session: MmOwnerSession,
    requests: Sequence[ImageSpanRequest],
    keys: List[ImageSpanKey],
    plan: OwnerPlan,
    features: Dict[SpanKey, torch.Tensor],
    encode: SpanEncoder,
) -> Dict[int, torch.Tensor]:
    rank = session.group.rank_in_group
    buffers: Dict[int, torch.Tensor] = {}
    owned: List[int] = []
    for index, (action, owner) in enumerate(zip(plan.actions, plan.owners)):
        if action == LOCAL_HIT:
            continue
        if owner != rank:
            try:
                buffers[index] = _new_span_buffer(session, keys[index])
            except Exception as exc:
                raise RuntimeError(
                    f"receive buffer for image hash {keys[index].hash} shape "
                    f"{(keys[index].span_len, session.width)} {session.dtype} "
                    f"failed: {type(exc).__name__}: {exc}"
                ) from exc
        elif action == OWNER_CACHE_BROADCAST:
            key = (keys[index].hash, keys[index].span_len)
            buffers[index] = features[key].contiguous()
        else:
            owned.append(index)
    if owned:
        owned_hashes = [keys[index].hash for index in owned]
        try:
            encoded = encode([requests[index].item for index in owned])
        except Exception as exc:
            raise RuntimeError(
                f"owner encode of image hashes {owned_hashes} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        spans = _split_spans(encoded, [keys[index].span_len for index in owned])
        for index, span in zip(owned, spans):
            buffers[index] = _validated_span(session, keys[index], span)
    return buffers


def _new_span_buffer(session: MmOwnerSession, key: ImageSpanKey) -> torch.Tensor:
    return torch.empty(
        (key.span_len, session.width), device=session.device, dtype=session.dtype
    )


def _split_spans(
    encoded: torch.Tensor | List[torch.Tensor], span_lens: List[int]
) -> List[torch.Tensor]:
    if isinstance(encoded, list):
        if len(encoded) != len(span_lens):
            raise ValueError(
                f"encoder returned {len(encoded)} spans for {len(span_lens)} images"
            )
        return [span.reshape(-1, span.shape[-1]) for span in encoded]
    encoded = encoded.reshape(-1, encoded.shape[-1])
    if encoded.shape[0] != sum(span_lens):
        raise ValueError(
            f"encoder returned {encoded.shape[0]} rows for spans of {span_lens}"
        )
    return list(torch.split(encoded, span_lens, dim=0))


def _validated_span(
    session: MmOwnerSession, key: ImageSpanKey, span: torch.Tensor
) -> torch.Tensor:
    expected = (key.span_len, session.width)
    if tuple(span.shape) != expected or span.dtype != session.dtype:
        raise ValueError(
            f"encoded span for hash={key.hash} has shape {tuple(span.shape)} "
            f"dtype {span.dtype}; expected {expected} {session.dtype}"
        )
    if span.device != session.device:
        span = span.to(session.device)
    return span.contiguous()


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def _describe(session: MmOwnerSession, stage: str, exc: BaseException) -> str:
    return (
        f"multimodal owner protocol failed during {stage} on group rank "
        f"{session.group.rank_in_group} (global rank "
        f"{session.group.ranks[session.group.rank_in_group]}, rids={list(session.rids)}): "
        f"{type(exc).__name__}: {exc}"
    )


def _raise_first_error(
    statuses: List[RankStatus], cause: Optional[BaseException]
) -> None:
    for status in statuses:
        if status.error is not None:
            raise MmOwnerProtocolError(status.error) from cause
