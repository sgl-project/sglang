# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BRANCH_COND = 0
BRANCH_UNCOND = 1
BRANCHES = (BRANCH_COND, BRANCH_UNCOND)


@dataclass
class DreamZeroCachePool:
    """Physical DreamZero cache pool addressed by logical session slots."""

    capacity: int
    local_attn_size: int = -1

    session_ids: list[str | None] = field(init=False)
    current_start_frames: list[int] = field(init=False)
    visual_valid: list[bool] = field(init=False)
    prompt_hashes: dict[int, list[str | None]] = field(init=False)
    prompt_valid: dict[int, list[bool]] = field(init=False)
    kv_valid: dict[int, list[bool]] = field(init=False)
    kv_lengths: dict[int, list[int]] = field(init=False)
    crossattn_valid: dict[int, list[bool]] = field(init=False)

    kv_cache1: list[torch.Tensor] = field(default_factory=list)
    kv_cache_neg: list[torch.Tensor] = field(default_factory=list)
    crossattn_cache: list[dict[str, Any]] = field(default_factory=list)
    crossattn_cache_neg: list[dict[str, Any]] = field(default_factory=list)
    cached_prompt_embs: dict[int, torch.Tensor | None] = field(init=False)

    clip_feas: torch.Tensor | None = None
    ys: torch.Tensor | None = None
    latent_video: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.session_ids = [None] * self.capacity
        self.current_start_frames = [0] * self.capacity
        self.visual_valid = [False] * self.capacity
        self.prompt_hashes = {
            branch: [None] * self.capacity for branch in BRANCHES
        }
        self.prompt_valid = {
            branch: [False] * self.capacity for branch in BRANCHES
        }
        self.kv_valid = {
            branch: [False] * self.capacity for branch in BRANCHES
        }
        self.kv_lengths = {
            branch: [0] * self.capacity for branch in BRANCHES
        }
        self.crossattn_valid = {
            branch: [False] * self.capacity for branch in BRANCHES
        }
        self.cached_prompt_embs = {branch: None for branch in BRANCHES}

    def branch_kv_cache(self, branch: int) -> list[torch.Tensor]:
        return self.kv_cache1 if branch == BRANCH_COND else self.kv_cache_neg

    def set_branch_kv_cache(self, branch: int, value: list[torch.Tensor]) -> None:
        if branch == BRANCH_COND:
            self.kv_cache1 = value
        else:
            self.kv_cache_neg = value

    def branch_crossattn_cache(self, branch: int) -> list[dict[str, Any]]:
        return (
            self.crossattn_cache
            if branch == BRANCH_COND
            else self.crossattn_cache_neg
        )

    def set_branch_crossattn_cache(
        self, branch: int, value: list[dict[str, Any]]
    ) -> None:
        if branch == BRANCH_COND:
            self.crossattn_cache = value
        else:
            self.crossattn_cache_neg = value

    def reset_slot(self, slot: int, *, preserve_text: bool = False) -> None:
        self.current_start_frames[slot] = 0
        self.visual_valid[slot] = False
        for branch in BRANCHES:
            self.kv_valid[branch][slot] = False
            self.kv_lengths[branch][slot] = 0
            self.crossattn_valid[branch][slot] = False
            if not preserve_text:
                self.prompt_valid[branch][slot] = False
                self.prompt_hashes[branch][slot] = None
            if not any(self.kv_valid[branch]):
                self.set_branch_kv_cache(branch, [])
            if not any(self.crossattn_valid[branch]):
                self.set_branch_crossattn_cache(branch, [])
        if not any(self.visual_valid):
            self.clip_feas = None
            self.ys = None
            self.latent_video = None

    def reset_stream(self, *, preserve_text: bool) -> None:
        for slot in range(self.capacity):
            if self.session_ids[slot] is not None:
                self.reset_slot(slot, preserve_text=preserve_text)

    def prompt_reusable(self, branch: int, slot: int, prompt_hash: str | None) -> bool:
        return (
            prompt_hash is not None
            and self.prompt_valid[branch][slot]
            and self.prompt_hashes[branch][slot] == prompt_hash
            and self.cached_prompt_embs[branch] is not None
        )

    def gather_prompt(self, branch: int, slots: list[int]) -> torch.Tensor | None:
        prompt_pool = self.cached_prompt_embs[branch]
        if prompt_pool is None:
            return None
        return prompt_pool.index_select(0, _slot_tensor(slots, prompt_pool.device))

    def scatter_prompt(
        self,
        branch: int,
        slots: list[int],
        values: torch.Tensor,
        prompt_hashes: list[str | None],
    ) -> None:
        if self.cached_prompt_embs[branch] is None:
            shape = (_pool_capacity_for_slots(None, slots, self.capacity), *values.shape[1:])
            self.cached_prompt_embs[branch] = values.new_zeros(shape)
        prompt_pool = self.cached_prompt_embs[branch]
        if prompt_pool is None:
            raise RuntimeError("DreamZero prompt pool was not initialized")
        prompt_pool = _ensure_mutable_tensor(prompt_pool)
        target_capacity = _pool_capacity_for_slots(
            prompt_pool,
            slots,
            self.capacity,
        )
        same_value_shape = prompt_pool.shape[1:] == values.shape[1:]
        if prompt_pool.shape[0] < target_capacity or not same_value_shape:
            shape = (target_capacity, *values.shape[1:])
            new_pool = values.new_zeros(shape)
            if same_value_shape:
                new_pool[: prompt_pool.shape[0]] = prompt_pool
            self.cached_prompt_embs[branch] = new_pool
            prompt_pool = new_pool
            if not same_value_shape:
                self.prompt_valid[branch] = [False] * self.capacity
        self.cached_prompt_embs[branch] = prompt_pool
        prompt_pool.index_copy_(0, _slot_tensor(slots, values.device), values.detach())
        for slot, prompt_hash in zip(slots, prompt_hashes, strict=True):
            self.prompt_valid[branch][slot] = True
            self.prompt_hashes[branch][slot] = prompt_hash

    def gather_visual(
        self, slots: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.clip_feas is None or self.ys is None or self.latent_video is None:
            raise RuntimeError("DreamZero visual cache pool is not initialized")
        missing = [slot for slot in slots if not self.visual_valid[slot]]
        if missing:
            raise RuntimeError(f"DreamZero visual cache missing slots {missing}")
        index = _slot_tensor(slots, self.clip_feas.device)
        return (
            self.clip_feas.index_select(0, index),
            self.ys.index_select(0, _slot_tensor(slots, self.ys.device)),
            self.latent_video.index_select(
                0, _slot_tensor(slots, self.latent_video.device)
            ),
        )

    def scatter_visual(
        self,
        slots: list[int],
        *,
        clip_feas: torch.Tensor | None = None,
        ys: torch.Tensor | None = None,
        latent_video: torch.Tensor | None = None,
    ) -> None:
        if clip_feas is not None:
            self.clip_feas = _scatter_dim0_pool(
                self.clip_feas,
                slots,
                clip_feas,
                pool_capacity=self.capacity,
            )
        if ys is not None:
            self.ys = _scatter_dim0_pool(
                self.ys,
                slots,
                ys,
                pool_capacity=self.capacity,
            )
        if latent_video is not None:
            self.latent_video = _scatter_dim0_pool(
                self.latent_video,
                slots,
                latent_video,
                pool_capacity=self.capacity,
            )
        for slot in slots:
            self.visual_valid[slot] = True

    def gather_kv(self, branch: int, slots: list[int]) -> list[torch.Tensor]:
        lengths = [self.kv_lengths[branch][slot] for slot in slots]
        if len(set(lengths)) != 1:
            raise ValueError(
                "DreamZero v1 requires uniform KV lengths within one batch: "
                f"{lengths}"
            )
        if not all(self.kv_valid[branch][slot] for slot in slots):
            missing = [slot for slot in slots if not self.kv_valid[branch][slot]]
            raise RuntimeError(f"DreamZero KV cache missing slots {missing}")
        length = lengths[0]
        index = None
        gathered = []
        for tensor in self.branch_kv_cache(branch):
            if index is None:
                index = _slot_tensor(slots, tensor.device)
            gathered.append(tensor[:, index, :length].contiguous())
        return gathered

    def scatter_kv(self, branch: int, slots: list[int], values: list[torch.Tensor]) -> None:
        pool = self.branch_kv_cache(branch)
        if not pool:
            pool = [None] * len(values)
        new_pool: list[torch.Tensor] = []
        for existing, value in zip(pool, values, strict=True):
            new_pool.append(
                _scatter_kv_pool(
                    existing,
                    slots,
                    value.detach(),
                    pool_capacity=self.capacity,
                    seq_capacity=self.local_attn_size,
                )
            )
        self.set_branch_kv_cache(branch, new_pool)
        kv_length = int(values[0].shape[2]) if values else 0
        for slot in slots:
            self.kv_valid[branch][slot] = True
            self.kv_lengths[branch][slot] = kv_length

    def gather_crossattn(self, branch: int, slots: list[int]) -> list[dict[str, Any]]:
        cache = self.branch_crossattn_cache(branch)
        if not cache:
            raise RuntimeError("DreamZero cross-attention cache pool is empty")
        if not all(self.crossattn_valid[branch][slot] for slot in slots):
            missing = [
                slot for slot in slots if not self.crossattn_valid[branch][slot]
            ]
            raise RuntimeError(f"DreamZero cross-attention cache missing slots {missing}")
        gathered: list[dict[str, Any]] = []
        for layer_cache in cache:
            if not layer_cache.get("is_init", False):
                gathered.append({"is_init": False})
                continue
            layer = {"is_init": True}
            for key, value in layer_cache.items():
                if key == "is_init":
                    continue
                if torch.is_tensor(value):
                    layer[key] = value.index_select(
                        0, _slot_tensor(slots, value.device)
                    ).contiguous()
                else:
                    layer[key] = value
            gathered.append(layer)
        return gathered

    def scatter_crossattn(
        self, branch: int, slots: list[int], values: list[dict[str, Any]]
    ) -> None:
        existing = self.branch_crossattn_cache(branch)
        if not existing:
            existing = [{"is_init": False} for _ in values]
        new_cache: list[dict[str, Any]] = []
        for layer_existing, layer_value in zip(existing, values, strict=True):
            if not layer_value.get("is_init", False):
                new_cache.append(layer_existing)
                continue
            layer = {"is_init": True}
            for key, value in layer_value.items():
                if key == "is_init":
                    continue
                if torch.is_tensor(value):
                    layer[key] = _scatter_dim0_pool(
                        layer_existing.get(key),
                        slots,
                        value.detach(),
                        pool_capacity=self.capacity,
                    )
                else:
                    layer[key] = value
            new_cache.append(layer)
        self.set_branch_crossattn_cache(branch, new_cache)
        for slot in slots:
            self.crossattn_valid[branch][slot] = True


class DreamZeroCachePoolManager:
    """Bounded logical-session table over a dense physical cache pool."""

    def __init__(self, max_sessions: int = 10) -> None:
        if max_sessions < 1:
            raise ValueError("DreamZero max_sessions must be at least 1")
        self.max_sessions = max_sessions
        self.pool = DreamZeroCachePool(capacity=max_sessions)
        self._sessions: OrderedDict[str, int] = OrderedDict()
        self._free_slots = list(range(max_sessions))
        self._active_reset_apply_token: str | None = None
        self._applied_reset_tokens: set[tuple[str, int]] = set()

    def lookup_or_allocate(
        self,
        logical_session_id: str,
        *,
        local_attn_size: int,
    ) -> tuple[int, bool]:
        slot = self._sessions.get(logical_session_id)
        if slot is not None:
            self._sessions.move_to_end(logical_session_id)
            return slot, True
        if not self._free_slots:
            raise RuntimeError(
                "DreamZero session cache pool is full. v1 disables eviction to "
                "preserve logical-session correctness; increase dreamzero_max_sessions."
            )
        slot = self._free_slots.pop(0)
        self._sessions[logical_session_id] = slot
        self.pool.session_ids[slot] = logical_session_id
        self.pool.local_attn_size = local_attn_size
        return slot, False

    def reset_slot(self, logical_session_id: str, *, preserve_text: bool = False) -> None:
        slot = self._sessions.get(logical_session_id)
        if slot is not None:
            self.pool.reset_slot(slot, preserve_text=preserve_text)

    def get_active_count(self) -> int:
        return len(self._sessions)


@dataclass
class DreamZeroRequestCache:
    """Broadcast-safe request-local logical session mapping.

    This view intentionally does not hold DreamZeroCachePool or tensors. Each
    stage resolves the physical pool through its rank-local cache manager.
    """

    cache_id: str | None
    logical_session_ids: list[str]
    slot_indices: list[int]
    reset_mask: list[bool]
    cache_hit: list[bool]
    prompt_hashes: list[str | None]
    neg_prompt_hashes: list[str | None]
    prompt_reusable: list[bool]
    neg_prompt_reusable: list[bool]
    persistent: bool

    @property
    def batch_size(self) -> int:
        return len(self.logical_session_ids)

    @property
    def has_logical_batch(self) -> bool:
        return self.batch_size > 1

    def pool(self, cache_manager: DreamZeroCachePoolManager | None) -> DreamZeroCachePool:
        if cache_manager is None:
            raise RuntimeError(
                "DreamZero request cache view requires a rank-local cache manager"
            )
        return cache_manager.pool

    def current_start_frames(
        self,
        cache_manager: DreamZeroCachePoolManager | None,
    ) -> list[int]:
        pool = self.pool(cache_manager)
        return [int(pool.current_start_frames[slot]) for slot in self.slot_indices]

    def uniform_current_start_frame(
        self,
        cache_manager: DreamZeroCachePoolManager | None,
    ) -> int:
        frames = self.current_start_frames(cache_manager)
        if len(set(frames)) != 1:
            raise ValueError(
                "DreamZero slot cache v1 requires uniform current_start_frame "
                f"within one request batch, got {frames}"
            )
        return frames[0]

    def mark_current_start_frame(
        self,
        cache_manager: DreamZeroCachePoolManager | None,
        value: int,
    ) -> None:
        pool = self.pool(cache_manager)
        for slot in self.slot_indices:
            pool.current_start_frames[slot] = int(value)


def record_session_timing(batch, key: str, elapsed_ms: float) -> None:
    timings = getattr(batch, "dreamzero_session_timing", None)
    if timings is None:
        timings = {}
        batch.dreamzero_session_timing = timings
    timings[key] = timings.get(key, 0.0) + float(elapsed_ms)


def normalize_batched_session_fields(
    *,
    session_ids: Any,
    reset_mask: Any,
    batch_size: int,
) -> tuple[list[str], list[bool]]:
    if isinstance(session_ids, (list, tuple)):
        normalized_session_ids = [str(session_id) for session_id in session_ids]
    else:
        raise TypeError("dreamzero_session_ids must be a list of session strings")
    normalized_session_ids = [
        session_id.strip() for session_id in normalized_session_ids
    ]
    if any(not session_id for session_id in normalized_session_ids):
        raise ValueError("dreamzero_session_ids cannot contain empty session ids")
    if len(normalized_session_ids) != batch_size:
        raise ValueError(
            "dreamzero_session_ids length must match batch size: "
            f"got {len(normalized_session_ids)} session ids for batch_size={batch_size}"
        )
    if len(set(normalized_session_ids)) != len(normalized_session_ids):
        raise ValueError(
            "DreamZero cache manager forbids duplicate logical session ids in one batch"
        )

    if reset_mask is None:
        raise ValueError("dreamzero_reset_mask is required when using session cache")
    if torch.is_tensor(reset_mask):
        normalized_reset_mask = [bool(value) for value in reset_mask.flatten().tolist()]
    elif isinstance(reset_mask, (list, tuple)):
        normalized_reset_mask = [bool(value) for value in reset_mask]
    else:
        raise TypeError("dreamzero_reset_mask must be a list of bools")
    if len(normalized_reset_mask) != batch_size:
        raise ValueError(
            "dreamzero_reset_mask length must match batch size: "
            f"got {len(normalized_reset_mask)} values for batch_size={batch_size}"
        )
    return normalized_session_ids, normalized_reset_mask


def normalize_batched_prompt_fields(value: Any, batch_size: int) -> list[str | None]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, (list, tuple)):
        values = [None if item is None else str(item) for item in value]
    else:
        raise TypeError("DreamZero prompt fields must be a string or a list")
    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    if len(values) != batch_size:
        raise ValueError(
            "DreamZero prompt field length must match batch size: "
            f"got {len(values)} values for batch_size={batch_size}"
        )
    return values


def _logical_session_fields(batch, batch_size: int) -> tuple[list[str], list[bool]]:
    extra = getattr(batch, "extra", {})
    session_ids = getattr(batch, "dreamzero_session_ids", None)
    if session_ids is None:
        session_ids = extra.get("dreamzero_session_ids")
    if session_ids is None:
        raise ValueError("DreamZero session cache requires dreamzero_session_ids")
    reset_mask = getattr(batch, "dreamzero_reset_mask", None)
    if reset_mask is None:
        reset_mask = extra.get("dreamzero_reset_mask")
    return normalize_batched_session_fields(
        session_ids=session_ids,
        reset_mask=reset_mask,
        batch_size=batch_size,
    )


def _normalize_optional_bool_list(
    value: Any,
    size: int,
    *,
    field_name: str,
    default: bool,
) -> list[bool]:
    if value is None:
        return [default] * size
    if isinstance(value, list):
        values = [bool(item) for item in value]
    elif isinstance(value, tuple):
        values = [bool(item) for item in value]
    elif torch.is_tensor(value):
        values = [bool(item) for item in value.flatten().tolist()]
    else:
        raise TypeError(f"{field_name} must be a list of bools")
    if len(values) != size:
        raise ValueError(
            f"{field_name} length must match batch size: "
            f"got {len(values)} values for batch_size={size}"
        )
    return values


def _normalize_optional_reason_list(value: Any, size: int) -> list[str | None]:
    if value is None:
        return [None] * size
    if isinstance(value, str):
        return [value] * size
    if isinstance(value, (list, tuple)):
        values = [None if item is None else str(item) for item in value]
    else:
        raise TypeError("dreamzero_session_reset_reason must be a string or a list")
    if len(values) != size:
        raise ValueError(
            "dreamzero_session_reset_reason length must match batch size: "
            f"got {len(values)} values for batch_size={size}"
        )
    return values


def _batch_prompt_hashes(batch, batch_size: int) -> tuple[list[str | None], list[str | None]]:
    extra = getattr(batch, "extra", {})
    prompts = normalize_batched_prompt_fields(
        extra.get("dreamzero_prompts"), batch_size
    )
    neg_prompts = normalize_batched_prompt_fields(
        extra.get("dreamzero_negative_prompts"), batch_size
    )
    inputs = getattr(batch, "dreamzero_inputs", {})
    prompt_hashes = [
        _prompt_hash(prompt, inputs.get("text"), index)
        for index, prompt in enumerate(prompts)
    ]
    neg_prompt_hashes = [
        _prompt_hash(neg_prompt, inputs.get("text_negative"), index)
        for index, neg_prompt in enumerate(neg_prompts)
    ]
    return prompt_hashes, neg_prompt_hashes


def resolve_request_cache(
    batch,
    cache_manager: DreamZeroCachePoolManager | None,
    *,
    local_attn_size: int,
    batch_size: int,
) -> DreamZeroRequestCache:
    if cache_manager is None:
        raise RuntimeError("DreamZero session cache requires a cache manager")
    logical_session_ids, reset_mask = _logical_session_fields(batch, batch_size)
    prompt_hashes, neg_prompt_hashes = _batch_prompt_hashes(batch, batch_size)
    start_time = time.perf_counter()
    persistent = True

    state = cache_manager.pool
    state.local_attn_size = local_attn_size
    slot_indices: list[int] = []
    cache_hit: list[bool] = []
    prompt_reusable: list[bool] = []
    neg_prompt_reusable: list[bool] = []

    for index, session_id in enumerate(logical_session_ids):
        slot, hit = cache_manager.lookup_or_allocate(
            session_id,
            local_attn_size=local_attn_size,
        )
        slot_indices.append(slot)
        cache_hit.append(bool(hit and not reset_mask[index]))
        prompt_reusable.append(
            bool(hit and not reset_mask[index] and state.prompt_reusable(
                BRANCH_COND, slot, prompt_hashes[index]
            ))
        )
        neg_prompt_reusable.append(
            bool(hit and not reset_mask[index] and state.prompt_reusable(
                BRANCH_UNCOND, slot, neg_prompt_hashes[index]
            ))
        )

    request_cache = DreamZeroRequestCache(
        cache_id="slot_pool" if persistent else None,
        logical_session_ids=logical_session_ids,
        slot_indices=slot_indices,
        reset_mask=reset_mask,
        cache_hit=cache_hit,
        prompt_hashes=prompt_hashes,
        neg_prompt_hashes=neg_prompt_hashes,
        prompt_reusable=prompt_reusable,
        neg_prompt_reusable=neg_prompt_reusable,
        persistent=persistent,
    )
    batch.dreamzero_cache = request_cache
    batch.dreamzero_cache_id = request_cache.cache_id
    batch.dreamzero_session_ids = logical_session_ids
    batch.dreamzero_session_slots = slot_indices
    batch.dreamzero_reset_mask = reset_mask
    batch.dreamzero_session_cache_hit = cache_hit
    batch.dreamzero_prompt_reusable = prompt_reusable
    batch.dreamzero_neg_prompt_reusable = neg_prompt_reusable
    batch.dreamzero_session_persistent = persistent
    if getattr(batch, "dreamzero_reset_apply_token", None) is None:
        batch.dreamzero_reset_apply_token = uuid.uuid4().hex
    record_session_timing(
        batch,
        "session_gather_ms",
        (time.perf_counter() - start_time) * 1000,
    )
    return request_cache


def apply_request_lifecycle_resets(
    batch,
    cache_manager: DreamZeroCachePoolManager | None,
    request_cache: DreamZeroRequestCache,
) -> None:
    """Apply logical request/lifecycle resets to this rank's physical pool.

    Reset decisions may be produced on rank 0 and carried by a broadcasted
    batch. Applied markers therefore live on the rank-local cache manager, not
    on the batch.
    """
    if cache_manager is None:
        raise RuntimeError("DreamZero session cache requires a cache manager")
    size = request_cache.batch_size
    lifecycle_mask = _normalize_optional_bool_list(
        getattr(batch, "dreamzero_lifecycle_reset_mask", None),
        size,
        field_name="dreamzero_lifecycle_reset_mask",
        default=False,
    )
    lifecycle_preserve_text = _normalize_optional_bool_list(
        getattr(batch, "dreamzero_lifecycle_reset_preserve_text", None),
        size,
        field_name="dreamzero_lifecycle_reset_preserve_text",
        default=True,
    )
    request_reasons = _normalize_optional_reason_list(
        getattr(batch, "dreamzero_session_reset_reason", None),
        size,
    )
    token = getattr(batch, "dreamzero_reset_apply_token", None)
    if token is None:
        token = uuid.uuid4().hex
        batch.dreamzero_reset_apply_token = token
    if cache_manager._active_reset_apply_token != token:
        cache_manager._active_reset_apply_token = token
        cache_manager._applied_reset_tokens.clear()

    pool = request_cache.pool(cache_manager)
    for index, slot in enumerate(request_cache.slot_indices):
        request_reset = bool(request_cache.reset_mask[index])
        lifecycle_reset = bool(lifecycle_mask[index])
        if not request_reset and not lifecycle_reset:
            continue
        preserve_text = bool(lifecycle_preserve_text[index]) and not request_reset
        key = (f"{token}:{index}", int(slot))
        if key in cache_manager._applied_reset_tokens:
            continue
        pool.reset_slot(slot, preserve_text=preserve_text)
        cache_manager._applied_reset_tokens.add(key)
        if request_reset and request_reasons[index] is None:
            request_reasons[index] = "request_reset"

    batch.dreamzero_session_reset_reason = request_reasons


def enter_request_cache(
    batch,
    cache_manager: DreamZeroCachePoolManager | None,
    *,
    local_attn_size: int,
    batch_size: int,
) -> tuple[DreamZeroRequestCache, DreamZeroCachePool]:
    request_cache = resolve_request_cache(
        batch,
        cache_manager,
        local_attn_size=local_attn_size,
        batch_size=batch_size,
    )
    apply_request_lifecycle_resets(batch, cache_manager, request_cache)
    return request_cache, request_cache.pool(cache_manager)


def session_metadata_from_batch(batch) -> list[dict[str, Any]]:
    request_cache = getattr(batch, "dreamzero_cache", None)
    if isinstance(request_cache, DreamZeroRequestCache):
        frame_value = getattr(batch, "dreamzero_current_start_frame", None)
        if isinstance(frame_value, list):
            frames = [int(value) for value in frame_value]
        elif frame_value is None:
            frames = [None] * request_cache.batch_size
        else:
            frames = [int(frame_value)] * request_cache.batch_size
        return [
            {
                "session_id": session_id,
                "cache_id": request_cache.cache_id,
                "slot": int(slot),
                "cache_hit": bool(cache_hit),
                "prompt_reusable": bool(prompt_reusable),
                "neg_prompt_reusable": bool(neg_prompt_reusable),
                "current_start_frame": None if frame is None else int(frame),
            }
            for (
                session_id,
                slot,
                cache_hit,
                prompt_reusable,
                neg_prompt_reusable,
                frame,
            ) in zip(
                request_cache.logical_session_ids,
                request_cache.slot_indices,
                request_cache.cache_hit,
                request_cache.prompt_reusable,
                request_cache.neg_prompt_reusable,
                frames,
                strict=True,
            )
        ]
    raise ValueError("DreamZero session metadata requires resolved request cache")


def _prompt_hash(prompt: str | None, tensor: Any, index: int) -> str | None:
    if prompt is not None:
        return "str:" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    if torch.is_tensor(tensor):
        row = tensor[index].detach().cpu().contiguous()
        return "tensor:" + hashlib.sha1(row.numpy().tobytes()).hexdigest()
    return None


def _slot_tensor(slots: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(slots, dtype=torch.long, device=device)


def _pool_capacity_for_slots(
    existing: torch.Tensor | None,
    slots: list[int],
    pool_capacity: int,
) -> int:
    required = max(slots) + 1 if slots else 0
    if required > pool_capacity:
        raise IndexError(
            "DreamZero cache pool slot index exceeds capacity: "
            f"required={required}, capacity={pool_capacity}"
        )
    if existing is not None:
        required = max(required, int(existing.shape[0]))
    return _bounded_geometric_capacity(required, pool_capacity)


def _kv_pool_capacity_for_slots(
    existing: torch.Tensor | None,
    slots: list[int],
    pool_capacity: int,
) -> int:
    required = max(slots) + 1 if slots else 0
    if required > pool_capacity:
        raise IndexError(
            "DreamZero KV cache pool slot index exceeds capacity: "
            f"required={required}, capacity={pool_capacity}"
        )
    if existing is not None:
        required = max(required, int(existing.shape[1]))
    return _bounded_geometric_capacity(required, pool_capacity)


def _bounded_geometric_capacity(required: int, pool_capacity: int) -> int:
    if required <= 0:
        return 0
    capacity = 1 << (int(required) - 1).bit_length()
    return min(max(capacity, required), int(pool_capacity))


def _kv_seq_capacity(
    existing: torch.Tensor | None,
    values: torch.Tensor,
    seq_capacity: int,
) -> int:
    required = int(values.shape[2])
    if seq_capacity > 0:
        required = max(required, int(seq_capacity))
    if existing is not None:
        required = max(required, int(existing.shape[2]))
    return required


def _ensure_mutable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "is_inference") and torch.is_inference(tensor):
        return tensor.clone()
    return tensor


def _scatter_dim0_pool(
    existing: torch.Tensor | None,
    slots: list[int],
    values: torch.Tensor,
    *,
    pool_capacity: int,
) -> torch.Tensor:
    if existing is not None:
        existing = _ensure_mutable_tensor(existing)
    capacity = _pool_capacity_for_slots(existing, slots, pool_capacity)
    if (
        existing is None
        or existing.shape[0] < capacity
        or existing.shape[1:] != values.shape[1:]
    ):
        pool = values.new_zeros((capacity, *values.shape[1:]))
        if existing is not None and existing.shape[1:] == values.shape[1:]:
            pool[: existing.shape[0]] = existing
        existing = pool
    existing.index_copy_(0, _slot_tensor(slots, values.device), values.detach())
    return existing


def _scatter_kv_pool(
    existing: torch.Tensor | None,
    slots: list[int],
    values: torch.Tensor,
    *,
    pool_capacity: int,
    seq_capacity: int,
) -> torch.Tensor:
    if existing is not None:
        existing = _ensure_mutable_tensor(existing)
    capacity = _kv_pool_capacity_for_slots(existing, slots, pool_capacity)
    seq_len = _kv_seq_capacity(existing, values, seq_capacity)
    target_shape = (values.shape[0], capacity, seq_len, *values.shape[3:])
    if existing is None:
        pool = values.new_zeros(target_shape)
    elif (
        existing.shape[0] != values.shape[0]
        or existing.shape[1] != capacity
        or existing.shape[3:] != values.shape[3:]
        or existing.shape[2] < seq_len
    ):
        pool = values.new_zeros(
            (values.shape[0], capacity, seq_len, *values.shape[3:])
        )
        seq_copy = min(existing.shape[2], seq_len)
        slot_copy = min(existing.shape[1], capacity)
        pool[:, :slot_copy, :seq_copy] = existing[:, :slot_copy, :seq_copy]
    else:
        pool = existing
    pool[:, _slot_tensor(slots, values.device), : values.shape[2]] = values.detach()
    return pool
