from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch


class HostSharedPageSlotState(str, Enum):
    """String-valued enum compatible with Python 3.10 (StrEnum is 3.11+)."""

    SLOT_FREE = "slot_free"
    SLOT_RESERVED = "slot_reserved"
    GET_IN_FLIGHT = "get_in_flight"
    SLOT_RESIDENT = "slot_resident"
    PUT_IN_FLIGHT = "put_in_flight"
    SLOT_INVALID = "slot_invalid"
    SLOT_RETIRING = "slot_retiring"


class HostSharedPageSlotError(RuntimeError):
    pass


class HostSharedPageSlotStateError(HostSharedPageSlotError):
    pass


class HostSharedPageSlotStaleTokenError(HostSharedPageSlotError):
    pass


@dataclass(frozen=True, slots=True)
class HostSharedPageSlotToken:
    slot_index: int
    slot_generation: int


@dataclass(frozen=True, slots=True)
class HostSharedPageSlotSnapshot:
    slot_index: int
    slot_generation: int
    page_start: int
    state: HostSharedPageSlotState
    pin_count: int
    logical_key: str | None


class HostSharedPageSlotTracker:
    def __init__(self, page_size: int, page_num: int) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        if page_num <= 0:
            raise ValueError(f"page_num must be positive, got {page_num}")
        self.page_size = page_size
        self.page_num = page_num
        self.reset()

    def reset(self) -> None:
        self._states = [HostSharedPageSlotState.SLOT_FREE for _ in range(self.page_num)]
        self._generations = [0 for _ in range(self.page_num)]
        self._pin_counts = [0 for _ in range(self.page_num)]
        self._logical_keys: list[str | None] = [None for _ in range(self.page_num)]

    def page_start_for_slot_index(self, slot_index: int) -> int:
        self._validate_slot_index(slot_index)
        return slot_index * self.page_size

    def slot_index_for_page_start(self, page_start: int) -> int:
        if page_start < 0:
            raise ValueError(f"page_start must be non-negative, got {page_start}")
        if page_start % self.page_size != 0:
            raise ValueError(
                f"page_start must align to page_size={self.page_size}, got {page_start}"
            )
        slot_index = page_start // self.page_size
        self._validate_slot_index(slot_index)
        return slot_index

    def snapshot(self, slot_index: int) -> HostSharedPageSlotSnapshot:
        self._validate_slot_index(slot_index)
        return HostSharedPageSlotSnapshot(
            slot_index=slot_index,
            slot_generation=self._generations[slot_index],
            page_start=self.page_start_for_slot_index(slot_index),
            state=self._states[slot_index],
            pin_count=self._pin_counts[slot_index],
            logical_key=self._logical_keys[slot_index],
        )

    def current_token(self, slot_index: int) -> HostSharedPageSlotToken:
        self._validate_slot_index(slot_index)
        return HostSharedPageSlotToken(
            slot_index=slot_index,
            slot_generation=self._generations[slot_index],
        )

    def reserve_slots(
        self,
        slot_indices: Sequence[int],
        logical_keys: Sequence[str] | None = None,
    ) -> tuple[HostSharedPageSlotToken, ...]:
        normalized_indices = self._normalize_slot_indices(slot_indices)
        self._validate_optional_logical_keys(normalized_indices, logical_keys)
        self._ensure_states(
            normalized_indices,
            allowed_states={HostSharedPageSlotState.SLOT_FREE},
            operation="reserve_slots",
        )
        for offset, slot_index in enumerate(normalized_indices):
            self._states[slot_index] = HostSharedPageSlotState.SLOT_RESERVED
            self._pin_counts[slot_index] = 1
            if logical_keys is not None:
                self._logical_keys[slot_index] = logical_keys[offset]
            else:
                self._logical_keys[slot_index] = None
        return tuple(
            self.current_token(slot_index) for slot_index in normalized_indices
        )

    def mark_get_inflight(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._ensure_states(
            normalized_indices,
            allowed_states={HostSharedPageSlotState.SLOT_RESERVED},
            operation="mark_get_inflight",
        )
        for slot_index in normalized_indices:
            self._states[slot_index] = HostSharedPageSlotState.GET_IN_FLIGHT

    def commit_get_success(
        self,
        slot_tokens: Sequence[HostSharedPageSlotToken],
        logical_keys: Sequence[str] | None = None,
    ) -> None:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._validate_optional_logical_keys(normalized_indices, logical_keys)
        self._ensure_states(
            normalized_indices,
            allowed_states={
                HostSharedPageSlotState.SLOT_RESERVED,
                HostSharedPageSlotState.GET_IN_FLIGHT,
            },
            operation="commit_get_success",
        )
        for offset, slot_index in enumerate(normalized_indices):
            self._states[slot_index] = HostSharedPageSlotState.SLOT_RESIDENT
            self._pin_counts[slot_index] = max(0, self._pin_counts[slot_index] - 1)
            if logical_keys is not None:
                self._logical_keys[slot_index] = logical_keys[offset]

    def fail_get(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._ensure_states(
            normalized_indices,
            allowed_states={
                HostSharedPageSlotState.SLOT_RESERVED,
                HostSharedPageSlotState.GET_IN_FLIGHT,
            },
            operation="fail_get",
        )
        for slot_index in normalized_indices:
            self._states[slot_index] = HostSharedPageSlotState.SLOT_INVALID
            self._pin_counts[slot_index] = max(0, self._pin_counts[slot_index] - 1)

    def begin_put(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._ensure_states(
            normalized_indices,
            allowed_states={HostSharedPageSlotState.SLOT_RESIDENT},
            operation="begin_put",
        )
        for slot_index in normalized_indices:
            self._states[slot_index] = HostSharedPageSlotState.PUT_IN_FLIGHT
            self._pin_counts[slot_index] += 1

    def finish_put(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._ensure_states(
            normalized_indices,
            allowed_states={HostSharedPageSlotState.PUT_IN_FLIGHT},
            operation="finish_put",
        )
        for slot_index in normalized_indices:
            if self._pin_counts[slot_index] <= 0:
                raise HostSharedPageSlotStateError(
                    f"finish_put requires positive pin_count for slot {slot_index}"
                )
            self._pin_counts[slot_index] -= 1
            self._states[slot_index] = HostSharedPageSlotState.SLOT_RESIDENT

    def mark_invalid(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._ensure_states(
            normalized_indices,
            allowed_states={
                HostSharedPageSlotState.SLOT_RESERVED,
                HostSharedPageSlotState.GET_IN_FLIGHT,
                HostSharedPageSlotState.SLOT_RESIDENT,
            },
            operation="mark_invalid",
        )
        for slot_index in normalized_indices:
            if self._states[slot_index] in {
                HostSharedPageSlotState.SLOT_RESERVED,
                HostSharedPageSlotState.GET_IN_FLIGHT,
            }:
                self._pin_counts[slot_index] = max(0, self._pin_counts[slot_index] - 1)
            self._states[slot_index] = HostSharedPageSlotState.SLOT_INVALID

    def begin_retire(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._ensure_states(
            normalized_indices,
            allowed_states={
                HostSharedPageSlotState.SLOT_RESIDENT,
                HostSharedPageSlotState.SLOT_INVALID,
            },
            operation="begin_retire",
        )
        for slot_index in normalized_indices:
            if self._pin_counts[slot_index] != 0:
                raise HostSharedPageSlotStateError(
                    f"slot {slot_index} cannot retire while pin_count={self._pin_counts[slot_index]}"
                )
            self._states[slot_index] = HostSharedPageSlotState.SLOT_RETIRING

    def finish_retire(
        self, slot_tokens: Sequence[HostSharedPageSlotToken]
    ) -> tuple[HostSharedPageSlotToken, ...]:
        normalized_indices = self._resolve_slot_tokens(slot_tokens)
        self._ensure_states(
            normalized_indices,
            allowed_states={HostSharedPageSlotState.SLOT_RETIRING},
            operation="finish_retire",
        )
        for slot_index in normalized_indices:
            self._states[slot_index] = HostSharedPageSlotState.SLOT_FREE
            self._generations[slot_index] += 1
            self._pin_counts[slot_index] = 0
            self._logical_keys[slot_index] = None
        return tuple(
            self.current_token(slot_index) for slot_index in normalized_indices
        )

    def retire_slots(
        self, slot_tokens: Sequence[HostSharedPageSlotToken]
    ) -> tuple[HostSharedPageSlotToken, ...]:
        self.begin_retire(slot_tokens)
        return self.finish_retire(slot_tokens)

    def _normalize_slot_indices(self, slot_indices: Sequence[int]) -> tuple[int, ...]:
        normalized_indices = tuple(int(slot_index) for slot_index in slot_indices)
        if not normalized_indices:
            raise ValueError("slot_indices must not be empty")
        for slot_index in normalized_indices:
            self._validate_slot_index(slot_index)
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("slot_indices must not contain duplicates")
        return normalized_indices

    def _resolve_slot_tokens(
        self, slot_tokens: Sequence[HostSharedPageSlotToken]
    ) -> tuple[int, ...]:
        normalized_indices = self._normalize_slot_indices(
            [slot_token.slot_index for slot_token in slot_tokens]
        )
        token_by_slot = {
            slot_token.slot_index: slot_token for slot_token in slot_tokens
        }
        for slot_index in normalized_indices:
            slot_token = token_by_slot[slot_index]
            current_generation = self._generations[slot_index]
            if slot_token.slot_generation != current_generation:
                raise HostSharedPageSlotStaleTokenError(
                    "stale slot token for slot "
                    f"{slot_index}: expected generation {current_generation}, got {slot_token.slot_generation}"
                )
        return normalized_indices

    def _ensure_states(
        self,
        slot_indices: Sequence[int],
        allowed_states: set[HostSharedPageSlotState],
        operation: str,
    ) -> None:
        for slot_index in slot_indices:
            current_state = self._states[slot_index]
            if current_state not in allowed_states:
                allowed = ", ".join(state.value for state in sorted(allowed_states))
                raise HostSharedPageSlotStateError(
                    f"{operation} requires slot {slot_index} to be in {{{allowed}}}, got {current_state.value}"
                )

    def _validate_optional_logical_keys(
        self,
        slot_indices: Sequence[int],
        logical_keys: Sequence[str] | None,
    ) -> None:
        if logical_keys is None:
            return
        if len(logical_keys) != len(slot_indices):
            raise ValueError(
                f"logical_keys length {len(logical_keys)} does not match slot count {len(slot_indices)}"
            )

    def _validate_slot_index(self, slot_index: int) -> None:
        if slot_index < 0 or slot_index >= self.page_num:
            raise ValueError(
                f"slot_index must be in [0, {self.page_num}), got {slot_index}"
            )


class HostSharedPageSlotManager:
    """Tensorcast-owned lifecycle manager for direct host-shared page slots."""

    def __init__(self, page_size: int, page_num: int) -> None:
        self.page_size = page_size
        self.page_num = page_num
        self._lock = threading.RLock()
        self.tracker = HostSharedPageSlotTracker(
            page_size=page_size,
            page_num=page_num,
        )

    def reset(self) -> None:
        with self._lock:
            self.tracker.reset()

    def retire_released_page_slots(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0 or indices.numel() % self.page_size != 0:
            return
        with self._lock:
            retire_tokens: list[HostSharedPageSlotToken] = []
            for offset in range(0, indices.numel(), self.page_size):
                page_start = int(indices[offset].item())
                slot_index = self.tracker.slot_index_for_page_start(page_start)
                snapshot = self.tracker.snapshot(slot_index)
                if snapshot.state == HostSharedPageSlotState.SLOT_FREE:
                    continue
                if snapshot.state in {
                    HostSharedPageSlotState.SLOT_RESIDENT,
                    HostSharedPageSlotState.SLOT_INVALID,
                }:
                    retire_tokens.append(self.tracker.current_token(slot_index))
                    continue
                raise HostSharedPageSlotStateError(
                    "cannot free host page slots while they are reserved or in flight"
                )
            if retire_tokens:
                self.tracker.retire_slots(retire_tokens)

    def slot_tokens_for_page_starts(
        self, page_starts: torch.Tensor | Sequence[int]
    ) -> tuple[HostSharedPageSlotToken, ...]:
        with self._lock:
            return tuple(
                self.tracker.current_token(slot_index)
                for slot_index in self._slot_indices_from_page_starts(page_starts)
            )

    def describe_page_slot(self, page_start: int) -> HostSharedPageSlotSnapshot:
        with self._lock:
            slot_index = self.tracker.slot_index_for_page_start(int(page_start))
            return self.tracker.snapshot(slot_index)

    def reserve_page_slots(
        self,
        page_starts: torch.Tensor | Sequence[int],
        logical_keys: Sequence[str] | None = None,
    ) -> tuple[HostSharedPageSlotToken, ...]:
        with self._lock:
            slot_indices = self._slot_indices_from_page_starts(page_starts)
            return self.tracker.reserve_slots(slot_indices, logical_keys)

    def mark_page_get_inflight(
        self, slot_tokens: Sequence[HostSharedPageSlotToken]
    ) -> None:
        with self._lock:
            self.tracker.mark_get_inflight(slot_tokens)

    def commit_page_get_success(
        self,
        slot_tokens: Sequence[HostSharedPageSlotToken],
        logical_keys: Sequence[str] | None = None,
    ) -> None:
        with self._lock:
            self.tracker.commit_get_success(slot_tokens, logical_keys)

    def fail_page_get(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        with self._lock:
            self.tracker.fail_get(slot_tokens)

    def begin_page_put(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        with self._lock:
            self.tracker.begin_put(slot_tokens)

    def finish_page_put(self, slot_tokens: Sequence[HostSharedPageSlotToken]) -> None:
        with self._lock:
            self.tracker.finish_put(slot_tokens)

    def mark_page_slots_invalid(
        self, slot_tokens: Sequence[HostSharedPageSlotToken]
    ) -> None:
        with self._lock:
            self.tracker.mark_invalid(slot_tokens)

    def retire_page_slots(
        self, slot_tokens: Sequence[HostSharedPageSlotToken]
    ) -> tuple[HostSharedPageSlotToken, ...]:
        with self._lock:
            return self.tracker.retire_slots(slot_tokens)

    def _slot_indices_from_page_starts(
        self, page_starts: torch.Tensor | Sequence[int]
    ) -> tuple[int, ...]:
        if isinstance(page_starts, torch.Tensor):
            normalized_page_starts = tuple(int(value) for value in page_starts.tolist())
        else:
            normalized_page_starts = tuple(int(value) for value in page_starts)
        return tuple(
            self.tracker.slot_index_for_page_start(page_start)
            for page_start in normalized_page_starts
        )
