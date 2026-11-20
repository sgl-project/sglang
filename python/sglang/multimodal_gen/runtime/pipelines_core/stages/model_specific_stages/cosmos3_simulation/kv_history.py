# SPDX-License-Identifier: Apache-2.0
"""Committed K/V of a causal Cosmos-Dreams rollout, kept in place.

Every GEN layer attends to one contiguous ``[text | committed history | current
chunk]`` block. The buffers here hold that block per layer for the whole
rollout: a forward writes its chunk right behind the committed history and
reads the block as a view, and a commit moves a length instead of copying the
history. A layer's keys and values share one allocation, so the kernel that
norms and rotates the keys can write both in one launch.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
    CosmosDreamsKVSlot,
    KVPair,
)

# History room an open-ended session starts with; the buffers double on demand.
# Arbitrary: small enough that a short session stays cheap.
OPEN_ENDED_INITIAL_FRAMES = 16
# Most spare frames a sliding history drifts through before it moves back to the
# front. Arbitrary.
MAX_SPARE_FRAMES = 16


class CosmosDreamsKVHistory:
    """Per-layer ``[text | committed history | current chunk]`` K/V buffers.

    The kept tail stays where it is and ``[text | sink]`` is rewritten just in
    front of it, so the block attention reads drifts forward through the spare
    room and moves back to the front when that runs out.
    """

    def __init__(
        self,
        text_kv: Sequence[KVPair],
        *,
        sink_tokens: int,
        live_tokens: int,
        spare_tokens: int = 0,
        initial_tokens: int | None = None,
    ) -> None:
        """``live_tokens`` bounds committed history plus one chunk."""
        if not text_kv:
            raise ValueError("Cosmos-Dreams K/V history needs at least one layer.")
        if sink_tokens < 0 or live_tokens <= 0 or spare_tokens < 0:
            raise ValueError(
                "Cosmos-Dreams K/V history needs sink_tokens >= 0, live_tokens > 0 and "
                f"spare_tokens >= 0, got {sink_tokens}, {live_tokens}, {spare_tokens}."
            )
        self.text_tokens = int(text_kv[0][0].shape[1])
        self.sink_tokens = int(sink_tokens)
        self.committed_tokens = 0
        self._offset = 0
        self._max_capacity = self.text_tokens + live_tokens + spare_tokens
        self._capacity = (
            self._max_capacity
            if initial_tokens is None
            else min(self._max_capacity, self.text_tokens + max(1, initial_tokens))
        )
        # [text | sink] of every layer, copied aside at the first eviction.
        self._pinned: list[torch.Tensor] | None = None
        # Per layer [2, 1, capacity, H_kv, D]: keys at 0, values at 1.
        self._kv: list[torch.Tensor] = []
        for key, value in text_kv:
            if (
                key.shape != value.shape
                or key.dtype != value.dtype
                or key.ndim != 4
                or key.shape[0] != 1
                or int(key.shape[1]) != self.text_tokens
            ):
                raise ValueError(
                    "Cosmos-Dreams text K/V must be matching [1, S, H_kv, D] tensors of one "
                    f"length, got {tuple(key.shape)} and {tuple(value.shape)}."
                )
            buffer = key.new_empty((2, 1, self._capacity, *key.shape[2:]))
            buffer[0, :, : self.text_tokens].copy_(key)
            buffer[1, :, : self.text_tokens].copy_(value)
            self._kv.append(buffer)

    def committed_kv(self, layer_idx: int) -> torch.Tensor:
        start = self._offset + self.text_tokens
        return self._kv[layer_idx][:, :, start : start + self.committed_tokens]

    def committed(self, layer_idx: int) -> KVPair:
        keys, values = self.committed_kv(layer_idx)
        return keys, values

    def clear(self) -> None:
        if self._offset:
            raise ValueError(
                "A Cosmos-Dreams K/V history that evicted cannot be cleared."
            )
        self.committed_tokens = 0

    def slots(self, chunk_tokens: int) -> list[CosmosDreamsKVSlot]:
        if chunk_tokens <= 0:
            raise ValueError(
                f"Cosmos-Dreams K/V chunk must be positive, got {chunk_tokens}."
            )
        self._make_room(chunk_tokens)
        start = self._offset
        chunk_start = start + self.text_tokens + self.committed_tokens
        end = chunk_start + chunk_tokens
        slots = []
        for kv in self._kv:
            block = kv[:, :, start:end]
            chunk = kv[:, :, chunk_start:end]
            slots.append(
                CosmosDreamsKVSlot(
                    keys=block[0],
                    values=block[1],
                    chunk_keys=chunk[0],
                    chunk_values=chunk[1],
                    chunk_kv=chunk,
                )
            )
        return slots

    def commit(self, chunk_tokens: int, *, tail_tokens: int | None = None) -> None:
        """Keep the chunk the last forward wrote. ``tail_tokens`` bounds what stays
        behind the sink, and None keeps everything."""
        total = self.committed_tokens + chunk_tokens
        if (
            chunk_tokens <= 0
            or self._offset + self.text_tokens + total > self._capacity
        ):
            raise ValueError(
                f"Cosmos-Dreams K/V history cannot commit {chunk_tokens} tokens after "
                f"{self.committed_tokens}; ask for the slots of the chunk first."
            )
        kept = total if tail_tokens is None else self.sink_tokens + tail_tokens
        if total <= kept:
            self.committed_tokens = total
            return
        pinned_tokens = self.text_tokens + self.sink_tokens
        if self._pinned is None:
            start = self._offset
            self._pinned = [
                kv[:, :, start : start + pinned_tokens].clone() for kv in self._kv
            ]
        # The kept tail ends where the history ended, so [text | sink] moves up
        # to it and overwrites only its old copy and the evicted tokens.
        self._offset += total - kept
        start = self._offset
        for pinned, kv in zip(self._pinned, self._kv, strict=True):
            kv[:, :, start : start + pinned_tokens].copy_(pinned)
        self.committed_tokens = kept

    def _make_room(self, chunk_tokens: int) -> None:
        live = self.text_tokens + self.committed_tokens
        needed = live + chunk_tokens
        if self._offset + needed <= self._capacity:
            return
        if needed > self._max_capacity:
            raise ValueError(
                f"Cosmos-Dreams K/V history holds {self.committed_tokens} tokens and "
                f"cannot take {chunk_tokens} more within {self._max_capacity}."
            )
        capacity = self._capacity
        if needed > capacity:
            capacity = min(max(needed, 2 * capacity), self._max_capacity)
        start = self._offset
        for layer_idx, buffer in enumerate(self._kv):
            source = buffer[:, :, start : start + live]
            if capacity != self._capacity:
                target = buffer.new_empty((2, 1, capacity, *buffer.shape[3:]))
                self._kv[layer_idx] = target
            else:
                target = buffer
                if start < live:
                    # Source and destination overlap.
                    source = source.clone()
            target[:, :, :live].copy_(source)
        self._offset = 0
        self._capacity = capacity


class SelectedKVHistory:
    """``[text | selected archive frames | current chunk]`` buffers for frustum retrieval.

    ``select`` gathers the chosen frames once per chunk, and ``commit`` also
    appends the chunk to the archive so later chunks can retrieve it.
    """

    def __init__(
        self,
        archive: CosmosDreamsKVHistory,
        text_kv: Sequence[KVPair],
        *,
        selected_tokens: int,
        chunk_tokens: int,
    ) -> None:
        self._archive = archive
        self._read = CosmosDreamsKVHistory(
            text_kv, sink_tokens=0, live_tokens=selected_tokens + chunk_tokens
        )

    def select(
        self, frame_ids: Sequence[int], *, tokens_per_frame: int
    ) -> SelectedKVHistory:
        """The selected frames, and the chunks committed after them, stay readable
        until the next ``select``."""
        stored = self._archive.committed_tokens // tokens_per_frame
        ids = list(frame_ids)
        if ids != sorted(set(ids)) or (ids and (ids[0] < 0 or ids[-1] >= stored)):
            raise ValueError(
                f"Selected frames must be ordered unique committed frames, got {ids} "
                f"with {stored} committed."
            )
        read = self._read
        read.clear()
        if not ids:
            return self
        selected_tokens = len(ids) * tokens_per_frame
        slots = read.slots(selected_tokens)
        positions = torch.tensor(
            [
                frame * tokens_per_frame + token
                for frame in ids
                for token in range(tokens_per_frame)
            ],
            dtype=torch.long,
            device=slots[0].keys.device,
        )
        for layer_idx, slot in enumerate(slots):
            torch.index_select(
                self._archive.committed_kv(layer_idx), 2, positions, out=slot.chunk_kv
            )
        read.commit(selected_tokens)
        return self

    def slots(self, chunk_tokens: int) -> list[CosmosDreamsKVSlot]:
        return self._read.slots(chunk_tokens)

    def commit(self, chunk_tokens: int, *, tail_tokens: int | None = None) -> None:
        written = self._read.slots(chunk_tokens)
        for source, target in zip(
            written, self._archive.slots(chunk_tokens), strict=True
        ):
            target.chunk_kv.copy_(source.chunk_kv)
        self._archive.commit(chunk_tokens, tail_tokens=tail_tokens)
        self._read.commit(chunk_tokens)


def new_kv_history(
    text_kv: Sequence[KVPair],
    *,
    tokens_per_frame: int,
    sink_frames: int,
    window_frames: int | None,
    chunk_frames: int,
    rollout_frames: int | None,
) -> CosmosDreamsKVHistory:
    """``rollout_frames`` is every frame the rollout writes, None for an open-ended
    session. ``window_frames`` None keeps every frame."""
    bound = None if window_frames is None else sink_frames + window_frames
    if rollout_frames is not None and (bound is None or rollout_frames <= bound):
        return CosmosDreamsKVHistory(
            text_kv,
            sink_tokens=sink_frames * tokens_per_frame,
            live_tokens=rollout_frames * tokens_per_frame,
        )
    if bound is None:
        raise ValueError(
            "An open-ended Cosmos-Dreams session needs a bounded K/V history window."
        )
    spare_frames = max(chunk_frames, min(bound // 4, MAX_SPARE_FRAMES))
    return CosmosDreamsKVHistory(
        text_kv,
        sink_tokens=sink_frames * tokens_per_frame,
        live_tokens=(bound + chunk_frames) * tokens_per_frame,
        spare_tokens=spare_frames * tokens_per_frame,
        initial_tokens=(
            None
            if rollout_frames is not None
            else (OPEN_ENDED_INITIAL_FRAMES + chunk_frames) * tokens_per_frame
        ),
    )
