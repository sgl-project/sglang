# SPDX-License-Identifier: Apache-2.0
"""MinWM raw-K cache metadata and window selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)


@dataclass(slots=True)
class MinWMCausalAttentionKVView:
    k: torch.Tensor
    v: torch.Tensor
    query_position_ids: torch.Tensor
    key_position_ids: torch.Tensor


@dataclass(slots=True)
class MinWMCausalSelfAttentionKVCache(CausalSelfAttentionKVCache):
    """One MinWM layer's unrotated K/V and position metadata.

    MinWM commit 4220c8a caches K after RMSNorm but before RoPE. RoPE is
    reconstructed after selecting the visible sink/pin/tail window so that
    ``block_relative`` positions follow the exact visible token order.
    """

    rope_position_mode: str = "absolute"
    rope_max_frame_gap: int = 1
    prompt_first_frame_pin_enabled: bool = False
    scene_cut_rope_offset: int = 0
    scene_cut_sink_enabled: bool = False
    position_ids: torch.Tensor | None = None
    rope_position_ids: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    current_position_ids: torch.Tensor | None = None
    rope_temporal_offset: int = 0
    pinned_token_start: int | None = None
    pinned_token_end: int | None = None
    prompt_pin_frame: int | None = None
    pending_prompt_switch: bool = False
    pending_scene_cut_pin: bool = False

    def __post_init__(self) -> None:
        CausalSelfAttentionKVCache.__post_init__(self)
        if self.rope_position_mode not in {"absolute", "block_relative"}:
            raise ValueError(
                f"unsupported MinWM rope_position_mode={self.rope_position_mode!r}"
            )
        if self.rope_max_frame_gap < 1:
            raise ValueError("MinWM rope_max_frame_gap must be >= 1")

    def reset_indices(self) -> None:
        CausalSelfAttentionKVCache.reset_indices(self)
        self.position_ids = None
        self.rope_position_ids = None
        self.token_ids = None
        self.current_position_ids = None
        self.rope_temporal_offset = 0
        self.pinned_token_start = None
        self.pinned_token_end = None
        self.prompt_pin_frame = None
        self.pending_prompt_switch = False
        self.pending_scene_cut_pin = False

    def can_direct_current_attention(self, num_new_tokens: int) -> bool:
        del num_new_tokens
        # Even a one-block request must initialize raw K and position metadata.
        return False

    def set_current_position_ids(self, position_ids: torch.Tensor) -> None:
        if position_ids.ndim != 2 or position_ids.shape[1] != 3:
            raise ValueError("MinWM position_ids must have shape [tokens, 3]")
        self.current_position_ids = position_ids

    def mark_prompt_switch(self) -> None:
        self.prompt_pin_frame = None
        self.pending_prompt_switch = self.prompt_first_frame_pin_enabled

    def mark_scene_cut(self) -> None:
        self.prompt_pin_frame = None
        self.pending_prompt_switch = False
        if self.rope_position_mode == "block_relative" and self.scene_cut_rope_offset:
            raise ValueError(
                "MinWM block_relative RoPE does not support nonzero scene-cut offset"
            )
        self.rope_temporal_offset += int(self.scene_cut_rope_offset)
        self.pending_scene_cut_pin = self.scene_cut_sink_enabled

    @staticmethod
    def _cache_head_slice(
        cache: torch.Tensor,
        value: torch.Tensor,
        cache_head_start: int | None,
        debug_name: str,
    ) -> slice | None:
        if cache.shape[2] == value.shape[2]:
            return None
        if cache_head_start is None:
            raise ValueError(
                f"{debug_name} requires cache_head_start when cache heads "
                f"({cache.shape[2]}) differ from input heads ({value.shape[2]})"
            )
        cache_head_end = cache_head_start + value.shape[2]
        if cache_head_start < 0 or cache_head_end > cache.shape[2]:
            raise ValueError(f"{debug_name} cache head slice is out of bounds")
        return slice(cache_head_start, cache_head_end)

    @staticmethod
    def _head_view(tensor: torch.Tensor, head_slice: slice | None) -> torch.Tensor:
        if head_slice is None:
            return tensor
        return tensor[:, :, head_slice, :]

    def _effective_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        effective = position_ids.clone()
        effective[:, 0] += int(self.rope_temporal_offset)
        return effective

    def _pinned_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.pinned_token_start is None or self.pinned_token_end is None:
            return token_ids.new_empty((0,))
        return torch.nonzero(
            (token_ids >= self.pinned_token_start)
            & (token_ids < self.pinned_token_end),
            as_tuple=False,
        ).flatten()

    def _window_indices(
        self,
        token_ids: torch.Tensor,
        max_tokens: int,
    ) -> tuple[torch.Tensor, int]:
        """Return chronological sink / dynamic-pin / tail indices."""
        total_tokens = int(token_ids.numel())
        device = token_ids.device
        if total_tokens <= max_tokens:
            return torch.arange(total_tokens, device=device), 0

        sink_tokens = min(int(self.sink_tokens), total_tokens)
        if sink_tokens >= max_tokens:
            raise ValueError("MinWM sink_size must be smaller than local_attn_size")

        pinned = self._pinned_indices(token_ids)
        pin_start = int(pinned[0].item()) if pinned.numel() else -1
        pin_len = int(pinned.numel())
        pin_is_sink_suffix = pin_start == sink_tokens
        protected_sink_tokens = sink_tokens + pin_len if pin_is_sink_suffix else sink_tokens
        tail_start_without_pinned = max(
            protected_sink_tokens,
            total_tokens - (max_tokens - protected_sink_tokens),
        )
        include_pinned = (
            pin_len > 0
            and not pin_is_sink_suffix
            and pin_start >= protected_sink_tokens
            and pin_start < tail_start_without_pinned
        )
        extra_pinned_tokens = pin_len if include_pinned else 0
        tail_tokens = max_tokens - protected_sink_tokens - extra_pinned_tokens
        if tail_tokens <= 0:
            raise ValueError("MinWM dynamic pin must be smaller than local_attn_size")

        pieces = []
        if protected_sink_tokens:
            pieces.append(torch.arange(protected_sink_tokens, device=device))
        if include_pinned:
            pieces.append(pinned)
        tail_start = max(protected_sink_tokens, total_tokens - tail_tokens)
        if tail_start < total_tokens:
            pieces.append(torch.arange(tail_start, total_tokens, device=device))
        return torch.cat(pieces), tail_start

    def _select_visible_indices(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        max_tokens = int(self.attention_window_size)
        indices, _ = self._window_indices(token_ids, max_tokens)
        if self.prompt_pin_frame is None:
            return indices

        marked = torch.nonzero(
            position_ids[:, 0] == self.prompt_pin_frame, as_tuple=False
        ).flatten()
        if marked.numel() == 0:
            self.prompt_pin_frame = None
            return indices
        selected_marked = torch.isin(marked, indices)
        if bool(selected_marked.all()):
            return indices

        self.pinned_token_start = int(token_ids[marked[0]].item())
        self.pinned_token_end = int(token_ids[marked[-1]].item()) + 1
        self.prompt_pin_frame = None
        indices, _ = self._window_indices(token_ids, max_tokens)
        return indices

    def _attention_position_ids(
        self,
        *,
        num_query_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position_ids is None or self.rope_position_ids is None:
            raise RuntimeError("MinWM cache position metadata is not initialized")
        if self.rope_position_mode == "absolute":
            key_position_ids = self.rope_position_ids
        else:
            key_position_ids = self.position_ids.clone()
            frame_position, frame_index = torch.unique_consecutive(
                key_position_ids[:, 0], return_inverse=True
            )
            frame_gap = torch.cat(
                [
                    frame_position.new_zeros(1),
                    frame_position[1:] - frame_position[:-1],
                ]
            )
            compressed = torch.cumsum(
                frame_gap.clamp(max=self.rope_max_frame_gap), dim=0
            )
            key_position_ids[:, 0] = compressed[frame_index]
        return key_position_ids[-num_query_tokens:], key_position_ids

    def _build_view(
        self,
        *,
        num_query_tokens: int,
        head_slice: slice | None,
    ) -> MinWMCausalAttentionKVView:
        _, local_end = self._read_indices()
        query_position_ids, key_position_ids = self._attention_position_ids(
            num_query_tokens=num_query_tokens
        )
        return MinWMCausalAttentionKVView(
            k=self._head_view(self.k[:, :local_end], head_slice),
            v=self._head_view(self.v[:, :local_end], head_slice),
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
        )

    def update_and_get_attention_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        current_chunk_start: int,
        cache_head_start: int | None = None,
        recent_window_tokens: int | None = None,
        debug_name: str = "MinWM causal KV cache",
        position_ids: torch.Tensor | None = None,
    ) -> MinWMCausalAttentionKVView:
        if recent_window_tokens is not None:
            raise ValueError("MinWM position-aware cache does not use recent_window_tokens")
        if key.shape != value.shape:
            raise ValueError("MinWM attention key/value shapes must match")
        position_ids = (
            self.current_position_ids if position_ids is None else position_ids
        )
        if position_ids is None:
            raise ValueError("MinWM cache requires current position_ids")
        if int(position_ids.shape[0]) != int(key.shape[1]):
            raise ValueError("MinWM position_ids length must match the current K/V")

        num_new_tokens = int(key.shape[1])
        current_chunk_end = int(current_chunk_start) + num_new_tokens
        current_token_ids = torch.arange(
            current_chunk_start,
            current_chunk_end,
            dtype=torch.long,
            device=position_ids.device,
        )
        head_slice = self._cache_head_slice(
            self.k, key, cache_head_start, debug_name
        )
        global_end, local_end = self._read_indices()

        if self.token_ids is not None and current_chunk_end <= global_end:
            start_matches = torch.nonzero(
                self.token_ids == current_chunk_start, as_tuple=False
            ).flatten()
            if start_matches.numel() != 1:
                raise RuntimeError(
                    f"{debug_name} cannot recompute an evicted/non-unique chunk"
                )
            local_start = int(start_matches[0].item())
            local_stop = local_start + num_new_tokens
            if local_stop > local_end or not torch.equal(
                self.token_ids[local_start:local_stop], current_token_ids
            ):
                raise RuntimeError(f"{debug_name} current chunk is not contiguous")
            if not torch.equal(
                self.position_ids[local_start:local_stop], position_ids
            ):
                raise ValueError(
                    "MinWM active chunk position changed before final cache update"
                )
            if head_slice is None:
                self.k[:, local_start:local_stop] = key
                self.v[:, local_start:local_stop] = value
            else:
                self.k[:, local_start:local_stop, head_slice, :] = key
                self.v[:, local_start:local_stop, head_slice, :] = value
            return self._build_view(
                num_query_tokens=num_new_tokens, head_slice=head_slice
            )

        if current_chunk_start < global_end:
            raise RuntimeError(f"{debug_name} cannot append over evicted history")

        if self.pending_prompt_switch:
            self.prompt_pin_frame = int(position_ids[0, 0].item())
            self.pending_prompt_switch = False

        old_position_ids = self.position_ids
        old_rope_position_ids = self.rope_position_ids
        old_token_ids = self.token_ids
        if old_position_ids is None:
            combined_position_ids = position_ids
            combined_rope_position_ids = self._effective_position_ids(position_ids)
            combined_token_ids = current_token_ids
        else:
            combined_position_ids = torch.cat([old_position_ids, position_ids])
            combined_rope_position_ids = torch.cat(
                [old_rope_position_ids, self._effective_position_ids(position_ids)]
            )
            combined_token_ids = torch.cat([old_token_ids, current_token_ids])

        required_tokens = int(combined_token_ids.numel())
        if self.allow_growth:
            self._grow_to_fit(required_tokens)
        indices = self._select_visible_indices(
            combined_token_ids, combined_position_ids
        )

        old_k = self._head_view(self.k[:, :local_end], head_slice)
        old_v = self._head_view(self.v[:, :local_end], head_slice)
        combined_k = key if local_end == 0 else torch.cat([old_k, key], dim=1)
        combined_v = value if local_end == 0 else torch.cat([old_v, value], dim=1)
        selected_k = combined_k[:, indices].contiguous()
        selected_v = combined_v[:, indices].contiguous()
        selected_len = int(indices.numel())
        if selected_len > self.cache_size:
            raise RuntimeError(f"{debug_name} selected window exceeds cache capacity")
        if head_slice is None:
            self.k[:, :selected_len] = selected_k
            self.v[:, :selected_len] = selected_v
        else:
            self.k[:, :selected_len, head_slice, :] = selected_k
            self.v[:, :selected_len, head_slice, :] = selected_v

        self.position_ids = combined_position_ids[indices].contiguous()
        self.rope_position_ids = combined_rope_position_ids[indices].contiguous()
        self.token_ids = combined_token_ids[indices].contiguous()
        if self.pending_scene_cut_pin:
            pin_tokens = min(
                int(self.sink_tokens),
                int(current_token_ids.numel()),
            )
            if pin_tokens > 0:
                self.pinned_token_start = int(current_token_ids[0].item())
                self.pinned_token_end = self.pinned_token_start + pin_tokens
            self.pending_scene_cut_pin = False
        self._write_indices(
            global_end_index=current_chunk_end,
            local_end_index=selected_len,
        )
        return self._build_view(
            num_query_tokens=num_new_tokens, head_slice=head_slice
        )

    def copy_committed_history_from(
        self, other: "MinWMCausalSelfAttentionKVCache"
    ) -> None:
        """Copy positive-branch self history while preserving cross KV elsewhere."""
        if self.rope_position_mode != other.rope_position_mode:
            raise ValueError("MinWM CFG cache policy mismatch")
        other_global_end, other_local_end = other._read_indices()
        self._grow_to_fit(other_local_end)
        self.k[:, :other_local_end].copy_(other.k[:, :other_local_end])
        self.v[:, :other_local_end].copy_(other.v[:, :other_local_end])
        self._write_indices(
            global_end_index=other_global_end,
            local_end_index=other_local_end,
        )
        self.position_ids = (
            None if other.position_ids is None else other.position_ids.clone()
        )
        self.rope_position_ids = (
            None
            if other.rope_position_ids is None
            else other.rope_position_ids.clone()
        )
        self.token_ids = None if other.token_ids is None else other.token_ids.clone()
        self.rope_temporal_offset = other.rope_temporal_offset
        self.pinned_token_start = other.pinned_token_start
        self.pinned_token_end = other.pinned_token_end
        self.prompt_pin_frame = other.prompt_pin_frame
        self.pending_prompt_switch = other.pending_prompt_switch
        self.pending_scene_cut_pin = other.pending_scene_cut_pin
