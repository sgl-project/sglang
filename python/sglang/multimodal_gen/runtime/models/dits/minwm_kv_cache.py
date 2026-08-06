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
    rotated_k: torch.Tensor | None = None
    rotated_k_is_valid: bool = False
    current_local_start: int = 0
    current_local_end: int = 0
    is_recompute: bool = False
    query_cos: torch.Tensor | None = None
    query_sin: torch.Tensor | None = None
    key_cos: torch.Tensor | None = None
    key_sin: torch.Tensor | None = None


@dataclass(slots=True)
class MinWMCausalAttentionKVPlan:
    """Layer-independent cache selection for one MinWM transformer pass."""

    state_key: tuple
    current_position_ids: torch.Tensor
    current_chunk_start: int
    current_chunk_end: int
    num_new_tokens: int
    global_end_before: int
    local_end_before: int
    current_local_start: int
    current_local_end: int
    required_tokens: int
    selected_len: int
    old_selected_indices: torch.Tensor | None
    new_selected_indices: torch.Tensor | None
    preserves_all_history: bool
    position_ids: torch.Tensor
    rope_position_ids: torch.Tensor
    token_ids: torch.Tensor
    query_position_ids: torch.Tensor
    key_position_ids: torch.Tensor
    is_recompute: bool
    rope_temporal_offset: int
    pinned_token_start: int | None
    pinned_token_end: int | None
    prompt_pin_frame: int | None
    pending_prompt_switch: bool
    pending_scene_cut_pin: bool
    query_cos: torch.Tensor | None = None
    query_sin: torch.Tensor | None = None
    key_cos: torch.Tensor | None = None
    key_sin: torch.Tensor | None = None


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
    rotated_k: torch.Tensor | None = None
    rotated_k_is_valid: bool = False
    prepared_attention_plan: MinWMCausalAttentionKVPlan | None = None
    last_attention_plan: MinWMCausalAttentionKVPlan | None = None

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
        self.rotated_k_is_valid = False
        self.prepared_attention_plan = None
        self.last_attention_plan = None

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
        self.last_attention_plan = None

    def mark_scene_cut(self) -> None:
        self.prompt_pin_frame = None
        self.pending_prompt_switch = False
        if self.rope_position_mode == "block_relative" and self.scene_cut_rope_offset:
            raise ValueError(
                "MinWM block_relative RoPE does not support nonzero scene-cut offset"
            )
        self.rope_temporal_offset += int(self.scene_cut_rope_offset)
        self.pending_scene_cut_pin = self.scene_cut_sink_enabled
        self.last_attention_plan = None

    def _grow_to_fit(self, required_tokens: int) -> None:
        old_cache_size = self.cache_size
        old_rotated_k = self.rotated_k
        CausalSelfAttentionKVCache._grow_to_fit(self, required_tokens)
        if self.cache_size == old_cache_size or old_rotated_k is None:
            return
        self.rotated_k = self.k.new_empty(self.k.shape)
        self.rotated_k[:, :old_cache_size].copy_(old_rotated_k)

    def _ensure_rotated_k(self) -> torch.Tensor:
        if self.rotated_k is None or self.rotated_k.shape != self.k.shape:
            self.rotated_k = torch.empty_like(self.k)
            self.rotated_k_is_valid = False
        return self.rotated_k

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
        protected_sink_tokens = (
            sink_tokens + pin_len if pin_is_sink_suffix else sink_tokens
        )
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

    def _attention_plan_state_key(
        self,
        *,
        current_chunk_start: int,
        num_new_tokens: int,
        position_ids: torch.Tensor,
    ) -> tuple:
        global_end, local_end = self._read_indices()
        return (
            current_chunk_start,
            num_new_tokens,
            id(position_ids),
            global_end,
            local_end,
            self.rope_temporal_offset,
            self.pinned_token_start,
            self.pinned_token_end,
            self.prompt_pin_frame,
            self.pending_prompt_switch,
            self.pending_scene_cut_pin,
        )

    def prepare_attention_plan(
        self,
        *,
        current_chunk_start: int,
        position_ids: torch.Tensor,
    ) -> MinWMCausalAttentionKVPlan:
        """Select cache metadata once so all transformer layers can reuse it."""
        if position_ids.ndim != 2 or position_ids.shape[1] != 3:
            raise ValueError("MinWM position_ids must have shape [tokens, 3]")
        num_new_tokens = int(position_ids.shape[0])
        state_key = self._attention_plan_state_key(
            current_chunk_start=current_chunk_start,
            num_new_tokens=num_new_tokens,
            position_ids=position_ids,
        )
        if (
            self.last_attention_plan is not None
            and self.last_attention_plan.state_key == state_key
        ):
            return self.last_attention_plan

        current_chunk_end = int(current_chunk_start) + num_new_tokens
        current_token_ids = torch.arange(
            current_chunk_start,
            current_chunk_end,
            dtype=torch.long,
            device=position_ids.device,
        )
        global_end, local_end = self._read_indices()

        if self.token_ids is not None and current_chunk_end <= global_end:
            start_matches = torch.nonzero(
                self.token_ids == current_chunk_start, as_tuple=False
            ).flatten()
            if start_matches.numel() != 1:
                raise RuntimeError(
                    "MinWM causal KV cache cannot recompute an evicted/non-unique chunk"
                )
            local_start = int(start_matches[0].item())
            local_stop = local_start + num_new_tokens
            if local_stop > local_end or not torch.equal(
                self.token_ids[local_start:local_stop], current_token_ids
            ):
                raise RuntimeError(
                    "MinWM causal KV cache current chunk is not contiguous"
                )
            if not torch.equal(self.position_ids[local_start:local_stop], position_ids):
                raise ValueError(
                    "MinWM active chunk position changed before final cache update"
                )
            query_position_ids, key_position_ids = self._attention_position_ids(
                num_query_tokens=num_new_tokens
            )
            plan = MinWMCausalAttentionKVPlan(
                state_key=state_key,
                current_position_ids=position_ids,
                current_chunk_start=current_chunk_start,
                current_chunk_end=current_chunk_end,
                num_new_tokens=num_new_tokens,
                global_end_before=global_end,
                local_end_before=local_end,
                current_local_start=local_start,
                current_local_end=local_stop,
                required_tokens=local_end,
                selected_len=local_end,
                old_selected_indices=None,
                new_selected_indices=None,
                preserves_all_history=True,
                position_ids=self.position_ids,
                rope_position_ids=self.rope_position_ids,
                token_ids=self.token_ids,
                query_position_ids=query_position_ids,
                key_position_ids=key_position_ids,
                is_recompute=True,
                rope_temporal_offset=self.rope_temporal_offset,
                pinned_token_start=self.pinned_token_start,
                pinned_token_end=self.pinned_token_end,
                prompt_pin_frame=self.prompt_pin_frame,
                pending_prompt_switch=self.pending_prompt_switch,
                pending_scene_cut_pin=self.pending_scene_cut_pin,
            )
            self.last_attention_plan = plan
            return plan

        if current_chunk_start < global_end:
            raise RuntimeError(
                "MinWM causal KV cache cannot append over evicted history"
            )
        if self.pending_prompt_switch:
            self.prompt_pin_frame = int(position_ids[0, 0].item())
            self.pending_prompt_switch = False

        if self.position_ids is None:
            combined_position_ids = position_ids
            combined_rope_position_ids = self._effective_position_ids(position_ids)
            combined_token_ids = current_token_ids
        else:
            combined_position_ids = torch.cat([self.position_ids, position_ids])
            combined_rope_position_ids = torch.cat(
                [self.rope_position_ids, self._effective_position_ids(position_ids)]
            )
            combined_token_ids = torch.cat([self.token_ids, current_token_ids])

        required_tokens = int(combined_token_ids.numel())
        indices = self._select_visible_indices(
            combined_token_ids, combined_position_ids
        )
        selected_len = int(indices.numel())
        preserves_all_history = selected_len == required_tokens
        if preserves_all_history:
            selected_position_ids = combined_position_ids
            selected_rope_position_ids = combined_rope_position_ids
            selected_token_ids = combined_token_ids
            old_selected_indices = None
            new_selected_indices = None
        else:
            selected_position_ids = combined_position_ids[indices].contiguous()
            selected_rope_position_ids = combined_rope_position_ids[
                indices
            ].contiguous()
            selected_token_ids = combined_token_ids[indices].contiguous()
            old_selected_indices = indices[indices < local_end]
            new_selected_indices = indices[indices >= local_end] - local_end

        if self.pending_scene_cut_pin:
            pin_tokens = min(int(self.sink_tokens), int(current_token_ids.numel()))
            if pin_tokens > 0:
                self.pinned_token_start = int(current_token_ids[0].item())
                self.pinned_token_end = self.pinned_token_start + pin_tokens
            self.pending_scene_cut_pin = False

        if self.rope_position_mode == "absolute":
            key_position_ids = selected_rope_position_ids
        else:
            key_position_ids = selected_position_ids.clone()
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
        query_position_ids = key_position_ids[-num_new_tokens:]
        plan = MinWMCausalAttentionKVPlan(
            state_key=state_key,
            current_position_ids=position_ids,
            current_chunk_start=current_chunk_start,
            current_chunk_end=current_chunk_end,
            num_new_tokens=num_new_tokens,
            global_end_before=global_end,
            local_end_before=local_end,
            current_local_start=selected_len - num_new_tokens,
            current_local_end=selected_len,
            required_tokens=required_tokens,
            selected_len=selected_len,
            old_selected_indices=old_selected_indices,
            new_selected_indices=new_selected_indices,
            preserves_all_history=preserves_all_history,
            position_ids=selected_position_ids,
            rope_position_ids=selected_rope_position_ids,
            token_ids=selected_token_ids,
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
            is_recompute=False,
            rope_temporal_offset=self.rope_temporal_offset,
            pinned_token_start=self.pinned_token_start,
            pinned_token_end=self.pinned_token_end,
            prompt_pin_frame=self.prompt_pin_frame,
            pending_prompt_switch=self.pending_prompt_switch,
            pending_scene_cut_pin=self.pending_scene_cut_pin,
        )
        self.last_attention_plan = plan
        return plan

    def set_prepared_attention_plan(self, plan: MinWMCausalAttentionKVPlan) -> None:
        self.current_position_ids = plan.current_position_ids
        self.prepared_attention_plan = plan

    @staticmethod
    def _select_kv_with_plan(
        old_value: torch.Tensor,
        new_value: torch.Tensor,
        plan: MinWMCausalAttentionKVPlan,
    ) -> torch.Tensor:
        pieces = []
        if plan.old_selected_indices is not None and plan.old_selected_indices.numel():
            pieces.append(old_value[:, plan.old_selected_indices])
        if plan.new_selected_indices is not None and plan.new_selected_indices.numel():
            pieces.append(new_value[:, plan.new_selected_indices])
        if not pieces:
            return new_value[:, :0].contiguous()
        if len(pieces) == 1:
            return pieces[0].contiguous()
        return torch.cat(pieces, dim=1).contiguous()

    def _apply_attention_plan(
        self,
        *,
        plan: MinWMCausalAttentionKVPlan,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_head_start: int | None,
        debug_name: str,
    ) -> MinWMCausalAttentionKVView:
        head_slice = self._cache_head_slice(self.k, key, cache_head_start, debug_name)
        if self.global_end_index_int is not None and (
            self.global_end_index_int != plan.global_end_before
            or self.local_end_index_int != plan.local_end_before
        ):
            raise RuntimeError(f"{debug_name} layer metadata is out of sync")

        if plan.is_recompute:
            local_start = plan.current_local_start
            local_stop = plan.current_local_end
            if head_slice is None:
                self.k[:, local_start:local_stop] = key
                self.v[:, local_start:local_stop] = value
            else:
                self.k[:, local_start:local_stop, head_slice, :] = key
                self.v[:, local_start:local_stop, head_slice, :] = value
        else:
            if self.allow_growth:
                self._grow_to_fit(plan.required_tokens)
            if plan.selected_len > self.cache_size:
                raise RuntimeError(
                    f"{debug_name} selected window exceeds cache capacity"
                )
            if plan.preserves_all_history:
                local_start = plan.local_end_before
                local_stop = plan.selected_len
                if head_slice is None:
                    self.k[:, local_start:local_stop] = key
                    self.v[:, local_start:local_stop] = value
                else:
                    self.k[:, local_start:local_stop, head_slice, :] = key
                    self.v[:, local_start:local_stop, head_slice, :] = value
            else:
                old_k = self._head_view(self.k[:, : plan.local_end_before], head_slice)
                old_v = self._head_view(self.v[:, : plan.local_end_before], head_slice)
                selected_k = self._select_kv_with_plan(old_k, key, plan)
                selected_v = self._select_kv_with_plan(old_v, value, plan)
                if head_slice is None:
                    self.k[:, : plan.selected_len] = selected_k
                    self.v[:, : plan.selected_len] = selected_v
                else:
                    self.k[:, : plan.selected_len, head_slice, :] = selected_k
                    self.v[:, : plan.selected_len, head_slice, :] = selected_v
            self.rotated_k_is_valid = False

        self.position_ids = plan.position_ids
        self.rope_position_ids = plan.rope_position_ids
        self.token_ids = plan.token_ids
        self.rope_temporal_offset = plan.rope_temporal_offset
        self.pinned_token_start = plan.pinned_token_start
        self.pinned_token_end = plan.pinned_token_end
        self.prompt_pin_frame = plan.prompt_pin_frame
        self.pending_prompt_switch = plan.pending_prompt_switch
        self.pending_scene_cut_pin = plan.pending_scene_cut_pin
        self._write_indices(
            global_end_index=plan.current_chunk_end,
            local_end_index=plan.selected_len,
        )
        self.last_attention_plan = plan
        rotated_k = self._head_view(
            self._ensure_rotated_k()[:, : plan.selected_len], head_slice
        )
        return MinWMCausalAttentionKVView(
            k=self._head_view(self.k[:, : plan.selected_len], head_slice),
            v=self._head_view(self.v[:, : plan.selected_len], head_slice),
            query_position_ids=plan.query_position_ids,
            key_position_ids=plan.key_position_ids,
            rotated_k=rotated_k,
            rotated_k_is_valid=self.rotated_k_is_valid,
            current_local_start=plan.current_local_start,
            current_local_end=plan.current_local_end,
            is_recompute=plan.is_recompute,
            query_cos=plan.query_cos,
            query_sin=plan.query_sin,
            key_cos=plan.key_cos,
            key_sin=plan.key_sin,
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
            raise ValueError(
                "MinWM position-aware cache does not use recent_window_tokens"
            )
        if key.shape != value.shape:
            raise ValueError("MinWM attention key/value shapes must match")
        position_ids = (
            self.current_position_ids if position_ids is None else position_ids
        )
        if position_ids is None:
            raise ValueError("MinWM cache requires current position_ids")
        if int(position_ids.shape[0]) != int(key.shape[1]):
            raise ValueError("MinWM position_ids length must match the current K/V")

        plan = self.prepared_attention_plan
        self.prepared_attention_plan = None
        if plan is None:
            plan = self.prepare_attention_plan(
                current_chunk_start=current_chunk_start,
                position_ids=position_ids,
            )
        elif (
            plan.current_chunk_start != current_chunk_start
            or plan.num_new_tokens != int(key.shape[1])
        ):
            raise RuntimeError("MinWM prepared cache plan does not match current K/V")
        return self._apply_attention_plan(
            plan=plan,
            key=key,
            value=value,
            cache_head_start=cache_head_start,
            debug_name=debug_name,
        )

    def copy_committed_history_from(
        self, other: MinWMCausalSelfAttentionKVCache
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
            None if other.rope_position_ids is None else other.rope_position_ids.clone()
        )
        self.token_ids = None if other.token_ids is None else other.token_ids.clone()
        if other.rotated_k_is_valid and other.rotated_k is not None:
            self._ensure_rotated_k()[:, :other_local_end].copy_(
                other.rotated_k[:, :other_local_end]
            )
            self.rotated_k_is_valid = True
        else:
            self.rotated_k_is_valid = False
        self.rope_temporal_offset = other.rope_temporal_offset
        self.pinned_token_start = other.pinned_token_start
        self.pinned_token_end = other.pinned_token_end
        self.prompt_pin_frame = other.prompt_pin_frame
        self.pending_prompt_switch = other.pending_prompt_switch
        self.pending_scene_cut_pin = other.pending_scene_cut_pin
        self.prepared_attention_plan = None
        self.last_attention_plan = None
