# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CausalAttentionKVView:
    k: torch.Tensor
    v: torch.Tensor
    local_start_index: int
    local_end_index: int
    visible_local_end: int
    visible_global_end: int


@dataclass(slots=True)
class CausalSelfAttentionKVCache:
    """one transformer block's causal self-attn K/V cache and write cursors"""

    k: torch.Tensor
    v: torch.Tensor
    # the right bound of the valid global token range
    # e.g., 12000 means [0, 12000) has been generated and cached
    global_end_index: torch.Tensor
    # the right bound of the valid local token range within the buffer (when cache is unfilled)
    local_end_index: torch.Tensor
    global_end_index_int: int | None = None
    local_end_index_int: int | None = None
    cache_size: int = 0
    sink_tokens: int = 0
    attention_window_size: int = 0
    allow_growth: bool = False

    def __post_init__(self) -> None:
        if self.cache_size == 0:
            self.cache_size = self.k.shape[1]
        if self.attention_window_size == 0:
            self.attention_window_size = self.cache_size

    def reset_indices(self) -> None:
        self.global_end_index.zero_()
        self.local_end_index.zero_()
        if self.global_end_index_int is not None:
            self.global_end_index_int = 0
        if self.local_end_index_int is not None:
            self.local_end_index_int = 0

    def _read_indices(self) -> tuple[int, int]:
        global_end_index = self.global_end_index_int
        local_end_index = self.local_end_index_int
        if global_end_index is None or local_end_index is None:
            global_end_index = int(self.global_end_index.item())
            local_end_index = int(self.local_end_index.item())
            self.global_end_index_int = global_end_index
            self.local_end_index_int = local_end_index
        return global_end_index, local_end_index

    def _write_indices(self, *, global_end_index: int, local_end_index: int) -> None:
        if (
            self.global_end_index_int == global_end_index
            and self.local_end_index_int == local_end_index
        ):
            return
        if self.global_end_index_int is not None:
            self.global_end_index_int = global_end_index
        if self.local_end_index_int is not None:
            self.local_end_index_int = local_end_index
        self.global_end_index.fill_(global_end_index)
        self.local_end_index.fill_(local_end_index)

    def _grow_to_fit(self, required_tokens: int) -> None:
        if required_tokens <= self.cache_size:
            return
        old_cache_size = self.cache_size
        new_cache_size = max(required_tokens, old_cache_size * 2)

        new_k = self.k.new_zeros(
            self.k.shape[0],
            new_cache_size,
            self.k.shape[2],
            self.k.shape[3],
        )
        new_v = self.v.new_zeros(
            self.v.shape[0],
            new_cache_size,
            self.v.shape[2],
            self.v.shape[3],
        )
        new_k[:, :old_cache_size] = self.k
        new_v[:, :old_cache_size] = self.v
        self.k = new_k
        self.v = new_v
        self.cache_size = new_cache_size
        if self.attention_window_size == old_cache_size:
            self.attention_window_size = new_cache_size

    def update_and_get_attention_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        current_chunk_start: int,
        debug_name: str = "causal KV cache",
    ) -> CausalAttentionKVView:
        """write kv into the cache, returns the part visible to the current chunk

        Args:
            current_chunk_start: the global position of the start of the chunk

        """
        num_new_tokens = key.shape[1]
        current_chunk_end = current_chunk_start + num_new_tokens
        kv_cache_size = self.cache_size
        sink_tokens = self.sink_tokens
        global_end_index, local_end_index_prev = self._read_indices()

        # local_end_index: the local position of the end of current chunk
        # updated_local_end: the updated local end
        # updated_global_end: the updated global end

        # the global position of the start of the buffer
        window_start = global_end_index - local_end_index_prev

        if current_chunk_end <= global_end_index:
            # the window stays as previous
            # cache layout:
            # [sink tokens, recent window tokens, current chunk tokens, uninitialized tokens (optional)]
            local_start_index = current_chunk_start - window_start
            local_end_index = local_start_index + num_new_tokens

            # the local end and global end remains unchanged (since the chunk hasn't proceed)
            updated_local_end = local_end_index_prev
            updated_global_end = global_end_index
        else:
            # the chunk window has proceed, append new tokens, and evict earliest (if have to)
            appended_tokens = current_chunk_end - global_end_index
            if self.allow_growth:
                self._grow_to_fit(local_end_index_prev + appended_tokens)
                kv_cache_size = self.cache_size
            if local_end_index_prev + appended_tokens > kv_cache_size:
                # the new tokens can't fit in the remaining space (after local_end_index_prev), start evicting:
                # before:
                # [sink tokens, evicted tokens, rolled tokens, remaining space]
                #                                            ^ end of previous chunk
                # after:
                # [sink tokens, rolled tokens,           remaining space      ]

                # 1. keep sink tokens ([0: sink_tokens]) untouched
                # 2. evict obsolete tokens in: [sink_tokens:sink_tokens + num_evicted_tokens]
                num_evicted_tokens = (
                    local_end_index_prev + appended_tokens - kv_cache_size
                )

                # number of tokens to move
                num_rolled_tokens = max(
                    0,
                    local_end_index_prev - num_evicted_tokens - sink_tokens,
                )
                if num_rolled_tokens > 0:
                    self.k[:, sink_tokens : sink_tokens + num_rolled_tokens] = self.k[
                        :,
                        sink_tokens
                        + num_evicted_tokens : sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()
                    self.v[:, sink_tokens : sink_tokens + num_rolled_tokens] = self.v[
                        :,
                        sink_tokens
                        + num_evicted_tokens : sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()

                # if we move the minimum number of tokens, the right bound of the append token would be aligned with end of the buffer
                local_end_index = kv_cache_size
            else:
                # enough space, directly append new tokens after end of previous chunk
                local_end_index = local_end_index_prev + appended_tokens
            local_start_index = local_end_index - num_new_tokens
            updated_local_end = local_end_index
            # after filling in the proceeded new chunk, the global end aligns with the global end of the current chunk
            updated_global_end = current_chunk_end

        if (
            local_start_index < 0
            or local_end_index > kv_cache_size
            or local_end_index - local_start_index != num_new_tokens
        ):
            raise RuntimeError(
                f"Invalid {debug_name} write range: "
                f"local=[{local_start_index}, {local_end_index}), "
                f"global_end={global_end_index}, "
                f"prev_local_end={local_end_index_prev}, "
                f"kv_cache_size={kv_cache_size}, "
                f"num_new_tokens={num_new_tokens}, "
                f"current_start={current_chunk_start}, current_end={current_chunk_end}"
            )

        if self.k.requires_grad:
            self.k = self.k.detach()
        if self.v.requires_grad:
            self.v = self.v.detach()
        self.k[:, local_start_index:local_end_index] = key
        self.v[:, local_start_index:local_end_index] = value

        attn_start_index = max(0, updated_local_end - self.attention_window_size)
        self._write_indices(
            global_end_index=updated_global_end,
            local_end_index=updated_local_end,
        )
        return CausalAttentionKVView(
            k=self.k[:, attn_start_index:updated_local_end],
            v=self.v[:, attn_start_index:updated_local_end],
            local_start_index=local_start_index,
            local_end_index=local_end_index,
            visible_local_end=updated_local_end,
            visible_global_end=updated_global_end,
        )


@dataclass(slots=True)
class CrossAttentionKVCache:
    """one transformer block's cross-attn condition K/V cache"""

    k: torch.Tensor
    v: torch.Tensor
    is_init: bool = False

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k.detach()
        self.v = v.detach()
        self.is_init = True

    def reset(self) -> None:
        self.is_init = False
