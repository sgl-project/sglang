# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, ClassVar

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
    # the position within global token sequence
    global_end_index: torch.Tensor
    # the position within current cache buffer
    local_end_index: torch.Tensor
    global_end_index_int: int | None = None
    local_end_index_int: int | None = None

    _FIELD_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "k",
            "v",
            "global_end_index",
            "local_end_index",
            "global_end_index_int",
            "local_end_index_int",
        }
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._FIELD_NAMES:
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._FIELD_NAMES and getattr(self, key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._FIELD_NAMES:
            return default
        value = getattr(self, key)
        return default if value is None else value

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

    def _write_indices(
        self, *, visible_global_end: int, visible_local_end: int
    ) -> None:
        if self.global_end_index_int is not None:
            self.global_end_index_int = visible_global_end
        if self.local_end_index_int is not None:
            self.local_end_index_int = visible_local_end
        self.global_end_index.fill_(visible_global_end)
        self.local_end_index.fill_(visible_local_end)

    def update_and_get_attention_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        current_start: int,
        sink_tokens: int,
        attention_window_size: int | None,
        debug_name: str = "causal KV cache",
    ) -> CausalAttentionKVView:
        num_new_tokens = key.shape[1]
        current_end = current_start + num_new_tokens
        kv_cache_size = self.k.shape[1]
        global_end_index, local_end_index_prev = self._read_indices()
        window_start = global_end_index - local_end_index_prev

        if current_end <= global_end_index:
            local_start_index = current_start - window_start
            local_end_index = local_start_index + num_new_tokens
            visible_local_end = local_end_index_prev
            visible_global_end = global_end_index
        else:
            appended_tokens = current_end - global_end_index
            if local_end_index_prev + appended_tokens > kv_cache_size:
                num_evicted_tokens = (
                    local_end_index_prev + appended_tokens - kv_cache_size
                )
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
                local_end_index = (
                    local_end_index_prev + appended_tokens - num_evicted_tokens
                )
            else:
                local_end_index = local_end_index_prev + appended_tokens
            local_start_index = local_end_index - num_new_tokens
            visible_local_end = local_end_index
            visible_global_end = current_end

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
                f"current_start={current_start}, current_end={current_end}"
            )

        self.k = self.k.detach()
        self.v = self.v.detach()
        self.k[:, local_start_index:local_end_index] = key
        self.v[:, local_start_index:local_end_index] = value

        if attention_window_size is None:
            attn_start_index = 0
        else:
            attn_start_index = max(0, visible_local_end - attention_window_size)
        self._write_indices(
            visible_global_end=visible_global_end,
            visible_local_end=visible_local_end,
        )
        return CausalAttentionKVView(
            k=self.k[:, attn_start_index:visible_local_end],
            v=self.v[:, attn_start_index:visible_local_end],
            local_start_index=local_start_index,
            local_end_index=local_end_index,
            visible_local_end=visible_local_end,
            visible_global_end=visible_global_end,
        )


@dataclass(slots=True)
class CrossAttentionKVCache:
    """one transformer block's cross-attn condition K/V cache"""

    k: torch.Tensor
    v: torch.Tensor
    is_init: bool = False

    _FIELD_NAMES: ClassVar[frozenset[str]] = frozenset({"k", "v", "is_init"})

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._FIELD_NAMES:
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._FIELD_NAMES

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._FIELD_NAMES:
            return default
        return getattr(self, key)

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k.detach()
        self.v = v.detach()
        self.is_init = True

    def reset(self) -> None:
        self.is_init = False
