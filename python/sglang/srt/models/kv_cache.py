# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import torch
import transformers

__all__ = ["JetCache"]


class JetNemotronCache(transformers.cache_utils.Cache):

    def __init__(
        self,
        seen_tokens: int = 0
    ) -> JetNemotronCache:

        self.states: list[dict[str, Any]] = []
        self.layer_wise_states: dict[str, Any] = {}

        self._base_seen_tokens = seen_tokens 
        self._seen_tokens = []  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: tuple[torch.Tensor, torch.Tensor] = None,
        conv_state: tuple[torch.Tensor] = None,
        ffn_state: torch.Tensor = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        increase_seen_tokens: bool = True,
        cache_kwargs: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            recurrent_state (`torch.Tensor`, `optional`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`, `optional`):
                The new convolution state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """
        if len(self._seen_tokens) <= layer_idx:
            self._seen_tokens.append(self._base_seen_tokens)

        # Update the number of seen tokens
        if increase_seen_tokens:
            self.increase_seen_tokens(layer_idx, offset)
            
        if attn_state is not None:
            input_size = attn_state[0].shape[-2]
            window_size = cache_kwargs.get('window_size', None)
            if not isinstance(attn_state, tuple) or len(attn_state) != 2:
                raise ValueError("`attn_state` must be a tuple of two tensors for key/value states")
        if len(self.states) <= layer_idx:
            # in prefilling stage
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state
            )
            if attn_state is not None and window_size is not None:
                # in prefilling stage, the cached and returned key/value states are different
                # original key/value states are returned, but the cached states are the last `window_size` tokens
                _key_state = attn_state[0][..., -window_size:, :]
                _value_state = attn_state[1][..., -window_size:, :]

                _attn_state = (_key_state, _value_state)
                _state = dict(
                    recurrent_state=recurrent_state,
                    attn_state=_attn_state,
                    conv_state=conv_state,
                    ffn_state=ffn_state
                )
                self.states.append(_state)
            else:
                self.states.append(state)
        else:
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                key_state, value_state = state['attn_state']
                assert window_size is None or key_state.shape[-2] <= window_size
                if window_size is not None and key_state.shape[-2] == window_size and input_size == 1:
                    # DO NOT allocate new memory if the cache is full
                    # only works in decoding stage
                    # roll the key/value states to the left by `input_size`
                                        
                    key_state = key_state.roll(-input_size, -2)
                    value_state = value_state.roll(-input_size, -2)
                                        
                    # replace the last `input_size` tokens with the new key/value states
                    key_state[..., -input_size:, :] = attn_state[0]
                    value_state[..., -input_size:, :] = attn_state[1]
                    
                    attn_state = (key_state, value_state)
                else:
                    # <= window_size or not sliding window or chunk-prefilling (input_size > 1)
                    attn_state = (torch.cat([key_state, attn_state[0]], -2),
                                  torch.cat([value_state, attn_state[1]], -2),)
                state['attn_state'] = attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        assert len(self.states) == len(self._seen_tokens)

        return state

    def trim_attn_state(self, layer_idx: int, window_size: int) -> None:
        # handle the case when the input length of SWA > 1 and has a cache, especially the chunk-prefilling case
        # this function is called after attention is donw
        assert layer_idx < len(self.states), f"Layer index {layer_idx} out of range for states with length {len(self.states)}"
        state = self.states[layer_idx]
        assert state["attn_state"] is not None, f"Layer {layer_idx} does not have an attention state"
        key_state, value_state = state["attn_state"]
        if key_state.shape[-2] > window_size:
            state["attn_state"] = (
                key_state[..., -window_size:, :],
                value_state[..., -window_size:, :],
            )

    def increase_seen_tokens(self, layer_idx: int, offset: int = 1) -> None:
        """Increases the number of seen tokens for the layer `layer_idx` by `offset`."""
        self._seen_tokens[layer_idx] += offset

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self._seen_tokens) <= layer_idx:
            return self._base_seen_tokens
        return self._seen_tokens[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> tuple:
        return tuple(self.states)

    def print_kv_sizes(self) -> None:
        """Returns the size of the cached key/value states."""
        for layer_idx, state in enumerate(self.states):
            if state.get("attn_state", None) is not None:
                key_state, value_state = state["attn_state"]
                # compute state size in MB
                key_size = key_state.element_size() * key_state.nelement() / (1024**2)
                value_size = value_state.element_size() * value_state.nelement() / (1024**2)
                print(key_state.shape, value_state.shape)
                print(f"Layer {layer_idx}: Attention. cache size: {key_size + value_size:.2f} MB")
            if state.get("conv_state", None) is not None:
                conv_state = state["conv_state"]
                # compute state size in MB
                conv_sizes = []
                for conv in conv_state:
                    conv_size = conv.element_size() * conv.nelement() / (1024**2)
                    conv_sizes.append(conv_size)
                conv_size = sum(conv_sizes)
                print(f"Layer {layer_idx}: Convolution. cache size: {conv_size:.2f} MB")
            if state.get("ffn_state", None) is not None:
                ffn_state = state["ffn_state"]
                # compute state size in MB
                ffn_size = ffn_state.element_size() * ffn_state.nelement() / (1024**2)
                print(f"Layer {layer_idx}: FFN. cache size: {ffn_size:.2f} MB")
            if state.get("recurrent_state", None) is not None:
                recurrent_state = state["recurrent_state"]
                # compute state size in MB
                recurrent_size = recurrent_state.element_size() * recurrent_state.nelement() / (1024**2)
                print(f"Layer {layer_idx}: Recurrent. cache size: {recurrent_size:.2f} MB")
