"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Memory pool."""

import logging

import torch

logger = logging.getLogger(__name__)


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size: int, max_context_len: int):
        self.size = size
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self.req_to_token = torch.empty(
            (size, max_context_len), dtype=torch.int32, device="cuda"
        )
        self.can_use_mem_size = size

    def alloc(self, need_size: int):
        if need_size > self.can_use_mem_size:
            return None

        select_index = (
            torch.nonzero(self.mem_state).squeeze(1)[:need_size].to(torch.int32)
        )
        self.mem_state[select_index] = False
        self.can_use_mem_size -= need_size

        return select_index

    def free(self, free_index):
        self.mem_state[free_index] = True
        if isinstance(free_index, (int,)):
            self.can_use_mem_size += 1
        else:
            self.can_use_mem_size += free_index.shape[0]

    def clear(self):
        self.mem_state.fill_(True)
        self.can_use_mem_size = len(self.mem_state)


class TokenToKVPool:
    """A memory pool that maps a token to its kv cache locations"""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ):
        self.size = size

        # We also add one slot. This slot is used for writing dummy output from padded tokens.
        self.mem_state = torch.ones((self.size + 1,), dtype=torch.bool, device="cuda")

        # [size, head_num, head_dim] for each layer
        self.k_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]

        # Prefetch buffer
        self.prefetch_buffer = torch.empty(0, device="cuda", dtype=torch.int32)
        self.prefetch_chunk_size = 512

        self.can_use_mem_size = self.size
        self.clear()

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def available_size(self):
        return self.can_use_mem_size + len(self.prefetch_buffer)

    def alloc(self, need_size: int):
        buffer_len = len(self.prefetch_buffer)
        if need_size <= buffer_len:
            select_index = self.prefetch_buffer[:need_size]
            self.prefetch_buffer = self.prefetch_buffer[need_size:]
            return select_index

        addition_size = need_size - buffer_len
        alloc_size = max(addition_size, self.prefetch_chunk_size)
        select_index = (
            torch.nonzero(self.mem_state).squeeze(1)[:alloc_size].to(torch.int32)
        )

        if select_index.shape[0] < addition_size:
            return None

        self.mem_state[select_index] = False
        self.can_use_mem_size -= len(select_index)

        self.prefetch_buffer = torch.cat((self.prefetch_buffer, select_index))
        ret_index = self.prefetch_buffer[:need_size]
        self.prefetch_buffer = self.prefetch_buffer[need_size:]

        return ret_index

    def free(self, free_index: torch.Tensor):
        self.mem_state[free_index] = True
        self.can_use_mem_size += len(free_index)

    def clear(self):
        self.prefetch_buffer = torch.empty(0, device="cuda", dtype=torch.int32)

        self.mem_state.fill_(True)
        self.can_use_mem_size = self.size

        # We also add one slot. This slot is used for writing dummy output from padded tokens.
        self.mem_state[0] = False
