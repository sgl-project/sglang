"""Memory pool."""

import logging

import torch

logger = logging.getLogger(__name__)


class ReqToTokenPool:
    def __init__(self, size, max_context_len):
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self.can_use_mem_size = size
        self.req_to_token = torch.empty(
            (size, max_context_len), dtype=torch.int32, device="cuda"
        )

    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            return None

        select_index = torch.nonzero(self.mem_state).squeeze(1)[:need_size]
        self.mem_state[select_index] = False
        self.can_use_mem_size -= need_size

        return select_index.to(torch.int32)

    def free(self, free_index):
        if isinstance(free_index, (int,)):
            self.can_use_mem_size += 1
        else:
            self.can_use_mem_size += free_index.shape[0]

        self.mem_state[free_index] = True

    def clear(self):
        self.mem_state.fill_(True)
        self.can_use_mem_size = len(self.mem_state)


class TokenToKVPool:
    def __init__(self, size, dtype, head_num, head_dim, layer_num):
        self.size = size

        # This can be promised:
        # assert torch.all(mem_state <= 1) and torch.all(mem_state >= 0)
        # We also add one slot. This slot is used for writing dummy output from padded tokens.
        self.mem_state = torch.ones((self.size + 1,), dtype=torch.bool, device="cuda")
        self.can_use_mem_size = self.size

        # [size, key/value, head_num, head_dim] for each layer
        self.kv_data = [
            torch.empty((size + 1, 2, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]

        # Prefetch buffer
        self.prefetch_buffer = torch.empty(0, device="cuda", dtype=torch.int32)
        self.prefetch_chunk_size = 512

        self.clear()

    def get_key_buffer(self, layer_id):
        return self.kv_data[layer_id][:, 0]

    def get_value_buffer(self, layer_id):
        return self.kv_data[layer_id][:, 1]

    def alloc(self, need_size):
        buffer_len = len(self.prefetch_buffer)
        if need_size <= buffer_len:
            select_index = self.prefetch_buffer[:need_size]
            self.prefetch_buffer = self.prefetch_buffer[need_size:]
            return select_index

        addition_size = need_size - buffer_len
        alloc_size = max(addition_size, self.prefetch_chunk_size)
        select_index = torch.nonzero(self.mem_state).squeeze(1)[:alloc_size]
        select_index = select_index.to(torch.int32)

        if select_index.shape[0] < addition_size:
            return None

        self.add_refs(select_index)

        self.prefetch_buffer = torch.cat((self.prefetch_buffer, select_index))
        ret_index = self.prefetch_buffer[:need_size]
        self.prefetch_buffer = self.prefetch_buffer[need_size:]

        return ret_index

    def available_size(self):
        return self.can_use_mem_size + len(self.prefetch_buffer)

    def add_refs(self, token_index: torch.Tensor):
        self.can_use_mem_size -= len(token_index)
        self.mem_state[token_index] = False

    def dec_refs(self, token_index: torch.Tensor):
        self.can_use_mem_size += len(token_index)
        self.mem_state[token_index] = True

    def clear(self):
        self.mem_state.fill_(True)
        self.can_use_mem_size = self.size

        # We also add one slot. This slot is used for writing dummy output from padded tokens.
        self.mem_state[0] = False
