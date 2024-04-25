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
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= need_size
        return select_index.to(torch.int32)

    def free(self, free_index):
        if isinstance(free_index, (int,)):
            self.can_use_mem_size += 1
        else:
            self.can_use_mem_size += free_index.shape[0]
        self.mem_state[free_index] = 1

    def clear(self):
        self.mem_state.fill_(1)
        self.can_use_mem_size = len(self.mem_state)


class TokenToKVPool:
    def __init__(self, size, dtype, head_num, head_dim, layer_num):
        self.mem_state = torch.zeros((size,), dtype=torch.int16, device="cuda")
        self.total_ref_ct = 0

        # [size, key/value, head_num, head_dim] for each layer
        self.kv_data = [
            torch.empty((size, 2, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id):
        return self.kv_data[layer_id][:, 0]

    def get_value_buffer(self, layer_id):
        return self.kv_data[layer_id][:, 1]

    def alloc(self, need_size):
        select_index = torch.nonzero(self.mem_state == 0).squeeze(1)[:need_size]
        if select_index.shape[0] < need_size:
            return None

        self.add_refs(select_index)
        return select_index.to(torch.int32)

    def alloc_contiguous(self, need_size):
        empty_index = torch.nonzero(self.mem_state == 0).squeeze(1)[:need_size]
        if empty_index.shape[0] < need_size:
            return None
        empty_size = len(empty_index)
        loc_sum = (
            empty_index[need_size - 1 :] - empty_index[: empty_size - (need_size - 1)]
        )
        can_used_loc = empty_index[: empty_size - (need_size - 1)][
            loc_sum == need_size - 1
        ]
        if can_used_loc.shape[0] == 0:
            return None

        start_loc = can_used_loc[0].item()
        select_index = torch.arange(start_loc, start_loc + need_size, device="cuda")
        self.add_refs(select_index)
        return select_index.to(torch.int32), start_loc, start_loc + need_size

    def used_size(self):
        return len(torch.nonzero(self.mem_state).squeeze(1))

    def available_size(self):
        return torch.sum(self.mem_state == 0).item()

    def add_refs(self, token_index: torch.Tensor):
        self.total_ref_ct += len(token_index)
        self.mem_state[token_index] += 1

    def dec_refs(self, token_index: torch.Tensor):
        self.total_ref_ct -= len(token_index)
        self.mem_state[token_index] -= 1

        num_freed = torch.sum(self.mem_state[token_index] == 0)

        return num_freed

    def clear(self):
        self.mem_state.fill_(0)
        self.total_ref_ct = 0
