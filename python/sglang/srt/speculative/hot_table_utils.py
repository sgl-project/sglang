import heapq
from typing import List, Optional, Union

import torch


class HotVocabTable:
    def __init__(self, initial_tokens, num_dynamic_tokens=256):
        self.topk = len(initial_tokens)
        self.heap = [
            (self.topk - i, token_id)
            for i, token_id in enumerate(initial_tokens[:-num_dynamic_tokens])
        ]
        self.heap.extend(
            [
                (0, token_id)
                for i, token_id in enumerate(initial_tokens[-num_dynamic_tokens:])
            ]
        )
        heapq.heapify(self.heap)
        self.counters = {
            token_id: self.topk - i
            for i, token_id in enumerate(initial_tokens[:-num_dynamic_tokens])
        }
        self.counters.update(
            {token_id: 0 for token_id in initial_tokens[-num_dynamic_tokens:]}
        )
        self.pos = {token_id: idx for idx, (_, token_id) in enumerate(self.heap)}
        self.token_ids = torch.tensor(initial_tokens, dtype=torch.int32, device="cuda")

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i][0] < self.heap[parent][0]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                self.pos[self.heap[i][1]], self.pos[self.heap[parent][1]] = i, parent
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            left, right, smallest = 2 * i + 1, 2 * i + 2, i
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self.pos[self.heap[i][1]], self.pos[self.heap[smallest][1]] = i, smallest
            i = smallest

    def add_token(self, token_ids: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        if not isinstance(token_ids, list):
            token_ids = [token_ids]

        for t in token_ids:
            if t.dim() != 1:
                t = t.flatten()
            for item in t:
                self._add_to_heap(item.item())

    def _add_to_heap(self, token_id):
        self.counters[token_id] = self.counters.get(token_id, 0) + 1
        current_count = self.counters[token_id]

        if token_id in self.pos:
            idx = self.pos[token_id]
            old_count, _ = self.heap[idx]
            if current_count == old_count:
                return
            self.heap[idx] = (current_count, token_id)
            self._sift_down(idx) if current_count > old_count else self._sift_up(idx)
        else:
            if current_count > self.heap[0][0]:
                old_token = self.heap[0][1]
                del self.pos[old_token]
                self.heap[0] = (current_count, token_id)
                self.token_ids[self.token_ids == old_token] = token_id
                self.pos[token_id] = 0
                self._sift_down(0)

    def get_hot_token_ids(self):
        return self.token_ids
