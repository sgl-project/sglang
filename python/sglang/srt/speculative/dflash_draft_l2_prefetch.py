"""SGLANG_OPT_DFLASH_DRAFT_L2_PREFETCH: a branch forked at the draft CUDA graph's head
prefetches the draft's small non-Linear tensors into L2; the graph's tail joins it."""

from itertools import chain
from typing import List

import torch

from sglang.kernels.ops.speculative.l2_prefetch import (
    build_l2_prefetch_table,
    l2_prefetch,
)
from sglang.srt.layers.linear import LinearBase

_L2_PREFETCH_MAX_TENSOR_BYTES = 4 << 20


def _own_tensors(module):
    attrs = (v for v in vars(module).values() if isinstance(v, torch.Tensor))
    return chain(module.parameters(recurse=False), module.buffers(recurse=False), attrs)


def draft_l2_prefetch_tensors(draft_model) -> List[torch.Tensor]:
    tensors, seen = [], set()

    def add(t):
        if (
            t.is_cuda
            and t.numel() * t.element_size() <= _L2_PREFETCH_MAX_TENSOR_BYTES
            and t.data_ptr() not in seen
        ):
            seen.add(t.data_ptr())
            tensors.append(t)

    for layer in draft_model.layers:
        for module in layer.modules():
            if isinstance(module, LinearBase):
                continue
            for t in _own_tensors(module):
                add(t)
    for module in draft_model.norm.modules():
        for t in _own_tensors(module):
            add(t)
    return tensors


class DraftL2Prefetch:
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensors = tensors
        self.table = build_l2_prefetch_table(tensors)
        self._stream = torch.cuda.Stream(device=self.table.device)

    def capture_head_hook(self, runner, forward_batch) -> None:
        del runner, forward_batch
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            l2_prefetch(self.table)

    def capture_tail_hook(self, runner, out, forward_batch, num_tokens) -> None:
        del runner, out, forward_batch, num_tokens
        torch.cuda.current_stream().wait_stream(self._stream)
