import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


def compute_split_seq_index(
        forward_mode: "ForwardMode",
        num_tokens: int,
        extend_lens: Optional[Sequence[int]],
) -> Optional[int]:
    if forward_mode.is_extend():
        assert extend_lens is not None
        return _split_array_by_half_sum(extend_lens)
    elif forward_mode.is_decode():
        return num_tokens // 2
    elif forward_mode.is_idle():
        assert num_tokens == 0
        return 0
    else:
        raise NotImplementedError


def _split_array_by_half_sum(arr: Sequence[int]) -> int:
    overall_sum = sum(arr)
    accumulator, split_index = 0, 0
    for value in arr[:-1]:
        accumulator += value
        split_index += 1
        if accumulator >= overall_sum // 2:
            break
    return split_index


def compute_split_token_index(
        split_seq_index: int,
        forward_mode: "ForwardMode",
        extend_seq_lens: Optional[Sequence[int]],
) -> int:
    if forward_mode.is_extend():
        assert extend_seq_lens is not None
        return sum(extend_seq_lens[:split_seq_index])
    elif forward_mode.is_decode():
        return split_seq_index
    elif forward_mode.is_idle():
        assert split_seq_index == 0
        return 0
    else:
        raise NotImplementedError

