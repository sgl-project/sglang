from typing import (
    TYPE_CHECKING,
    Optional,
    Sequence,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode


def compute_split_seq_index(
    forward_mode: "ForwardMode",
    num_tokens: int,
    extend_lens: Optional[Sequence[int]],
) -> Optional[int]:
    if forward_mode.is_extend():
        assert extend_lens is not None
        split_seq_index = _split_array_by_half_sum(extend_lens)
        num_seqs = len(extend_lens)
    elif forward_mode.is_decode():
        split_seq_index = num_tokens // 2
        num_seqs = num_tokens
    else:
        raise NotImplementedError

    if split_seq_index == 0 or split_seq_index == num_seqs:
        return None

    return split_seq_index


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
    else:
        raise NotImplementedError
