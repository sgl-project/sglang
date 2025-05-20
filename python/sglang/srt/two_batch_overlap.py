from typing import (
    TYPE_CHECKING,
    Optional,
    Sequence,
)

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.utils import DeepEPMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


# TODO: may smartly disable TBO when batch size is too small b/c it will slow down
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


class TboDPAttentionPreparer:
    def prepare_all_gather(self, local_batch, deepep_mode, enable_deepep_moe, enable_two_batch_overlap):
        self.enable_two_batch_overlap = enable_two_batch_overlap

        if local_batch is not None:
            self.local_tbo_split_seq_index = compute_split_seq_index(
                forward_mode=local_batch.forward_mode,
                num_tokens=local_batch.input_ids.shape[0],
                extend_lens=local_batch.extend_lens,
            )
            resolved_deepep_mode = deepep_mode.resolve(local_batch.forward_mode)
            local_can_run_tbo = (self.local_tbo_split_seq_index is not None) and not (
                local_batch.forward_mode.is_extend()
                and enable_deepep_moe
                and (resolved_deepep_mode == DeepEPMode.low_latency)
            )
        else:
            self.local_tbo_split_seq_index = 0
            local_can_run_tbo = True

        local_forward_mode = self._compute_local_forward_mode(local_batch)

        return local_can_run_tbo, local_forward_mode

    def compute_output(self, partial_global_info):
        local_can_run_tbo_aggregated = min(partial_global_info[:, 0, 0].tolist())
        forward_modes = partial_global_info[:, 0, 1].tolist()

        global_forward_mode, forward_mode_agree = self._compute_global_forward_mode(forward_modes)

        can_run_tbo = (
            self.enable_two_batch_overlap
            and local_can_run_tbo_aggregated
            and forward_mode_agree
        )

        tbo_split_seq_index = self.local_tbo_split_seq_index if can_run_tbo else None
        global_forward_mode = global_forward_mode if can_run_tbo else None
        return tbo_split_seq_index, global_forward_mode

    @staticmethod
    def _compute_local_forward_mode(local_batch):
        return (
            local_batch.forward_mode
            if local_batch is not None
            else ForwardMode.IDLE
        ).value

    @staticmethod
    def _compute_global_forward_mode(forward_modes):
        converted_forward_modes = [
            ForwardMode.DECODE.value if x == ForwardMode.IDLE.value else x
            for x in forward_modes
        ]
        forward_mode_agree = TboDPAttentionPreparer._is_all_same(converted_forward_modes)
        global_forward_mode = (
            ForwardMode(converted_forward_modes[0]) if forward_mode_agree else None
        )
        return global_forward_mode, forward_mode_agree

    @staticmethod
    def _is_all_same(x):
        return all(value == x[0] for value in x)
