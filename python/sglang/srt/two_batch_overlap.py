import os
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


def compute_split_seq_index(
        forward_mode: ForwardMode,
        num_tokens: int,
        extend_lens: Optional[Sequence[int]],
) -> Optional[int]:
    if forward_mode.is_extend():
        assert extend_lens is not None
        split_seq_index = _split_array_by_half_sum(extend_lens)
    elif forward_mode.is_decode():
        split_seq_index = num_tokens // 2
    else:
        raise NotImplementedError

    if split_seq_index == 0 or split_seq_index == num_tokens:
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
        forward_mode: ForwardMode,
        extend_lens: Optional[Sequence[int]],
) -> int:
    if forward_mode.is_extend():
        assert extend_lens is not None
        return sum(extend_lens[:split_seq_index])
    elif forward_mode.is_decode():
        return split_seq_index
    else:
        raise NotImplementedError


def model_forward_split_inputs(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
) -> Tuple[Dict, Dict]:
    return tuple(
        *[
            _model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                output_forward_batch=output_forward_batch,
            )
            for output_forward_batch in forward_batch.tbo_children
        ]
    )


def _model_forward_filter_inputs(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        output_forward_batch: ForwardBatch,
) -> Dict:
    token_slice = slice(*output_forward_batch.tbo_parent_token_range)
    return dict(
        hidden_states=hidden_states[token_slice],
        residual=residual[token_slice],
        positions=positions[token_slice],
        forward_batch=output_forward_batch,
    )


def model_forward_merge_outputs(output_a, output_b):
    hidden_states_a, residual_a = output_a
    hidden_states_b, residual_b = output_b

    hidden_states = torch.concat([hidden_states_a, hidden_states_b], dim=0)
    residual = torch.concat([residual_a, residual_b], dim=0)

    return hidden_states, residual


_ENABLE_PROFILE = bool(
    int(os.environ.get("SGLANG_MULTI_BATCH_EXECUTOR_ENABLE_PROFILE", "0"))
)


def model_forward_execute_two_batch(
        inputs,
        stages: List[Callable],
        delta_stages: int,
):
    splitted_inputs = model_forward_split_inputs(**inputs)
    inputs_a, inputs_b = splitted_inputs
    output_a, output_b = _execute_two_batch_raw(
        inputs_a, inputs_b, stages, delta_stages=delta_stages
    )
    return model_forward_merge_outputs(output_a, output_b)


def _execute_two_batch_raw(inputs_a, inputs_b, stages, delta_stages: int):
    executor_a = _StageExecutor("a", stages, inputs=inputs_a)
    executor_b = _StageExecutor("b", stages, inputs=inputs_b)

    for _ in range(delta_stages):
        executor_a.next()

    for _ in range(executor_a.num_stages - delta_stages):
        executor_a.next()
        executor_b.next()

    for _ in range(delta_stages):
        executor_b.next()

    assert executor_a.done and executor_b.done
    return executor_a.output, executor_b.output


class _StageExecutor:
    def __init__(self, debug_name: str, stages: List[Callable], inputs):
        self._debug_name = debug_name
        self._stages = stages
        self._index = 0
        self._stage_state = None
        self._stage_output = inputs

    def next(self):
        assert not self.done

        if _ENABLE_PROFILE:
            ctx = torch.profiler.record_function(
                f"Stage-{self._debug_name}{self._index}"
            )
        else:
            ctx = nullcontext()

        with ctx:
            stage = self._stages[self._index]
            self._stage_state, self._stage_output = stage(
                state=self._stage_state, **(self._stage_output or {})
            )

        self._index += 1

    @property
    def output(self):
        assert self.done
        return self._stage_output

    @property
    def done(self):
        return self._index >= self.num_stages

    @property
    def num_stages(self):
        return len(self._stages)
