import os
from typing import Sequence, Optional, Tuple, Dict

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from torch._dynamo.eval_frame import null_context


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


# ------------------------------------------ TODO ------------------------------------------


def split_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
) -> Optional[Tuple[Dict, Dict]]:
    output_a = _filter_inputs(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        child_mode="a",
    )
    output_b = _filter_inputs(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        child_mode="b",
    )

    # _log(
    #     f'split_inputs {forward_batch.input_ids.tolist()=} {output_a["forward_batch"].input_ids.tolist()=} {output_b["forward_batch"].input_ids.tolist()=}')
    # _log(
    #     f'split_inputs (input){forward_batch=} {output_a=} {output_b=}')

    return output_a, output_b


def _filter_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    child_index: int,
) -> Dict:
    # _log(
    #     f'filter_inputs {start_token_index=} {end_token_index=} {start_seq_index=} {end_seq_index=} {output_global_num_tokens=}')
    # TODO improve, e.g. make it `children`?
    output_forward_batch = forward_batch.tbo_children[child_index]
    token_slice = slice(*output_forward_batch.tbo_parent_token_range)
    return dict(
        hidden_states=hidden_states[token_slice],
        residual=residual[token_slice],
        positions=positions[token_slice],
        forward_batch=output_forward_batch,
    )


def _merge_outputs(output_a, output_b):
    hidden_states_a, residual_a = output_a
    hidden_states_b, residual_b = output_b

    hidden_states = torch.concat([hidden_states_a, hidden_states_b], dim=0)
    residual = torch.concat([residual_a, residual_b], dim=0)

    return hidden_states, residual


# ------------------------------------------ TODO ------------------------------------------

_ENABLE_PROFILE = bool(
    int(os.environ.get("SGLANG_MULTI_BATCH_EXECUTOR_ENABLE_PROFILE", "0"))
)


def execute_maybe_two_batch(
    inputs,
    fn,
    delta_stages: int,
    enable_two_batch_overlap: bool,
    split_inputs,
    merge_outputs,
):
    # TODO maybe optimize these nested `if`s
    if enable_two_batch_overlap:
        splitted_inputs = split_inputs(**inputs)
        if splitted_inputs is not None:
            inputs_a, inputs_b = splitted_inputs
            output_a, output_b = execute_two_batch(
                inputs_a, inputs_b, fn, delta_stages=delta_stages
            )
            return merge_outputs(output_a, output_b)

    return execute_single_batch(inputs, fn)


def execute_single_batch(inputs, fn):
    generator = fn(**inputs)
    while True:
        try:
            next(generator)
        except StopIteration as e:
            return e.value


def execute_two_batch(inputs_a, inputs_b, fn, delta_stages: int):
    generator_a = _WrappedGenerator("a", fn(**inputs_a))
    generator_b = _WrappedGenerator("b", fn(**inputs_b))

    # print('hack: run generator_a to the end, this is NOT OVERLAP')
    # while not generator_a.done:
    #     generator_a.next()
    # print('hack: run generator_b to the end, this is NOT OVERLAP')
    # while not generator_b.done:
    #     generator_b.next()

    for _ in range(delta_stages):
        generator_a.next()

    while not generator_a.done:
        generator_a.next()
        generator_b.next()

    for _ in range(delta_stages):
        generator_b.next()

    assert generator_a.done and generator_b.done
    return generator_a.output, generator_b.output


class _WrappedGenerator:
    def __init__(self, debug_name: str, generator):
        self._debug_name = debug_name
        self._generator = generator
        self._count = 0
        self.output = None

    def next(self):
        assert not self.done

        if _ENABLE_PROFILE:
            ctx = torch.profiler.record_function(f"Gen-{self._debug_name}{self._count}")
        else:
            ctx = null_context()

        try:
            with ctx:
                next(self._generator)
        except StopIteration as e:
            assert e.value is not None
            self.output = e.value

        self._count += 1

    @property
    def done(self):
        return self.output is not None
