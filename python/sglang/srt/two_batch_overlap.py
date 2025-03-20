import os

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch._dynamo.eval_frame import null_context


# TODO refactor
def compute_middle_split_token_and_seq_index(self):
    num_tokens = self.input_ids.shape[0]

    if self.forward_mode.is_extend():
        split_token_index, split_seq_index = 0, 0
        for extend_seq_len in self.extend_lens[:-1]:
            split_token_index += extend_seq_len
            split_seq_index += 1
            if split_token_index >= num_tokens // 2:
                break
    elif self.forward_mode.is_decode():
        split_token_index = split_seq_index = num_tokens // 2
    else:
        raise NotImplementedError

    # print(
    #     f'[TP{get_tensor_model_parallel_rank()}] compute_middle_split_token_and_seq_index {num_tokens=} {split_token_index=} {self.input_ids.tolist()=} {self.forward_mode=}')
    if split_token_index == 0 or split_token_index == num_tokens:
        return -1, -1

    return split_token_index, split_seq_index


def compute_split_seq_index() -> int:
    return TODO


def compute_split_token_index(split_seq_index: int) -> int:
    return TODO


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
