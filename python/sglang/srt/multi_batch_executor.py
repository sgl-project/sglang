import os
from typing import Optional

import torch
from torch._dynamo.eval_frame import null_context

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ENABLE_PROFILE = bool(int(os.environ.get('SGLANG_MULTI_BATCH_EXECUTOR_ENABLE_PROFILE', '0')))

def execute_maybe_two_batch(
        inputs, fn, delta_stages: int, enable_two_batch_overlap: bool,
        split_inputs, merge_outputs,
):
    if enable_two_batch_overlap:
        inputs_a, inputs_b = split_inputs(**inputs)
        output_a, output_b = execute_two_batch(inputs_a, inputs_b, fn, delta_stages=delta_stages)
        return merge_outputs(output_a, output_b)
    else:
        return execute_single_batch(inputs, fn)

def execute_single_batch(inputs, fn):
    generator = fn(**inputs)
    while True:
        try:
            next(generator)
        except StopIteration as e:
            return e.value


def execute_two_batch(inputs_a, inputs_b, fn, delta_stages: int):
    generator_a = _WrappedGenerator('a', fn(**inputs_a))
    generator_b = _WrappedGenerator('b', fn(**inputs_b))

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
            ctx = torch.profiler.record_function(f'Gen-{self._debug_name}{self._count}')
        else:
            ctx = null_context()

        try:
            next(self._generator)
        except StopIteration as e:
            assert e.value is not None
            self.output = e.value

        self._count += 1

    @property
    def done(self):
        return self.output is not None
