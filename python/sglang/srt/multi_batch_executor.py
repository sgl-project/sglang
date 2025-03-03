from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def execute_single_batch(inputs, fn):
    generator = fn(**inputs)
    while True:
        try:
            next(generator)
        except StopIteration as e:
            return e.value


def execute_two_batch(inputs_a, inputs_b, fn, delta_stages: int):
    generator_a = _WrappedGenerator(fn(**inputs_a))
    generator_b = _WrappedGenerator(fn(**inputs_b))

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
    def __init__(self, generator):
        self._generator = generator
        self.output = None

    def next(self):
        assert not self.done
        try:
            next(self._generator)
        except StopIteration as e:
            assert e.value is not None
            self.output = e.value

    @property
    def done(self):
        return self.output is not None
