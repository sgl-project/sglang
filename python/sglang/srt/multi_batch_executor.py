def execute_single_batch(inputs, fn):
    generator = fn(**inputs)
    while True:
        try:
            next(generator)
        except StopIteration as e:
            return e.value



def execute_two_batch(inputs, fn, delta_stages: int):
    inputs_a, inputs_b = _split_inputs(inputs)
    output_a, output_b = _execute_two_batch_raw(inputs_a, inputs_b, fn, delta_stages)
    return _merge_outputs(output_a, output_b)

def _split_inputs(inputs):
    return TODO

def _merge_outputs(output_a, output_b):
    return TODO


def _execute_two_batch_raw(inputs_a, inputs_b, fn, delta_stages: int):
    output_a = output_b = None

    generator_a = fn(**inputs_a)
    generator_b = fn(**inputs_b)

    for _ in range(delta_stages):
        next(generator_a)

    while output_a is None:
        try:
            next(generator_a)
        except StopIteration as e:
            assert e.value is not None
            output_a = e.value

        next(generator_b)

    for _ in range(delta_stages - 1):
        next(generator_b)

    try:
        next(generator_b)
    except StopIteration as e:
        assert e.value is not None
        output_b = e.value

    assert (output_a is not None) and (output_b is not None)
    return output_a, output_b
