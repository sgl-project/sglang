def execute_single_batch(inputs, fn):
    generator = fn(**inputs)
    while True:
        try:
            next(generator)
        except StopIteration as e:
            return e.value


def execute_two_batch(inputs, fn):
    TODO
