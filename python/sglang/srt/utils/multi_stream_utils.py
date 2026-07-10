# Adapted from trtllm.

from typing import Any, Callable, Optional

import torch

from sglang.srt.runtime_context import get_forward


def set_do_multi_stream(enable: bool):
    get_forward().set("multi_stream", enable)


def do_multi_stream() -> bool:
    return get_forward().multi_stream


def with_multi_stream(enable: bool):
    return get_forward().scoped(multi_stream=enable)


def maybe_execute_in_parallel(
    fn0: Callable,
    fn1: Callable,
    events: list[torch.cuda.Event],
    aux_stream: Optional[torch.cuda.Stream] = None,
) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        events (list[torch.cuda.Event]): cuda events for callables
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    multi_stream = do_multi_stream() and aux_stream is not None

    if multi_stream:
        events[0].record()
        result0 = fn0()

        with torch.cuda.stream(aux_stream):
            events[0].wait()
            result1 = fn1()
            events[1].record()
        events[1].wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)
