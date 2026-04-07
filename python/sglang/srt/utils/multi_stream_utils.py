# Adapted from trtllm.

import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch

from sglang.srt.server_args import get_global_server_args

class do_multi_stream_local(threading.local):

    def __init__(self):
        self.do_multi_stream = False


_local = do_multi_stream_local()


def set_do_multi_stream(enable: bool):
    _local.do_multi_stream = enable


def do_multi_stream() -> bool:
    return _local.do_multi_stream


@contextmanager
def with_multi_stream(enable: bool):
    prev_do_multi_stream = _local.do_multi_stream
    set_do_multi_stream(enable)
    try:
        yield
    finally:
        set_do_multi_stream(prev_do_multi_stream)


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

class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class MultiStreamUtils(metaclass=Singleton):
    def __init__(self):
        if get_global_server_args().enable_longcat_double_stream:
            self.stream_8 = torch.npu.Stream()
            self.stream_16 = torch.npu.Stream()
            torch.npu.set_stream_limit(self.stream_8, 8, 16)
            torch.npu.set_stream_limit(self.stream_16, 16, 32)
            self.main_stream = None
            self.forward_moe_func = None

            self.first_attn_finished = torch.npu.Event()
            self.mlp_attn0_finished = torch.npu.Event()
            self.moe_dispatch_finished = torch.npu.Event()
            self.moe_gemm_finished = torch.npu.Event()