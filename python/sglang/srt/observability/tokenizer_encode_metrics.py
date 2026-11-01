"""Wall-clock timing for encode calls, independent of process CPU accounting."""

import time
from contextlib import contextmanager, nullcontext

from sglang.srt.observability.frontend_request_timing import current_frontend_timing


def measure_tokenizer_encode(
    collector, *, source: str, batch_size: int = 1, operation: str = "encode"
):
    """Observe one actual invocation, including a batch, without dividing its time.

    The caller must enter this context inside the execution thread, after any
    queueing, and outside template rendering unless operation="chat_template".
    Missing collectors keep the metrics-disabled path free of clock reads.
    """
    if collector is None:
        return nullcontext()
    return _measure(collector, source, batch_size, operation)


@contextmanager
def _measure(collector, source: str, batch_size: int, operation: str):
    started = time.perf_counter()
    result = "success"
    try:
        yield
    except BaseException:
        result = "error"
        raise
    finally:
        finished = time.perf_counter()
        elapsed = finished - started
        timing = current_frontend_timing.get()
        if timing is not None:
            timing.add_span("text", started, finished)
            # Keep the inseparable template portion visible as a subset of
            # text preparation rather than silently reporting zero encoding.
            if operation == "chat_template":
                timing.add_span("chat_template", started, finished)
        collector.observe_tokenizer_encode(
            elapsed,
            source=source,
            input_type="single" if batch_size == 1 else "batch",
            operation=operation,
            result=result,
        )
