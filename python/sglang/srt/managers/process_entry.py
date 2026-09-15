"""Lazy entry points for worker processes.

`multiprocessing.Process(target=...)` pickles the target by qualified name, so
the launcher can reference these without importing the scheduler / DP
controller / detokenizer modules (several seconds of imports) itself. The child
(or the preloaded forkserver) resolves the real function on first call.
"""


def run_scheduler_process(*args, **kwargs):
    from sglang.srt.managers.scheduler import run_scheduler_process as fn

    return fn(*args, **kwargs)


def run_data_parallel_controller_process(*args, **kwargs):
    from sglang.srt.managers.data_parallel_controller import (
        run_data_parallel_controller_process as fn,
    )

    return fn(*args, **kwargs)


def run_detokenizer_process(*args, **kwargs):
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process as fn

    return fn(*args, **kwargs)
