import logging
from contextlib import contextmanager
from functools import wraps

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_NVTX_ENV_ENABLED = envs.SGLANG_SCHEDULER_ENABLE_NVTX.get()
_nvtx_module = None
if _NVTX_ENV_ENABLED:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        logger.warning(
            "SGLANG_SCHEDULER_ENABLE_NVTX is set, but the `nvtx` package is missing. "
            "Scheduler NVTX annotations are disabled."
        )

NVTX_ENABLED = _nvtx_module is not None

_NVTX_COLOR_MAP = {
    # Scheduler overlap pipeline
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.launch_batch_sample_if_needed": "yellow",
    "scheduler.process_batch_result": "cyan",
    "scheduler.prepare_mlp_sync_batch": "orange",
    # Disaggregated decode pipeline
    "scheduler.disagg_decode.event_loop_normal": "blue",
    "scheduler.disagg_decode.event_loop_overlap": "blue",
    "scheduler.disagg_decode.prepare_idle_batch": "yellow",
    "scheduler.disagg_decode.get_next_batch": "green",
    "scheduler.disagg_decode.get_new_prebuilt_batch": "green",
    "scheduler.disagg_decode.process_decode_queue": "purple",
    # CUDA graph runner
    "cuda_graph_runner.replay_prepare": "green",
    "cuda_graph_runner.replay": "blue",
    # TP worker helpers
    "tp_worker.batch_init_new": "green",
}


@contextmanager
def nvtx_range(debug_name: str):
    if _nvtx_module is None:
        yield
        return

    color = _NVTX_COLOR_MAP.get(debug_name)
    with torch.autograd.profiler.record_function(debug_name):
        with _nvtx_module.annotate(debug_name, color=color):
            yield


def nvtx_annotated_method(debug_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            range_func = getattr(self, "_nvtx_range", None)
            if range_func is None:
                ctx = nvtx_range(debug_name)
            else:
                ctx = range_func(debug_name)

            with ctx:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
