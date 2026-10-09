"""Test-only tokenizer GC observations for the capture latency experiment."""

import os
import sys

from sglang.srt.managers import tokenizer_manager
from sglang.test.training_capture_diagnostics import install_tokenizer_gc_observer

tokenizer_manager.configure_gc_warning = install_tokenizer_gc_observer


def install_lifecycle_probe():
    """Observe request lifetime only in the explicitly enabled experiment."""
    import gc
    import weakref
    from collections import deque
    from functools import wraps

    import psutil

    from sglang.srt.entrypoints.http_server import app, get_global_state
    from sglang.srt.utils import gc_control

    references = deque(maxlen=16384)
    created = 0
    original_init = tokenizer_manager.ReqState.__init__

    @wraps(original_init)
    def tracked_init(self, *args, **kwargs):
        nonlocal created
        original_init(self, *args, **kwargs)
        references.append(weakref.ref(self))
        created += 1

    tokenizer_manager.ReqState.__init__ = tracked_init

    @app.post("/test_training_capture_gc_state")
    async def snapshot(collect: bool = False):
        collected = None
        cycle_collected = None
        if collect:

            class Cycle:
                pass

            cycle = Cycle()
            cycle.link = cycle
            reference = weakref.ref(cycle)
            del cycle
            collected = gc.collect()
            cycle_collected = reference() is None
        with gc_control._lock:
            policy = {
                "serving_freeze_requested": gc_control._permanent,
                "temporary_scopes": gc_control._temporary_depth,
            }
        return {
            "pid": os.getpid(),
            "rss_bytes": psutil.Process().memory_info().rss,
            "enabled": gc.isenabled(),
            "thresholds": gc.get_threshold(),
            "counts": gc.get_count(),
            "stats": gc.get_stats(),
            "freeze_count": gc.get_freeze_count(),
            "policy": policy,
            "active_requests": len(get_global_state().tokenizer_manager.rid_to_state),
            "created_request_states": created,
            "retained_weakrefs": len(references),
            "dropped_weakrefs": max(0, created - len(references)),
            "live_request_states": sum(ref() is not None for ref in references),
            "collected": collected,
            "new_cycle_collected": cycle_collected,
        }


if os.environ.get("SGLANG_TEST_CAPTURE_GC_LIFECYCLE") == "1":
    install_lifecycle_probe()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
