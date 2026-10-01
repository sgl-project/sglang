"""Profiler-only CPU scopes for attributing capture kernels and DMA events."""

import os
import sys
from functools import wraps

import torch

from sglang.srt.training_capture import coordinator
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.kv_staging import KVStaging


def scoped(name, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function("training_capture." + name):
            return function(*args, **kwargs)

    return wrapped


coordinator.capture_teacher = scoped("teacher", coordinator.capture_teacher)
SelectedLayerKVExporter.export = scoped("kv", SelectedLayerKVExporter.export)
KVStaging.flush = scoped("kv_d2h", KVStaging.flush)
RequestCaptureContext.record_teacher_range = scoped(
    "teacher_d2h", RequestCaptureContext.record_teacher_range
)
RequestCaptureContext.record_positions = scoped(
    "positions_d2h", RequestCaptureContext.record_positions
)


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
