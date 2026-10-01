"""Observe the actual prefill graph buffers without changing capture behavior."""

import itertools
import os
import sys

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.dspark_capture_observer import install_capture_observer

_load_batch = PrefillCudaGraphRunner.load_batch
_replays = itertools.count()


def load_batch(self, forward_batch, **kwargs):
    static = _load_batch(self, forward_batch, **kwargs)
    forward_batch.training_capture_test_prefill_graph = {
        "replay_id": next(_replays),
        "raw_tokens": forward_batch.input_ids.numel(),
        "padded_tokens": static.input_ids.numel(),
        "input_buffer": static.input_ids.data_ptr(),
        "request_slots": self._capture_req_slots if self._is_full_backend else None,
    }
    return static


PrefillCudaGraphRunner.load_batch = load_batch
install_capture_observer()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
