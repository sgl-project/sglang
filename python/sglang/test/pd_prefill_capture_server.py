"""P/D handoff faults, source parity and actual prefill graph observations."""

import os
import sys

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test import (  # noqa: F401
    pd_capture_server,
    training_capture_prefill_server,
)

_forward = ModelRunner.forward


def forward(self, forward_batch, *args, **kwargs):
    if not self.is_draft_worker and self.server_args.disaggregation_mode == "decode":
        assert forward_batch.forward_mode.name in ("DECODE", "TARGET_VERIFY", "IDLE"), (
            "D must consume transferred target KV without another target prefill",
            forward_batch.forward_mode,
        )
    return _forward(self, forward_batch, *args, **kwargs)


ModelRunner.forward = forward


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
