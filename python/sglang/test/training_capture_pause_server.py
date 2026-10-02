"""Test-only tokenizer GC observations for the capture latency experiment."""

import os
import sys

from sglang.srt.managers import tokenizer_manager
from sglang.test.training_capture_diagnostics import install_tokenizer_gc_observer

tokenizer_manager.configure_gc_warning = install_tokenizer_gc_observer


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
