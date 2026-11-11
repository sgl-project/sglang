"""Helpers for startup overlap engine tests."""

import os
import sys
import tempfile
import threading
from contextlib import contextmanager


@contextmanager
def capture_worker_stderr():
    """Keep worker logs visible while retaining them for startup assertions."""
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as logs:
        sys.stderr.flush()
        original_stderr = os.dup(2)
        stopped = threading.Event()

        def forward_logs():
            # A separate cursor must not seek the workers' inherited descriptor.
            with open(logs.name, "rb") as source:
                while True:
                    chunk = source.read(65536)
                    if chunk:
                        remaining = memoryview(chunk)
                        while remaining:
                            remaining = remaining[
                                os.write(original_stderr, remaining) :
                            ]
                    elif stopped.is_set():
                        return
                    else:
                        stopped.wait(0.05)

        forwarder = threading.Thread(target=forward_logs, daemon=True)
        forwarder.start()
        try:
            os.dup2(logs.fileno(), 2)
            yield logs
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr, 2)
            stopped.set()
            forwarder.join()
            os.close(original_stderr)
