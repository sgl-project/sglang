"""Shared host submission boundary for the asynchronous NPU graph probe."""

import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)
_submission_lock = threading.RLock()
_attested_sources = set()


@contextmanager
def npu_graph_submission(device_module, update_stream, *, source):
    # Keep wait -> replay -> update contiguous across model worker threads.
    # Waiting after replay would make the update depend on its own consumer.
    with _submission_lock:
        compute_stream = device_module.current_stream()
        update_stream.wait_stream(compute_stream)
        yield
        if source not in _attested_sources:
            logger.info(
                "NPU async submission completed: source=%s compute=%s update=%s",
                source,
                compute_stream,
                update_stream,
            )
            _attested_sources.add(source)
