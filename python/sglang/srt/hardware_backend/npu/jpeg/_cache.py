"""Own DVPP sessions used by the public JPEG loader."""

import atexit
import os
import threading

from sglang.srt.hardware_backend.npu.dvpp import Dvpp


class _DvppCache:
    """Keep one session per live worker thread, bound to its selected NPU."""

    def __init__(self):
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._sessions = {}
        self._closed = False

    def get(self, device):
        if self._pid != os.getpid():
            raise RuntimeError(
                "NPU JPEG decoders must be initialized after worker spawn"
            )
        thread = threading.current_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("NPU JPEG decoder cache has been closed")
            for owner, session in list(self._sessions.items()):
                if not owner.is_alive() or (
                    owner is thread
                    and (session.closed or session.device_index != device)
                ):
                    session.close()
                    del self._sessions[owner]
            if thread not in self._sessions:
                self._sessions[thread] = Dvpp()
            return self._sessions[thread]

    def close(self):
        """Close every session after workers join; retain failures for retry."""
        if self._pid != os.getpid():
            return
        first_error = None
        with self._lock:
            self._closed = True
            for owner, session in list(self._sessions.items()):
                try:
                    session.close()
                    del self._sessions[owner]
                except Exception as error:
                    if first_error is None:
                        first_error = error
                    else:
                        first_error.add_note(
                            f"Additional DVPP cleanup failure: {error}"
                        )
        if first_error is not None:
            raise first_error


cached_decoders = _DvppCache()
atexit.register(cached_decoders.close)
