"""ACL host-callback support for sparsity-driven KV offload."""

import atexit
import logging
import threading

import acl
import torch

_REPORTERS = {}
logger = logging.getLogger(__name__)


def _get_stream_ptr(stream) -> int:
    for attr in ("npu_stream", "stream_ptr", "cuda_stream"):
        if hasattr(stream, attr):
            value = getattr(stream, attr)
            value = value() if callable(value) else value
            return int(value)
    raise RuntimeError("cannot get raw NPU stream ptr")


class _AclReportThread:
    def __init__(self, device_index: int, context: int):
        self.device_index = device_index
        self.context = context
        self.startup_error = None
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread_id = None
        self.streams = set()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=30):
            self.stop.set()
            raise RuntimeError("ACL callback reporter initialization timed out")
        if self.startup_error is not None:
            raise RuntimeError(
                "ACL callback reporter initialization failed"
            ) from self.startup_error
        atexit.register(self.close)

    def _loop(self):
        try:
            # The stream belongs to the caller's existing context. Initializing
            # another device context here can fail and strand the main thread
            # at ready.wait(), especially during large host-pool startup.
            ret = acl.rt.set_context(self.context)
            if ret != 0:
                raise RuntimeError(f"acl.rt.set_context failed, ret={ret}")
            self.thread_id = threading.current_thread().ident
        except Exception as error:
            self.startup_error = error
            logger.exception("ACL callback reporter failed to initialize")
            return
        finally:
            self.ready.set()

        while not self.stop.is_set():
            acl.rt.process_report(100)

    def subscribe(self, stream):
        stream_ptr = _get_stream_ptr(stream)
        if stream_ptr in self.streams:
            return

        ret = acl.rt.subscribe_report(self.thread_id, stream_ptr)
        if ret != 0:
            raise RuntimeError(
                f"acl.rt.subscribe_report failed, ret={ret}, "
                f"thread_id={self.thread_id}, stream_ptr={stream_ptr}"
            )
        self.streams.add(stream_ptr)

    def close(self):
        for stream_ptr in list(self.streams):
            try:
                acl.rt.unsubscribe_report(self.thread_id, stream_ptr)
            except Exception:
                logger.debug("ACL callback unsubscribe failed", exc_info=True)
        self.streams.clear()
        self.stop.set()
        self.thread.join(timeout=1)


def register_npu_host_callback_stream(stream, device):
    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()

    context, ret = acl.rt.get_context()
    if ret != 0 or context is None:
        raise RuntimeError(f"acl.rt.get_context failed, ret={ret}")
    key = (device_index, context)
    reporter = _REPORTERS.get(key)
    if reporter is None:
        reporter = _AclReportThread(device_index, context)
        _REPORTERS[key] = reporter

    reporter.subscribe(stream)
