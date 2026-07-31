"""ACL host-callback support for sparsity-driven KV offload."""

import atexit
import threading

import acl
import torch

_REPORTERS = {}


def _get_stream_ptr(stream) -> int:
    for attr in ("npu_stream", "stream_ptr", "cuda_stream"):
        if hasattr(stream, attr):
            value = getattr(stream, attr)
            value = value() if callable(value) else value
            return int(value)
    raise RuntimeError("cannot get raw NPU stream ptr")


class _AclReportThread:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread_id = None
        self.streams = set()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.ready.wait()
        atexit.register(self.close)

    def _loop(self):
        torch.npu.set_device(self.device_index)
        try:
            acl.rt.set_device(self.device_index)
        except Exception:
            pass

        self.thread_id = threading.current_thread().ident
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
                pass
        self.streams.clear()
        self.stop.set()
        self.thread.join(timeout=1)


def register_npu_host_callback_stream(stream, device):
    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()

    reporter = _REPORTERS.get(device_index)
    if reporter is None:
        reporter = _AclReportThread(device_index)
        _REPORTERS[device_index] = reporter

    reporter.subscribe(stream)
