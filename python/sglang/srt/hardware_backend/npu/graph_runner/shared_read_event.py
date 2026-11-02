"""CANN 9.2 graph-to-stream events for the shared metadata read barrier."""

import ctypes
import weakref
from functools import cache

ACL_EVENT_SYNC = 0x01
ACL_EVENT_RECORD_EXTERNAL = 0x01


def _check_acl_error(result, function, args):
    if result:
        raise RuntimeError(f"{function.__name__} failed with ACL error {result}")
    return result


@cache
def _event_api():
    library = ctypes.CDLL("libascendcl.so")
    signatures = {
        "aclrtCreateEventExWithFlag": (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
        ),
        "aclrtRecordEventWithFlag": (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ),
        "aclrtStreamWaitEvent": (ctypes.c_void_p, ctypes.c_void_p),
        "aclrtDestroyEvent": (ctypes.c_void_p,),
    }
    for name, argtypes in signatures.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        function.errcheck = _check_acl_error
    return library


class NPUSharedReadEvent:
    """Record a graph node that the scheduler can wait on outside the graph.

    Unlike torch.npu.ExternalEvent, this event uses CreateEventExWithFlag and
    must not be reset. Only the capture-time record uses the EXTERNAL flag;
    the scheduler consumes it through the ordinary Stream.wait_event protocol.
    """

    def __init__(self, device_module):
        self._device_module = device_module
        self._api = _event_api()
        self._event = ctypes.c_void_p()
        self._api.aclrtCreateEventExWithFlag(ctypes.byref(self._event), ACL_EVENT_SYNC)
        self._finalizer = weakref.finalize(
            self, self._api.aclrtDestroyEvent, self._event
        )
        # The runtime owns process-exit cleanup; it may shut down before Python
        # finalizers. Normal object disposal still releases the event.
        self._finalizer.atexit = False

    def record(self, stream=None):
        if stream is None:
            stream = self._device_module.current_stream()
        # Reading npu_stream drains torch_npu's host submission queue, without
        # waiting for device completion. Do not cache the raw stream handle.
        self._api.aclrtRecordEventWithFlag(
            self._event, stream.npu_stream, ACL_EVENT_RECORD_EXTERNAL
        )

    def wait(self, stream=None):
        if stream is None:
            stream = self._device_module.current_stream()
        self._api.aclrtStreamWaitEvent(stream.npu_stream, self._event)
