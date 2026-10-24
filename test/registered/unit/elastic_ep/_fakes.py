"""Shared fakes for the elastic EP scale-down unit tests."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0, suite="base-a-test-cpu", disabled="elastic EP scale-down test fakes"
)


class RecordingDriver:
    """Driver stub: ``on_*`` hooks and ``announce_departure`` only record their name."""

    def __init__(self, *, idle=True, barrier_ready=True, departure_ready=True):
        self.calls = []
        self.idle = idle
        self.barrier_ready = barrier_ready
        self.departure_ready = departure_ready
        self.consumed = []
        self._handle_seq = 0

    def __getattr__(self, name):
        if name.startswith("on_") or name == "announce_departure":
            return lambda sm: self.calls.append(name)
        raise AttributeError(name)

    def _new_handle(self, tag):
        self._handle_seq += 1
        return f"{tag}-{self._handle_seq}"

    def local_idle(self, sm):
        self.calls.append("local_idle")
        return self.idle

    def post_drain_barrier(self, sm):
        self.calls.append("post_drain_barrier")
        return self._new_handle("drain")

    def post_nixl_retire_barrier(self, sm):
        self.calls.append("post_nixl_retire_barrier")
        return self._new_handle("nixl")

    def departure_cleared(self, sm):
        return self.departure_ready

    def check_barrier(self, handle, *, block_s=None, keep_serving=False):
        return self.barrier_ready

    def consume_barrier(self, handle):
        self.consumed.append(handle)
