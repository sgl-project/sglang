"""Installs a fake ``mooncake.store`` when the real one is missing (it ships Linux wheels
only); the fake keeps the behaviors the backend relies on: one status per put_batch,
get_into's codes, and -704 for a missing key on remove."""

import ctypes
import importlib.util
import sys
import types

OBJECT_NOT_FOUND = -704
BUFFER_TOO_SMALL = -600
TRANSFER_FAIL = -800


class FakeReplicateConfig:
    def __init__(self):
        self.replica_num = 1
        self.with_hard_pin = False


class FakeMooncakeStore:
    instances: list["FakeMooncakeStore"] = []

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.setup_config = None
        self.put_calls: list[tuple[list[str], object]] = []
        self.remove_calls: list[tuple[list[str], bool]] = []
        self.get_calls: list[str] = []
        # Registered buffers: start address -> size.
        self.registered: dict[int, int] = {}
        # Statuses for the next put_batch calls; each may keep the first N keys stored.
        self.put_script: list[tuple[int, int]] = []
        self.closed = False
        FakeMooncakeStore.instances.append(self)

    def setup(self, config):
        self.setup_config = config
        return 0

    def put_batch(self, keys, values, config):
        self.put_calls.append((list(keys), config))
        status, kept = self.put_script.pop(0) if self.put_script else (0, len(keys))
        for key, value in list(zip(keys, values))[:kept]:
            self.objects[key] = bytes(value)
        return status

    def register_buffer(self, ptr, size):
        self.registered[ptr] = size
        return 0

    def unregister_buffer(self, ptr):
        return 0 if self.registered.pop(ptr, None) is not None else -1

    def get_into(self, key, ptr, size):
        self.get_calls.append(key)
        # Like RDMA: a destination outside every registered buffer fails the transfer.
        if not any(
            b <= ptr and ptr + size <= b + n for b, n in self.registered.items()
        ):
            return TRANSFER_FAIL
        if key not in self.objects:
            return OBJECT_NOT_FOUND
        data = self.objects[key]
        if len(data) > size:
            return BUFFER_TOO_SMALL
        ctypes.memmove(ptr, data, len(data))
        return len(data)

    def batch_remove(self, keys, force=False):
        self.remove_calls.append((list(keys), force))
        return [
            0 if self.objects.pop(key, None) is not None else OBJECT_NOT_FOUND
            for key in keys
        ]

    def close(self):
        self.closed = True


USING_FAKE_MOONCAKE = importlib.util.find_spec("mooncake") is None

if USING_FAKE_MOONCAKE:
    mooncake = types.ModuleType("mooncake")
    store = types.ModuleType("mooncake.store")
    store.MooncakeDistributedStore = FakeMooncakeStore
    store.ReplicateConfig = FakeReplicateConfig
    mooncake.store = store
    sys.modules["mooncake"] = mooncake
    sys.modules["mooncake.store"] = store
