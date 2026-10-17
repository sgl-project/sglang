"""Installs a fake ``mooncake.store`` when the real one is missing (it ships Linux wheels
only); the fake keeps the behaviors the backend relies on: one status per put_batch,
signed-byte buffer exports, and -704 for a missing key on remove."""

import importlib.util
import sys
import types

import numpy as np

OBJECT_NOT_FOUND = -704


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

    def batch_get_buffer(self, keys):
        return [
            np.frombuffer(self.objects[key], dtype=np.int8).copy()
            if key in self.objects
            else None
            for key in keys
        ]

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
