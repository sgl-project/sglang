# SPDX-License-Identifier: Apache-2.0
"""An "rdma" Mooncake store must start on RDMA in a PD worker that forces its
KV transfer engine onto multi-node NVLink with MC_FORCE_MNNVL."""

import os
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import mooncake_transfer_engine
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FORCE = "MC_FORCE_MNNVL"
IB_DEVICE = "rdma_dev0"
PD_ENGINE = object()


class _FakeStore:
    """MooncakeDistributedStore stand-in that records its setup() call."""

    instances = []

    def __init__(self):
        self.objects = {}
        type(self).instances.append(self)

    def setup(self, *args, **kwargs):
        self.protocol, self.transfer_engine = args[4], args[7]
        self.force_at_setup = os.environ.get(FORCE)
        return 0

    def register_buffer(self, *args, **kwargs):
        return 0

    def put(self, key, value, *args):
        self.objects[key] = value
        return 0

    def is_exist(self, key):
        return 1 if key in self.objects else 0

    def get(self, key):
        return self.objects.get(key)


def _fake_mooncake_modules():
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = _FakeStore
    mooncake_store.ReplicateConfig = type("ReplicateConfig", (), {})
    return {"mooncake": types.ModuleType("mooncake"), "mooncake.store": mooncake_store}


def _pd_engine():
    """The PD worker's shared transfer engine, started under MC_FORCE_MNNVL."""
    engine = object.__new__(mooncake_transfer_engine.MooncakeTransferEngine)
    engine.ib_device = IB_DEVICE
    engine.session_id = "127.0.0.1:17777"
    engine.engine = SimpleNamespace(get_engine=lambda: PD_ENGINE)
    return engine


def _storage_config():
    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        tp_lcm_size=None,
        should_split_heads=False,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "rdma",
            "device_name": IB_DEVICE,
            "check_server": False,
            "global_segment_size": 1024 * 1024,
        },
        unified_suffixes=None,
        unified_chunk_spans=None,
    )


def _start_store(disaggregation_mode):
    with (
        temp_set_env(MC_FORCE_MNNVL="1"),
        patch.dict("sys.modules", _fake_mooncake_modules()),
        patch.object(
            mooncake_transfer_engine, "_mooncake_transfer_engine", _pd_engine()
        ),
        published_topology(
            disaggregation_mode=disaggregation_mode,
            disaggregation_transfer_backend="mooncake",
        ),
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        MooncakeStore(_storage_config())
        # The PD engine reads the force again on any later engine start.
        force_after = os.environ.get(FORCE)
    return _FakeStore.instances[-1], force_after


class TestMooncakeStorePdMnnvl(CustomTestCase):
    def test_pd_store_starts_its_own_engine_without_the_force(self):
        # Mooncake reads MC_FORCE_MNNVL process-wide, so a store set up under it,
        # or on the PD worker's NVLink engine, fails every put to RDMA segments.
        store, force_after = _start_store("prefill")
        self.assertEqual(store.protocol, "rdma")
        self.assertIsNone(store.force_at_setup)
        self.assertIsNone(store.transfer_engine)
        self.assertEqual(force_after, "1")

    def test_store_outside_pd_keeps_the_force(self):
        # Without PD the variable is the store's own setting.
        store, _ = _start_store("null")
        self.assertEqual(store.force_at_setup, "1")


if __name__ == "__main__":
    unittest.main()
