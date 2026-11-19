"""Unit tests for the EGM-backed Mooncake Store pool (enable_egm_store_pool)."""

import contextlib
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStoreConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_MODULE = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store"


class _FakeStore:
    instances = []

    def __init__(self):
        self.setup_calls = []
        self.objects = {}
        type(self).instances.append(self)

    def setup(self, *args, **kwargs):
        self.setup_calls.append((args, kwargs))
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


class _EgmMooncakeStore(_FakeStore):
    instances = []

    def setup(self, *args, **kwargs):
        """setup(self, config: dict) -> int

        enable_egm_store_pool: Use EGM DRAM for the global Store pool.
        """
        return super().setup(*args, **kwargs)


class _PreEgmMooncakeStore(_FakeStore):
    instances = []

    def setup(self, *args, **kwargs):
        """setup(self, config: dict) -> int"""
        return super().setup(*args, **kwargs)


def _fake_mooncake_modules(store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = store_cls
    mooncake_store.ReplicateConfig = type("ReplicateConfig", (), {})
    return {"mooncake": mooncake, "mooncake.store": mooncake_store}


def _make_storage_config(**extra):
    return SimpleNamespace(
        tp_rank=0,
        tp_size=2,
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
        unified_suffixes=None,
        unified_chunk_spans=None,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "check_server": False,
            "global_segment_size": "4gb",
            "protocol": "nvlink",
            "enable_egm_store_pool": True,
            **extra,
        },
    )


def _build_store(store_cls, storage_config):
    store_cls.instances = []
    with patch.dict("sys.modules", _fake_mooncake_modules(store_cls)):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        with envs.MOONCAKE_LOCAL_HOSTNAME.override("127.0.0.2:12345"):
            MooncakeStore(storage_config)
    return store_cls.instances[-1]


class TestMooncakeEgmStorePoolSetup(CustomTestCase):
    def test_egm_pool_is_set_up_through_the_config_dict(self):
        """The EGM keys exist only in Mooncake's setup(config) overload, with the
        key names and string values of Mooncake's ConfigDict; the positional
        overload would build an ordinary pool that NVLink peers cannot read."""
        store = _build_store(
            _EgmMooncakeStore,
            _make_storage_config(
                egm_numa_nodes="1", tenant_id="tenant-a", local_buffer_size=33554432
            ),
        )

        self.assertEqual(len(store.setup_calls), 1)
        args, kwargs = store.setup_calls[0]
        self.assertEqual(kwargs, {})
        self.assertEqual(
            args,
            (
                {
                    "local_hostname": "127.0.0.2:12345",
                    "metadata_server": "http://127.0.0.1:8080/metadata",
                    # 4 GiB split over tp_size=2 ranks.
                    "global_segment_size": "2147483648",
                    "local_buffer_size": "33554432",
                    "protocol": "nvlink",
                    "rdma_devices": "",
                    "master_server_addr": "127.0.0.1:50051",
                    "enable_egm_store_pool": "true",
                    "egm_numa_nodes": "1",
                    "tenant_id": "tenant-a",
                },
            ),
        )

    def test_egm_numa_nodes_default_to_the_rank_gpu_node(self):
        """Unset egm_numa_nodes places the rank's segment on its own GPU's host
        NUMA node, and falls back to Mooncake's "auto" when that is unknown."""
        for gpu_node, expected in ((1, "1"), (None, "auto")):
            with (
                self.subTest(gpu_node=gpu_node),
                patch(f"{_MODULE}._current_gpu_host_numa_node", return_value=gpu_node),
            ):
                store = _build_store(_EgmMooncakeStore, _make_storage_config())
                self.assertEqual(store.setup_calls[0][0][0]["egm_numa_nodes"], expected)

    def test_egm_pool_requires_mooncake_support(self):
        """A Mooncake without EGM support ignores unknown setup keys, so SGLang
        must refuse instead of silently building an ordinary pool."""
        with self.assertRaisesRegex(RuntimeError, "enable_egm_store_pool"):
            _build_store(_PreEgmMooncakeStore, _make_storage_config())
        self.assertEqual(_PreEgmMooncakeStore.instances[-1].setup_calls, [])


class TestMooncakeEgmStorePoolConfig(CustomTestCase):
    _BASE = {"master_server_address": "127.0.0.1:50051"}

    def test_enable_flag_and_numa_nodes_parsing(self):
        cases = [
            ({"enable_egm_store_pool": "false"}, None, False, None),
            ({"enable_egm_store_pool": "true"}, None, True, None),
            ({}, True, True, None),
            ({"enable_egm_store_pool": False}, True, False, None),
            ({"egm_numa_nodes": [0, 1]}, None, False, "0,1"),
            ({"egm_numa_nodes": " auto "}, None, False, "auto"),
        ]
        for extra, env_value, expected_enabled, expected_nodes in cases:
            with self.subTest(extra=extra, env_value=env_value):
                env = (
                    contextlib.nullcontext()
                    if env_value is None
                    else envs.MOONCAKE_ENABLE_EGM_STORE_POOL.override(env_value)
                )
                with env:
                    cfg = MooncakeStoreConfig.load_from_extra_config(
                        {**self._BASE, **extra}
                    )
                self.assertEqual(cfg.enable_egm_store_pool, expected_enabled)
                self.assertEqual(cfg.egm_numa_nodes, expected_nodes)


if __name__ == "__main__":
    unittest.main()
