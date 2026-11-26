"""Unit tests for the EGM-backed Mooncake Store pool (enable_egm_store_pool)."""

import contextlib
import json
import os
import tempfile
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

    def test_numa_probe_failure_falls_back_to_auto(self):
        """A missing cuda-bindings package or a driver that rejects the
        attribute must not abort setup; the segment falls back to "auto"."""
        fake_torch_cuda = SimpleNamespace(current_device=lambda: 0)
        cases = {
            "no cuda-bindings": {"cuda.bindings": None},
            "driver rejects the attribute": {
                "cuda": types.ModuleType("cuda"),
                "cuda.bindings": types.ModuleType("cuda.bindings"),
                "cuda.bindings.driver": SimpleNamespace(
                    CUdevice_attribute=SimpleNamespace(),  # no HOST_NUMA_ID
                ),
            },
        }
        for name, modules in cases.items():
            with (
                self.subTest(case=name),
                patch.dict("sys.modules", modules),
                patch(f"{_MODULE}.torch.cuda", fake_torch_cuda),
            ):
                store = _build_store(_EgmMooncakeStore, _make_storage_config())
                self.assertEqual(store.setup_calls[0][0][0]["egm_numa_nodes"], "auto")

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
            ({"enable_egm_store_pool": "yes"}, None, True, None),
            ({"enable_egm_store_pool": " true "}, None, True, None),
            ({}, "yes", True, None),
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

    def test_env_and_extra_config_accept_the_same_spellings(self):
        """The env var and the extra_config key go through the same EnvBool
        parser, so "yes" enables the pool from either source."""
        with (
            envs.MOONCAKE_MASTER.override("127.0.0.1:50051"),
            envs.MOONCAKE_ENABLE_EGM_STORE_POOL.override("yes"),
        ):
            self.assertTrue(MooncakeStoreConfig.load_from_env().enable_egm_store_pool)
        cfg = MooncakeStoreConfig.load_from_extra_config(
            {**self._BASE, "enable_egm_store_pool": "yes"}
        )
        self.assertTrue(cfg.enable_egm_store_pool)


def _storage_config_with(extra_config):
    cfg = _make_storage_config()
    cfg.extra_config = extra_config
    return cfg


def _is_egm_setup(store):
    (args, kwargs) = store.setup_calls[0]
    return len(args) == 1 and args[0].get("enable_egm_store_pool") == "true"


class TestMooncakeEgmKeysOverrideEndpointSource(CustomTestCase):
    """The EGM keys in extra_config configure this rank's own segment, so they
    apply even when the master endpoint comes from the env or a config file
    (extra_config then carries no server address)."""

    def test_extra_config_enables_egm_with_env_endpoint(self):
        with (
            envs.MOONCAKE_MASTER.override("127.0.0.1:50051"),
            envs.MOONCAKE_PROTOCOL.override("nvlink"),
            envs.MOONCAKE_CHECK_SERVER.override(False),
        ):
            store = _build_store(
                _EgmMooncakeStore,
                _storage_config_with(
                    {"enable_egm_store_pool": True, "egm_numa_nodes": "1"}
                ),
            )
        self.assertTrue(_is_egm_setup(store))
        self.assertEqual(store.setup_calls[0][0][0]["egm_numa_nodes"], "1")

    def test_extra_config_disables_egm_from_config_file(self):
        file_config = {
            "master_server_address": "127.0.0.1:50051",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "protocol": "nvlink",
            "check_server": False,
            "enable_egm_store_pool": True,
            "egm_numa_nodes": "0",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(file_config, f)
        self.addCleanup(os.remove, f.name)
        with envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(f.name):
            from_file = _build_store(_EgmMooncakeStore, _storage_config_with({}))
            overridden = _build_store(
                _EgmMooncakeStore,
                _storage_config_with({"enable_egm_store_pool": False}),
            )
        self.assertTrue(_is_egm_setup(from_file))
        self.assertFalse(_is_egm_setup(overridden))


class TestMooncakeEmbeddingStoreRejectsEgm(CustomTestCase):
    def test_embedding_store_refuses_egm_before_mounting_a_segment(self):
        """The embedding store mounts its segment through the positional setup(),
        which builds an ordinary pool that NVLink peers cannot read."""
        _EgmMooncakeStore.instances = []
        with (
            patch.dict("sys.modules", _fake_mooncake_modules(_EgmMooncakeStore)),
            envs.MOONCAKE_MASTER.override("127.0.0.1:50051"),
            envs.MOONCAKE_PROTOCOL.override("nvlink"),
            envs.MOONCAKE_ENABLE_EGM_STORE_POOL.override(True),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
                MooncakeEmbeddingStore,
            )

            with self.assertRaisesRegex(ValueError, "enable_egm_store_pool"):
                MooncakeEmbeddingStore()
        self.assertEqual(_EgmMooncakeStore.instances[-1].setup_calls, [])


class TestSharedEngineOnlyForRdmaHiCache(CustomTestCase):
    """MooncakeStore reuses the shared transfer engine only with protocol "rdma",
    so startup must not create one that HiCache alone would never use."""

    _TE = "sglang.srt.distributed.device_communicators.mooncake_transfer_engine"

    def _engine_created(self, *, protocol, disaggregation_mode="null"):
        from sglang.srt.distributed.device_communicators import (
            mooncake_transfer_engine as te,
        )

        memory = SimpleNamespace(
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config=json.dumps(
                {"master_server_address": "127.0.0.1:50051", "protocol": protocol}
            ),
        )
        disagg = SimpleNamespace(
            disaggregation_mode=disaggregation_mode,
            disaggregation_transfer_backend="mooncake",
            disaggregation_ib_device=None,
            encoder_only=False,
            language_only=False,
            encoder_transfer_backend="mooncake",
        )
        exec_ = SimpleNamespace(
            moe=SimpleNamespace(
                enable_elastic_expert_backup=False,
                elastic_ep_backend=None,
                mooncake_ib_device=None,
            )
        )
        with (
            patch.object(te, "get_memory", return_value=memory),
            patch.object(te, "get_disagg", return_value=disagg),
            patch.object(te, "get_exec", return_value=exec_),
            patch.object(te, "get_local_ip_auto", return_value="127.0.0.1"),
            patch.object(te, "init_mooncake_transfer_engine") as init,
        ):
            te.maybe_init_shared_mooncake_transfer_engine(gpu_id=0)
        return init.called

    def test_hicache_only_engine_follows_the_store_protocol(self):
        self.assertTrue(self._engine_created(protocol="rdma"))
        self.assertFalse(self._engine_created(protocol="nvlink"))

    def test_pd_transfer_still_gets_the_engine(self):
        self.assertTrue(
            self._engine_created(protocol="nvlink", disaggregation_mode="prefill")
        )


if __name__ == "__main__":
    unittest.main()
