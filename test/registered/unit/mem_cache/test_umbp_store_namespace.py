"""UMBPStore anchor-key namespaces, with real CPU host pools and a fake MoRI client."""

import ctypes
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.storage.umbp import umbp_store
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ClientConfig:
    @staticmethod
    def from_environment():
        return SimpleNamespace(
            role=None,
            dram=SimpleNamespace(capacity_bytes=1 << 20),
            ssd=SimpleNamespace(
                enabled=False, ssd_backend="file", storage_dir="unused"
            ),
            distributed=None,
            standalone_process=None,
        )


class _PointerClient:
    """In-memory MoRI client. Stores share one dict, like workers sharing a cache."""

    def __init__(self, objects):
        self.objects = objects

    def is_distributed(self):
        return False

    def flush(self):
        return True

    def batch_put_from_ptr(self, keys, ptrs, sizes):
        result = []
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            if key in self.objects:
                result.append(False)
            else:
                self.objects[key] = ctypes.string_at(ptr, size)
                result.append(True)
        return result

    def batch_get_into_ptr(self, keys, ptrs, sizes):
        result = []
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            value = self.objects.get(key)
            valid = value is not None and len(value) == size
            if valid:
                ctypes.memmove(ptr, value, size)
            result.append(valid)
        return result

    def batch_exists_consecutive(self, keys):
        return next(
            (index for index, key in enumerate(keys) if key not in self.objects),
            len(keys),
        )


class TestUMBPStoreNamespace(CustomTestCase):
    def setUp(self):
        super().setUp()
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("UMBP_")
            and key not in ("SGLANG_DP_RANK", "SGLANG_DP_SIZE", "LOCAL_RANK")
        }
        self.enterContext(mock.patch.dict(os.environ, environment, clear=True))

    def _host(self):
        device = MHATokenToKVPool(
            dtype=torch.bfloat16,
            size=8,
            page_size=1,
            layer_num=2,
            start_layer=0,
            end_layer=2,
            head_num=1,
            head_dim=8,
            v_head_dim=8,
            device="cpu",
            enable_memory_saver=False,
        )
        with host_memory_budget_scope(1 << 20):
            host = MHATokenToKVPoolHost(
                buffer_info=device.get_device_pool_infos()[0].buffer_info,
                host_to_device_ratio=2,
                host_size=0,
                page_size=1,
                layout="page_first",
                pin_memory=False,
                device="cpu",
                device_capacity_tokens=device.size,
                start_layer=0,
                end_layer=2,
                write_back_staging=device.hicache_write_back_staging,
            )
        self.addCleanup(host.destroy)
        return host

    def _store(self, host, objects, *, model="org/model-a", tag="tenant-a"):
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name=model,
            extra_config={
                "ssd_enabled": False,
                "dram_capacity_bytes": 1 << 20,
                "extra_backend_tag": tag,
            },
        )
        client = _PointerClient(objects)
        roles = SimpleNamespace(
            Standalone="standalone",
            SharedSSDLeader="leader",
            SharedSSDFollower="follower",
        )
        boundary = (lambda config: client, _ClientConfig, roles) + (None,) * 5
        with mock.patch.object(
            umbp_store, "_import_umbp_client", return_value=boundary
        ):
            store = umbp_store.UMBPStore(config, host)
        self.addCleanup(store.close)
        return store

    def _assert_page(self, host, index, k, v):
        torch.testing.assert_close(
            host.k_buffer[index], torch.full_like(host.k_buffer[index], k)
        )
        torch.testing.assert_close(
            host.v_buffer[index], torch.full_like(host.v_buffer[index], v)
        )

    def test_different_model_or_tag_does_not_read_other_pages(self):
        for identity in ({"model": "org/model-b"}, {"tag": "tenant-b"}):
            with self.subTest(identity=identity):
                objects = {}
                a_host, b_host, peer_host = self._host(), self._host(), self._host()
                a = self._store(a_host, objects)
                b = self._store(b_host, objects, **identity)
                a_peer = self._store(peer_host, objects)
                a_host.k_buffer.fill_(11)
                a_host.v_buffer.fill_(22)
                b_host.k_buffer.fill_(33)
                b_host.v_buffer.fill_(44)
                indices = torch.tensor([0], dtype=torch.int64)

                self.assertEqual(a.batch_set_v1(["page"], indices), [True])
                self.assertEqual(b.batch_exists(["page"]), 0)
                self.assertEqual(b.batch_get_v1(["page"], indices), [False])
                self._assert_page(b_host, 0, 33, 44)
                self.assertEqual(b.batch_set_v1(["page"], indices), [True])

                for host in (a_host, b_host, peer_host):
                    host.kv_buffer.zero_()
                for store in (a, b, a_peer):
                    self.assertEqual(store.batch_get_v1(["page"], indices), [True])
                self._assert_page(a_host, 0, 11, 22)
                self._assert_page(b_host, 0, 33, 44)
                self._assert_page(peer_host, 0, 11, 22)

    def test_empty_prefix_keeps_legacy_keys(self):
        for model, tag in ((None, None), ("", None), (None, "")):
            with self.subTest(model=model, tag=tag):
                objects = {}
                writer_host, reader_host = self._host(), self._host()
                writer = self._store(writer_host, objects, model=model, tag=tag)
                reader = self._store(reader_host, objects, model=model, tag=tag)
                configured_host = self._host()
                configured = self._store(configured_host, objects)
                writer_host.k_buffer.fill_(11)
                writer_host.v_buffer.fill_(22)
                configured_host.k_buffer.fill_(-7)
                configured_host.v_buffer.fill_(-9)
                indices = torch.tensor([0], dtype=torch.int64)

                self.assertEqual(writer.batch_set_v1(["page"], indices), [True])
                self.assertEqual(set(objects), {"page_0_k", "page_0_v"})
                self.assertEqual(reader.batch_get_v1(["page"], indices), [True])
                self._assert_page(reader_host, 0, 11, 22)
                self.assertEqual(configured.batch_exists(["page"]), 0)
                self.assertEqual(configured.batch_get_v1(["page"], indices), [False])
                self._assert_page(configured_host, 0, -7, -9)

    def test_anchor_keys_up_to_255_bytes_round_trip(self):
        page_hash = "a" * 64
        indices = torch.tensor([0], dtype=torch.int64)
        # prefix + "_" + 64-char hash + "_0_k" is exactly 255 bytes.
        for prefix in ("x" * 186, "é" * 93):
            with self.subTest(prefix_bytes=len(prefix.encode("utf-8"))):
                objects = {}
                host = self._host()
                store = self._store(host, objects, model=prefix, tag=None)
                host.k_buffer.fill_(11)
                host.v_buffer.fill_(22)
                self.assertEqual(store.batch_set_v1([page_hash], indices), [True])
                self.assertEqual(
                    [len(key.encode("utf-8")) for key in objects], [255, 255]
                )
                host.kv_buffer.zero_()
                self.assertEqual(store.batch_get_v1([page_hash], indices), [True])
                self._assert_page(host, 0, 11, 22)

                too_long = self._store(self._host(), {}, model=prefix + "x", tag=None)
                with self.assertRaises(ValueError):
                    too_long.batch_set_v1([page_hash], indices)

    def test_overlong_anchor_batch_is_rejected_before_client_io(self):
        objects = {}
        host = self._host()
        store = self._store(host, objects, model="x" * 187, tag=None)
        host.k_buffer.fill_(-7)
        host.v_buffer.fill_(-9)
        # The first page fits. The second is one byte over the limit.
        keys = ["short", "a" * 64]
        indices = torch.tensor([0, 1], dtype=torch.int64)
        operations = {
            "put": lambda: store.batch_set_v1(keys, indices),
            "put_with_depth": lambda: store.batch_set_v1(
                keys, indices, HiCacheStorageExtraInfo(prefix_keys=["parent"])
            ),
            "get": lambda: store.batch_get_v1(keys, indices),
            "exists": lambda: store.batch_exists(keys),
            "exists_v2": lambda: store.batch_exists_v2(keys, []),
        }
        with mock.patch.object(store, "client", wraps=store.client) as client:
            for name, operation in operations.items():
                with self.subTest(operation=name):
                    with self.assertRaises(ValueError):
                        operation()
                    self.assertEqual(client.mock_calls, [])
        self.assertEqual(objects, {})
        for index in (0, 1):
            self._assert_page(host, index, -7, -9)


if __name__ == "__main__":
    unittest.main()
