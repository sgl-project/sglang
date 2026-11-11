"""UMBP anchor isolation with real CPU pools and a shared external-client fixture.

Two models with equal KV geometry must not accept each other's cached bytes.
Only the MoRI client/configuration boundary is substituted; the protocol-only
layout cases below use a small host descriptor instead of claiming GPU coverage.
"""

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
    """External store contract: shared keys and exact-size pointer transfers."""

    def __init__(self, objects):
        self.objects = objects
        self.depths = {}

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

    def batch_put_from_ptr_with_depth(self, keys, ptrs, sizes, depths):
        self.depths.update(zip(keys, depths, strict=True))
        return self.batch_put_from_ptr(keys, ptrs, sizes)

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


class _ProtocolPool:
    """Pointer-order fixture only; it does not emulate model transfer kernels."""

    page_size = 1

    def __init__(self, objects_per_page, *, envelope=False, split=False):
        self.layout = "page_head" if split else "page_first"
        self.stores_page_envelope = envelope
        self.kv_buffer = torch.empty((4, objects_per_page, 8), dtype=torch.float32)

    def get_page_buffer_meta(self, indices):
        buffers = [row for index in indices.tolist() for row in self.kv_buffer[index]]
        return [row.data_ptr() for row in buffers], [32] * len(buffers)

    def get_split_heads_page_buffer_meta(self, indices, split_factor):
        assert self.kv_buffer.shape[1] == split_factor * 2
        return self.get_page_buffer_meta(indices)


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

    def _store(
        self, host, objects, *, model="org/model-a", tag="tenant-a", rank=0, **kwargs
    ):
        config = HiCacheStorageConfig(
            tp_rank=rank,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=kwargs.pop("is_mla_model", False),
            enable_storage_metrics=False,
            is_page_first_layout=host.layout == "page_first",
            model_name=model,
            extra_config={
                "ssd_enabled": False,
                "dram_capacity_bytes": 1 << 20,
                "extra_backend_tag": tag,
            },
            **kwargs,
        )
        client = _PointerClient(objects)
        boundary = (
            lambda config: client,
            _ClientConfig,
            SimpleNamespace(
                Standalone="standalone",
                SharedSSDLeader="leader",
                SharedSSDFollower="follower",
            ),
            None,
            None,
            None,
            None,
            None,
        )
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

    def test_models_and_tags_isolate_equal_sized_kv(self):
        for identity in ({"model": "org/model-b"}, {"tag": "tenant-b"}):
            with self.subTest(identity=identity):
                objects = {}
                first_host, second_host, reader_host = (
                    self._host(),
                    self._host(),
                    self._host(),
                )
                first = self._store(first_host, objects)
                second = self._store(second_host, objects, **identity)
                same_namespace = self._store(reader_host, objects)
                first_host.k_buffer.fill_(11)
                first_host.v_buffer.fill_(22)
                second_host.k_buffer.fill_(33)
                second_host.v_buffer.fill_(44)
                indices = torch.tensor([0], dtype=torch.int64)
                self.assertEqual(first.batch_set_v1(["page"], indices), [True])
                read_result = second.batch_get_v1(["page"], indices)
                self._assert_page(second_host, 0, 33, 44)
                self.assertEqual(read_result, [False])
                self.assertEqual(second.batch_exists(["page"]), 0)
                self.assertEqual(second.batch_set_v1(["page"], indices), [True])

                # Both namespaces retain their own bytes; matching peers can share.
                first_host.kv_buffer.zero_()
                second_host.kv_buffer.zero_()
                reader_host.kv_buffer.zero_()
                for store in (first, second, same_namespace):
                    self.assertEqual(store.batch_exists(["page"]), 1)
                    self.assertEqual(store.batch_get_v1(["page"], indices), [True])
                self._assert_page(first_host, 0, 11, 22)
                self._assert_page(second_host, 0, 33, 44)
                self._assert_page(reader_host, 0, 11, 22)
                other_rank = self._store(self._host(), objects, rank=1)
                self.assertEqual(other_rank.batch_exists(["page"]), 0)
                self.assertEqual(first.batch_exists(["different-page"]), 0)

    def test_unnamespaced_compatibility_does_not_enable_legacy_fallback(self):
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
                self.assertEqual(reader.batch_exists(["page"]), 1)
                self.assertEqual(reader.batch_exists([]), 0)
                self.assertEqual(reader.batch_get_v1(["page"], indices), [True])
                self._assert_page(reader_host, 0, 11, 22)
                self.assertEqual(configured.batch_exists(["page"]), 0)
                self.assertEqual(configured.batch_get_v1(["page"], indices), [False])
                self._assert_page(configured_host, 0, -7, -9)

    def test_depth_aware_writes_and_prefix_queries_use_the_same_namespace(self):
        objects = {}
        host = self._host()
        store = self._store(host, objects)
        host.k_buffer.fill_(11)
        host.v_buffer.fill_(22)
        indices = torch.tensor([0, 1], dtype=torch.int64)
        self.assertEqual(
            store.batch_set_v1(
                ["a", "b"],
                indices,
                HiCacheStorageExtraInfo(prefix_keys=["parent-0", "parent-1"]),
            ),
            [True, True],
        )
        self.assertEqual(
            store.client.depths,
            {
                "tenant-a_org-model-a_a_0_k": 2,
                "tenant-a_org-model-a_a_0_v": 2,
                "tenant-a_org-model-a_b_0_k": 3,
                "tenant-a_org-model-a_b_0_v": 3,
            },
        )
        self.assertEqual(store.batch_exists(["a", "b"]), 2)
        self.assertEqual(store.batch_exists(["a", "missing", "b"]), 1)
        host.kv_buffer.zero_()
        self.assertEqual(store.batch_get_v1(["a", "b"], indices), [True, True])
        for index in (0, 1):
            self._assert_page(host, index, 11, 22)

    def test_anchor_key_byte_limit_preserves_component_suffixes(self):
        page_hash = "a" * 64
        indices = torch.tensor([0], dtype=torch.int64)
        for prefix in ("x" * 186, "é" * 93):
            with self.subTest(prefix_bytes=len(prefix.encode("utf-8"))):
                objects = {}
                host = self._host()
                store = self._store(host, objects, model=prefix, tag=None)
                host.k_buffer.fill_(11)
                host.v_buffer.fill_(22)
                self.assertEqual(store.batch_set_v1([page_hash], indices), [True])
                self.assertEqual(len(objects), 2)
                self.assertTrue(all(len(key.encode("utf-8")) == 255 for key in objects))
                host.kv_buffer.zero_()
                self.assertEqual(store.batch_get_v1([page_hash], indices), [True])
                self._assert_page(host, 0, 11, 22)

                too_long = self._store(self._host(), {}, model=prefix + "x", tag=None)
                with self.assertRaisesRegex(ValueError, "255 UTF-8 bytes"):
                    too_long.batch_set_v1([page_hash], indices)

        # The envelope suffix is one byte longer than an ordinary K/V suffix.
        envelope = self._store(
            _ProtocolPool(1, envelope=True), {}, model="x" * 186, tag=None
        )
        with self.assertRaisesRegex(ValueError, "255 UTF-8 bytes"):
            envelope.batch_set_v1([page_hash], indices)

    def test_overlong_anchor_batch_is_rejected_before_client_io(self):
        objects = {}
        host = self._host()
        store = self._store(host, objects, model="x" * 187, tag=None)
        host.k_buffer.fill_(-7)
        host.v_buffer.fill_(-9)
        # The first page fits; the second would lose its K/V suffix on the wire.
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
                    with self.assertRaisesRegex(ValueError, "255 UTF-8 bytes"):
                        operation()
                    self.assertEqual(client.mock_calls, [])
        self.assertEqual(objects, {})
        self.assertEqual(store.client.depths, {})
        for index in (0, 1):
            self._assert_page(host, index, -7, -9)

    def test_protocol_layouts_keep_namespace_and_component_order(self):
        cases = (
            ("mla", 1, {"is_mla_model": True}, {}, ("_k",)),
            ("envelope", 1, {}, {"envelope": True}, ("0_kv",)),
            (
                "split_mha",
                4,
                {"should_split_heads": True, "tp_lcm_size": 4},
                {"split": True},
                ("0_k", "0_v", "1_k", "1_v"),
            ),
        )
        for name, count, config, geometry, suffixes in cases:
            with self.subTest(layout=name):
                objects = {}
                host = _ProtocolPool(count, **geometry)
                store = self._store(host, objects, **config)
                host.kv_buffer[0].copy_(
                    torch.arange(count * 8, dtype=torch.float32).reshape(count, 8)
                )
                expected = host.kv_buffer[0].clone()
                indices = torch.tensor([0], dtype=torch.int64)
                self.assertEqual(store.batch_set_v1(["page"], indices), [True])
                self.assertEqual(
                    set(objects),
                    {f"tenant-a_org-model-a_page_{suffix}" for suffix in suffixes},
                )
                for index, suffix in enumerate(suffixes):
                    component = torch.arange(
                        index * 8, (index + 1) * 8, dtype=torch.float32
                    )
                    self.assertEqual(
                        objects[f"tenant-a_org-model-a_page_{suffix}"],
                        bytes(component.view(torch.uint8).tolist()),
                    )
                self.assertEqual(store.batch_exists(["page"]), 1)
                host.kv_buffer[1].fill_(-1)
                self.assertEqual(
                    store.batch_get_v1(["page"], torch.tensor([1])), [True]
                )
                torch.testing.assert_close(host.kv_buffer[1], expected)


if __name__ == "__main__":
    unittest.main()
