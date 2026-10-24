"""Real Mooncake IO through striped HiCache host pools, without a model.

Run with ``python3 -m unittest discover -s test/manual/hicache \
    -p test_kv_shard_mooncake_io.py -v``. Requires CUDA and Mooncake binaries.
The test starts local TCP services unless MOONCAKE_MASTER is already configured;
an external service also needs MOONCAKE_TE_META_DATA_SERVER and storage capacity.
Every run uses a unique namespace and never clears the shared store.
"""

import os
import socket
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    _shard_storage_io,
    storage_model_name,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.page_interleave import (
    PageInterleaveHostPool,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.mooncake_utils import MooncakeTestServices

SHARDS = 2
PAGE_SIZE = 16
PAGES_PER_RANK = 16
LAYERS = 3


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA host-memory support")
class TestKVShardMooncakeIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            import mooncake.store  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("requires mooncake-transfer-engine") from exc
        if not os.environ.get("MOONCAKE_MASTER"):
            services = MooncakeTestServices(
                protocol="tcp", store_segment_size=128 * 1024**2
            )
            services.start()
            cls.addClassCleanup(services.stop)
            environment = patch.dict(os.environ, services.server_env())
            environment.start()
            cls.addClassCleanup(environment.stop)

    def _host_pool(self, rank, is_mla):
        common = dict(
            size=PAGES_PER_RANK * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            layer_num=LAYERS,
            device="cuda",
            enable_memory_saver=False,
        )
        if is_mla:
            device_pool = MLATokenToKVPool(
                **common, kv_lora_rank=128, qk_rope_head_dim=64
            )
            host_cls = MLATokenToKVPoolHost
            host_kwargs = {"target_layer_num": LAYERS}
        else:
            device_pool = MHATokenToKVPool(**common, head_num=2, head_dim=64)
            host_cls = MHATokenToKVPoolHost
            host_kwargs = {}
        placement = PageInterleavePlacement(
            PageShardSpec(
                shard_rank=rank,
                shard_size=SHARDS,
                page_size=PAGE_SIZE,
                max_prefix_tokens=SHARDS * PAGE_SIZE,
                chunk_tokens=PAGE_SIZE,
            )
        )
        native = host_cls(
            device_pool.get_device_pool_infos()[0].buffer_info,
            2.0,
            0,
            PAGE_SIZE,
            "page_first",
            pin_memory=True,
            device="cpu",
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            **host_kwargs,
        )
        host = PageInterleaveHostPool(native, placement)
        # Keep the captured device tensors alive until their host mirror dies.
        self.addCleanup(lambda: device_pool)
        self.addCleanup(host.destroy)
        return host

    def _store(self, host, rank, is_mla, namespace):
        # GQA tests one head shard replicated across CP. MLA tests distinct
        # owners across plain TP. Physical ranks deliberately differ from the
        # GQA key rank so a rank-dependent key cannot accidentally pass.
        key_rank = rank if is_mla else 1
        config = HiCacheStorageConfig(
            tp_rank=rank if is_mla else rank * 2 + 1,
            tp_size=SHARDS if is_mla else SHARDS * 2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0 if is_mla else rank,
            attn_cp_size=1 if is_mla else SHARDS,
            is_mla_model=is_mla,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name=storage_model_name("kv-shard-io", host) + "-atp2",
            key_tp_rank=key_rank,
            key_tp_size=2,
            extra_config={
                "master_server_address": os.environ["MOONCAKE_MASTER"],
                "metadata_server": os.environ["MOONCAKE_TE_META_DATA_SERVER"],
                "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME")
                or socket.gethostbyname(socket.gethostname()),
                "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
                "device_name": os.environ.get("MOONCAKE_DEVICE", ""),
                "global_segment_size": 0,
                "check_server": False,
                "extra_backend_tag": namespace,
            },
        )
        self.assertTrue(config.is_storage_writer)
        backend = MooncakeStore(config, host)
        backend.register_mem_pool_host(host)
        # The underlying client's close releases registrations before host.destroy.
        self.addCleanup(backend.store.close)
        controller = SimpleNamespace(storage_backend=backend)
        get, put = _shard_storage_io(
            pool=host,
            page_get=lambda *args: HiCacheController._page_get_zero_copy(
                controller, *args
            ),
            page_set=lambda *args: HiCacheController._page_set_zero_copy(
                controller, *args
            ),
        )
        return backend, get, put

    def _round_trip(self, is_mla):
        hosts = [self._host_pool(rank, is_mla) for rank in range(SHARDS)]
        namespace = f"kv-shard-io-{uuid.uuid4().hex}"
        stores = [
            self._store(host, rank, is_mla, namespace)
            for rank, host in enumerate(hosts)
        ]
        self.assertEqual(len({host.storage_format_tag for host in hosts}), 1)
        page_count = 7
        keys = [f"page-{page}" for page in range(page_count)]
        source = hosts[0].alloc(page_count * PAGE_SIZE, start_owner=0)
        for host in hosts[1:]:
            self.assertTrue(
                torch.equal(host.alloc(page_count * PAGE_SIZE, start_owner=0), source)
            )
        expected = []
        for page in range(page_count):
            logical = source[page * PAGE_SIZE]
            owner = int(logical) // PAGE_SIZE % SHARDS
            host = hosts[owner]
            local = int(host.placement.local_index(logical))
            shape = host.get_data_page(local).shape
            data = (
                (torch.arange(shape.numel(), dtype=torch.int32) + 17 * page) % 239
            ).to(host.dtype)
            host.set_from_flat_data_page(local, data)
            expected.append(data)
        for _, _, put in stores:
            self.assertTrue(put(keys, source))
        for backend, _, _ in stores:
            self.assertEqual(backend.batch_exists(keys), page_count)

        # Reuse the same storage objects under the opposite owner rotation.
        for host in hosts:
            host.free(source)
            host.kv_buffer.zero_()
        destination = hosts[0].alloc(page_count * PAGE_SIZE, start_owner=1)
        for host in hosts[1:]:
            self.assertTrue(
                torch.equal(
                    host.alloc(page_count * PAGE_SIZE, start_owner=1), destination
                )
            )
        placement = hosts[0].placement
        self.assertTrue(
            torch.all(placement.owner_of(source) != placement.owner_of(destination))
        )
        operation = SimpleNamespace(request_id="shifted-owner")
        prefixes = [get(operation, keys, destination) for _, get, _ in stores]
        self.assertEqual(min(prefixes), page_count)
        self._assert_pages(hosts, destination, expected)

        # A real missing storage object on one owner must clamp the group's
        # prefix in logical-page units, while peers report a longer prefix.
        hole = 3
        partial_keys = list(keys)
        partial_keys[hole] = "missing-" + uuid.uuid4().hex
        for host in hosts:
            host.kv_buffer.zero_()
        prefixes = [get(operation, partial_keys, destination) for _, get, _ in stores]
        self.assertEqual(min(prefixes), hole)
        self.assertIn(page_count, prefixes)
        self._assert_pages(hosts, destination, expected[:hole])

        # One rank owns no page in this smaller logical batch.
        prefixes = [
            get(operation, keys[:1], destination[:PAGE_SIZE]) for _, get, _ in stores
        ]
        self.assertEqual(prefixes, [1] * SHARDS)

    def _assert_pages(self, hosts, indices, expected):
        for page, data in enumerate(expected):
            logical = indices[page * PAGE_SIZE]
            owner = int(logical) // PAGE_SIZE % SHARDS
            host = hosts[owner]
            local = int(host.placement.local_index(logical))
            torch.testing.assert_close(host.get_data_page(local), data, rtol=0, atol=0)

    def test_gqa_cp_shifted_owner_and_partial_hit(self):
        self._round_trip(is_mla=False)

    def test_mla_tp_shifted_owner_and_partial_hit(self):
        self._round_trip(is_mla=True)


if __name__ == "__main__":
    unittest.main()
