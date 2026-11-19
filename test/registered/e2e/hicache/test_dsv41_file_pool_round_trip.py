"""CUDA C1/C2 pages must survive D2H, File storage and H2D byte-for-byte.

This uses the real low-ratio device pools and the same HostPool assembler as
serving, without loading model weights. Nonzero, permuted page offsets and a
poisoned destination distinguish a physical restore from a logical anchor hit.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.dsv41_decode_storage import build_decode_storage_pool_plan
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _dsv4_low_ratio_entries,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-large")

PAGE_SIZE = 256
FULL_SIZE = 6 * PAGE_SIZE
PERSISTENT_POOLS = (
    PoolName.DEEPSEEK_V4_C1,
    PoolName.DEEPSEEK_V4_C1_INDEXER,
    PoolName.DEEPSEEK_V4_C2,
    PoolName.DEEPSEEK_V4_C2_INDEXER,
)


def _indices(pages, device="cpu"):
    return torch.cat(
        [torch.arange(page * PAGE_SIZE, (page + 1) * PAGE_SIZE) for page in pages]
    ).to(device)


def _sha256(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


@unittest.skipUnless(torch.cuda.is_available() and not is_hip(), "Requires CUDA")
class TestDSV41FilePoolRoundTrip(CustomTestCase):
    def setUp(self):
        super().setUp()
        context = get_context().override_server_args(
            model_path="dsv41-physical-pool-test",
            page_size=PAGE_SIZE,
            dsv4_attn_backend="flashmla",
            hicache_mem_layout="page_first",
        )
        context.install()
        self.addCleanup(context.restore)
        directory = tempfile.TemporaryDirectory(prefix="hicache_dsv41_gpu_pages_")
        self.addCleanup(directory.cleanup)
        storage = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(directory.name)
        storage.__enter__()
        self.addCleanup(storage.__exit__, None, None, None)
        self.backend = HiCacheFile(
            HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=True,
                enable_storage_metrics=False,
                is_page_first_layout=True,
                model_name="dsv41-physical-pool-test",
                extra_config={
                    "enable_metadata_cache": False,
                    "max_size": 0,
                    "min_free_space": 0,
                },
            )
        )
        self.pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=2,
            swa_size=2 * PAGE_SIZE,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=PAGE_SIZE,
            swa_page_size=128,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=40,
            start_layer=0,
            end_layer=40,
            device="cuda",
            enable_memory_saver=False,
            compression_ratios=[0] * 2 + [2] * 18 + [1] * 20,
            kv_source_layers=(2, 8, 14, 20),
            full_size=FULL_SIZE,
            kv_layout=KVLayout.V4,
        )
        infos = {info.pool_name: info for info in self.pool.get_device_pool_infos()}
        self.entries = _dsv4_low_ratio_entries(self.pool, infos, PAGE_SIZE, 6, 40)
        self.assertEqual({entry.name for entry in self.entries}, set(PERSISTENT_POOLS))
        self.backend.register_mem_host_pool_v2(
            LogicalHostPool(6 * PAGE_SIZE, PAGE_SIZE), PoolName.KV
        )
        for entry in self.entries:
            self.backend.register_mem_host_pool_v2(entry.host_pool, entry.name)
        plan = build_decode_storage_pool_plan(
            self.pool,
            PAGE_SIZE,
            [
                SidecarPoolSpec(info.pool_name, info.indices_from_pool)
                for info in infos.values()
                if info.indices_from_pool == PoolName.KV
            ],
        )
        self.assertEqual(plan.pool_names, set(PERSISTENT_POOLS))
        self.backend.configure_persistent_pool_bundle(plan)

    def _committed_pool_object_path(self, key, name):
        manifest = self.backend._persistent_bundle.decode_manifest(
            Path(self.backend._bundle_manifest_path(key)).read_bytes(), key
        )
        self.assertIsNotNone(manifest, "File did not publish a valid pool bundle")
        self.assertEqual(set(manifest["objects"]), set(map(str, PERSISTENT_POOLS)))
        path = Path(self.backend.file_path) / (
            self.backend._bundle_object_key(key, manifest, name) + ".bin"
        )
        raw = path.read_bytes()
        self.assertEqual(len(raw), manifest["objects"][str(name)]["size"])
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), manifest["objects"][str(name)]["sha256"]
        )
        return path

    def test_all_persistent_pages_round_trip_and_faults(self):
        keys = ["gpu-logical-page-0", "gpu-logical-page-1"]
        source_pages = [1, 3]
        destination_pages = [2, 4]
        host_source = _indices([4, 1])
        host_destination = _indices([3, 5])
        source_indices = _indices(source_pages, "cuda")
        destination_indices = _indices(destination_pages, "cuda")
        expected = {}
        for pool_id, entry in enumerate(self.entries):
            rows = []
            for layer_id, buffer in enumerate(entry.host_pool.device_buffers):
                raw = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
                pattern = torch.arange(raw.numel(), device="cuda", dtype=torch.int64)
                raw.copy_(
                    ((pattern + pool_id * 17 + layer_id * 31) % 251)
                    .to(torch.uint8)
                    .reshape_as(raw)
                )
                rows.append(raw[source_pages].clone())
            expected[entry.name] = rows
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool, host_source, source_indices, "kernel"
            )
        # File may read host memory only after the actual device copy completes.
        torch.cuda.synchronize()
        transfers = [
            PoolTransfer(name=name, keys=keys, host_indices=host_source)
            for name in (PoolName.KV, *PERSISTENT_POOLS)
        ]
        self.assertEqual(
            self.backend.batch_set_v2(transfers),
            {transfer.name: [True, True] for transfer in transfers},
        )
        records = []
        for entry in self.entries:
            host = entry.host_pool
            for page_id, key in enumerate(keys):
                wanted = torch.cat([rows[page_id] for rows in expected[entry.name]])
                actual = host.get_data_page(host_source[page_id * PAGE_SIZE])
                torch.testing.assert_close(actual, wanted.cpu(), rtol=0, atol=0)
                path = self._committed_pool_object_path(key, entry.name)
                self.assertEqual(path.read_bytes(), wanted.cpu().numpy().tobytes())
                records.append(
                    {
                        "pool": str(entry.name),
                        "page": page_id,
                        "bytes": wanted.numel(),
                        "sha256": _sha256(wanted),
                    }
                )
            # Clear L2 and poison every device page before restore.
            host.kv_buffer.fill_(0xC7)
            for buffer in host.device_buffers:
                buffer.view(torch.uint8).fill_(0xA5)
        query = self.backend.batch_exists_v2(keys, transfers[1:])
        self.assertEqual(query.kv_hit_pages, 2)
        self.assertEqual(query.restorable_prefix_pages, [1, 2])
        restored = [
            PoolTransfer(name=t.name, keys=keys, host_indices=host_destination)
            for t in transfers
        ]
        self.assertEqual(
            self.backend.batch_get_v2(restored),
            {transfer.name: [True, True] for transfer in restored},
        )
        for entry in self.entries:
            host = entry.host_pool
            for layer_id in range(host.layer_num):
                host.load_to_device_per_layer(
                    entry.device_pool,
                    host_destination.cuda(),
                    destination_indices,
                    layer_id,
                    "kernel",
                )
        torch.cuda.synchronize()
        for entry in self.entries:
            for layer_id, buffer in enumerate(entry.host_pool.device_buffers):
                raw = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
                torch.testing.assert_close(
                    raw[destination_pages],
                    expected[entry.name][layer_id],
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(torch.all(raw[0] == 0xA5))
                self.assertTrue(torch.all(raw[5] == 0xA5))
        # The fused CUDA indexer objects include the required scale bytes.
        for name in (PoolName.DEEPSEEK_V4_C1_INDEXER, PoolName.DEEPSEEK_V4_C2_INDEXER):
            path = self._committed_pool_object_path(keys[1], name)
            saved = path.read_bytes()
            try:
                path.unlink()
                # Keys-only queries also require the complete committed bundle.
                self.assertEqual(self.backend.batch_exists_v2(keys).kv_hit_pages, 1)
                query = self.backend.batch_exists_v2(keys, transfers[1:])
                self.assertEqual(query.kv_hit_pages, 1)
                self.assertEqual(query.restorable_prefix_pages, [1])
                path.write_bytes(saved)
                self.assertEqual(
                    self.backend.batch_exists_v2(keys, transfers[1:]).kv_hit_pages, 2
                )
                path.write_bytes(saved[:-1])
                for entry in self.entries:
                    entry.host_pool.kv_buffer.fill_(0xC7)
                self.assertEqual(
                    self.backend.batch_get_v2(restored[1:]),
                    {pool_name: [True, False] for pool_name in PERSISTENT_POOLS},
                )
                for entry in self.entries:
                    bad = entry.host_pool.get_data_page(host_destination[PAGE_SIZE])
                    self.assertTrue(torch.all(bad == 0xC7))
            finally:
                path.write_bytes(saved)
        print(
            "[DSV41 physical round trip] "
            + json.dumps(
                {
                    "cuda_device": torch.cuda.current_device(),
                    "layout": self.pool.pd_kv_layout(),
                    "pools": records,
                    "missing_scale_fallbacks": 2,
                    "short_reads_rejected": 2,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
