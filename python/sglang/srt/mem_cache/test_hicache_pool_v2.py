from __future__ import annotations

import os

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    NSAIndexerHostPool,
    PoolEntry,
)


class _RecordingPool:
    def __init__(self, page_size: int = 2):
        self.page_size = page_size
        self.layout = "page_first"
        self.device = "cpu"
        self.size = 64
        self.size_per_token = 1
        self.dtype = torch.uint8
        self.load_calls = []
        self.backup_calls = []

    def clear(self):
        return

    def alloc(self, need_size: int):
        return torch.arange(need_size, dtype=torch.int64)

    def free(self, indices: torch.Tensor):
        return len(indices)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        self.load_calls.append((host_indices.clone(), device_indices.clone(), layer_id))

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self.backup_calls.append((host_indices.clone(), device_indices.clone()))

    def get_data_page(self, index, flat: bool = True):
        return torch.zeros((4,), dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.zeros((4,), dtype=torch.uint8)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor):
        return

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        return [0], [4]

    def get_register_buffers(self):
        return []


class _FlatPagePool:
    def __init__(self, page_size: int = 2, page_bytes: int = 4):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self.layout = "page_first"
        self.device = "cpu"
        self.size = 64
        self.size_per_token = 1
        self.dtype = torch.uint8
        self.pages: dict[int, torch.Tensor] = {}

    def get_data_page(self, index, flat: bool = True):
        return self.pages[index].clone()

    def get_dummy_flat_data_page(self):
        return torch.zeros((self.page_bytes,), dtype=torch.uint8)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor):
        self.pages[index] = data_page.clone()

    def get_register_buffers(self):
        return []

    def get_ksize_per_token(self):
        return 1


class _FakeNSAAnchor:
    def __init__(self):
        self.layout = "page_first"
        self.page_size = 2
        self.size = 8
        self.device = "cpu"
        self.pin_memory = False
        self.layer_num = 2
        self.indexer_dtype = torch.uint8
        self.indexer_size_per_token = 3
        self.indexer_page_stride_size = 6
        self.index_k_with_scale_buffer = torch.zeros((4, 2, 1, 6), dtype=torch.uint8)

    def _get_indexer_page_indices(self, host_indices, device_indices):
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices


def _storage_config(layout: str = "page_first") -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        layout=layout,
        model_name="test",
    )


def test_host_pool_group_reuses_anchor_indices():
    anchor = _RecordingPool(page_size=2)
    sidecar = _RecordingPool(page_size=2)
    group = HostPoolGroup(
        [
            PoolEntry("kv", anchor, object(), lambda layer_id: layer_id, True),
            PoolEntry("nsa", sidecar, object(), lambda layer_id: layer_id),
        ]
    )

    anchor_host = torch.tensor([4, 5], dtype=torch.int64)
    anchor_device = torch.tensor([14, 15], dtype=torch.int64)
    transfer = PoolTransfer(
        name=PoolName.NSA,
        use_anchor_host_indices=True,
        use_anchor_device_indices=True,
    )

    group.load_to_device_per_layer(
        object(),
        anchor_host,
        anchor_device,
        layer_id=0,
        io_backend="direct",
        pool_transfers=[transfer],
    )
    group.backup_from_device_all_layer(
        object(),
        anchor_host,
        anchor_device,
        io_backend="direct",
        pool_transfers=[transfer],
    )

    assert sidecar.load_calls[0][0].tolist() == anchor_host.tolist()
    assert sidecar.load_calls[0][1].tolist() == anchor_device.tolist()
    assert sidecar.backup_calls[0][0].tolist() == anchor_host.tolist()
    assert sidecar.backup_calls[0][1].tolist() == anchor_device.tolist()


def test_nsa_indexer_host_pool_roundtrip_and_meta():
    anchor = _FakeNSAAnchor()
    pool = NSAIndexerHostPool(anchor)
    data_page = torch.arange(12, dtype=torch.uint8)

    pool.set_from_flat_data_page(2, data_page)
    assert torch.equal(pool.get_data_page(2), data_page)

    ptrs, sizes = pool.get_page_buffer_meta(torch.tensor([2, 3], dtype=torch.int64))
    assert len(ptrs) == 1
    assert sizes == [12]


def test_file_backend_nsa_key_compatibility_and_prefix(tmp_path):
    storage = HiCacheFile(_storage_config(), file_path=str(tmp_path))
    kv_pool = _FlatPagePool()
    nsa_pool = _FlatPagePool()
    storage.register_mem_pool_host(kv_pool)
    storage.register_mem_host_pool_v2(kv_pool, PoolName.KV.value)
    storage.register_mem_host_pool_v2(nsa_pool, PoolName.NSA.value)

    kv_pool.set_from_flat_data_page(0, torch.tensor([1, 2, 3, 4], dtype=torch.uint8))
    kv_pool.set_from_flat_data_page(4, torch.tensor([5, 6, 7, 8], dtype=torch.uint8))
    nsa_pool.set_from_flat_data_page(0, torch.tensor([9, 9, 9, 9], dtype=torch.uint8))

    storage.batch_set_v2(
        [
            PoolTransfer(
                name=PoolName.KV,
                host_indices=torch.tensor([0, 1, 4, 5], dtype=torch.int64),
                keys=["hash0", "hash2"],
            ),
            PoolTransfer(
                name=PoolName.NSA,
                host_indices=torch.tensor([0, 1], dtype=torch.int64),
                keys=["hash0"],
            ),
        ]
    )

    assert os.path.exists(tmp_path / "hash0__nsa_idx_test.bin")

    kv_only = storage.batch_exists_v2(["hash0", "hash1", "hash2"])
    assert kv_only.kv_hit_pages == 1

    hit = storage.batch_exists_v2(
        ["hash0", "hash2"],
        [
            PoolTransfer(
                name=PoolName.NSA,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
        ],
    )
    assert hit.kv_hit_pages == 1
