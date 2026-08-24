from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.storage.backend_factory import _get_hf3fs_bytes_per_page
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_hf3fs_uses_marker_page_for_logical_anchor():
    logical_pool = SimpleNamespace(
        kv_buffer=None,
        layout="page_first_direct",
        page_size=16,
        dtype=torch.uint8,
        get_ksize_per_token=lambda: 0,
        get_size_per_token=lambda: 0,
    )

    assert _get_hf3fs_bytes_per_page(logical_pool) == 4096


def test_hf3fs_logical_anchor_v1_operations_are_noops():
    backend = object.__new__(HiCacheHF3FS)
    backend._logical_anchor = True
    keys = ["a", "b"]
    host_indices = torch.tensor([0, 1], dtype=torch.int64)

    assert backend.batch_exists(keys) == len(keys)
    assert backend.batch_get_v1(keys, host_indices) == [True, True]
    assert backend.batch_set_v1(keys, host_indices) == [True, True]
