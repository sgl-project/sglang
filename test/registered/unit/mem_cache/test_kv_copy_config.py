from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("stride_bytes", "expected"),
    [
        (
            4095,
            {
                "bytes_per_tile": 128,
                "byte_tiles": 32,
                "num_warps": 4,
                "num_locs_upper": 256,
            },
        ),
        (
            4096,
            {
                "bytes_per_tile": 256,
                "byte_tiles": 16,
                "num_warps": 4,
                "num_locs_upper": 256,
            },
        ),
        (
            8192,
            {
                "bytes_per_tile": 512,
                "byte_tiles": 16,
                "num_warps": 8,
                "num_locs_upper": 128,
            },
        ),
    ],
)
def test_default_kv_copy_config_is_unchanged(stride_bytes, expected):
    pool = object.__new__(MHATokenToKVPool)
    assert pool._get_kv_copy_config(stride_bytes) == expected


def test_kv_copy_warmup_uses_subclass_config():
    class CustomPool(MHATokenToKVPool):
        def _get_kv_copy_config(self, stride_bytes):
            assert stride_bytes == 1024
            return {
                "bytes_per_tile": 128,
                "byte_tiles": 8,
                "num_warps": 4,
                "num_locs_upper": 64,
            }

    pool = object.__new__(CustomPool)
    pool.layer_num = 1
    pool.data_strides = torch.tensor([1024])
    pool.data_ptrs = torch.tensor([0])
    pool.device = torch.device("cpu")

    with patch("sglang.srt.mem_cache.memory_pool.copy_all_layer_kv_cache_func") as copy:
        pool._init_kv_copy_and_warmup()

    assert pool._kv_copy_config["num_locs_upper"] == 64
    assert copy.call_args.args[5] == 64
    assert copy.call_args.args[6] == pool._kv_copy_config
