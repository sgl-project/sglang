import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.pool_host import mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def device_pool(v_head_dim=8):
    return SimpleNamespace(
        size=1024,
        page_size=16,
        head_num=2,
        head_dim=8,
        v_head_dim=v_head_dim,
        layer_num=2,
        store_dtype=torch.bfloat16,
        device="cpu",
        start_layer=0,
        end_layer=1,
        hicache_write_back_staging=None,
    )


class TestHiCacheStagingPreparation(unittest.TestCase):
    def test_host_construction_reuses_staging_after_resize(self):
        for v_dim in (8, 16):
            with (
                self.subTest(v_dim=v_dim),
                patch.object(mha, "_is_cuda", True),
                patch.object(mha, "can_use_write_back_jit_kernel", return_value=True),
                # Exercise host-pool wiring without the Linux mmap allocator.
                patch.dict(
                    mha.ALLOC_MEMORY_FUNCS,
                    {
                        "cpu": lambda dims, dtype, **kwargs: torch.empty(
                            dims, dtype=dtype
                        )
                    },
                ),
            ):
                pool = device_pool(v_dim)
                staging = mha.prepare_mha_write_back_staging(
                    pool, layer_num=pool.layer_num, page_size=pool.page_size
                )
                # Final KV sizing shrinks the device pool before the host pool exists.
                pool.size = 64
                host_cls = mha.get_mha_host_pool_cls(pool)
                with patch.object(
                    mha.WriteBackStaging,
                    "allocate",
                    side_effect=AssertionError("allocated twice"),
                ):
                    host = host_cls(
                        pool,
                        host_to_device_ratio=0.2,
                        host_size=0,
                        page_size=pool.page_size,
                        layout="page_first",
                        pin_memory=False,
                    )
                self.assertEqual(host.size, 16)
                self.assertEqual(host.staging_page_capacity, 1)
                self.assertEqual(
                    host.staging_k_buffer.data_ptr(), staging.buffers[0].data_ptr()
                )
                self.assertEqual(
                    host.staging_v_buffer.data_ptr(), staging.buffers[1].data_ptr()
                )
                self.assertEqual(host.staging_v_buffer.shape[-1], v_dim)

    def test_unsupported_kernel_does_not_allocate(self):
        with (
            patch.object(mha, "_is_cuda", True),
            patch.object(mha, "can_use_write_back_jit_kernel", return_value=False),
            patch.object(
                mha.WriteBackStaging,
                "allocate",
                side_effect=AssertionError("allocated"),
            ),
        ):
            pool = device_pool()
            self.assertIsNone(
                mha.prepare_mha_write_back_staging(pool, layer_num=2, page_size=16)
            )
            self.assertIsNone(pool.hicache_write_back_staging)


if __name__ == "__main__":
    unittest.main()
