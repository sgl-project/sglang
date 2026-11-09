"""MLU HND cache writes, movement and unsupported configurations.

Cover the MLU writer and HND storage without loading model weights.
Run with SGLANG_PLUGINS=none on one free MLU:
    python test/registered/kernels/ops/kvcache/test_mlu_kv_cache.py -f
"""

import unittest
from types import SimpleNamespace

import torch
import torch_mlu  # noqa: F401

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlu.memory_pool_mlu import MLUMHATokenToKVPool
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

# Exercises the MLU reshape_paged_cache writer and HND cache movement.
register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestMLUKVCache(CustomTestCase):
    def _pool(self, *, dtype=torch.bfloat16, layers=2, start=0, layout=None, heads=2):
        pool = MLUMHATokenToKVPool(
            size=128,
            page_size=16,
            dtype=dtype,
            head_num=heads,
            head_dim=128,
            layer_num=layers,
            start_layer=start,
            end_layer=start + layers - 1,
            device="mlu",
            enable_memory_saver=False,
            kv_cache_layout=layout,
        )
        for buf in pool.k_buffer + pool.v_buffer:
            buf.zero_()
        return pool

    def _write(self, pool, expected, layer, loc, offset=0):
        k = self._values(loc.numel(), pool.head_num, 128, pool.dtype, offset)
        v = -k
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=layer), loc, k.to("mlu"), v.to("mlu")
        )
        slots = loc.cpu().long()
        expected[0][layer - pool.start_layer, slots] = k
        expected[1][layer - pool.start_layer, slots] = v
        self._assert_cache(pool, expected)

    @staticmethod
    def _snapshot(pool):
        # Independent logical [layer, physical_slot, head, dim] reference.
        return tuple(
            torch.stack(
                [
                    buf.cpu()
                    .permute(0, 2, 1, 3)
                    .reshape(-1, pool.head_num, buf.shape[-1])
                    for buf in buffers
                ]
            )
            for buffers in (pool.k_buffer, pool.v_buffer)
        )

    def _assert_cache(self, pool, expected):
        for actual, reference in zip(self._snapshot(pool), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    @staticmethod
    def _values(count, heads, dim, dtype, offset=0):
        # Small exact integers make bit-exact BF16/FP16 copy assertions possible.
        return (
            (torch.arange(count * heads * dim).reshape(count, heads, dim) + offset)
            % 127
            - 63
        ).to(dtype)

    def test_hnd_writes_overwrite_scaling_and_layer_override(self):
        """Writes cross pages, preserve other slots and honor scales/layer offsets."""
        with envs.SGLANG_USE_HND_KVCACHE.override(None):
            envs.SGLANG_USE_HND_KVCACHE.clear()
            pool = self._pool(start=5)
        self.assertTrue(pool.use_hnd)
        expected = self._snapshot(pool)
        loc = torch.tensor([143, 16, 31, 0, 32, 15], device="mlu")
        self._write(pool, expected, 6, loc)
        loc = loc[::2].contiguous().int()
        k = self._values(loc.numel(), pool.head_num, 128, torch.float32, 37)
        v = -k
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=5),
            KVWriteLoc(loc=loc),
            k.to("mlu"),
            v.to("mlu"),
            k_scale=2,
            v_scale=4,
            layer_id_override=6,
        )
        expected[0][1, loc.cpu().long()] = (k / 2).to(pool.dtype)
        expected[1][1, loc.cpu().long()] = (v / 4).to(pool.dtype)
        self._assert_cache(pool, expected)

    def test_invalid_configuration_is_rejected(self):
        # Explicit overrides bypass shared HND initialization, even "hnd".
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            for layout in ("hnd", "nhd"):
                with self.subTest(layout=layout):
                    with self.assertRaisesRegex(
                        NotImplementedError, "remove any explicit"
                    ):
                        self._pool(layout=layout)
        with envs.SGLANG_USE_HND_KVCACHE.override(False):
            with self.assertRaisesRegex(
                NotImplementedError, "got kv_cache_layout='nhd'"
            ):
                self._pool()

        with self.assertRaisesRegex(NotImplementedError, "got dtype=torch.float32"):
            self._pool(dtype=torch.float32)

    def test_dcp_mask_rejected_before_input_or_cache_mutation(self):
        """Unsupported DCP must fail before scaling inputs or writing local KV."""
        pool = self._pool(layers=1)
        expected = self._snapshot(pool)
        loc = torch.tensor([15, 16], device="mlu")
        k = self._values(2, pool.head_num, 128, torch.float32).to("mlu")
        v = self._values(2, pool.head_num, 128, torch.float32, 17).to("mlu")
        original_k, original_v = k.clone(), v.clone()
        mask = torch.tensor([True, False], device="mlu")
        with self.assertRaisesRegex(NotImplementedError, "dcp_kv_mask.*not supported"):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=0),
                KVWriteLoc(loc=loc),
                k,
                v,
                k_scale=2,
                v_scale=4,
                dcp_kv_mask=mask,
            )
        torch.testing.assert_close(k, original_k, rtol=0, atol=0)
        torch.testing.assert_close(v, original_v, rtol=0, atol=0)
        self._assert_cache(pool, expected)

    def test_move_overlapping_slots_uses_original_sources_all_layers(self):
        pool = self._pool()
        loc = torch.arange(16, 144, device="mlu")
        for layer_id in range(pool.layer_num):
            k = self._values(128, pool.head_num, 128, pool.dtype, layer_id * 29)
            v = self._values(128, pool.head_num, 128, pool.dtype, layer_id * 29 + 13)
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=layer_id), loc, k.to("mlu"), v.to("mlu")
            )
        expected = self._snapshot(pool)
        src, dst = [16, 31, 32, 47, 80], [31, 32, 47, 80, 143]
        for reference in expected:
            reference[:, dst] = reference[:, src].clone()
        pool.move_kv_cache(
            torch.tensor(dst, device="mlu"), torch.tensor(src, device="mlu")
        )
        self._assert_cache(pool, expected)


if __name__ == "__main__":
    unittest.main()
