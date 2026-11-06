"""Exercise the ROCm fused KV writers' slot contract with a CPU kernel boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRocmHiSparseFusedKV(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.cache = object()
        self.pool = self.make_pool(DSATokenToKVPool)
        self.attn = SimpleNamespace(
            kv_cache_dtype="bfloat16",
            current_attention_backend="dsa",
            w_kc=torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn),
            w_scale=None,
            attn_mqa=SimpleNamespace(layer_id=7, k_scale=1.0),
            rotary_emb=SimpleNamespace(
                cos_cache=None, sin_cache=None, is_neox_style=False
            ),
        )

    def make_pool(self, pool_type):
        # Supply storage without allocating a model's cache; keep real accessors.
        pool = pool_type.__new__(pool_type)
        pool.kv_buffer = [self.cache]
        pool.start_layer = 7
        pool.layer_transfer_counter = None
        pool.dtype = pool.store_dtype = torch.bfloat16
        return pool

    def use_hisparse(self):
        self.pool = self.make_pool(HiSparseDSATokenToKVPool)
        mapping = torch.zeros(65, dtype=torch.int64)
        mapping[17], mapping[18], mapping[-1] = 3, 5, -1
        self.pool.register_mapping(mapping)
        return mapping

    def invoke(self, locations):
        """Slots each fused KV writer hands its AITER kernel, keyed by writer."""
        slots = {}

        def kernel(writer, kv_cache_arg):
            def record(*args, **kwargs):
                self.assertIs(args[kv_cache_arg], self.cache)
                slots[writer] = args[kv_cache_arg + 1]

            return record

        with (
            patch.object(forward_mla_rocm, "get_token_to_kv_pool", lambda: self.pool),
            patch.object(
                forward_mla_rocm,
                "fused_qk_rope_cat_and_cache_mla",
                kernel("rope_cache", kv_cache_arg=4),
                create=True,
            ),
            patch.object(
                forward_mla_rocm,
                "fused_fp8_bmm_rope_cat_and_cache_mla",
                kernel("bmm_rope_cache", kv_cache_arg=6),
                create=True,
            ),
            # The gfx1250 fused branch excludes DSA, so these DSA pools use the gfx95 one.
            patch.object(forward_mla_rocm, "_use_gfx1250_fused_mla_rope", False),
        ):
            forward_mla_rocm._fused_rope_cat_and_cache(
                self.attn, torch.empty(0), None, None, None, None, locations
            )
            forward_mla_rocm._fused_bmm_rope_cat_and_cache(
                self.attn, torch.empty(0, 0, 0), None, None, None, None, locations
            )
        self.assertEqual(slots.keys(), {"rope_cache", "bmm_rope_cache"})
        return slots

    def test_resident_locations_are_unchanged(self):
        locations = torch.tensor([17, 0, -1], dtype=torch.int64)
        for writer, actual in self.invoke(locations).items():
            with self.subTest(writer=writer):
                self.assertIs(actual, locations)

    def test_resident_strided_locations(self):
        """Strided locations must not send interleaved storage values to AITER."""
        storage = torch.tensor([3, 21, 5, 22, -1, 23])
        locations = storage[::2]
        for writer, actual in self.invoke(locations).items():
            with self.subTest(writer=writer):
                self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(actual, torch.tensor([3, 5, -1]))
        torch.testing.assert_close(storage, torch.tensor([3, 21, 5, 22, -1, 23]))

    def test_hisparse_maps_logical_slots(self):
        """Logical slots beyond device capacity must write their physical rows,
        whichever fused writer runs."""
        self.use_hisparse()
        locations = torch.tensor([17, 18], dtype=torch.int64)
        for writer, actual in self.invoke(locations).items():
            with self.subTest(writer=writer):
                torch.testing.assert_close(actual, torch.tensor([3, 5]))
        torch.testing.assert_close(locations, torch.tensor([17, 18]))

    def test_hisparse_padding_and_unmapped_slots(self):
        self.use_hisparse()
        locations = torch.tensor([17, -1, 19, 0])
        for writer, actual in self.invoke(locations).items():
            with self.subTest(writer=writer):
                torch.testing.assert_close(actual, torch.tensor([3, -1, 0, 0]))
        torch.testing.assert_close(locations, torch.tensor([17, -1, 19, 0]))


if __name__ == "__main__":
    unittest.main()
