"""Test MLU MHA and MLA KV-pool reads, writes, and configuration validation."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlu.memory_pool import (
    MLUMHATokenToKVPool,
    MLUMLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestMLUMHATokenToKVPool(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        # These tests construct MHA pools without platform configuration.
        stack.enter_context(envs.SGLANG_USE_HND_KVCACHE.override(True))
        torch.manual_seed(0)

    def _create_pool(self, *, dtype=torch.bfloat16, layers=2, start=0, layout=None):
        pool = MLUMHATokenToKVPool(
            size=128,
            page_size=16,
            dtype=dtype,
            head_num=2,
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

    @staticmethod
    def _snapshot_cache_in_token_order(pool):
        """Snapshot all K/V slots on CPU in [layer, physical_slot, head, dim] order.

        Convert each HND buffer from [page, head, offset_in_page, dim] to
        [physical_slot, head, dim], where physical_slot = page * page_size
        + offset_in_page. This reads the entire cache without selecting slots.
        """
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

    def _assert_cache_matches(self, pool, expected):
        """Compare all K/V slots across all layers against token-order references."""
        for actual, reference in zip(
            self._snapshot_cache_in_token_order(pool), expected
        ):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_write_overwrite_and_layer_override(self):
        """Writes cross pages, preserve other slots and honor scales/layer offsets."""
        # Create the pool and input tensors.
        pool = self._create_pool(start=5)
        self.assertTrue(pool.use_hnd, "MLU KV cache should use HND layout by default.")
        loc = torch.tensor([143, 16, 31, 0, 32, 15])
        k = torch.randn(loc.numel(), pool.head_num, 128).to(pool.dtype)
        v = -k
        overwrite_loc = loc[::2].contiguous().int()
        overwrite_k = torch.randn(overwrite_loc.numel(), pool.head_num, 128)
        overwrite_v = -overwrite_k

        # Compute the expected cache contents on CPU.
        expected = self._snapshot_cache_in_token_order(pool)
        expected[0][1, loc] = k
        expected[1][1, loc] = v
        expected_overwrite = tuple(buffer.clone() for buffer in expected)
        expected_overwrite[0][1, overwrite_loc.long()] = (overwrite_k / 2).to(
            pool.dtype
        )
        expected_overwrite[1][1, overwrite_loc.long()] = (overwrite_v / 4).to(
            pool.dtype
        )

        # Write and overwrite the cache on MLU.
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=6), loc.to("mlu"), k.to("mlu"), v.to("mlu")
        )
        actual = self._snapshot_cache_in_token_order(pool)
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=5),
            KVWriteLoc(loc=overwrite_loc.to("mlu")),
            overwrite_k.to("mlu"),
            overwrite_v.to("mlu"),
            k_scale=2,
            v_scale=4,
            layer_id_override=6,
        )

        # Check the cache after each write against the CPU reference.
        for buffer, reference in zip(actual, expected):
            torch.testing.assert_close(buffer, reference, rtol=0, atol=0)
        self._assert_cache_matches(pool, expected_overwrite)

    def test_invalid_configuration_is_rejected(self):
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            with self.assertRaises(NotImplementedError):
                self._create_pool(layout="nhd")

        with envs.SGLANG_USE_HND_KVCACHE.override(False):
            with self.assertRaises(NotImplementedError):
                self._create_pool()

        with self.assertRaises(NotImplementedError):
            self._create_pool(dtype=torch.float32)


class TestMLUMLATokenToKVPool(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(get_parallel().override(dcp_enabled=False))

    @staticmethod
    def _create_pool(rope_dim=64, dtype=torch.bfloat16, use_dsa=False):
        pool = MLUMLATokenToKVPool(
            size=32,
            page_size=16,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=rope_dim,
            layer_num=2,
            start_layer=5,
            end_layer=6,
            device="mlu",
            enable_memory_saver=False,
            use_dsa=use_dsa,
        )
        for buf in pool.kv_buffer:
            buf.zero_()
        return pool

    def test_write_and_gather_preserve_slots_and_layers(self):
        """Paged writes and latent gathers agree across overwrites and layer offsets."""
        # Create the pool and input tensors.
        pool = self._create_pool()
        layer = SimpleNamespace(layer_id=5)
        # Out-of-order slots span a page boundary and the last allocated slot.
        loc = torch.tensor([47, 16, 0, 15])
        values = torch.randn(4, 1, 576).to(torch.bfloat16)
        replacement = torch.randn(2, 1, 576)
        overwrite_loc = loc[:2]

        # Compute the expected cache contents on CPU.
        expected = torch.zeros(2, 48, 1, 576, dtype=torch.bfloat16)
        expected[0, loc] = values
        expected[0, overwrite_loc] = replacement.to(torch.bfloat16)
        expected[1, loc] = -values

        # Write and overwrite the cache on MLU.
        mlu_loc = loc.to("mlu")
        pool.set_kv_buffer(layer, mlu_loc, values.to("mlu"), None)
        pool.set_kv_buffer(
            layer, KVWriteLoc(loc=overwrite_loc.to("mlu")), replacement.to("mlu"), None
        )
        pool.set_kv_buffer(
            layer, mlu_loc, (-values).to("mlu"), None, layer_id_override=6
        )

        # Check the full cache and gathered rows against the CPU reference.
        for layer_id in (5, 6):
            torch.testing.assert_close(
                pool.get_key_buffer(layer_id).cpu(),
                expected[layer_id - 5],
                rtol=0,
                atol=0,
            )
            nope, rope = pool.get_mla_kv_buffer(
                layer, mlu_loc, layer_id_override=layer_id
            )
            torch.testing.assert_close(
                torch.cat((nope, rope), dim=-1).cpu(),
                expected[layer_id - 5, loc],
                rtol=0,
                atol=0,
            )

    def test_gather_dtype_and_optional_rope(self):
        """The pool preserves latent/RoPE components when selecting output dtype."""
        layer = SimpleNamespace(layer_id=5)
        loc = torch.tensor([16, 1], device="mlu")

        for rope_dim in (64, 0):
            with self.subTest(rope_dim=rope_dim):
                # Create the pool and input tensor.
                pool = self._create_pool(rope_dim=rope_dim)
                values = torch.randn(2, 1, 512 + rope_dim).to(torch.bfloat16)

                # Convert the reference values to FP32 on CPU.
                expected = values.float()

                # Write the cache and gather FP32 outputs on MLU.
                pool.set_kv_buffer(layer, loc, values.to("mlu"), None)
                nope, rope = pool.get_mla_kv_buffer(layer, loc, dst_dtype=torch.float32)

                # Check the latent values and optional RoPE output.
                torch.testing.assert_close(
                    nope.cpu(), expected[..., :512], rtol=0, atol=0
                )
                if rope_dim:
                    torch.testing.assert_close(
                        rope.cpu(), expected[..., 512:], rtol=0, atol=0
                    )
                else:
                    self.assertIsNone(rope)

    def test_unsupported_cache_formats_rejected(self):
        """Reject layouts and dtypes the dense BF16 cache writer cannot represent."""
        for dtype, use_dsa in ((torch.float16, False), (torch.bfloat16, True)):
            with self.subTest(dtype=dtype, use_dsa=use_dsa):
                with self.assertRaises(NotImplementedError):
                    self._create_pool(dtype=dtype, use_dsa=use_dsa)


if __name__ == "__main__":
    unittest.main()
