# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Round-trip tests for MHATokenToKVPool CPU offload under both KV layouts.

`get_cpu_copy` / `load_cpu_copy` move KV cache between device and host by slot
id. The NHD layout stores a slot as a contiguous row `[slot, :, :]`, while the
HND layout (SGLANG_USE_HND_KVCACHE) folds (page, head) and stores a slot as
`[page, :, off, :]`. These tests write sentinel values, offload to CPU, clear
the device buffers, load back, and assert the original values are restored for
both layouts.

    python -m pytest \
        test/registered/unit/mem_cache/test_mha_pool_cpu_offload_hnd.py -v
"""

import types
import unittest

import torch

from sglang.srt import environ as srt_environ
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestMHAPoolCPUOffload(unittest.TestCase):
    def _build_pool(
        self,
        use_hnd: bool,
        page_size: int,
        device: str = "cpu",
        v_head_dim: int | None = None,
    ):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        with srt_environ.envs.SGLANG_USE_HND_KVCACHE.override(use_hnd):
            pool = MHATokenToKVPool(
                size=16,
                page_size=page_size,
                dtype=torch.bfloat16,
                head_num=2,
                head_dim=8,
                v_head_dim=v_head_dim,
                layer_num=2,
                device=device,
                enable_memory_saver=False,
            )
        self.assertEqual(pool.use_hnd, use_hnd)
        return pool

    def _fill_sentinels(self, pool, indices):
        layer = types.SimpleNamespace(layer_id=0)
        n = len(indices)
        for layer_id in range(pool.layer_num):
            layer.layer_id = layer_id
            token_values = torch.arange(n, dtype=torch.float32, device=pool.device)[
                :, None, None
            ]
            head_values = torch.arange(
                pool.head_num, dtype=torch.float32, device=pool.device
            )[None, :, None]
            cache_k = token_values * 24 + head_values * pool.head_dim
            cache_k = (
                cache_k
                + torch.arange(pool.head_dim, dtype=torch.float32, device=pool.device)[
                    None, None, :
                ]
            )
            cache_v = token_values * 24 + head_values * pool.v_head_dim
            cache_v = (
                cache_v
                + torch.arange(
                    pool.v_head_dim, dtype=torch.float32, device=pool.device
                )[None, None, :]
            )
            cache_k = (cache_k + layer_id * 128).to(torch.bfloat16)
            cache_v = (-cache_v - layer_id * 128 - 1).to(torch.bfloat16)
            pool.set_kv_buffer(layer, indices, cache_k.clone(), cache_v.clone())

    def _read_slots(self, pool, indices):
        out = []
        for layer_id in range(pool.layer_num):
            out.append(
                (
                    pool.get_key_buffer(layer_id)[indices].clone(),
                    pool.get_value_buffer(layer_id)[indices].clone(),
                )
                if not pool.use_hnd
                else (
                    pool.get_key_buffer(layer_id)[
                        indices // pool.page_size, :, indices % pool.page_size, :
                    ].clone(),
                    pool.get_value_buffer(layer_id)[
                        indices // pool.page_size, :, indices % pool.page_size, :
                    ].clone(),
                )
            )
        return out

    def _run_round_trip(
        self,
        use_hnd: bool,
        page_size: int,
        device: str = "cpu",
        chunk_size: int = 8192,
        v_head_dim: int | None = None,
    ):
        pool = self._build_pool(
            use_hnd, page_size, device=device, v_head_dim=v_head_dim
        )
        pool.cpu_offloading_chunk_size = chunk_size
        indices = torch.tensor([0, 3, 5, 8, 11], dtype=torch.int64, device=pool.device)

        self._fill_sentinels(pool, indices)
        expected = self._read_slots(pool, indices)

        kv_cache_cpu = pool.get_cpu_copy(indices)
        for layer_chunks in kv_cache_cpu:
            for chunk_cpu in layer_chunks:
                k_cpu, v_cpu = chunk_cpu[:2]
                self.assertEqual(k_cpu.device.type, "cpu")
                self.assertEqual(v_cpu.device.type, "cpu")

        for layer_id in range(pool.layer_num):
            pool.k_buffer[layer_id].zero_()
            pool.v_buffer[layer_id].zero_()

        pool.load_cpu_copy(kv_cache_cpu, indices)
        restored = self._read_slots(pool, indices)

        for layer_id in range(pool.layer_num):
            self.assertTrue(
                torch.equal(restored[layer_id][0], expected[layer_id][0]),
                f"K mismatch layer={layer_id} use_hnd={use_hnd} ps={page_size}",
            )
            self.assertTrue(
                torch.equal(restored[layer_id][1], expected[layer_id][1]),
                f"V mismatch layer={layer_id} use_hnd={use_hnd} ps={page_size}",
            )

    def test_round_trip_nhd(self):
        self._run_round_trip(use_hnd=False, page_size=1)

    def test_round_trip_hnd_page_size_1(self):
        self._run_round_trip(use_hnd=True, page_size=1)

    def test_round_trip_hnd_page_size_4(self):
        self._run_round_trip(use_hnd=True, page_size=4)

    def test_round_trip_hnd_different_value_head_dim(self):
        self._run_round_trip(use_hnd=True, page_size=4, v_head_dim=4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_round_trip_hnd_cuda_chunked(self):
        self._run_round_trip(use_hnd=True, page_size=4, device="cuda", chunk_size=2)

    def test_hnd_matches_nhd(self):
        """HND and NHD must offload/restore identical logical KV values."""
        page_size = 4
        indices = torch.tensor([0, 3, 5, 8, 11], dtype=torch.int64)

        pool_nhd = self._build_pool(use_hnd=False, page_size=page_size)
        pool_hnd = self._build_pool(use_hnd=True, page_size=page_size)
        self._fill_sentinels(pool_nhd, indices)
        self._fill_sentinels(pool_hnd, indices)

        cpu_nhd = pool_nhd.get_cpu_copy(indices)
        cpu_hnd = pool_hnd.get_cpu_copy(indices)
        for layer_id in range(pool_nhd.layer_num):
            for chunk_nhd, chunk_hnd in zip(cpu_nhd[layer_id], cpu_hnd[layer_id]):
                self.assertTrue(torch.equal(chunk_nhd[0], chunk_hnd[0]))
                self.assertTrue(torch.equal(chunk_nhd[1], chunk_hnd[1]))

    def test_hnd_rejects_quantized_kv_cache(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        with srt_environ.envs.SGLANG_USE_HND_KVCACHE.override(True):
            with self.assertRaisesRegex(
                ValueError, "Quantized KV cache does not support"
            ):
                MHATokenToKVPool(
                    size=16,
                    page_size=4,
                    dtype=torch.bfloat16,
                    head_num=2,
                    head_dim=8,
                    layer_num=2,
                    device="cpu",
                    enable_memory_saver=False,
                    quant_method=object(),
                )

    def test_quantized_nhd_round_trip_restores_scales(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
        )
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        with srt_environ.envs.SGLANG_USE_HND_KVCACHE.override(False):
            pool = MHATokenToKVPool(
                size=16,
                page_size=1,
                dtype=torch.bfloat16,
                head_num=2,
                head_dim=16,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                quant_method=FP4MXBlock16KVCacheMethod(),
            )
        pool.cpu_offloading_chunk_size = 2
        indices = torch.tensor([1, 3, 7], dtype=torch.int64)

        expected = []
        for layer_id in range(pool.layer_num):
            buffers = (
                pool.k_buffer[layer_id],
                pool.v_buffer[layer_id],
                pool.k_scale_buffer[layer_id],
                pool.v_scale_buffer[layer_id],
            )
            for value, buffer in enumerate(buffers, start=1):
                buffer[indices] = value + layer_id
            expected.append(tuple(buffer[indices].clone() for buffer in buffers))

        cpu_copy = pool.get_cpu_copy(indices)
        self.assertTrue(all(len(chunk) == 4 for chunks in cpu_copy for chunk in chunks))

        for layer_id in range(pool.layer_num):
            pool.k_buffer[layer_id][indices] = 0
            pool.v_buffer[layer_id][indices] = 0
            pool.k_scale_buffer[layer_id][indices] = 0
            pool.v_scale_buffer[layer_id][indices] = 0

        pool.load_cpu_copy(cpu_copy, indices)

        for layer_id in range(pool.layer_num):
            restored = (
                pool.k_buffer[layer_id][indices],
                pool.v_buffer[layer_id][indices],
                pool.k_scale_buffer[layer_id][indices],
                pool.v_scale_buffer[layer_id][indices],
            )
            for actual, wanted in zip(restored, expected[layer_id]):
                self.assertTrue(torch.equal(actual, wanted))

    def test_swa_filter_preserves_quantized_scale_chunks(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        pool = object.__new__(SWAKVPool)
        pool.swa_kv_pool = types.SimpleNamespace(cpu_offloading_chunk_size=2)
        layer_chunks = [
            [torch.arange(3) + offset for offset in (0, 10, 20, 30)],
            [torch.arange(3, 6) + offset for offset in (0, 10, 20, 30)],
        ]
        row_mask = torch.tensor([True, False, True, False, True, True])

        filtered = pool._filter_swa_cpu_copy([layer_chunks], row_mask)

        self.assertTrue(all(len(chunk) == 4 for chunk in filtered[0]))
        for tensor_id, offset in enumerate((0, 10, 20, 30)):
            actual = torch.cat([chunk[tensor_id] for chunk in filtered[0]])
            expected = (torch.arange(6) + offset)[row_mask]
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
