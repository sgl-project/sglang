"""Round-trip tests for the DeepSeek-V4 KV pool CPU offload path.

Each test seeds known bytes at physical slots, saves them via
``get_cpu_copy``, clobbers the device buffer, restores via
``load_cpu_copy`` at distinct slots, and asserts the bytes round-trip.
"""

import unittest

import torch

CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(CUDA_AVAILABLE, "Requires CUDA for DSv4 byte-paged pools")
class TestDeepSeekV4SingleKVPoolRoundTrip(unittest.TestCase):
    # Pinned to NopeFp8RopeBf16Pack.__post_init__'s assertions on the
    # currently-released DSv4 weights.
    NOPE_HEAD_DIM = 448
    ROPE_HEAD_DIM = 64

    def _make_pool(self, page_size=8, layer_num=2, size=64):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4SingleKVPool,
        )

        return DeepSeekV4SingleKVPool(
            size=size,
            page_size=page_size,
            dtype=torch.float8_e4m3fn,
            qk_nope_head_dim=self.NOPE_HEAD_DIM,
            qk_rope_head_dim=self.ROPE_HEAD_DIM,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
        )

    def _random_pack(self, pool, n_tokens):
        # k_nope_fp8 is uint8-viewed-as-fp8; the kernel touches raw bytes.
        from sglang.srt.layers.attention.dsv4.index_buf_accessor import (
            NopeFp8RopeBf16Pack,
            fp8_dtype,
        )

        nope_dim = pool.qk_nope_head_dim
        rope_dim = pool.qk_rope_head_dim
        scale_dim = pool.qk_nope_head_dim // pool.quantize_block_size
        k_nope_uint8 = torch.randint(
            0, 256, (n_tokens, nope_dim), dtype=torch.uint8, device="cuda"
        )
        k_rope = torch.randn(
            n_tokens, rope_dim, dtype=pool.rope_storage_dtype, device="cuda"
        )
        scale = torch.randint(
            0, 256, (n_tokens, scale_dim), dtype=torch.uint8, device="cuda"
        )
        return NopeFp8RopeBf16Pack(
            k_nope_fp8=k_nope_uint8.view(fp8_dtype),
            k_rope_bf16=k_rope,
            scale_k_nope_ue8m0=scale,
        )

    def test_get_then_load_with_distinct_indices(self):
        from sglang.srt.layers.attention.dsv4 import index_buf_accessor

        pool = self._make_pool()
        layer_num = pool.layer_num

        n_tokens = 12
        # Source slots span two pages; pick something that isn't page-aligned
        # so the page-internal offsets actually exercise the kernel.
        src_loc = torch.tensor(
            [1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 17],
            dtype=torch.int64,
            device="cuda",
        )

        per_layer_packs = []
        for layer_id in range(layer_num):
            pack = self._random_pack(pool, n_tokens)
            index_buf_accessor.SetKAndS.execute(
                pool=pool,
                buf=pool.kv_buffer[layer_id],
                loc=src_loc,
                nope_fp8_rope_bf16_pack=pack,
            )
            per_layer_packs.append(pack)

        cpu_copy = pool.get_cpu_copy(src_loc)

        for buf in pool.kv_buffer:
            buf.zero_()

        # Distinct dst_loc; the i-th input pack must land at the i-th dst slot.
        dst_loc = torch.tensor(
            [33, 34, 35, 37, 39, 41, 42, 43, 45, 46, 47, 49],
            dtype=torch.int64,
            device="cuda",
        )
        pool.load_cpu_copy(cpu_copy, dst_loc)

        for layer_id in range(layer_num):
            restored = index_buf_accessor.GetKAndS.execute(
                pool=pool, buf=pool.kv_buffer[layer_id], loc=dst_loc
            )
            expected = per_layer_packs[layer_id]
            torch.testing.assert_close(
                restored.k_nope_fp8.view(torch.uint8),
                expected.k_nope_fp8.view(torch.uint8),
                msg=f"k_nope mismatch on layer {layer_id}",
            )
            torch.testing.assert_close(
                restored.k_rope_bf16,
                expected.k_rope_bf16,
                msg=f"k_rope mismatch on layer {layer_id}",
            )
            torch.testing.assert_close(
                restored.scale_k_nope_ue8m0,
                expected.scale_k_nope_ue8m0,
                msg=f"scale mismatch on layer {layer_id}",
            )

    def test_empty_indices_short_circuits(self):
        pool = self._make_pool()
        empty = torch.empty(0, dtype=torch.int64, device="cuda")
        cpu_copy = pool.get_cpu_copy(empty)
        self.assertEqual(len(cpu_copy), pool.layer_num)
        pool.load_cpu_copy(cpu_copy, empty)


@unittest.skipUnless(CUDA_AVAILABLE, "Requires CUDA")
class TestDeepSeekV4IndexerPoolRoundTrip(unittest.TestCase):
    def _make_pool(self, page_size=64, layer_num=2, size=512):
        # index_buf_accessor.SetKAndS hard-codes buf_numel_per_page == 64*(128+4).
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4IndexerPool,
        )

        return DeepSeekV4IndexerPool(
            size=size,
            page_size=page_size,
            dtype=torch.uint8,
            index_head_dim=128,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
        )

    def test_per_token_round_trip(self):
        from sglang.srt.layers.attention.dsv4.index_buf_accessor import (
            fp8_dtype,
        )
        from sglang.srt.layers.attention.nsa import index_buf_accessor

        pool = self._make_pool()
        n_tokens = 6
        src_loc = torch.tensor(
            [3, 12, 50, 65, 130, 200], dtype=torch.int64, device="cuda"
        )
        per_layer = []
        for layer_id in range(pool.layer_num):
            k_uint = torch.randint(
                0,
                256,
                (n_tokens, pool.index_head_dim),
                dtype=torch.uint8,
                device="cuda",
            )
            index_k = k_uint.view(fp8_dtype)
            index_k_scale = torch.randn(
                n_tokens, 1, dtype=torch.float32, device="cuda"
            )
            buf = pool.index_k_with_scale_buffer[layer_id]
            index_buf_accessor.SetKAndS.execute(
                pool=pool,
                buf=buf,
                loc=src_loc,
                index_k=index_k,
                index_k_scale=index_k_scale,
            )
            expected_k_bytes, expected_s_bytes = pool._gather_per_token_index_k_scale(
                layer_id, src_loc
            )
            per_layer.append((expected_k_bytes.clone(), expected_s_bytes.clone()))

        cpu_copy = pool.get_cpu_copy(src_loc)
        for buf in pool.index_k_with_scale_buffer:
            buf.zero_()

        dst_loc = torch.tensor(
            [80, 81, 100, 132, 250, 300], dtype=torch.int64, device="cuda"
        )
        pool.load_cpu_copy(cpu_copy, dst_loc)

        for layer_id in range(pool.layer_num):
            restored_k, restored_s = pool._gather_per_token_index_k_scale(
                layer_id, dst_loc
            )
            expected_k, expected_s = per_layer[layer_id]
            torch.testing.assert_close(restored_k, expected_k)
            torch.testing.assert_close(restored_s, expected_s)


@unittest.skipUnless(CUDA_AVAILABLE, "Requires CUDA")
class TestCompressStatePoolRoundTrip(unittest.TestCase):
    def test_round_trip_with_duplicate_state_locs(self):
        from sglang.srt.mem_cache.deepseek_v4_compress_state import (
            CompressStatePool,
        )

        pool = CompressStatePool(
            size=64,
            ring_size=4,
            overlap=False,
            head_dim=16,
            dtype=torch.float32,
            device="cuda",
            enable_memory_saver=False,
            ratio=4,
        )

        old_state_locs = torch.tensor(
            [4, 5, 4, 6, 5], dtype=torch.int64, device="cuda"
        )
        unique_old = torch.unique(old_state_locs).tolist()
        seeded = {}
        for row in unique_old:
            v = torch.randn(
                pool.kv_score_buffer.kv_score.shape[1:],
                dtype=torch.float32,
                device="cuda",
            )
            pool.kv_score_buffer.kv_score[row] = v
            seeded[row] = v

        cpu = pool.get_cpu_copy(old_state_locs)
        pool.kv_score_buffer.kv_score.zero_()

        new_state_locs = torch.tensor(
            [12, 13, 12, 14, 13], dtype=torch.int64, device="cuda"
        )
        pool.load_cpu_copy(cpu, new_state_locs)

        old_to_new = {4: 12, 5: 13, 6: 14}
        for old_row, new_row in old_to_new.items():
            torch.testing.assert_close(
                pool.kv_score_buffer.kv_score[new_row],
                seeded[old_row],
                msg=f"row {old_row} -> {new_row}",
            )


class TestNoTryCatchInReleaseReq(unittest.TestCase):
    # The DSv4 fix lives at the pool layer; release_req must not catch
    # NotImplementedError (which would silently abort retracted requests).

    def test_release_req_does_not_swallow_not_implemented(self):
        import inspect

        from sglang.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.release_req)
        self.assertNotIn("NotImplementedError", src)


if __name__ == "__main__":
    unittest.main()
