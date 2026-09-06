"""Tests for the temporary KV buffer used by DSA CP+DCP prefill."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.dcp import comm as dcp_comm
from sglang.srt.layers.dcp import planner as dcp_planner
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
RAW_KV_ROW_WIDTH = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PACKED_NOPE_ROW_BYTES = 528
PACKED_ROPE_ROW_BYTES = 128
PACKED_KV_ROW_WIDTH = PACKED_NOPE_ROW_BYTES + PACKED_ROPE_ROW_BYTES


class _FakeTritonKernel:
    def __init__(self, callback):
        self.callback = callback

    def __getitem__(self, _grid):
        return self.callback


class TestDSADCPMetadataPlanner(CustomTestCase):
    def _prepare_metadata(self, kv_dtype: torch.dtype, row_width: int):
        prefix_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int32)

        def fill_prefix_indices(*args, **_kwargs):
            args[5].copy_(prefix_indices)

        def fill_dcp_indices(*args, **_kwargs):
            args[5].copy_(torch.arange(args[5].numel(), dtype=torch.int32))

        parallel = SimpleNamespace(dcp_enabled=True, dcp_size=2, dcp_rank=1)
        device = SimpleNamespace(device=torch.device("cpu"))
        with (
            patch.object(dcp_planner, "get_parallel", return_value=parallel),
            patch.object(dcp_planner, "get_device", return_value=device),
            patch.object(
                dcp_planner,
                "create_dcp_kv_indices",
                _FakeTritonKernel(fill_dcp_indices),
            ),
        ):
            return dcp_planner.prepare_decode_context_parallel_metadata(
                seq_lens=torch.tensor([3, 2], dtype=torch.int32),
                extend_prefix_lens=torch.tensor([2, 2], dtype=torch.int32),
                extend_prefix_lens_cpu=[2, 2],
                extend_seq_lens=torch.tensor([1, 0], dtype=torch.int32),
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
                req_to_token=torch.empty((2, 8), dtype=torch.int32),
                seq_lens_sum=5,
                kv_buffer_shape=torch.Size([128, 1, row_width]),
                kv_cache_dtype=kv_dtype,
                kv_cache_device=torch.device("cpu"),
                create_chunked_prefix_cache_kv_indices_fn=_FakeTritonKernel(
                    fill_prefix_indices
                ),
                kv_buffer_token_padding=64,
            )

    def test_preserves_raw_and_packed_kv_buffer_layouts(self):
        cases = [
            (torch.float16, RAW_KV_ROW_WIDTH),
            (torch.bfloat16, RAW_KV_ROW_WIDTH),
        ]
        if hasattr(torch, "float8_e4m3fn"):
            cases.append((torch.float8_e4m3fn, PACKED_KV_ROW_WIDTH))

        for kv_dtype, row_width in cases:
            with self.subTest(kv_dtype=kv_dtype, row_width=row_width):
                metadata = self._prepare_metadata(kv_dtype, row_width)
                self.assertEqual(metadata.dcp_kv_buffer.shape, (64, 1, row_width))
                self.assertEqual(metadata.dcp_kv_buffer.dtype, kv_dtype)
                torch.testing.assert_close(
                    metadata.dcp_kv_indptr,
                    torch.tensor([0, 3, 5], dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    metadata.dcp_local_prefix_kv_indices,
                    torch.tensor([0, 2], dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )

    def test_indexer_cache_covers_widened_allocator_padding(self):
        size = 4 * 64
        page_size = 64
        dcp_size = 8
        with get_parallel().override(
            dcp_enabled=True,
            attn_dcp_size=dcp_size,
        ):
            pool = DSATokenToKVPool(
                size=size,
                page_size=page_size,
                kv_lora_rank=KV_LORA_RANK,
                dtype=torch.bfloat16,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                layer_num=1,
                device="cpu",
                index_head_dim=128,
                enable_memory_saver=False,
                kv_cache_dim=RAW_KV_ROW_WIDTH,
            )

        indexer_capacity = pool.index_k_with_scale_buffer[0].shape[0] * page_size
        self.assertEqual(indexer_capacity, (size + page_size) * dcp_size)

    def test_flashmla_metadata_uses_kernel_head_bucket(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.dsa_index_topk = 2048
        cache_seqlens = torch.ones(2, dtype=torch.int32)
        # Default decode heads may differ from an explicit CP-prefill layout.
        for decode_heads, prefill_heads, expected in (
            (128, None, 128),
            (32, None, 64),
            (32, 128, 128),
            (8, None, 64),
        ):
            with (
                self.subTest(decode=decode_heads, prefill=prefill_heads),
                patch(
                    "sgl_kernel.flash_mla.get_mla_metadata", return_value=(None, None)
                ) as get_metadata,
            ):
                backend.flashmla_decode_q_heads = decode_heads
                backend._compute_flashmla_metadata(
                    cache_seqlens, seq_len_q=1, num_q_heads=prefill_heads
                )
                self.assertEqual(get_metadata.call_args.kwargs["num_heads_q"], expected)
                self.assertEqual(
                    get_metadata.call_args.kwargs["num_q_tokens_per_head_k"], expected
                )

    def test_prefill_page_table_preserves_original_batch_segments(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )

        backend = object.__new__(DeepseekSparseAttnBackend)
        dcp_meta = SimpleNamespace(
            dcp_kv_indptr=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
            dcp_kv_indices=torch.tensor([4, 5, 10, 11, 12, 20], dtype=torch.int32),
        )

        for batch_indices, seq_lens, expected in (
            (
                [0, 1, 2],
                [2, 3, 1],
                [[4, 5, -1], [10, 11, 12], [20, -1, -1]],
            ),
            ([1, 2], [3, 1], [[10, 11, 12], [20, -1, -1]]),
        ):
            with self.subTest(batch_indices=batch_indices):
                page_table = backend._build_dcp_prefill_page_table(
                    seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
                    dcp_meta=dcp_meta,
                    selected_batch_indices=batch_indices,
                )

                torch.testing.assert_close(
                    page_table,
                    torch.tensor(expected, dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )


class TestDSADCPKVBufferGather(CustomTestCase):
    def test_dcp_gather_restores_request_order(self):
        local_prefix = torch.tensor([0, 2, 10], dtype=torch.float16).view(3, 1, 1)
        gathered_padded = torch.tensor(
            [0, 1, 2, 99, 10, 11], dtype=torch.float16
        ).view(6, 1, 1)
        parallel = SimpleNamespace(dcp_enabled=True, dcp_size=2, dcp_rank=0)

        with (
            patch.object(dcp_comm, "get_parallel", return_value=parallel),
            patch.object(
                dcp_comm,
                "_all_gather_dcp_kv_cache",
                return_value=gathered_padded,
            ),
        ):
            actual = dcp_comm.all_gather_kv_cache_for_dcp(
                prefix_kv_a=local_prefix,
                prefix_k_pe=None,
                prefix_kv_lens_cpu=torch.tensor([3, 2], dtype=torch.int32),
            )

        expected = torch.tensor([0, 1, 2, 10, 11], dtype=torch.float16)
        torch.testing.assert_close(actual.flatten(), expected, rtol=0, atol=0)

    def test_raw_kv_gather_preserves_dtype_and_appends_extend_tokens(self):
        for kv_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(kv_dtype=kv_dtype):
                prefix = torch.arange(2 * RAW_KV_ROW_WIDTH, dtype=torch.float32)
                prefix = prefix.view(2, 1, RAW_KV_ROW_WIDTH).to(kv_dtype)
                cache_nope = torch.empty((1, 1, KV_LORA_RANK), dtype=kv_dtype)
                cache_rope = torch.empty((1, 1, QK_ROPE_HEAD_DIM), dtype=kv_dtype)
                pool = SimpleNamespace(
                    dsa_kv_cache_store_fp8=False,
                    get_mla_kv_buffer=MagicMock(
                        return_value=(cache_nope, cache_rope)
                    ),
                )
                attn_mqa = SimpleNamespace(layer_id=3)
                k_nope = torch.full((2, 1, KV_LORA_RANK), 7, dtype=kv_dtype)
                k_pe = torch.full((2, 1, QK_ROPE_HEAD_DIM), 11, dtype=kv_dtype)
                buffer = torch.full((8, 1, RAW_KV_ROW_WIDTH), 19, dtype=kv_dtype)

                with patch.object(
                    dcp_comm, "all_gather_kv_cache_for_dcp", return_value=prefix
                ):
                    dcp_comm.all_gather_kv_cache_for_mla_extend(
                        pool,
                        attn_mqa,
                        extend_prefix_lens_cpu=[2],
                        dcp_local_prefix_kv_indices=torch.tensor(
                            [0], dtype=torch.int32
                        ),
                        dcp_extend_prefix_lens_sum=2,
                        dcp_kv_buffer=buffer,
                        kv_lora_rank=KV_LORA_RANK,
                        k_nope=k_nope,
                        k_pe=k_pe,
                    )

                pool.get_mla_kv_buffer.assert_called_once()
                self.assertEqual(
                    pool.get_mla_kv_buffer.call_args.kwargs["dst_dtype"], kv_dtype
                )
                torch.testing.assert_close(buffer[:2], prefix)
                torch.testing.assert_close(
                    buffer[2:4, ..., :KV_LORA_RANK], k_nope
                )
                torch.testing.assert_close(
                    buffer[2:4, ..., KV_LORA_RANK:], k_pe
                )
                self.assertEqual(torch.count_nonzero(buffer[4:]).item(), 0)

    @unittest.skipUnless(
        hasattr(torch, "float8_e4m3fn"), "PyTorch FP8 support is required."
    )
    def test_packed_fp8_gather_is_byte_exact(self):
        persistent_bytes = (
            torch.arange(8 * PACKED_KV_ROW_WIDTH, dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
            .view(8, 1, PACKED_KV_ROW_WIDTH)
        )
        persistent = persistent_bytes.view(torch.float8_e4m3fn)
        pool = SimpleNamespace(
            dsa_kv_cache_store_fp8=True,
            get_key_buffer=MagicMock(return_value=persistent),
        )
        attn_mqa = SimpleNamespace(layer_id=5)
        gathered_prefix = persistent_bytes[[1, 3]].clone()
        current_nope = torch.full(
            (2, 1, PACKED_NOPE_ROW_BYTES), 17, dtype=torch.uint8
        )
        current_rope = torch.full(
            (2, 1, PACKED_ROPE_ROW_BYTES), 29, dtype=torch.uint8
        )
        buffer = torch.empty(
            (8, 1, PACKED_KV_ROW_WIDTH), dtype=torch.float8_e4m3fn
        )
        buffer.view(torch.uint8).fill_(255)

        with (
            patch.object(
                dcp_comm,
                "all_gather_kv_cache_for_dcp",
                return_value=gathered_prefix,
            ) as gather_mock,
            patch(
                "sglang.kernels.ops.attention.dsa.quant_k_cache."
                "quantize_k_cache_separate",
                return_value=(current_nope, current_rope),
            ),
        ):
            dcp_comm.all_gather_kv_cache_for_mla_extend(
                pool,
                attn_mqa,
                extend_prefix_lens_cpu=[2],
                dcp_local_prefix_kv_indices=torch.tensor([1, 3]),
                dcp_extend_prefix_lens_sum=2,
                dcp_kv_buffer=buffer,
                kv_lora_rank=KV_LORA_RANK,
                k_nope=torch.empty((2, 1, KV_LORA_RANK), dtype=torch.bfloat16),
                k_pe=torch.empty((2, 1, QK_ROPE_HEAD_DIM), dtype=torch.bfloat16),
            )

        pool.get_key_buffer.assert_called_once_with(5)
        gathered_input = gather_mock.call_args.args[0]
        self.assertEqual(gathered_input.dtype, torch.uint8)
        self.assertTrue(torch.equal(gathered_input, gathered_prefix))

        buffer_bytes = buffer.view(torch.uint8)
        self.assertTrue(torch.equal(buffer_bytes[:2], gathered_prefix))
        self.assertTrue(
            torch.equal(
                buffer_bytes[2:4, ..., :PACKED_NOPE_ROW_BYTES], current_nope
            )
        )
        self.assertTrue(
            torch.equal(
                buffer_bytes[2:4, ..., PACKED_NOPE_ROW_BYTES:], current_rope
            )
        )
        self.assertEqual(torch.count_nonzero(buffer_bytes[4:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
