"""Ascend coverage for the shared compact DCP Triton kernels."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1, suite="base-a-test-1-npu-a2")

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def _npu_is_available() -> bool:
    return torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()


@unittest.skipUnless(_npu_is_available(), "Ascend NPU is required")
class TestNpuDcpCompactKernels(CustomTestCase):
    device = "npu"

    def test_shared_dcp_prefix_index_kernel(self):
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_chunked_prefix_cache_kv_indices,
        )

        req_to_token = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
            device=self.device,
        )
        prefix_lens = torch.tensor([3, 2], dtype=torch.int32, device=self.device)
        cu_lens = torch.tensor([0, 3], dtype=torch.int32, device=self.device)
        indices = torch.empty(5, dtype=torch.int32, device=self.device)
        create_chunked_prefix_cache_kv_indices[(2,)](
            req_to_token,
            torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            torch.zeros(2, dtype=torch.int32, device=self.device),
            prefix_lens,
            cu_lens,
            indices,
            req_to_token.shape[1],
        )

        self.assertEqual(cu_lens.cpu().tolist(), [0, 3])
        self.assertEqual(indices.cpu().tolist(), [10, 11, 12, 20, 21])

    def test_shared_mla_dcp_page_table_kernel(self):
        from sglang.srt.hardware_backend.npu.attention.dcp_metadata import (
            build_mla_dcp_local_block_tables,
        )

        req_to_token = (
            torch.arange(16, dtype=torch.int64, device=self.device).view(1, 16) + 56
        )
        block_tables, local_lens = build_mla_dcp_local_block_tables(
            req_to_token,
            torch.tensor([0], dtype=torch.int64, device=self.device),
            torch.tensor([10], dtype=torch.int32, device=self.device),
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=1,
        )

        self.assertEqual(local_lens.cpu().tolist(), [5])
        self.assertEqual(block_tables.cpu().tolist(), [[7, 8]])

    def test_shared_page_tables_match_cyclic_reference(self):
        from sglang.srt.hardware_backend.npu.attention.dcp_metadata import (
            build_mla_dcp_local_block_tables,
        )

        # Non-monotonic page IDs, reordered requests, page boundaries, and an
        # empty row. A fixed-width graph table also has unused trailing pages.
        for dcp_size in (1, 2, 4):
            page_size = 4
            logical_page = page_size * dcp_size
            req_to_token = (
                torch.tensor([[7, 3, 11], [2, 9, 5], [4, 8, 6]])[:, :, None]
                * logical_page
                + torch.arange(logical_page)[None, None, :]
            ).reshape(3, -1)
            reqs = torch.tensor([2, 0, 1])
            seq_lens = torch.tensor([2 * logical_page + 1, logical_page - 1, 0])
            for rank in range(dcp_size):
                for width in (None, 5, 1):
                    with self.subTest(dcp_size=dcp_size, rank=rank, width=width):
                        table, local_lens = build_mla_dcp_local_block_tables(
                            req_to_token.to(self.device),
                            reqs.to(self.device),
                            seq_lens.to(self.device),
                            page_size,
                            dcp_size,
                            rank,
                            num_pages=width,
                        )
                        expected = torch.zeros_like(table, device="cpu")
                        expected_lens = torch.tensor(
                            [len(range(rank, int(n), dcp_size)) for n in seq_lens],
                            dtype=torch.int32,
                        )
                        for row, req in enumerate(reqs):
                            for col in range(expected.shape[1]):
                                pos = rank + col * logical_page
                                if col * page_size < expected_lens[row]:
                                    expected[row, col] = (
                                        req_to_token[req, pos] // logical_page
                                    )
                        torch.testing.assert_close(table.cpu(), expected)
                        torch.testing.assert_close(local_lens.cpu(), expected_lens)

    def test_page_table_bounds_mask_rounded_graph_columns(self):
        from sglang.srt.hardware_backend.npu.attention.dcp_metadata import (
            build_mla_dcp_local_block_tables,
        )

        # A padded graph length can reach beyond the available request-table
        # columns. Preserve the previous implementation's zero-filled tail.
        table, _ = build_mla_dcp_local_block_tables(
            torch.arange(56, 65, device=self.device).view(1, 9),
            torch.tensor([0], device=self.device),
            torch.tensor([16], device=self.device),
            4,
            2,
            1,
            num_pages=3,
        )
        self.assertEqual(table.cpu().tolist(), [[7, 0, 0]])

    def test_empty_page_table_does_not_launch_zero_grid(self):
        from sglang.srt.hardware_backend.npu.attention.dcp_metadata import (
            build_mla_dcp_local_block_tables,
        )

        table, lens = build_mla_dcp_local_block_tables(
            torch.empty((1, 16), dtype=torch.int64, device=self.device),
            torch.empty(0, dtype=torch.int64, device=self.device),
            torch.empty(0, dtype=torch.int32, device=self.device),
            4,
            2,
            0,
        )
        torch.npu.synchronize()
        self.assertEqual(table.shape, (0, 1))
        self.assertEqual(lens.shape, (0,))

    def test_shared_planner_runs_prefix_kernel_on_npu(self):
        from unittest.mock import patch

        from sglang.kernels.ops.kvcache.kv_indices import (
            create_chunked_prefix_cache_kv_indices,
        )
        from sglang.srt import runtime_context as rc
        from sglang.srt.layers.dcp.planner import (
            prepare_decode_context_parallel_metadata,
        )
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator

        translator = KVIndexTranslator.__new__(KVIndexTranslator)
        translator.is_translating = False
        backend = SimpleNamespace(
            dcp_use_packed_kv=False, kv_index_translator=translator
        )
        table = torch.arange(40, 64, dtype=torch.int32, device=self.device).view(3, 8)
        for prefixes in ([4, 8], [0, 4], [0, 0]):
            prefix_lens = torch.tensor(prefixes, dtype=torch.int32, device=self.device)
            extend_lens = torch.tensor([2, 3], dtype=torch.int32, device=self.device)
            for rank in range(4):
                with (
                    self.subTest(prefixes=prefixes, rank=rank),
                    rc.get_parallel().override(
                        dcp_enabled=True, dcp_size=4, dcp_rank=rank, attn_dcp_size=4
                    ),
                    patch(
                        "sglang.srt.layers.dcp.planner.get_attn_backend",
                        return_value=backend,
                    ),
                    patch(
                        "sglang.srt.layers.dcp.planner.get_device",
                        return_value=SimpleNamespace(device=self.device),
                    ),
                ):
                    metadata = prepare_decode_context_parallel_metadata(
                        seq_lens=prefix_lens + extend_lens,
                        extend_prefix_lens=prefix_lens,
                        extend_prefix_lens_cpu=prefixes,
                        extend_seq_lens=extend_lens,
                        req_pool_indices=torch.tensor([2, 0], device=self.device),
                        req_to_token=table,
                        seq_lens_sum=sum(prefixes) + 5,
                        kv_buffer_shape=torch.Size([32, 1]),
                        kv_cache_dtype=torch.bfloat16,
                        kv_cache_device=self.device,
                        create_chunked_prefix_cache_kv_indices_fn=create_chunked_prefix_cache_kv_indices,
                    )
                    expected = torch.cat(
                        [table[2, : prefixes[0]], table[0, : prefixes[1]]]
                    )
                    torch.testing.assert_close(
                        metadata.dcp_local_prefix_kv_indices, expected[rank::4] // 4
                    )
                    self.assertIsNone(metadata.dcp_kv_buffer)

    def test_compact_buffers_survive_graph_replay(self):
        """Real NPU graph kernels; local copy substitutes for HCCL transport."""
        from sglang.srt.layers.dcp.comm import (
            create_dcp_a2a_buffers,
            dcp_a2a_lse_reduce,
        )

        output = torch.randn(2, 12, 16, dtype=torch.bfloat16, device=self.device)
        lse = torch.randn(2, 12, dtype=torch.float32, device=self.device)
        group = SimpleNamespace(
            world_size=2, all_to_all_single=lambda dst, src: dst.copy_(src)
        )
        buffers = create_dcp_a2a_buffers(
            2, 2, 6, 16, dtype=output.dtype, device=self.device
        )
        ptrs = {key: value.data_ptr() for key, value in buffers.items()}
        for _ in range(3):
            dcp_a2a_lse_reduce(output, lse, group, cuda_graph_buffers=buffers)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            merged = dcp_a2a_lse_reduce(output, lse, group, cuda_graph_buffers=buffers)
        for empty_shard in (False, True, False):
            output.copy_(torch.randn_like(output))
            lse.copy_(torch.randn_like(lse))
            if empty_shard:
                lse[0] = float("-inf")
            graph.replay()
            eager = dcp_a2a_lse_reduce(output, lse, group)
            torch.testing.assert_close(merged, eager, atol=0, rtol=0)
            self.assertEqual(
                ptrs, {key: value.data_ptr() for key, value in buffers.items()}
            )

    def test_pack_preserves_bf16_output_and_fp32_lse_bits(self):
        from sglang.kernels.ops.attention.dcp_kernels import dcp_pack_a2a_send

        num_shards, batch_size, local_heads, head_dim = 2, 2, 6, 512
        total_heads = num_shards * local_heads
        output = torch.randn(
            batch_size,
            total_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        lse = torch.randn(
            batch_size, total_heads, device=self.device, dtype=torch.float32
        )
        packed = torch.empty(
            num_shards,
            batch_size,
            local_heads,
            head_dim + 2,
            device=self.device,
            dtype=torch.bfloat16,
        )

        dcp_pack_a2a_send(
            output,
            lse,
            packed[..., :head_dim],
            packed.view(torch.float32)[..., head_dim // 2],
        )

        expected_output = output.view(
            batch_size, num_shards, local_heads, head_dim
        ).permute(1, 0, 2, 3)
        expected_lse = lse.view(batch_size, num_shards, local_heads).permute(1, 0, 2)
        self.assertTrue(torch.equal(packed[..., :head_dim], expected_output))
        self.assertTrue(
            torch.equal(packed.view(torch.float32)[..., head_dim // 2], expected_lse)
        )

    def test_fused_unpack_merge_handles_empty_graph_row(self):
        from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton

        output = torch.randn(2, 2, 6, 512, device=self.device, dtype=torch.bfloat16)
        lse = torch.randn(2, 2, 6, device=self.device, dtype=torch.float32)
        lse[:, 0, 0] = float("-inf")

        merged, _ = dcp_lse_combine_triton(output, lse, is_lse_base_on_e=True)

        reference_weights = torch.softmax(lse.cpu(), dim=0)
        reference = (output.cpu().float() * reference_weights.unsqueeze(-1)).sum(dim=0)
        reference[0, 0] = 0
        torch.testing.assert_close(
            merged.cpu().float(), reference, atol=1e-2, rtol=1e-2
        )
        self.assertTrue(torch.equal(merged[0, 0], torch.zeros_like(merged[0, 0])))


if __name__ == "__main__":
    unittest.main()
