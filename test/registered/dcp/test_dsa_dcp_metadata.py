"""RoPE DSA sparse-index ownership and TRTLLM LSE integration on Blackwell."""

import math
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.srt.layers.dcp.layout import remap_dcp_sparse_indices
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# TRTLLM's RoPE sparse MLA cubins require SM100/SM103. This test emulates
# independent KV shards on one device; no distributed process group is needed.
register_cuda_ci(est_time=40, stage="base-b", runner_config="4-gpu-b200")


def _reference_remap(indices, dcp_size, rank, interleave_size=1):
    rows = []
    counts = []
    for row in indices.tolist():
        selected = []
        for slot in row:
            if slot < 0:
                continue
            block, offset = divmod(slot, interleave_size)
            local_block, owner = divmod(block, dcp_size)
            if owner == rank:
                selected.append(local_block * interleave_size + offset)
        counts.append(len(selected))
        rows.append(selected + [-1] * (len(row) - len(selected)))
    return torch.tensor(rows, dtype=indices.dtype), torch.tensor(
        counts, dtype=torch.int32
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA sparse-index transform")
class TestDsaDcpSparseIndices(CustomTestCase):
    def test_repeated_rows_match_folded_query_heads(self):
        """Repeated compact rows must stay aligned with reshaped query heads."""
        for dtype in (torch.int32, torch.int64):
            for width in (37, 2048):
                cpu = torch.arange(3 * width, dtype=dtype).reshape(3, width).flip(1)
                cpu[0, ::3] = -1
                cpu[1].fill_(-1)
                cpu[2] += 2**40 if dtype == torch.int64 else 2**24
                storage = torch.empty((6, 2 * width), dtype=dtype, device="cuda")
                source = storage[::2, ::2]
                source.copy_(cpu)
                for dcp_size in (1, 2, 4):
                    if dcp_size == 1:
                        expected = cpu
                        expected_counts = (cpu >= 0).sum(dim=-1, dtype=torch.int32)
                    else:
                        expected, expected_counts = _reference_remap(
                            cpu, dcp_size, dcp_size - 1
                        )
                    for repeat_rows in (2, 3):
                        actual, counts = remap_dcp_sparse_indices(
                            source,
                            dcp_size,
                            dcp_size - 1,
                            return_counts=True,
                            repeat_rows=repeat_rows,
                        )
                        # A folded [B, heads, D] query exposes each B's head
                        # groups consecutively, so their metadata must follow.
                        torch.testing.assert_close(
                            actual.cpu(), expected.repeat_interleave(repeat_rows, dim=0)
                        )
                        torch.testing.assert_close(
                            counts.cpu(), expected_counts.repeat_interleave(repeat_rows)
                        )

    def test_matches_independent_reference_with_wide_slots_and_strides(self):
        """Keep score order, exact large-slot owners and zero-owner rows."""
        generator = torch.Generator().manual_seed(41926)
        for dtype in (torch.int32, torch.int64):
            for width in (1, 37, 2048, 2051):
                cpu = torch.randint(0, 8192, (4, width), generator=generator).to(dtype)
                cpu[0].fill_(-1)
                cpu[1] = cpu[1] * 4
                cpu[2, ::3] = -1
                cpu[3] += 2**40 if dtype == torch.int64 else 2**24
                # Both row and column strides differ from the compact output.
                storage = torch.empty((8, width * 2), dtype=dtype, device="cuda")
                source = storage[::2, ::2]
                source.copy_(cpu)
                for dcp_size in (2, 4):
                    for interleave_size in (1, 3, 64):
                        for rank in range(dcp_size):
                            with self.subTest(
                                dtype=dtype,
                                width=width,
                                dcp_size=dcp_size,
                                rank=rank,
                                interleave=interleave_size,
                            ):
                                expected, counts = _reference_remap(
                                    cpu, dcp_size, rank, interleave_size
                                )
                                actual, actual_counts = remap_dcp_sparse_indices(
                                    source,
                                    dcp_size,
                                    rank,
                                    interleave_size,
                                    return_counts=True,
                                )
                                torch.testing.assert_close(actual.cpu(), expected)
                                torch.testing.assert_close(actual_counts.cpu(), counts)

    def test_prefill_and_decode_map_positions_before_owner_filter(self):
        """A fused top-k global slot must match an unfused position lookup."""
        generator = torch.Generator().manual_seed(31821)
        page_table_cpu = torch.stack(
            [torch.randperm(4096, generator=generator) for _ in range(2)]
        ).int()
        positions_cpu = torch.randint(0, 4096, (3, 2048), generator=generator).int()
        positions_cpu[:, 3::7] = -1
        expected_global = torch.full_like(positions_cpu, -1)
        for row, request in enumerate((0, 0, 1)):
            valid = positions_cpu[row] >= 0
            expected_global[row, valid] = page_table_cpu[
                request, positions_cpu[row, valid].long()
            ]

        positions = positions_cpu.cuda()
        page_table = page_table_cpu.cuda()
        prefill = transform_index_page_table_prefill(
            page_table=page_table,
            topk_indices=positions,
            extend_lens_cpu=[2, 1],
        )
        decode = transform_index_page_table_decode(
            page_table=page_table[[0, 0, 1]], topk_indices=positions
        )
        for global_slots in (prefill, decode, expected_global.cuda()):
            torch.testing.assert_close(global_slots.cpu(), expected_global)
            for dcp_size in (2, 4):
                for rank in range(dcp_size):
                    actual, counts = remap_dcp_sparse_indices(
                        global_slots, dcp_size, rank, return_counts=True
                    )
                    expected, expected_counts = _reference_remap(
                        expected_global, dcp_size, rank
                    )
                    torch.testing.assert_close(actual.cpu(), expected)
                    torch.testing.assert_close(counts.cpu(), expected_counts)

    def test_cuda_graph_replay_replaces_prefix_and_counts(self):
        """A captured remap must clear stale entries when ownership changes."""
        for repeat_rows in (1, 2):
            source = torch.tensor([[0, 4, 8, 12, -1]], dtype=torch.int32, device="cuda")
            remap_dcp_sparse_indices(
                source, 4, 0, return_counts=True, repeat_rows=repeat_rows
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                indices, counts = remap_dcp_sparse_indices(
                    source, 4, 0, return_counts=True, repeat_rows=repeat_rows
                )
            source.fill_(1)
            graph.replay()
            self.assertEqual(indices.cpu().tolist(), [[-1] * 5] * repeat_rows)
            self.assertEqual(counts.cpu().tolist(), [0] * repeat_rows)
            source.copy_(
                torch.tensor([[8, -1, 0, 4, 1]], dtype=torch.int32, device="cuda")
            )
            graph.replay()
            self.assertEqual(indices.cpu().tolist(), [[2, 0, 1, -1, -1]] * repeat_rows)
            self.assertEqual(counts.cpu().tolist(), [3] * repeat_rows)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "TRTLLM RoPE sparse MLA requires Blackwell SM100/SM103",
)
class TestDsaDcpSparseLse(CustomTestCase):
    def test_captured_folded_lengths_survive_eager_growth(self):
        """An eager grow must not release the offsets recorded by zero-KV fixup."""
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        rows, heads, groups, topk = 8192, 64, 2, 2048
        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.device = "cuda"
        backend._arange_buf = torch.arange(16384, dtype=torch.int32, device="cuda")
        backend.dcp_enabled, backend.dcp_size, backend.dcp_rank = True, 2, 0
        backend.dcp_head_groups = groups
        backend.num_q_heads, backend.num_dcp_q_heads = 32, heads
        backend.dsa_index_topk, backend.dsa_index_kpool = topk, 1
        backend.dsa_drop_wide_page_table = True
        backend.dsa_decode_impl = "trtllm"
        backend.real_page_size, backend.kv_cache_dim = 64, 576
        backend.qk_nope_head_dim, backend.kv_lora_rank, backend.qk_rope_head_dim = (
            128,
            512,
            64,
        )
        backend.kv_cache_dtype = torch.bfloat16
        backend.use_fused_topk = True
        backend.dsa_topk_backend = SimpleNamespace(
            is_sgl_kernel=lambda: True, should_use_topk_v2=lambda: False
        )
        backend.set_dsa_prefill_impl = lambda **kwargs: None
        backend._init_kpool_metadata_capture = lambda metadata, *args: metadata
        backend.req_to_token = torch.zeros((rows, 64), dtype=torch.int32, device="cuda")
        backend.decode_cuda_graph_metadata = dict(
            cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
            real_page_table=torch.zeros((rows, 1), dtype=torch.int32, device="cuda"),
        )
        # Indexer scheduling is unrelated to attention's captured offset lifetime.
        with patch.object(dsa_backend, "is_cuda", return_value=False):
            backend._build_forward_metadata_cuda_graph(
                rows,
                rows,
                None,
                torch.full((rows,), 64, dtype=torch.int32, device="cuda"),
                None,
                ForwardMode.DECODE,
                None,
            )
        backend.workspace_buffer = torch.empty(
            2 << 30, dtype=torch.uint8, device="cuda"
        )
        backend._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(
                torch.device("cuda"), heads, rows
            )
        )
        kv = torch.ones((64, 576), dtype=torch.bfloat16, device="cuda")
        backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: kv)
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=heads, head_dim=576, scaling=1 / math.sqrt(192)
        )
        query = torch.zeros((rows, heads, 576), dtype=torch.bfloat16, device="cuda")
        indices = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
        indices[1::2, :64] = torch.arange(64, dtype=torch.int32, device="cuda") * 2

        def forward():
            return backend._forward_trtllm(
                q=query,
                k=None,
                v=None,
                layer=layer,
                forward_batch=None,
                seq_lens=None,
                save_kv_cache=False,
                topk_indices=indices,
            )

        with (
            dsa_backend.get_parallel().override(dcp_group=None),
            patch.object(
                dsa_backend, "use_symmetric_memory", side_effect=lambda _: nullcontext()
            ),
        ):
            for _ in range(3):
                forward()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, lse = forward()
            captured_storage = weakref.ref(backend._arange_buf)
            # A 16K eager prefill requests 32769 folded offsets. Its metadata
            # is discarded when decode metadata is restored for graph replay.
            backend.get_device_int32_arange(16384 * groups + 1)
            backend.forward_metadata = backend.decode_cuda_graph_metadata[rows]
            churn = [
                torch.full((32768,), 7, dtype=torch.int32, device="cuda")
                for _ in range(8)
            ]
            # Fail safely before replaying dangling offsets on the old code.
            self.assertIsNotNone(
                captured_storage(),
                "captured folded offsets were freed after eager growth",
            )
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(output[1::2], torch.ones_like(output[1::2]))
                self.assertTrue(bool((output[::2] == 0).all()))
                self.assertTrue(bool(torch.isneginf(lse[::2]).all()))
            self.assertEqual(len(churn), 8)

    def test_workspace_covers_full_prefill_rows(self):
        """Exercise production head folding, workspace sizing and grid-Z chunks."""
        from sglang.srt.layers.attention.dsa_backend import _dcp_trtllm_sparse_attention
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )
        from sglang.srt.mem_cache.kv_cache_configurator import (
            dsa_dcp_head_groups,
            dsa_dcp_workspace_size_bytes,
        )

        topk = 2048
        kv = torch.ones((1, 1, 64, 576), dtype=torch.float8_e4m3fn, device="cuda")
        for rows, dcp_size, local_heads in (
            (16384, 2, 16),
            (16384, 4, 16),
            (32768, 4, 16),
            (65536, 4, 16),
            (16384, 4, 32),
            (16320, 2, 64),  # 1.25x dynamic chunk probe, 65280 folded rows
        ):
            with self.subTest(rows=rows, dcp=dcp_size, local_heads=local_heads):
                heads = local_heads * dcp_size
                groups = dsa_dcp_head_groups(heads)
                source = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
                source[:, :64] = (
                    torch.arange(64, dtype=torch.int32, device="cuda") * dcp_size
                )
                # Empty rows at both ends also cover the final bounded call.
                source[0].fill_(-1)
                source[-1].fill_(-1)
                indices, lengths = remap_dcp_sparse_indices(
                    source, dcp_size, 0, return_counts=True, repeat_rows=groups
                )
                del source
                query = torch.zeros(
                    (rows, 1, heads, 576), dtype=torch.float8_e4m3fn, device="cuda"
                )
                lse_buffer = torch.empty(
                    (rows, heads), dtype=torch.float32, device="cuda"
                )
                cu_seqlens_q = torch.arange(
                    rows * groups + 1, dtype=torch.int32, device="cuda"
                )
                workspace = torch.empty(
                    dsa_dcp_workspace_size_bytes(
                        num_q_heads=local_heads, dcp_size=dcp_size, max_query_rows=rows
                    ),
                    dtype=torch.uint8,
                    device="cuda",
                )
                counter = make_persistent_multi_ctas_kv_counter_buffer(
                    torch.device("cuda"), heads, rows
                )
                # The original 64-head vendor failure was intermittent; reuse
                # the same workspace/counter to cover repeated dispatches too.
                for repeat in range(3):
                    with self.subTest(repeat=repeat):
                        output, lse = _dcp_trtllm_sparse_attention(
                            query=query,
                            block_tables=indices[:, None, :],
                            seq_lens=lengths,
                            lse=lse_buffer,
                            head_groups=groups,
                            cu_seqlens_q=cu_seqlens_q,
                            kv_cache=kv,
                            workspace_buffer=workspace,
                            qk_nope_head_dim=128,
                            kv_lora_rank=512,
                            qk_rope_head_dim=64,
                            max_seq_len=64,
                            sparse_mla_top_k=topk,
                            bmm1_scale=1 / math.sqrt(192),
                            backend="trtllm-gen",
                            multi_ctas_kv_counter_buffer=counter,
                        )
                        self.assertEqual(output.shape, (rows, heads, 512))
                        self.assertEqual(lse.shape, (rows, heads))
                        torch.testing.assert_close(
                            output[1:-1],
                            torch.ones_like(output[:1]).expand(rows - 2, -1, -1),
                        )
                        torch.testing.assert_close(
                            lse[1:-1], torch.full_like(lse[1:-1], 6)
                        )
                        self.assertTrue(bool((output[[0, -1]] == 0).all()))
                        self.assertTrue(bool(torch.isneginf(lse[[0, -1]]).all()))
                        del output, lse
                del (
                    query,
                    workspace,
                    lse_buffer,
                    cu_seqlens_q,
                    counter,
                    indices,
                    lengths,
                )

    def test_folded_heads_match_independent_attention_calls(self):
        """Random heads and uneven counts expose a folded-row permutation."""
        for heads in (32, 64, 128):
            with self.subTest(head_groups=heads // 32):
                self._check_folded_heads_match_independent_attention_calls(heads)

    def _check_folded_heads_match_independent_attention_calls(self, heads):
        import flashinfer

        from sglang.srt.layers.attention.dsa_backend import _dcp_trtllm_sparse_attention
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )
        from sglang.srt.mem_cache.kv_cache_configurator import dsa_dcp_head_groups

        rows, topk, dcp_size = 4, 2048, 4
        groups = dsa_dcp_head_groups(heads)
        generator = torch.Generator().manual_seed(41926)
        source = torch.full((rows, topk), -1, dtype=torch.int32)
        for row, count in enumerate((1, 63, 64, 0)):
            source[row, :count] = (
                torch.randperm(64, generator=generator)[:count] * dcp_size
            )
        reference_indices, reference_lengths = _reference_remap(source, dcp_size, 0)
        reference_indices = reference_indices.cuda()
        reference_lengths = reference_lengths.cuda()
        folded_indices, folded_lengths = remap_dcp_sparse_indices(
            source.cuda(), dcp_size, 0, return_counts=True, repeat_rows=groups
        )
        query_cpu = torch.randn(rows, 1, heads, 576, generator=generator) * 0.25
        kv_cpu = torch.randn(1, 1, 64, 576, generator=generator) * 0.25
        workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
        counter = make_persistent_multi_ctas_kv_counter_buffer(
            torch.device("cuda"), heads, rows
        )
        cu_original = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
        cu_folded = torch.arange(rows * groups + 1, dtype=torch.int32, device="cuda")

        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            with self.subTest(dtype=dtype):
                query, kv = query_cpu.cuda().to(dtype), kv_cpu.cuda().to(dtype)
                kwargs = dict(
                    kv_cache=kv,
                    workspace_buffer=workspace,
                    qk_nope_head_dim=128,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    max_seq_len=64,
                    sparse_mla_top_k=topk,
                    bmm1_scale=1 / math.sqrt(192),
                    backend="trtllm-gen",
                    enable_pdl=False,
                    multi_ctas_kv_counter_buffer=counter,
                )
                output, lse = _dcp_trtllm_sparse_attention(
                    query=query,
                    block_tables=folded_indices[:, None, :],
                    seq_lens=folded_lengths,
                    lse=torch.empty((rows, heads), dtype=torch.float32, device="cuda"),
                    head_groups=groups,
                    cu_seqlens_q=cu_folded,
                    **kwargs,
                )
                references, reference_lses = [], []
                for begin in range(0, heads, 32):
                    reference, reference_lse = (
                        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                            query=query[:, :, begin : begin + 32].contiguous(),
                            block_tables=reference_indices[:, None, :],
                            seq_lens=reference_lengths,
                            return_lse=True,
                            **kwargs,
                        )
                    )
                    reference = reference.view(rows, 32, 512)
                    reference_lse = reference_lse.view(rows, 32)
                    fixup_zero_kv_rows(
                        reference, reference_lse, reference_lengths, cu_original, 1
                    )
                    references.append(reference)
                    reference_lses.append(reference_lse)
                torch.testing.assert_close(
                    output, torch.cat(references, dim=1), atol=0.003, rtol=0.02
                )
                torch.testing.assert_close(
                    lse, torch.cat(reference_lses, dim=1), atol=0.003, rtol=0.002
                )
                self.assertTrue(bool((output[-1] == 0).all()))
                self.assertTrue(bool(torch.isneginf(lse[-1]).all()))
                empty_output, empty_lse = _dcp_trtllm_sparse_attention(
                    query=query[:0],
                    block_tables=folded_indices[:0, None, :],
                    seq_lens=folded_lengths[:0],
                    lse=lse[:0],
                    head_groups=groups,
                    cu_seqlens_q=cu_folded[:1],
                    **kwargs,
                )
                self.assertEqual(empty_output.shape, (0, heads, 512))
                self.assertEqual(empty_output.dtype, torch.bfloat16)
                self.assertEqual(empty_lse.shape, (0, heads))

    def test_combined_attention_matches_unsharded_sparse_attention(self):
        """Counted local prefixes and base-2 LSE must preserve sparse attention.

        Rows selecting only rank-zero KV exercise empty contributions on all
        other ranks. High global slots also exceed a shard's physical capacity.
        """
        import flashinfer

        generator = torch.Generator().manual_seed(41926)
        capacity, topk, num_heads = 8192, 2048, 32
        indices_cpu = torch.full((4, topk), -1, dtype=torch.int32)
        indices_cpu[0, 0] = capacity - 4
        indices_cpu[1, :127] = (
            torch.randperm(capacity // 4, generator=generator)[:127] * 4
        )
        indices_cpu[2] = torch.randperm(capacity, generator=generator)[:topk]
        # Deliberately unbalanced rank weights expose a base-e/base-2 mix-up.
        indices_cpu[3, :11] = torch.arange(capacity - 8, capacity - 52, -4)
        indices_cpu[3, 11:13] = torch.tensor([capacity - 3, capacity - 2])
        indices = indices_cpu.cuda()
        lengths = torch.tensor([1, 127, topk, 13], dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.arange(5, dtype=torch.int32, device="cuda")
        query_cpu = torch.randn(4, 1, num_heads, 576, generator=generator) * 0.25
        kv_cpu = torch.randn(capacity, 576, generator=generator) * 0.25
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        scale = 1 / math.sqrt(192)

        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            query = query_cpu.cuda().to(dtype)
            global_kv = kv_cpu.cuda().to(dtype)

            def check_reference(output, lse, kv, selected, counts):
                for row, count in enumerate(counts.cpu().tolist()):
                    if count == 0:
                        continue
                    keys = kv.float()[selected[row, :count].long()]
                    logits = query[row, 0].float() @ keys.T * scale
                    numerators = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
                    denominator = numerators.sum(dim=-1, keepdim=True)
                    if dtype == torch.float8_e4m3fn:
                        # TRTLLM scales softmax numerators to E4M3's maximum
                        # before BMM2; the denominator and LSE remain FP32.
                        fp8_max = torch.finfo(dtype).max
                        numerators = (numerators * fp8_max).to(dtype).float() / fp8_max
                    reference = (numerators @ keys[:, :512]) / denominator
                    reference_lse = torch.logsumexp(logits, dim=-1) / math.log(2)
                    torch.testing.assert_close(
                        output[row].float(), reference, atol=0.003, rtol=0.02
                    )
                    torch.testing.assert_close(
                        lse[row], reference_lse, atol=0.003, rtol=0.002
                    )

            def attend(kv, selected, counts):
                output, lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                    query=query,
                    kv_cache=kv.reshape(-1, 1, 64, 576),
                    workspace_buffer=workspace,
                    qk_nope_head_dim=128,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    block_tables=selected.unsqueeze(1),
                    seq_lens=counts,
                    max_seq_len=topk,
                    sparse_mla_top_k=topk,
                    bmm1_scale=scale,
                    backend="trtllm-gen",
                    return_lse=True,
                )
                output = output.squeeze(1)
                fixup_zero_kv_rows(output, lse, counts, cu_seqlens_q, 1)
                check_reference(output, lse, kv, selected, counts)
                return output, lse

            expected, expected_lse = attend(global_kv, indices, lengths)

            for dcp_size in (2, 4):
                with self.subTest(dtype=dtype, dcp_size=dcp_size):
                    outputs, lses = [], []
                    for rank in range(dcp_size):
                        local_indices, counts = remap_dcp_sparse_indices(
                            indices, dcp_size, rank, return_counts=True
                        )
                        output, lse = attend(
                            global_kv[rank::dcp_size].contiguous(),
                            local_indices,
                            counts,
                        )
                        if rank:
                            self.assertEqual(counts[:2].cpu().tolist(), [0, 0])
                            self.assertTrue((output[:2] == 0).all())
                            self.assertTrue(torch.isneginf(lse[:2]).all())
                        outputs.append(output)
                        lses.append(lse)
                    combined, combined_lse = dcp_lse_combine_triton(
                        torch.stack(outputs),
                        torch.stack(lses),
                        is_lse_base_on_e=False,
                        return_lse=True,
                    )
                    # Sharding changes the max used to quantize FP8 softmax
                    # numerators. Both paths match the independent quantized
                    # reference above at 0.003, but need 0.008 against each
                    # other (measured max 0.0078125 on the unbalanced row).
                    atol = 0.008 if dtype == torch.float8_e4m3fn else 0.003
                    torch.testing.assert_close(combined, expected, atol=atol, rtol=0.02)
                    torch.testing.assert_close(
                        combined_lse, expected_lse, atol=0.003, rtol=0.002
                    )


if __name__ == "__main__":
    unittest.main()
