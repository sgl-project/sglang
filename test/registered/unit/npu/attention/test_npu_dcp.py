"""CPU-only tests for the NPU DSA+DCP implementation.

The index arithmetic, buffer geometry, and metadata contracts run on CPU. NPU
extension modules and operators are stubbed because their numerical kernels are
covered by the operator-library tests.
"""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# Avoid importing the real extensions on CPU.  ``layers.dcp`` performs optional
# kernel discovery during import, so temporarily force the platform probe to
# report CPU while loading the modules under test.
sys.modules.setdefault("sgl_kernel_npu", types.ModuleType("sgl_kernel_npu"))
sys.modules.setdefault("torch_npu", MagicMock(name="torch_npu"))
import sglang.srt.utils as _utils
from sglang.srt.utils import common as _common_utils

_real_is_npu = _common_utils.is_npu
_real_utils_is_npu = _utils.is_npu
_common_utils.is_npu = lambda: False
_utils.is_npu = _common_utils.is_npu
try:
    from sglang.srt import runtime_context as rc
    from sglang.srt.hardware_backend.npu.attention.dsa_dcp import (
        forward_dcp_sparse_attention,
    )
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mla_npu
    from sglang.srt.layers.dcp.layout import (
        filter_dcp_local_chunk_kv_indices,
        get_dcp_chain_spec_lens,
        get_dcp_lens,
        remap_dcp_sparse_indices,
    )
finally:
    _common_utils.is_npu = _real_is_npu
    _utils.is_npu = _real_utils_is_npu


class _Mode:
    def __init__(self, target_verify: bool):
        self._target_verify = target_verify

    def is_target_verify(self):
        return self._target_verify


class _FakeGroup:
    """Deterministic all-to-all transport around the real NPU LSE operator."""

    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.calls = []

    def all_to_all_single(self, output, input_tensor):
        self.calls.append((output.shape, input_tensor.shape))
        # Exchange rank 0/1 chunks. The real npu_attention_update below then
        # merges the exchanged partials.
        if self.world_size == 2:
            output.copy_(input_tensor.flip(0))
        else:
            output.copy_(input_tensor)


class TestNpuDcpLengths(unittest.TestCase):
    def test_owner_counts_cover_every_position(self):
        for size in (1, 2, 3, 4, 8):
            for length in range(0, 65):
                lens = [
                    int(get_dcp_lens(torch.tensor([length]), size, rank)[0])
                    for rank in range(size)
                ]
                self.assertEqual(sum(lens), length)

    def test_owner_counts_follow_absolute_start(self):
        for size in (2, 3, 4, 8):
            for start in (0, 1, 2, 7, 31):
                for length in (0, 1, 2, 5, 17, 64):
                    got = [
                        int(
                            get_dcp_lens(
                                torch.tensor([length]),
                                size,
                                rank,
                                start=torch.tensor([start]),
                            )[0]
                        )
                        for rank in range(size)
                    ]
                    expected = [
                        sum(pos % size == rank for pos in range(start, start + length))
                        for rank in range(size)
                    ]
                    self.assertEqual(got, expected)

    def test_chain_lengths_are_request_major(self):
        total = torch.tensor([8, 11], dtype=torch.int32)
        got = get_dcp_chain_spec_lens(total, 3, dcp_size=4, dcp_rank=1)
        # Global frontiers are [6, 7, 8] and [9, 10, 11]. Rank 1 owns
        # positions 1,5,9,..., hence [2, 2, 2] and [2, 3, 3].
        self.assertEqual(got.tolist(), [2, 2, 2, 2, 3, 3])

    def test_chain_lengths_zero_short_requests(self):
        total = torch.tensor([0, 1, 2], dtype=torch.int64)
        for rank in range(4):
            got = get_dcp_chain_spec_lens(total, 3, 4, rank)
            self.assertEqual(got.tolist(), [0] * 9)

    def test_chain_lengths_reject_non_positive_speculation(self):
        with self.assertRaisesRegex(ValueError, "tokens_per_req"):
            get_dcp_chain_spec_lens(torch.tensor([4]), 0, 2, 0)

    def test_chunk_filter_uses_start_phase(self):
        starts = torch.tensor([1, 7, 12])
        lengths = torch.tensor([5, 4, 6])
        values = torch.arange(15)
        with rc.get_parallel().override(dcp_enabled=True, dcp_size=4, dcp_rank=2):
            got = filter_dcp_local_chunk_kv_indices(values, starts, lengths)
        # The first owned offset is (rank-start) mod size for each chunk.
        expected = torch.tensor([1, 8, 11])
        self.assertTrue(torch.equal(got, expected))


class TestNpuDcpSparseIndexRemap(unittest.TestCase):
    def test_rank_zero_compacts_valid_entries_stably(self):
        topk = torch.tensor([[7, 4, 2, -1, 8, 0], [3, 1, -1, 6, 5, -1]])
        got = remap_dcp_sparse_indices(topk, dcp_size=2, dcp_rank=0)
        expected = torch.tensor([[2, 1, 4, 0, -1, -1], [3, -1, -1, -1, -1, -1]])
        self.assertTrue(torch.equal(got, expected))

    def test_rank_one_preserves_score_order_before_padding(self):
        topk = torch.tensor([[7, 4, 2, -1, 8, 0]])
        got = remap_dcp_sparse_indices(topk, dcp_size=2, dcp_rank=1)
        self.assertEqual(got.tolist(), [[3, -1, -1, -1, -1, -1]])

    def test_negative_padding_never_becomes_a_valid_index(self):
        topk = torch.tensor([[-1, -3, 2, 6, -1]], dtype=torch.int64)
        for rank in range(3):
            got = remap_dcp_sparse_indices(topk, 3, rank)
            self.assertTrue(torch.all((got == -1) | (got >= 0)))
            expected_count = int(((topk >= 0) & (topk % 3 == rank)).sum())
            self.assertEqual(int((got >= 0).sum()), expected_count)

    def test_single_rank_is_identity_object(self):
        topk = torch.tensor([[0, 3, -1]])
        got = remap_dcp_sparse_indices(topk, 1, 0)
        self.assertTrue(torch.equal(got, topk))

    def test_wide_integer_values_use_float32_owner_arithmetic(self):
        # Values around 2**24 expose accidental float16 arithmetic.  The
        # implementation deliberately promotes to float32 before remainder.
        topk = torch.tensor(
            [[16_777_216, 16_777_219, 16_777_218, -1]], dtype=torch.int64
        )
        got = remap_dcp_sparse_indices(topk, 4, 2)
        self.assertEqual(got.tolist(), [[4_194_304, -1, -1, -1]])

    def test_shape_and_dtype_are_preserved(self):
        topk = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        got = remap_dcp_sparse_indices(topk, 4, 0)
        self.assertEqual(got.shape, topk.shape)
        self.assertEqual(got.dtype, topk.dtype)


class TestNpuDcpSparseAttentionContract(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_sparse_attention(**kwargs):
            self.calls.append(kwargs)
            rows, heads, dim = kwargs["query"].shape
            out = torch.arange(rows * heads * dim, dtype=torch.float32).view(
                rows, heads, dim
            )
            softmax_max = torch.zeros(heads, rows, 1)
            softmax_sum = torch.full((heads, rows, 1), 2.0)
            return out, softmax_max, softmax_sum

        self.fake_op = fake_sparse_attention

    def _metadata(self, rows=2):
        return SimpleNamespace(
            dcp_block_tables=torch.tensor([[10, 11]] * rows, dtype=torch.int32),
            dcp_seq_lens=torch.arange(5, 5 + rows, dtype=torch.int32),
            dcp_spec_block_tables=torch.arange(rows * 2).view(rows, 2),
            dcp_spec_seq_lens=torch.arange(3, 3 + rows, dtype=torch.int32),
        )

    def _inputs(self, rows=2):
        q_nope = torch.ones(rows, 2, 4, dtype=torch.float16)
        q_rope = torch.zeros_like(q_nope)
        k_nope = torch.ones(rows, 2, 4, dtype=torch.float16)
        k_rope = torch.zeros_like(k_nope)
        topk = torch.tensor([[0, 2, -1], [1, -1, -1]] * rows)[:rows]
        return q_nope, q_rope, k_nope, k_rope, topk

    def test_decode_passes_rank_local_metadata_to_operator(self):
        q, rope, k_nope, k_rope, topk = self._inputs()
        batch = SimpleNamespace(forward_mode=_Mode(False))
        metadata = self._metadata()
        with patch.object(
            torch.ops.npu,
            "sgl_sparse_flash_attention",
            self.fake_op,
            create=True,
        ):
            out, lse = forward_dcp_sparse_attention(
                q_nope=q,
                q_rope=rope,
                k_nope=k_nope,
                k_rope=k_rope,
                topk_indices=topk,
                actual_seq_lengths_query=torch.tensor([1, 2], dtype=torch.int64),
                forward_metadata=metadata,
                forward_batch=batch,
                speculative_num_draft_tokens=None,
                scaling=0.25,
            )
        call = self.calls[0]
        self.assertEqual(call["actual_seq_lengths_query"].dtype, torch.int32)
        self.assertEqual(call["actual_seq_lengths_query"].tolist(), [1, 2])
        self.assertIs(call["block_table"], metadata.dcp_block_tables)
        self.assertEqual(call["sparse_mode"], 0)
        self.assertEqual(call["attention_mode"], 2)
        self.assertEqual(call["layout_kv"], "PA_BSND")
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(lse.shape, (2, 2))
        expected_lse = torch.full((2, 2), torch.log(torch.tensor(2.0)))
        self.assertTrue(torch.allclose(lse, expected_lse))

    def test_verify_replaces_query_lengths_and_uses_spec_metadata(self):
        rows = 4
        q, rope, k_nope, k_rope, topk = self._inputs(rows)
        batch = SimpleNamespace(forward_mode=_Mode(True))
        metadata = self._metadata(rows=rows)
        with patch.object(
            torch.ops.npu,
            "sgl_sparse_flash_attention",
            self.fake_op,
            create=True,
        ):
            forward_dcp_sparse_attention(
                q_nope=q,
                q_rope=rope,
                k_nope=k_nope,
                k_rope=k_rope,
                topk_indices=topk,
                actual_seq_lengths_query=torch.tensor([99]),
                forward_metadata=metadata,
                forward_batch=batch,
                speculative_num_draft_tokens=2,
                scaling=1.0,
            )
        call = self.calls[0]
        self.assertEqual(call["actual_seq_lengths_query"].tolist(), [1, 2, 3, 4])
        self.assertTrue(
            torch.equal(call["block_table"], metadata.dcp_spec_block_tables)
        )
        self.assertTrue(
            torch.equal(call["actual_seq_lengths_kv"], metadata.dcp_spec_seq_lens)
        )

    def test_missing_rank_local_metadata_fails_loudly(self):
        q, rope, _, _, topk = self._inputs()
        bad = SimpleNamespace(dcp_block_tables=None, dcp_seq_lens=None)
        with self.assertRaisesRegex(AssertionError, "rank-local paged-KV"):
            forward_dcp_sparse_attention(
                q_nope=q,
                q_rope=rope,
                k_nope=q,
                k_rope=rope,
                topk_indices=topk,
                actual_seq_lengths_query=torch.tensor([1, 2]),
                forward_metadata=bad,
                forward_batch=SimpleNamespace(forward_mode=_Mode(False)),
                speculative_num_draft_tokens=None,
                scaling=1.0,
            )

    def test_verify_row_mismatch_fails_before_operator(self):
        q, rope, _, _, topk = self._inputs(2)
        metadata = self._metadata(rows=3)
        with self.assertRaisesRegex(AssertionError, "block-table rows"):
            forward_dcp_sparse_attention(
                q_nope=q,
                q_rope=rope,
                k_nope=q,
                k_rope=rope,
                topk_indices=topk,
                actual_seq_lengths_query=torch.tensor([1]),
                forward_metadata=metadata,
                forward_batch=SimpleNamespace(forward_mode=_Mode(True)),
                speculative_num_draft_tokens=2,
                scaling=1.0,
            )
        self.assertEqual(self.calls, [])


class TestNpuDcpBufferAndLseHelpers(unittest.TestCase):
    def test_transfer_view_keeps_page_shape_and_strides_global_slots(self):
        pool = SimpleNamespace(dcp_size=4)
        raw = torch.empty(4, 8, 1, 2, 3)
        view = NPUMLATokenToKVPool._get_disagg_buffer_view(
            pool, raw, page_size=8, uses_global_slots=True
        )
        self.assertEqual(view.shape, (1, 32, 1, 2, 3))
        self.assertEqual(view.numel(), raw.numel())
        self.assertEqual(view.data_ptr(), raw.data_ptr())

    def test_transfer_view_rank_local_does_not_multiply_page_size(self):
        pool = SimpleNamespace(dcp_size=4)
        raw = torch.empty(3, 8, 1, 1, 2)
        view = NPUMLATokenToKVPool._get_disagg_buffer_view(
            pool, raw, page_size=8, uses_global_slots=False
        )
        self.assertEqual(view.shape, (3, 8, 1, 1, 2))

    def test_transfer_view_rejects_non_integral_global_pages(self):
        pool = SimpleNamespace(dcp_size=4)
        raw = torch.empty(1, 8, 1, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "integral transfer pages"):
            NPUMLATokenToKVPool._get_disagg_buffer_view(
                pool, raw, page_size=8, uses_global_slots=True
            )

    def test_lse_merge_world_size_one_is_identity(self):
        group = _FakeGroup(world_size=1)
        out = torch.randn(2, 3, 4)
        lse = torch.randn(2, 3)
        got = cp_lse_ag_out_rs_mla_npu(out, lse, group)
        self.assertIs(got, out)
        self.assertEqual(group.calls, [])

    def test_lse_merge_calls_npu_update_for_multi_rank(self):
        group = _FakeGroup(world_size=2)
        out = torch.arange(2 * 4 * 3, dtype=torch.float16).view(2, 4, 3)
        lse = torch.zeros(2, 4)
        updates = []

        def fake_update(lses, outputs, _zero):
            updates.append((len(lses), len(outputs)))
            return outputs[0], lses[0]

        fake_torch_npu = types.SimpleNamespace(npu_attention_update=fake_update)
        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            got = cp_lse_ag_out_rs_mla_npu(out, lse, group)
        self.assertEqual(updates, [(2, 2)])
        self.assertEqual(group.calls[0][0], group.calls[0][1])
        self.assertEqual(got.shape, (2, 2, 3))
        self.assertEqual(got.dtype, out.dtype)


if __name__ == "__main__":
    unittest.main()
