"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.
"""

import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Loaded before the torch_npu mock below: importing transformers runs
# accelerate's find_spec("torch_npu"), which fails on a mock without __spec__.
import sglang.srt.hf_transformers  # noqa: F401
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="extra-a-test-npu")

# The backend imports ragged_verify at module level, so load the REAL module
# before the stub block below shadows its package parent.
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode  # noqa: F401

for mod in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.operators",
    "sgl_kernel_npu.operators.attention",
    "sgl_kernel_npu.operators.indexer",
    "sgl_kernel_npu.operators.mem_cache",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sgl_kernel_npu.kvcacheio",
    "sgl_kernel_npu.norm",
    "sgl_kernel_npu.norm.add_rmsnorm_bias",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
    "sglang.srt.speculative.base_spec_worker",
):
    sys.modules.setdefault(mod, MagicMock())

# Stub deepseek_v2._is_hip to avoid importing the heavy model.
_ds2_stub = ModuleType("sglang.srt.models.deepseek_v2")
_ds2_stub._is_hip = False
sys.modules.setdefault("sglang.srt.models.deepseek_v2", _ds2_stub)

# Stub eagle_utils with a faithful per_step_draft_out_cache_loc.
_eagle_stub = ModuleType("sglang.srt.speculative.eagle_utils")


def _per_step_draft_out_cache_loc(out_cache_loc, batch_size, topk, num_steps):
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


_eagle_stub.per_step_draft_out_cache_loc = _per_step_draft_out_cache_loc
sys.modules.setdefault("sglang.srt.speculative", ModuleType("sglang.srt.speculative"))
sys.modules.setdefault("sglang.srt.speculative.eagle_utils", _eagle_stub)

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    C4IndexerAscendBackendMixin,
    CompressorAscendBackendMixin,
    DeepseekV4AscendAttnBackend,
    DeepseekV4AscendMultiStepDraftBackend,
    _apply_hadamard,
    _get_kv_indices,
    _sparse_attn_kv_quant_kwargs,
    _sparse_attn_ops,
    _walsh_hadamard_matrix,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    dsv4_state_payloads,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DSV4NPUTokenToKVPool,
    NPUCompressStatePool,
)

_BACKEND_NPU_OPS = (
    "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.npu_attention_ops"
)


class TestVerifyCompressPositions(unittest.TestCase):
    @staticmethod
    def _backend():
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend._dsv4_compress_ratios = (4, 128)
        return backend

    def _assert_device_path_matches_cpu_reference(
        self,
        *,
        positions,
        live_seq_lens,
        n_draft,
        ratio,
        dst_size,
    ):
        backend = self._backend()
        positions = torch.tensor(positions, dtype=torch.int64)
        live_seq_lens = torch.tensor(live_seq_lens, dtype=torch.int32)
        final_seq_lens = torch.where(
            live_seq_lens > 0,
            live_seq_lens + int(n_draft),
            live_seq_lens,
        )
        expected = torch.full((dst_size,), -1, dtype=torch.int64)
        actual = torch.full((dst_size,), -2, dtype=torch.int64)

        backend._fill_verify_positions_cmp_padding_one(
            positions,
            expected,
            ratio=ratio,
            seq_lens_cpu=final_seq_lens,
            n_draft=n_draft,
        )
        backend._fill_verify_positions_cmp_padding_one_device(
            positions,
            actual,
            ratio=ratio,
            live_seq_lens=live_seq_lens,
            n_draft=n_draft,
        )

        self.assertEqual(actual.tolist(), expected.tolist())

    def test_uses_group_start_rope_position(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend._dsv4_compress_ratios = (4, 128)

        # Two linear three-token verify trees.  Their completed C4 groups end
        # at token positions 7 and 11, whose compressed RoPE positions are the
        # corresponding group starts 4 and 8.
        positions = torch.tensor([7, 8, 9, 10, 11, 12], dtype=torch.int64)
        final_seq_lens = torch.tensor([10, 13], dtype=torch.int32)
        dst = torch.full((4,), -1, dtype=torch.int64)

        backend._fill_verify_positions_cmp_padding_one(
            positions,
            dst,
            ratio=4,
            seq_lens_cpu=final_seq_lens,
            n_draft=3,
        )

        self.assertEqual(dst.tolist(), [4, 8, 0, 0])

    def test_c128_uses_group_start_rope_position(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend._dsv4_compress_ratios = (4, 128)
        dst = torch.full((2,), -1, dtype=torch.int64)

        backend._fill_verify_positions_cmp_padding_one(
            torch.tensor([126, 127, 128], dtype=torch.int64),
            dst,
            ratio=128,
            seq_lens_cpu=torch.tensor([129], dtype=torch.int32),
            n_draft=3,
        )

        self.assertEqual(dst.tolist(), [0, 0])

    def test_no_completed_group_clears_destination(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend._dsv4_compress_ratios = (4, 128)
        dst = torch.full((2,), -1, dtype=torch.int64)

        backend._fill_verify_positions_cmp_padding_one(
            torch.tensor([5, 6], dtype=torch.int64),
            dst,
            ratio=4,
            seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
            n_draft=2,
        )

        self.assertEqual(dst.tolist(), [0, 0])

    def test_zero_length_graph_padding_does_not_emit_a_position(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend._dsv4_compress_ratios = (4, 128)
        dst = torch.full((2,), -1, dtype=torch.int64)

        backend._fill_verify_positions_cmp_padding_one(
            torch.tensor([0, 0, 0, 10, 11, 12], dtype=torch.int64),
            dst,
            ratio=4,
            seq_lens_cpu=torch.tensor([0, 13], dtype=torch.int32),
            n_draft=3,
        )

        self.assertEqual(dst.tolist(), [8, 0])

    def test_device_path_matches_reference_across_boundaries_and_padding(self):
        cases = (
            # Existing C4 and C128 boundary examples.
            dict(
                positions=[7, 8, 9, 10, 11, 12],
                live_seq_lens=[7, 10],
                n_draft=3,
                ratio=4,
                dst_size=4,
            ),
            dict(
                positions=[126, 127, 128],
                live_seq_lens=[126],
                n_draft=3,
                ratio=128,
                dst_size=2,
            ),
            # A zero-length graph-padding row may appear before a live row.
            dict(
                positions=[0, 0, 0, 10, 11, 12],
                live_seq_lens=[0, 10],
                n_draft=3,
                ratio=4,
                dst_size=4,
            ),
            # More than one boundary per request and destination truncation.
            dict(
                positions=list(range(2, 11)) + list(range(7, 16)),
                live_seq_lens=[2, 7],
                n_draft=9,
                ratio=4,
                dst_size=3,
            ),
            # Preserve values from a non-linear tree position array rather than
            # reconstructing them arithmetically from sequence lengths.
            dict(
                positions=[50, 90, 51, 52, 70, 71, 120, 72],
                live_seq_lens=[3, 126],
                n_draft=4,
                ratio=4,
                dst_size=4,
            ),
            # Long-context C128 boundaries around 128K.
            dict(
                positions=[131070, 131071, 131072, 131073],
                live_seq_lens=[131070],
                n_draft=4,
                ratio=128,
                dst_size=2,
            ),
        )
        for case in cases:
            with self.subTest(case=case):
                self._assert_device_path_matches_cpu_reference(**case)

    def test_stable_compact_preserves_boolean_index_order_and_zero_tail(self):
        dst = torch.full((4,), -1, dtype=torch.int64)
        values = torch.tensor([11, 22, 33, 44, 55, 66], dtype=torch.int64)
        keep = torch.tensor([False, True, False, True, True, False])

        DeepseekV4AscendAttnBackend._stable_compact_1d(dst, values, keep)

        self.assertEqual(dst.tolist(), [22, 44, 55, 0])

    def test_stable_compact_matches_boolean_index_truncation(self):
        dst = torch.full((2,), -1, dtype=torch.int64)
        values = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int64)
        keep = torch.tensor([True, False, True, True, True])

        DeepseekV4AscendAttnBackend._stable_compact_1d(dst, values, keep)

        self.assertEqual(dst.tolist(), values[keep][:2].tolist())

    def test_stable_compact_matches_all_small_boolean_masks(self):
        values = torch.arange(1, 7, dtype=torch.int64)
        for mask_bits in range(1 << values.numel()):
            keep = torch.tensor(
                [(mask_bits >> index) & 1 for index in range(values.numel())],
                dtype=torch.bool,
            )
            for dst_size in range(1, values.numel() + 1):
                dst = torch.full((dst_size,), -1, dtype=torch.int64)
                DeepseekV4AscendAttnBackend._stable_compact_1d(dst, values, keep)
                selected = values[keep][:dst_size]
                expected = torch.zeros_like(dst)
                expected[: selected.numel()].copy_(selected)
                self.assertEqual(dst.tolist(), expected.tolist())


class TestMultiStepDraftCompressedLocs(unittest.TestCase):
    def test_draft_steps_skip_compressed_locs_but_preserve_full_and_swa(self):
        backend = DeepseekV4AscendMultiStepDraftBackend.__new__(
            DeepseekV4AscendMultiStepDraftBackend
        )
        backend.topk = 1
        backend.speculative_num_steps = 3
        bundle = SimpleNamespace(
            out_full_loc=torch.arange(6, dtype=torch.int64),
            out_swa_loc=torch.arange(10, 16, dtype=torch.int64),
            out_c4_loc=torch.tensor([101, 102], dtype=torch.int64),
            out_c128_loc=torch.tensor([201], dtype=torch.int64),
        )
        forward_batch = SimpleNamespace(
            batch_size=2,
            out_cache_loc=bundle.out_full_loc,
            out_cache_loc_dsv4=bundle,
            seq_lens=torch.tensor([7, 11], dtype=torch.int32),
        )

        with patch("torch.cumsum", side_effect=AssertionError("unexpected compaction")):
            step = backend._step_out_cache_loc_dsv4(forward_batch, step_id=1)

        self.assertEqual(step.out_full_loc.tolist(), [1, 4])
        self.assertEqual(step.out_swa_loc.tolist(), [11, 14])
        self.assertEqual(step.out_c4_loc.numel(), 0)
        self.assertEqual(step.out_c128_loc.numel(), 0)
        self.assertEqual(step.out_c4_loc.dtype, bundle.out_c4_loc.dtype)
        self.assertEqual(step.out_c128_loc.dtype, bundle.out_c128_loc.dtype)


class TestC4StateTransferLayout(unittest.TestCase):
    @staticmethod
    def _pool(stage_layers, c4_layers, c128_layers=(), ring_size=8):
        pool = DSV4NPUTokenToKVPool.__new__(DSV4NPUTokenToKVPool)
        pool.stage_layer_ids = tuple(stage_layers)
        pool.sources_by_ratio = {4: list(c4_layers), 128: list(c128_layers)}
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[
                torch.empty((12, 128, 1, 8), dtype=torch.bfloat16) for _ in stage_layers
            ]
        )
        pool.c128_kv_pool = SimpleNamespace(
            kv_buffer=[
                torch.empty((12, 1, 1, 8), dtype=torch.bfloat16) for _ in c128_layers
            ]
        )
        total_layers = max(stage_layers, default=-1) + 1
        pool.compress_state_pools = [None] * total_layers
        pool.indexer_compress_state_pools = [None] * total_layers
        for layer_id in c4_layers:
            for pools, width in (
                (pool.compress_state_pools, 8),
                (pool.indexer_compress_state_pools, 4),
            ):
                state = NPUCompressStatePool.__new__(NPUCompressStatePool)
                state.ratio = 4
                state.ring_size = ring_size
                state.swa_page_size = 128
                state.dummy_state_loc = 12 * state.ring_size
                state.kv_score_buffer = SimpleNamespace(
                    kv_score=torch.full(
                        (12 * state.ring_size, width), -1.0, dtype=torch.float32
                    )
                )
                pools[layer_id] = state
        pool.layer_router = SimpleNamespace(sub_pool=lambda _: None)
        pool._unified_kv = False
        pool.get_request_state_buf_infos = lambda: ([], [], [])
        return pool

    def test_payload_only_carries_separate_c128_kv(self):
        req_pool = SimpleNamespace(
            c128_page_size=1,
            req_to_c128_sidecar=torch.tensor(
                [[0, 0, 0], [0, 0, 0], [5, 7, 0]], dtype=torch.int32
            ),
        )
        payloads = dsv4_state_payloads(req_pool, 2, 384, page_size=128)
        self.assertEqual(set(payloads), {StateType.DSV4_C128})
        self.assertEqual(payloads[StateType.DSV4_C128]().tolist(), [5, 7])

    def test_swa_registers_c4_rings_and_c128_remains_separate(self):
        from sglang.srt.mem_cache.static_kv_storage import _static_state_entries

        pool = self._pool([2, 3, 4], [2, 4], [3])
        states = [
            p
            for pools in (
                pool.compress_state_pools,
                pool.indexer_compress_state_pools,
            )
            for p in pools
            if p is not None
        ]
        buffers = pool.swa_kv_pool.kv_buffer + [
            p.kv_score_buffer.kv_score for p in states
        ]
        expected_items = [b[0].nbytes for b in pool.swa_kv_pool.kv_buffer] + [
            p.kv_score_buffer.kv_score[0].nbytes * p.ring_size for p in states
        ]
        ptrs, lens, items = pool.get_state_buf_infos()
        self.assertEqual(ptrs, [b.data_ptr() for b in buffers])
        self.assertEqual(lens, [b.nbytes for b in buffers])
        self.assertEqual(items, expected_items)
        expected_ids = [2, 3, 4, 2, 4, 2, 4]
        self.assertEqual(pool.get_swa_state_layer_ids(), expected_ids)

        with patch("sglang.srt.mem_cache.static_kv_storage.is_npu", return_value=True):
            entries = _static_state_entries(
                pool,
                draft_token_to_kv_pool=None,
                total_kv_layers=5,
                state_pool=None,
            )
        self.assertEqual(
            [entry.state_type for entry in entries.wire_entries],
            [StateType.SWA, StateType.DSV4_C128],
        )
        self.assertEqual(entries.regions["state0"].ptrs, tuple(ptrs))
        self.assertEqual(entries.regions["state0"].item_lens, tuple(expected_items))
        self.assertEqual(entries.regions["state0"].layer_ids, tuple(expected_ids))
        self.assertEqual(entries.regions["state1"].layer_ids, (3,))

    def test_swa_layer_ids_pair_each_buffer_group_across_pp(self):
        from sglang.srt.disaggregation.utils import build_transfer_entry_pairs

        src = self._pool([2, 3, 4], [2, 4], [3])
        dst = self._pool([0, 1, 2, 3, 4], [0, 2, 4], [1, 3])
        src_ptrs = src.get_state_buf_infos()[0]
        dst_ptrs = dst.get_state_buf_infos()[0]
        pairs = build_transfer_entry_pairs(
            src.get_swa_state_layer_ids(),
            dst.get_swa_state_layer_ids(),
            len(src_ptrs),
            len(dst_ptrs),
        )
        self.assertEqual(
            pairs, [(0, 2), (1, 3), (2, 4), (3, 6), (4, 7), (5, 9), (6, 10)]
        )

    def test_swa_only_stage_and_draft_have_no_c4_entries(self):
        for stage, c128 in (([2, 3], [2, 3]), ([0], [])):
            with self.subTest(stage=stage, c128=c128):
                pool = self._pool(stage, [], c128)
                buffers = pool.swa_kv_pool.kv_buffer
                self.assertEqual(
                    pool.get_state_buf_infos(),
                    (
                        [b.data_ptr() for b in buffers],
                        [b.nbytes for b in buffers],
                        [b[0].nbytes for b in buffers],
                    ),
                )
                self.assertEqual(pool.get_swa_state_layer_ids(), stage)

    @staticmethod
    def _req_pool(kv_pool, swa_pages):
        req_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 384), dtype=torch.int64)
        )
        mapping = torch.zeros(24 * 128, dtype=torch.int64)
        for logical_page, swa_page in enumerate(swa_pages):
            full_page = 20 - 3 * logical_page
            full_locs = full_page * 128 + torch.arange(128)
            req_pool.req_to_token[0, logical_page * 128 : (logical_page + 1) * 128] = (
                full_locs
            )
            mapping[full_locs] = swa_page * 128 + torch.arange(128)
        kv_pool.translate_loc_from_full_to_swa = lambda locs: mapping[locs]
        return req_pool

    def test_swa_page_copy_preserves_decode_compressor_history(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
            _build_explicit_state_block_table,
        )

        src_pages, dst_pages = [7, 3, 9], [2, 8, 5]
        for ring_size in (8, 16):
            for seq_len in (0, 1, 4, 7, 8, 9, 13, 127, 128, 129, 255, 256, 257):
                with self.subTest(ring_size=ring_size, seq_len=seq_len):
                    src = self._pool([0], [0], ring_size=ring_size)
                    dst = self._pool([0], [0], ring_size=ring_size)
                    src_req = self._req_pool(src, src_pages)
                    dst_req = self._req_pool(dst, dst_pages)
                    src_items = src.get_state_buf_infos()[2]
                    dst_items = dst.get_state_buf_infos()[2]
                    self.assertEqual(src_items, dst_items)
                    for entry, attr in enumerate(
                        ("compress_state_pools", "indexer_compress_state_pools"),
                        start=1,
                    ):
                        src_state = getattr(src, attr)[0]
                        dst_state = getattr(dst, attr)[0]
                        src_data = src_state.kv_score_buffer.kv_score
                        dst_data = dst_state.kv_score_buffer.kv_score
                        dst_data.fill_(-2.0)
                        positions = list(range(max(0, seq_len - 8), seq_len))
                        for pos in positions:
                            swa_loc = src_req.req_to_token[0, pos]
                            swa_loc = src.translate_loc_from_full_to_swa(swa_loc)
                            row = src_state.translate_from_swa_loc_to_state_loc(swa_loc)
                            src_data[row] = pos + torch.arange(src_data.shape[1])

                        # Use the exported item size and the shared SWA page IDs.
                        # Source and destination physical page numbers differ.
                        item_elements = src_items[entry] // src_data.element_size()
                        for src_page, dst_page in zip(src_pages, dst_pages):
                            dst_data.view(-1).narrow(
                                0, dst_page * item_elements, item_elements
                            ).copy_(
                                src_data.view(-1).narrow(
                                    0, src_page * item_elements, item_elements
                                )
                            )
                        table = _build_explicit_state_block_table(
                            compress_ratio=4,
                            coff=2,
                            state_pool=dst_state,
                            token_to_kv_pool=dst,
                            req_to_token=dst_req.req_to_token,
                            req_pool_indices=torch.tensor([0]),
                            start_pos=torch.tensor([seq_len]),
                            cu_seqlens=torch.tensor([0, 1]),
                            seqused=torch.tensor([1]),
                            max_input_capacity=1,
                        )
                        history_rows = table[0, max(0, 8 - seq_len) : 8]
                        for pos, row in zip(positions, history_rows):
                            self.assertTrue(
                                torch.equal(
                                    dst_data[row],
                                    pos + torch.arange(dst_data.shape[1]),
                                )
                            )


class TestC4IndexerInitialization(unittest.TestCase):
    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.is_npu_arch35",
        return_value=True,
    )
    def test_arch35_indexer_uses_float8_kv(self, _):
        indexer = torch.nn.Module()
        indexer.head_dim = 8
        indexer.compressor = SimpleNamespace()
        backend = C4IndexerAscendBackendMixin.__new__(C4IndexerAscendBackendMixin)

        backend._ensure_npu_c4_indexer(indexer, torch.device("cpu"))

        self.assertEqual(indexer.compressor.li_kv_dtype, "float8")


class TestWalshHadamardMatrix(unittest.TestCase):
    def test_shape_n1(self):
        had = _walsh_hadamard_matrix(1, torch.float32, "cpu")
        self.assertEqual(had.shape, (1, 1))

    def test_shape_n2(self):
        had = _walsh_hadamard_matrix(2, torch.float32, "cpu")
        self.assertEqual(had.shape, (2, 2))

    def test_shape_n4(self):
        had = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertEqual(had.shape, (4, 4))

    def test_value_error_n3_not_power_of_two(self):
        with self.assertRaises(ValueError):
            _walsh_hadamard_matrix(3, torch.float32, "cpu")

    def test_value_error_n0(self):
        with self.assertRaises(ValueError):
            _walsh_hadamard_matrix(0, torch.float32, "cpu")

    def test_value_error_negative(self):
        with self.assertRaises(ValueError):
            _walsh_hadamard_matrix(-2, torch.float32, "cpu")

    def test_orthonormality_n2(self):
        had = _walsh_hadamard_matrix(2, torch.float32, "cpu").float()
        # bfloat16 truncates 1/sqrt(2), so use a looser tolerance
        self.assertTrue(torch.allclose(had @ had.T, torch.eye(2), atol=1e-2))

    def test_orthonormality_n4(self):
        had = _walsh_hadamard_matrix(4, torch.float32, "cpu").float()
        self.assertTrue(torch.allclose(had @ had.T, torch.eye(4), atol=1e-2))

    def test_orthonormality_n1(self):
        had = _walsh_hadamard_matrix(1, torch.float32, "cpu").float()
        self.assertTrue(torch.allclose(had @ had.T, torch.eye(1)))

    def test_caching_returns_same_object(self):
        h1 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        h2 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertIs(h1, h2)

    def test_caching_different_n_returns_different_object(self):
        h1 = _walsh_hadamard_matrix(2, torch.float32, "cpu")
        h2 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertIsNot(h1, h2)

    def test_dtype_always_bfloat16(self):
        had = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertEqual(had.dtype, torch.bfloat16)

    def test_dtype_argument_ignored_for_cache_key(self):
        h1 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        h2 = _walsh_hadamard_matrix(4, torch.bfloat16, "cpu")
        self.assertIs(h1, h2)

    def test_entries_are_plus_minus_norm(self):
        n = 4
        had = _walsh_hadamard_matrix(n, torch.float32, "cpu").float()
        expected_abs = 1.0 / math.sqrt(n)
        self.assertTrue(
            torch.allclose(had.abs(), torch.full_like(had, expected_abs), atol=1e-2)
        )


class TestApplyHadamard(unittest.TestCase):
    def test_shape_preserved_2d(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        inp = torch.randn(3, n, dtype=H.dtype)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.shape, inp.shape)

    def test_shape_preserved_3d(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        inp = torch.randn(2, 5, n, dtype=H.dtype)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.shape, inp.shape)

    def test_identity_times_hadamard_equals_hadamard(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        eye = torch.eye(n, dtype=H.dtype)
        out = _apply_hadamard(eye, H)
        self.assertTrue(torch.equal(out, H))

    def test_identity_times_hadamard_n2(self):
        n = 2
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        eye = torch.eye(n, dtype=H.dtype)
        out = _apply_hadamard(eye, H)
        self.assertTrue(torch.equal(out, H))

    def test_output_dtype_is_bfloat16(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        inp = torch.randn(3, n, dtype=H.dtype)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_output_dtype_bfloat16_from_float32_input(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.bfloat16, "cpu")
        inp = torch.randn(3, n, dtype=torch.bfloat16)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_3d_values(self):
        n = 2
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu").float()
        inp = torch.randn(2, 3, n, dtype=torch.float32)
        expected = inp.matmul(H).to(torch.bfloat16)
        out = _apply_hadamard(inp, H)
        self.assertTrue(torch.equal(out, expected))


class TestCompressorStateTableABI(unittest.TestCase):
    def test_graph_replay_refreshes_explicit_table_in_place(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend.req_to_token = torch.arange(32).view(2, 16)
        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda locs: locs,
        )
        backend._dsv4_state_pools_by_ratio = {
            4: SimpleNamespace(
                translate_from_swa_loc_to_state_loc=lambda locs: locs,
                dummy_state_loc=0,
            )
        }
        table = torch.zeros((1, 9), dtype=torch.int32)
        ctx = SimpleNamespace(
            bs=1,
            forward_batch=SimpleNamespace(req_pool_indices=torch.tensor([1, 0])),
            fm=SimpleNamespace(
                dsv4_explicit_state_block_tables={4: table},
                start_pos=torch.tensor([8]),
                actual_seq_lengths_q_pa=torch.tensor([0, 1]),
                seqused=torch.tensor([1]),
                dsv4_max_input_capacity=1,
            ),
        )

        backend._refresh_graph_explicit_state_block_tables(ctx)

        self.assertIs(ctx.fm.dsv4_explicit_state_block_tables[4], table)
        self.assertEqual(table.tolist(), [list(range(16, 25))])


class TestArch35SparseAttentionDispatch(unittest.TestCase):
    _ARCH35_PATCH_TARGET = (
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.is_npu_arch35"
    )

    @patch(_BACKEND_NPU_OPS)
    @patch(_ARCH35_PATCH_TARGET, return_value=True)
    def test_arch35_uses_kv_quant_ops_and_layout_kwargs(self, _, ops):
        metadata_op, attn_op = _sparse_attn_ops()
        self.assertIs(metadata_op, ops.mixed_quant_sparse_flash_mla_metadata)
        self.assertIs(attn_op, ops.mixed_quant_sparse_flash_mla)
        self.assertEqual(
            _sparse_attn_kv_quant_kwargs(), {"quant_mode": 1, "rope_head_dim": 64}
        )

    @patch(_BACKEND_NPU_OPS)
    @patch(_ARCH35_PATCH_TARGET, return_value=False)
    def test_pre_arch35_uses_the_smla_pair_without_quant_kwargs(self, _, ops):
        metadata_op, attn_op = _sparse_attn_ops()
        self.assertIs(metadata_op, ops.sparse_flash_mla_metadata)
        self.assertIs(attn_op, ops.sparse_flash_mla)
        self.assertEqual(_sparse_attn_kv_quant_kwargs(), {})


class TestSparseAttentionMetadata(unittest.TestCase):
    _ARCH35_PATCH_TARGET = (
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.is_npu_arch35"
    )

    def test_device_metadata_receives_sequence_lengths(self):
        cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32)
        seqused_kv = torch.tensor([8, 12], dtype=torch.int32)

        for is_arch35 in (False, True):
            with (
                self.subTest(is_arch35=is_arch35),
                patch(self._ARCH35_PATCH_TARGET, return_value=is_arch35),
                patch(_BACKEND_NPU_OPS) as ops,
            ):
                backend = DeepseekV4AscendAttnBackend.__new__(
                    DeepseekV4AscendAttnBackend
                )
                backend.forward_metadata = SimpleNamespace(
                    ori_sparse_indices=None,
                    ori_mask_mode=4,
                    ori_win_left=127,
                    ori_win_right=0,
                )
                backend._is_dspark_draft_worker = False
                backend._dsv4_sliding_window_size = 128
                backend._dsv4_q_head_num = 64
                backend._dsv4_kv_head_num = 1
                backend._dsv4_head_dim = 512
                backend._dsv4_has_c4 = True
                backend._dsv4_has_c128 = True
                backend._dsv4_index_topk = 512
                backend._dsv4_index_n_heads = 16
                backend._dsv4_index_head_dim = 128

                backend._kernel_metadata_from_parts(
                    bs=2,
                    actual_seq_lengths_q_pa=cu_seqlens_q,
                    actual_seq_lengths_kv=seqused_kv,
                )

                metadata_op = (
                    ops.mixed_quant_sparse_flash_mla_metadata
                    if is_arch35
                    else ops.sparse_flash_mla_metadata
                )
                self.assertEqual(metadata_op.call_count, 3)
                for call in metadata_op.call_args_list:
                    self.assertIs(call.kwargs["seqused_ori_kv"], seqused_kv)

    def test_dspark_metadata_uses_the_sparse_dialect(self):
        cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)
        seqused_kv = torch.tensor([8, 12], dtype=torch.int32)

        metadata_op = MagicMock()
        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._sparse_attn_ops",
            return_value=(metadata_op, MagicMock()),
        ):
            backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
            backend.forward_metadata = SimpleNamespace()
            backend._is_dspark_draft_block = lambda _fb: True
            backend._dsv4_sliding_window_size = 128
            backend._dsv4_q_head_num = 64
            backend._dsv4_kv_head_num = 1
            backend._dsv4_head_dim = 512
            backend._dsv4_has_c4 = False
            backend._dsv4_has_c128 = False

            forward_batch = SimpleNamespace(
                spec_info=SimpleNamespace(draft_token_num=2),
                seq_lens=seqused_kv,
                out_cache_loc=torch.zeros(4, dtype=torch.int64),
            )
            backend._init_dspark_sparse_metadata(forward_batch)
            backend._kernel_metadata_from_parts(
                bs=2,
                actual_seq_lengths_q_pa=cu_seqlens_q,
                actual_seq_lengths_kv=seqused_kv,
            )

        fm = backend.forward_metadata
        # Trailing window then the draft block, -1 padded; window+2 topk each.
        torch.testing.assert_close(
            fm.ori_sparse_indices[2, 0, :14], torch.arange(14, dtype=torch.int32)
        )
        torch.testing.assert_close(
            fm.ori_topk_length.flatten(),
            torch.tensor([10, 10, 14, 14], dtype=torch.int32),
        )
        metadata_op.assert_called_once()
        kwargs = metadata_op.call_args.kwargs
        self.assertEqual(kwargs["ori_mask_mode"], 0)
        self.assertEqual(kwargs["ori_topk"], 256)
        self.assertIs(kwargs["ori_topk_length"], fm.ori_topk_length)
        self.assertIs(kwargs["seqused_ori_kv"], seqused_kv)
        self.assertEqual(kwargs["ori_win_left"], 129)
        self.assertNotIn("seqused_kv", kwargs)


class TestGetKvIndices(unittest.TestCase):
    _PATCH_TARGET = (
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.get_attn_backend"
    )

    @patch(_PATCH_TARGET)
    def test_page_size_one_plain_slice(self, mock_get_attn_backend):
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=1)
        page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        # req_idx=0, seqlen=5, kv_len=5 -> logic_start=0, logic_end=5
        result = _get_kv_indices(MagicMock(), 5, page_table, 0, 5)
        expected = page_table[0, 0:5]
        self.assertEqual(result.tolist(), expected.tolist())

    @patch(_PATCH_TARGET)
    def test_page_size_one_partial_window(self, mock_get_attn_backend):
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=1)
        page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        # req_idx=0, seqlen=10, kv_len=4 -> logic_start=6, logic_end=10
        result = _get_kv_indices(MagicMock(), 4, page_table, 0, 10)
        expected = page_table[0, 6:10]
        self.assertEqual(result.tolist(), expected.tolist())

    @patch(_PATCH_TARGET)
    def test_page_size_gt_one_paged(self, mock_get_attn_backend):
        page_size = 4
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=page_size)
        page_table = torch.tensor(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.int32
        )
        # req_idx=0, seqlen=6, kv_len=6 -> logic_pos=[0..5]; block_id=[0,0,0,0,1,1]
        # page_table[0, block_id]=[10,10,10,10,20,20]; physical=[40,41,42,43,80,81]
        result = _get_kv_indices(MagicMock(), 6, page_table, 0, 6)
        expected = [40, 41, 42, 43, 80, 81]
        self.assertEqual(result.tolist(), expected)

    @patch(_PATCH_TARGET)
    def test_page_size_gt_one_partial_window(self, mock_get_attn_backend):
        page_size = 4
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=page_size)
        page_table = torch.tensor(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.int32
        )
        # req_idx=0, seqlen=10, kv_len=4 -> logic_pos=[6,7,8,9]; block_id=[1,1,2,2]
        # physical=[82,83,120,121]
        result = _get_kv_indices(MagicMock(), 4, page_table, 0, 10)
        expected = [82, 83, 120, 121]
        self.assertEqual(result.tolist(), expected)

    @patch(_PATCH_TARGET)
    def test_page_size_gt_one_second_request(self, mock_get_attn_backend):
        page_size = 4
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=page_size)
        page_table = torch.tensor(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.int32
        )
        # req_idx=1, seqlen=6 -> page_table[1, block_id]=[50,50,50,50,60,60]
        # physical=[200,201,202,203,240,241]
        result = _get_kv_indices(MagicMock(), 6, page_table, 1, 6)
        expected = [200, 201, 202, 203, 240, 241]
        self.assertEqual(result.tolist(), expected)

    @patch(_PATCH_TARGET)
    def test_kv_len_clamped_to_zero(self, mock_get_attn_backend):
        # kv_len > seqlen -> logic_start = max(0, seqlen - kv_len) = 0
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=1)
        page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        result = _get_kv_indices(MagicMock(), 100, page_table, 0, 3)
        expected = page_table[0, 0:3]
        self.assertEqual(result.tolist(), expected.tolist())


class TestStepOutCacheLoc(unittest.TestCase):
    def _make_backend(self, topk, speculative_num_steps):
        backend = object.__new__(DeepseekV4AscendMultiStepDraftBackend)
        backend.topk = topk
        backend.speculative_num_steps = speculative_num_steps
        return backend

    def test_none_out_cache_loc_returns_none(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        forward_batch = SimpleNamespace(out_cache_loc=None, batch_size=4)
        self.assertIsNone(backend._step_out_cache_loc(forward_batch, 0))

    def test_short_out_cache_loc_returns_as_is(self):
        # numel <= single_step_width (batch_size * topk) -> returned unchanged
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.tensor([10, 20, 30], dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=4)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_short_out_cache_loc_boundary_equal(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(8, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=4)
        # single_step_width = 4*2 = 8; numel=8 <= 8
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_indivisible_returns_as_is(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        # step_layout_width = 2*3 = 6; numel=10, 10 % 6 != 0
        loc = torch.arange(10, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_step_layout_width_zero_returns_as_is(self):
        backend = self._make_backend(topk=0, speculative_num_steps=3)
        loc = torch.arange(5, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_normal_case_step0(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        # view(2,2,3).permute(2,0,1).reshape(3,-1); step 0: [0,3,6,9]
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertEqual(result.tolist(), [0, 3, 6, 9])

    def test_normal_case_step1(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 1)
        self.assertEqual(result.tolist(), [1, 4, 7, 10])

    def test_normal_case_step2(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 2)
        self.assertEqual(result.tolist(), [2, 5, 8, 11])

    def test_normal_case_different_dimensions(self):
        backend = self._make_backend(topk=3, speculative_num_steps=2)
        loc = torch.arange(24, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=4)
        # view(4,3,2).permute(2,0,1).reshape(2,-1)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertEqual(result.tolist(), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

    def test_normal_case_returns_tensor(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIsInstance(result, torch.Tensor)


class TestCompressorEpilogEmptyWrite(unittest.TestCase):
    def setUp(self):
        probe = patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.is_npu_arch35",
            return_value=False,
        )
        probe.start()
        self.addCleanup(probe.stop)

    @staticmethod
    def _backend(*, loc, graph_mode=False):
        backend = CompressorAscendBackendMixin.__new__(CompressorAscendBackendMixin)
        backend.graph_mode = graph_mode
        backend.token_to_kv_pool = MagicMock()
        backend.forward_metadata = SimpleNamespace(c4_loc=loc, c128_loc=loc)
        return backend

    @staticmethod
    def _compressor(*, li_kv_dtype="bf16", is_in_indexer=False):
        return SimpleNamespace(
            ratio=128,
            layer_id=0,
            is_in_indexer=is_in_indexer,
            li_kv_dtype=li_kv_dtype,
        )

    @staticmethod
    def _verify_batch():
        forward_mode = MagicMock()
        forward_mode.is_target_verify.return_value = True
        return SimpleNamespace(forward_mode=forward_mode)

    def test_all_slots_masked_skips_compress_write(self):
        backend = self._backend(loc=torch.zeros(3, dtype=torch.int32))
        backend._compressor_epilog_npu(
            self._compressor(), torch.zeros(3, 512), self._verify_batch()
        )
        backend.token_to_kv_pool.set_compress_buffer.assert_not_called()

    def test_partially_masked_slots_writes_surviving_rows(self):
        backend = self._backend(loc=torch.tensor([0, 7, 0], dtype=torch.int32))
        kv = torch.arange(12, dtype=torch.float32).view(3, 4)
        backend._compressor_epilog_npu(self._compressor(), kv, self._verify_batch())

        backend.token_to_kv_pool.set_compress_buffer.assert_called_once()
        _, written_loc, written_kv, _, _ = (
            backend.token_to_kv_pool.set_compress_buffer.call_args.args
        )
        self.assertEqual(written_loc.tolist(), [7])
        self.assertEqual(written_kv.tolist(), [kv[1].tolist()])

    def test_graph_mode_keeps_static_shape_write(self):
        backend = self._backend(loc=torch.zeros(3, dtype=torch.int32), graph_mode=True)
        backend._compressor_epilog_npu(
            self._compressor(), torch.ones(3, 4), self._verify_batch()
        )

        backend.token_to_kv_pool.set_compress_buffer.assert_called_once()
        written_kv = backend.token_to_kv_pool.set_compress_buffer.call_args.args[2]
        self.assertEqual(written_kv.shape[0], 3)
        self.assertEqual(written_kv.abs().sum().item(), 0.0)

    def test_all_slots_masked_skips_fused_indexer_write(self):
        backend = self._backend(loc=torch.zeros(3, dtype=torch.int32))
        compressor = self._compressor(li_kv_dtype="float8", is_in_indexer=True)
        quant_cache = MagicMock()
        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.npu_cache_ops.indexer_quant_cache",
            quant_cache,
        ):
            backend._compressor_epilog_npu(
                compressor, torch.zeros(3, 512), self._verify_batch()
            )
        quant_cache.assert_not_called()


class TestMirrorlessReplayWidthHint(unittest.TestCase):
    """max_seq_len_host bounds the mirrorless replay-prep copy widths."""

    _PAGE = 128
    _WIDTH = 64  # captured block-table width, in pages

    @staticmethod
    def _verify_mode():
        mode = MagicMock()
        mode.is_target_verify.return_value = True
        mode.is_draft_extend_v2.return_value = False
        mode.is_idle.return_value = False
        return mode

    def _replay_backend(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend.page_size = self._PAGE
        backend._dsv4_has_c4 = True
        backend._dsv4_has_c128 = False
        backend._dsv4_graph_tokens_per_req = 8
        backend._is_dspark_algorithm = False
        backend._is_dspark_draft_worker = False
        backend.graph_metadata = {
            "block_tables": torch.zeros((2, self._WIDTH), dtype=torch.int32)
        }
        return backend

    def _replay_batch(self, hint):
        seq_lens = torch.tensor([300, 660], dtype=torch.int32)
        return SimpleNamespace(
            forward_mode=self._verify_mode(),
            batch_size=2,
            num_padding=0,
            spec_info=None,
            seq_lens=seq_lens,
            seq_lens_cpu=None,  # mirrorless
            req_pool_indices=torch.tensor([0, 1]),
            out_cache_loc=torch.zeros(16, dtype=torch.int64),
            max_seq_len_host=hint,
        )

    def test_replay_ctx_caps_compress_width_at_hint(self):
        backend = self._replay_backend()
        ctx = backend._build_dsv4_graph_replay_ctx(self._replay_batch(hint=300))
        # hint 300 -> ceil(300/128)+1 = 4 pages of slack -> 512 tokens, well
        # under the captured 64-page width.
        self.assertTrue(ctx.mirrorless)
        self.assertEqual(ctx.compress_seq_lens_max, 4 * self._PAGE)

    def test_replay_ctx_caps_compress_width_at_captured_width(self):
        backend = self._replay_backend()
        ctx = backend._build_dsv4_graph_replay_ctx(self._replay_batch(hint=None))
        self.assertEqual(ctx.compress_seq_lens_max, self._WIDTH * self._PAGE)

    def test_replay_ctx_hint_above_captured_width_never_overshoots(self):
        backend = self._replay_backend()
        ctx = backend._build_dsv4_graph_replay_ctx(self._replay_batch(hint=10**9))
        self.assertEqual(ctx.compress_seq_lens_max, self._WIDTH * self._PAGE)

    def _swa_refresh_copied_width(self, hint):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend.page_size = self._PAGE
        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda locs: locs
        )
        copied = {}
        backend._copy_page_table_into_graph = lambda key, src: copied.setdefault(
            "src", src
        )
        fm = SimpleNamespace(
            swa_loc=torch.zeros(4, dtype=torch.int64),
            block_tables_swa=torch.arange(16, dtype=torch.int32).reshape(2, 8),
        )
        ctx = SimpleNamespace(
            fm=fm,
            mirrorless=True,
            bs=2,
            forward_batch=SimpleNamespace(
                max_seq_len_host=hint,
                out_cache_loc=torch.zeros(4, dtype=torch.int64),
            ),
        )
        backend._refresh_graph_swa_metadata_direct(ctx)
        return copied["src"].shape[1]

    def test_swa_refresh_sliced_to_hint_plus_one_page(self):
        # hint 100 (+1 page slack) -> 2 of the 8 captured pages.
        self.assertEqual(self._swa_refresh_copied_width(100), 2)

    def test_swa_refresh_without_hint_copies_full_width(self):
        self.assertEqual(self._swa_refresh_copied_width(None), 8)


if __name__ == "__main__":
    unittest.main()
