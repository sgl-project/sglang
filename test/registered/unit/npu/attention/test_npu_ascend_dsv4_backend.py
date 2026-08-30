"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.
"""

import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="base-a-test-1-npu-a2")

for mod in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sgl_kernel_npu.norm",
    "sgl_kernel_npu.norm.add_rmsnorm_bias",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
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

from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    C4IndexerAscendBackendMixin,
    CompressorAscendBackendMixin,
    DeepseekV4AscendAttnBackend,
    DeepseekV4AscendMultiStepDraftBackend,
    _apply_hadamard,
    _build_cycle_state_block_table,
    _get_kv_indices,
    _sparse_attn_kv_quant_kwargs,
    _sparse_attn_ops,
    _walsh_hadamard_matrix,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import DSV4NPUTokenToKVPool


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
    def test_skips_unused_compressed_locs_but_preserves_full_and_swa_steps(self):
        backend = DeepseekV4AscendMultiStepDraftBackend.__new__(
            DeepseekV4AscendMultiStepDraftBackend
        )
        backend.topk = 1
        backend.speculative_num_steps = 3
        backend._needs_step_compressed_locs = False

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
    def test_registers_single_rows_instead_of_request_banks(self):
        attn_state = torch.empty((32, 5), dtype=torch.float32)
        indexer_state = torch.empty((32, 7), dtype=torch.float32)
        pool = DSV4NPUTokenToKVPool.__new__(DSV4NPUTokenToKVPool)
        pool.compress_state_pools = [
            SimpleNamespace(
                ratio=4,
                ring_size=8,
                kv_score_buffer=SimpleNamespace(kv_score=attn_state),
            )
        ]
        pool.indexer_compress_state_pools = [
            SimpleNamespace(
                ratio=4,
                ring_size=8,
                kv_score_buffer=SimpleNamespace(kv_score=indexer_state),
            )
        ]

        _, data_lens, item_lens = pool.get_c4_state_buf_infos()

        self.assertEqual(data_lens, [attn_state.nbytes, indexer_state.nbytes])
        self.assertEqual(
            item_lens,
            [attn_state[0].nbytes, indexer_state[0].nbytes],
        )


class TestC4IndexerInitialization(unittest.TestCase):
    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._is_npu_arch35",
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
    def test_arch35_cycle_table_is_one_bank_per_request(self):
        req_pool_indices = torch.tensor([7, 3], dtype=torch.int64)
        table = _build_cycle_state_block_table(req_pool_indices)
        self.assertEqual(tuple(table.shape), (2,))
        self.assertEqual(table.dtype, torch.int32)
        self.assertEqual(table.tolist(), [7, 3])

    def test_arch35_cycle_table_rejects_explicit_shape(self):
        with self.assertRaises(ValueError):
            _build_cycle_state_block_table(torch.zeros((2, 8), dtype=torch.int32))

    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._is_npu_arch35",
        return_value=True,
    )
    def test_arch35_eager_metadata_builds_cycle_table(self, _):
        backend = CompressorAscendBackendMixin.__new__(CompressorAscendBackendMixin)
        backend.forward_metadata = SimpleNamespace()
        backend.token_to_kv_pool = MagicMock()
        backend.req_to_token = torch.empty((0, 0), dtype=torch.int32)
        backend.req_to_token_pool = MagicMock()
        backend._dsv4_compress_ratios = ()
        backend._compute_compress_locs = MagicMock(return_value={})

        forward_mode = MagicMock()
        forward_mode.is_decode.return_value = True
        forward_mode.is_target_verify.return_value = False
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            req_pool_indices=torch.tensor([7, 3], dtype=torch.int64),
            seq_lens=torch.tensor([5, 9], dtype=torch.int32),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            out_cache_loc_dsv4=None,
            batch_size=2,
        )

        backend._build_npu_compress_metadata(forward_batch)

        table = getattr(backend.forward_metadata, "dsv4_cycle_state_block_table", None)
        self.assertIsNotNone(table)
        self.assertEqual(table.tolist(), [7, 3])
        self.assertEqual(table.dtype, torch.int32)

    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._is_npu_arch35",
        return_value=True,
    )
    def test_arch35_graph_replay_slices_static_req_pool_buffer_to_graph_bs(self, _):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        table = torch.zeros(1, dtype=torch.int32)
        graph_mode = MagicMock()
        graph_mode.is_decode.return_value = False
        graph_mode.is_target_verify.return_value = False
        ctx = SimpleNamespace(
            fm=SimpleNamespace(dsv4_cycle_state_block_table=table),
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.arange(7, 19, dtype=torch.int64)
            ),
            graph_mode=graph_mode,
            bs=1,
        )
        backend._build_dsv4_graph_replay_ctx = MagicMock(return_value=ctx)
        for name in (
            "_refresh_graph_seq_metadata",
            "_refresh_graph_compress_page_tables_direct",
            "_refresh_graph_explicit_state_block_tables",
            "_refresh_graph_swa_metadata_direct",
            "_refresh_graph_dspark_sparse_metadata",
            "_refresh_graph_kernel_metadata",
        ):
            setattr(backend, name, MagicMock())

        backend._apply_dsv4_graph_metadata(SimpleNamespace())

        self.assertIs(ctx.fm.dsv4_cycle_state_block_table, table)
        self.assertEqual(table.tolist(), [7])

    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._is_npu_arch35",
        return_value=True,
    )
    def test_arch35_graph_capture_allocates_cycle_table_buffer(self, _):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        metadata = SimpleNamespace()
        backend.device = "cpu"
        backend.graph_metadata = {
            2: metadata,
            "swa_page_table": torch.full((2, 4), -1, dtype=torch.int32),
            "c4_page_table": torch.full((2, 4), -1, dtype=torch.int32),
            "c128_page_table": torch.full((2, 4), -1, dtype=torch.int32),
            "kernel_metadata_c1a": torch.zeros(1024, dtype=torch.int32),
            "kernel_metadata_c4a": torch.zeros(1024, dtype=torch.int32),
            "kernel_metadata_c128a": torch.zeros(1024, dtype=torch.int32),
            "kernel_metadata_li_quant": torch.zeros(1024, dtype=torch.int32),
            "c4_topk_indices": torch.full((2, 1), -1, dtype=torch.int32),
        }
        backend._dsv4_graph_tokens_per_req = 1
        backend._dsv4_index_topk = 1
        backend._dsv4_state_pools_by_ratio = {}
        backend._dsv4_sliding_window_size = 128
        backend._is_dspark_draft_worker = False
        forward_mode = MagicMock()
        forward_mode.is_target_verify.return_value = False
        forward_mode.is_draft_extend_v2.return_value = False

        backend._init_dsv4_graph_metadata(2, forward_mode)

        table = getattr(metadata, "dsv4_cycle_state_block_table", None)
        self.assertIsNotNone(table)
        self.assertEqual(tuple(table.shape), (2,))
        self.assertEqual(table.dtype, torch.int32)

    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._is_npu_arch35",
        return_value=True,
    )
    def test_arch35_forward_reuses_batch_cycle_table(self, _):
        table = torch.tensor([7, 3], dtype=torch.int32)
        backend = CompressorAscendBackendMixin.__new__(CompressorAscendBackendMixin)
        backend.graph_mode = False
        backend.forward_metadata = SimpleNamespace(
            dsv4_cycle_state_block_table=table,
            positions_cmp_padding_c128=torch.empty(0, dtype=torch.int64),
            actual_seq_lengths_q_pa=torch.tensor([0, 1, 2], dtype=torch.int32),
            seqused=torch.ones(2, dtype=torch.int32),
            start_pos=torch.zeros(2, dtype=torch.int32),
            c128_loc=None,
        )
        backend.token_to_kv_pool = MagicMock()
        backend.token_to_kv_pool._get_state_pool.return_value = SimpleNamespace(
            state_cache_3d=torch.empty(0)
        )
        backend._ensure_compressor_hadamard = MagicMock()
        backend._ensure_fused_caches = MagicMock()
        backend._compressor_epilog_npu = MagicMock()

        compressor = SimpleNamespace(
            ratio=128,
            overlap=False,
            layer_id=0,
            is_in_indexer=False,
            freqs_cis=None,
            rotary_emb=None,
            _fused_wkv_w=torch.empty(0),
            _fused_wgate_w=torch.empty(0),
            ape=torch.empty(0),
            _fused_norm_weight_fp32=torch.empty(0),
            rope_head_dim=64,
            norm=SimpleNamespace(variance_epsilon=1e-6),
            rotate=False,
        )
        forward_mode = MagicMock()
        forward_mode.is_prefill.return_value = False
        forward_mode.is_target_verify.return_value = False
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([7, 3], dtype=torch.int64),
            forward_mode=forward_mode,
        )
        rope = MagicMock()
        rope.get_cos_sin.return_value = (torch.empty(0), torch.empty(0))

        with (
            patch(
                "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend."
                "Dsv4NpuRoPE.for_freqs",
                return_value=rope,
            ),
            patch.object(torch.ops, "custom", MagicMock(), create=True) as custom_ops,
            patch.object(torch.ops, "npu", MagicMock(), create=True) as npu_ops,
        ):
            custom_ops.compressor.return_value = torch.empty((0, 1))
            backend.forward_compress(compressor, torch.empty((2, 1)), forward_batch)
            backend.forward_compress(compressor, torch.empty((2, 1)), forward_batch)

        self.assertEqual(npu_ops.compressor.call_count, 0)
        self.assertIs(
            custom_ops.compressor.call_args_list[0].kwargs["state_block_table"], table
        )
        self.assertIs(
            custom_ops.compressor.call_args_list[1].kwargs["state_block_table"], table
        )


class TestArch35SparseAttentionDispatch(unittest.TestCase):
    _ARCH35_PATCH_TARGET = (
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._is_npu_arch35"
    )

    @patch(_ARCH35_PATCH_TARGET, return_value=True)
    def test_arch35_uses_kv_quant_ops_and_layout_kwargs(self, _):
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            metadata_op, attention_op = _sparse_attn_ops()
            kwargs = _sparse_attn_kv_quant_kwargs()

        self.assertIs(
            metadata_op, custom_ops.npu_kv_quant_sparse_attn_sharedkv_metadata
        )
        self.assertIs(attention_op, custom_ops.npu_kv_quant_sparse_attn_sharedkv)
        self.assertEqual(
            kwargs,
            {"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64},
        )

    @patch(_ARCH35_PATCH_TARGET, return_value=False)
    def test_pre_arch35_keeps_legacy_ops_without_quant_kwargs(self, _):
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            metadata_op, attention_op = _sparse_attn_ops()
            kwargs = _sparse_attn_kv_quant_kwargs()

        self.assertIs(metadata_op, custom_ops.npu_sparse_attn_sharedkv_metadata)
        self.assertIs(attention_op, custom_ops.npu_sparse_attn_sharedkv)
        self.assertEqual(kwargs, {})


class TestSparseAttentionMetadata(unittest.TestCase):
    _ARCH35_PATCH_TARGET = (
        "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend._is_npu_arch35"
    )

    def test_device_metadata_receives_sequence_lengths(self):
        cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32)
        seqused_kv = torch.tensor([8, 12], dtype=torch.int32)

        for is_arch35, metadata_op_name in (
            (False, "npu_sparse_attn_sharedkv_metadata"),
            (True, "npu_kv_quant_sparse_attn_sharedkv_metadata"),
        ):
            with (
                self.subTest(is_arch35=is_arch35),
                patch(self._ARCH35_PATCH_TARGET, return_value=is_arch35),
                patch("torch.ops.custom", MagicMock(), create=True) as custom_ops,
            ):
                backend = DeepseekV4AscendAttnBackend.__new__(
                    DeepseekV4AscendAttnBackend
                )
                backend.forward_metadata = SimpleNamespace()
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
                    block_tables=torch.zeros((2, 1), dtype=torch.int32),
                    max_seqlen_q=2,
                    is_nextn=False,
                )

                metadata_op = getattr(custom_ops, metadata_op_name)
                self.assertEqual(metadata_op.call_count, 3)
                for call in metadata_op.call_args_list:
                    self.assertIs(call.kwargs["cu_seqlens_q"], cu_seqlens_q)
                    self.assertIs(call.kwargs["seqused_kv"], seqused_kv)


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


class TestCommonTemplate(unittest.TestCase):
    def _make_backend(self, speculative_num_steps):
        backend = object.__new__(DeepseekV4AscendMultiStepDraftBackend)
        backend.speculative_num_steps = speculative_num_steps
        return backend

    def test_calls_call_fn_for_each_step(self):
        backend = self._make_backend(speculative_num_steps=4)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        # range(speculative_num_steps - 1) = range(3) -> i=0,1,2
        self.assertEqual(call_fn.call_count, 3)
        for i, call in enumerate(call_fn.call_args_list):
            self.assertEqual(call.args[0], i)
            self.assertIs(call.args[1], forward_batch)

    def test_single_step_no_calls(self):
        backend = self._make_backend(speculative_num_steps=1)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 0)

    def test_two_steps_one_call(self):
        backend = self._make_backend(speculative_num_steps=2)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 1)
        self.assertEqual(call_fn.call_args_list[0].args[0], 0)

    def test_asserts_spec_info_not_none(self):
        backend = self._make_backend(speculative_num_steps=3)
        forward_batch = SimpleNamespace(spec_info=None)
        call_fn = MagicMock()
        with self.assertRaises(AssertionError):
            backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 0)

    def test_call_fn_exception_propagates(self):
        backend = self._make_backend(speculative_num_steps=3)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 1)


class TestCompressorEpilogEmptyWrite(unittest.TestCase):
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
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            backend._compressor_epilog_npu(
                compressor, torch.zeros(3, 512), self._verify_batch()
            )
        custom_ops.indexer_compress_epilog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
