"""
Unit tests for sglang.srt.hardware_backend.npu.attention.mla_preprocess.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
    round_up,
    trans_rope_weight,
    transdata,
)
from sglang.test.test_utils import CustomTestCase


class TestMlaPrologInt8KvCache(CustomTestCase):
    """C8 decode must write the same 656-byte records sparse attention reads."""

    def setUp(self):
        self.preprocess = NPUFusedMLAPreprocess.__new__(NPUFusedMLAPreprocess)
        torch.nn.Module.__init__(self.preprocess)
        self.preprocess.no_rope_mlaprolog = None
        self.preprocess.kv_quant_scheme = None
        self.preprocess.has_preprocess_weights = False
        self.preprocess.layer_id = 0
        self.preprocess.weight_quant_mode = 2
        self.preprocess.is_npu_arch35 = False
        self.preprocess.q_lora_rank = 16
        self.preprocess.kv_lora_rank = 512
        self.preprocess.qk_rope_head_dim = 64
        self.preprocess.qk_head_dim = 192
        self.preprocess.num_local_heads = 1
        self.preprocess.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.ones(8, 64, dtype=torch.bfloat16), is_neox_style=False
        )
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            setattr(
                self.preprocess,
                name,
                SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
            )
        self.preprocess.qkv_a_proj = SimpleNamespace(
            input_size=16,
            weight=torch.zeros(16, 592, dtype=torch.int8),
            weight_scale=torch.ones(592),
        )
        self.preprocess.q_b_proj = SimpleNamespace(
            weight=torch.zeros(16, 192, dtype=torch.int8),
            weight_scale=torch.ones(192),
        )
        self.preprocess.w_kc = torch.empty(1, dtype=torch.bfloat16)
        self.hidden_states = torch.zeros(2, 16, dtype=torch.bfloat16)
        self.positions = torch.arange(2)
        self.batch = SimpleNamespace(out_cache_loc=torch.tensor([1, 129]))
        self.pool = SimpleNamespace(start_layer=0, index_head_dim=128)
        self.k_cache = torch.zeros(2, 128, 1, 656, dtype=torch.int8)
        self.kr_cache = torch.empty(2, 128, 1, 0, dtype=torch.bfloat16)
        self.pool.dsa_kv_cache_store_fp8 = True
        self.pool.get_kv_buffer = lambda layer_id: (self.k_cache, self.kr_cache)
        self.outputs = (
            torch.zeros(2, 1, 512, dtype=torch.bfloat16),
            torch.zeros(2, 1, 64, dtype=torch.bfloat16),
            torch.empty(0),
            torch.zeros(2, 16, dtype=torch.int8),
            torch.ones(2, 1),
        )
        self.prolog = Mock(return_value=self.outputs)
        for p in (
            patch(
                "sglang.srt.hardware_backend.npu.attention.mla_preprocess."
                "npu_format_cast",
                side_effect=lambda tensor: tensor,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.attention.mla_preprocess."
                "get_token_to_kv_pool",
                return_value=self.pool,
            ),
            patch.dict(
                "sys.modules",
                {"torch_npu": SimpleNamespace(npu_mla_prolog_v3=self.prolog)},
            ),
            patch.object(
                torch.ops.npu,
                "npu_dynamic_quant",
                create=True,
                return_value=(self.hidden_states.to(torch.int8), torch.ones(2)),
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_int8_cache_uses_per_token_group_quantization_with_bf16_query(self):
        result = self.preprocess.forward_mlaprolog(
            self.positions, self.hidden_states, self.batch
        )
        # These values are the CANN packed-record contract, shared with the
        # pool writer and npu_kv_quant_sparse_flash_attention reader.
        args = self.prolog.call_args.kwargs
        self.assertEqual(args["weight_quant_mode"], 2)
        self.assertEqual(args["kv_cache_quant_mode"], 3)
        self.assertEqual(args["query_quant_mode"], 0)
        self.assertEqual(args["ckvkr_repo_mode"], 1)
        self.assertEqual(args["quant_scale_repo_mode"], 1)
        self.assertEqual(args["tile_size"], 128)
        self.assertEqual(args["cache_mode"], "PA_BSND")
        self.assertIs(args["kv_cache"], self.k_cache)
        self.assertIs(args["kr_cache"], self.kr_cache)
        self.assertEqual(args["cache_index"].dtype, torch.int64)
        self.assertIsNone(args.get("quant_scale_ckv"))
        # CANN requires a device FP32 [1] clip factor for packed INT8 KV.
        # Alpha=1 matches the pool writer's unclipped dynamic quantization.
        alpha = args.get("k_nope_clip_alpha")
        self.assertIsNotNone(alpha, "MLAProlog requires k_nope_clip_alpha")
        self.assertEqual(alpha.device, self.k_cache.device)
        self.assertTrue(alpha.is_contiguous())
        torch.testing.assert_close(
            alpha, torch.ones(1, dtype=torch.float32, device=self.k_cache.device)
        )
        self.assertIs(result[0], self.outputs[1])
        self.assertIs(result[2], self.outputs[0])
        self.assertIs(result[3], self.k_cache)
        self.assertIs(result[4], self.outputs[3])
        torch.testing.assert_close(result[-2], torch.ones(2))
        self.assertIs(result[-1], self.outputs[2])

        # Warmup/capture must reuse the initialized tensor on later forwards.
        self.preprocess.forward_mlaprolog(
            self.positions, self.hidden_states, self.batch
        )
        self.assertIs(
            self.prolog.call_args.kwargs["k_nope_clip_alpha"], args["k_nope_clip_alpha"]
        )

    def test_bf16_cache_keeps_unquantized_cache_mode(self):
        self.pool.dsa_kv_cache_store_fp8 = False
        self.k_cache = torch.zeros(2, 128, 1, 512, dtype=torch.bfloat16)
        self.kr_cache = torch.zeros(2, 128, 1, 64, dtype=torch.bfloat16)
        result = self.preprocess.forward_mlaprolog(
            self.positions, self.hidden_states, self.batch
        )
        args = self.prolog.call_args.kwargs
        self.assertEqual(args["kv_cache_quant_mode"], 0)
        self.assertIsNone(args.get("k_nope_clip_alpha"))
        self.assertEqual(args.get("ckvkr_repo_mode", 0), 0)
        self.assertIs(args["kr_cache"], self.kr_cache)
        self.assertIs(result[2], self.outputs[0])

    def test_int8_cache_rejects_incompatible_projection_weights(self):
        self.preprocess.has_preprocess_weights = True
        for mode in (0, 1, 3):
            with self.subTest(mode=mode):
                self.preprocess.weight_quant_mode = mode
                with self.assertRaisesRegex(RuntimeError, "requires dynamic INT8"):
                    self.preprocess.forward_mlaprolog(
                        self.positions, self.hidden_states, self.batch
                    )

    def test_int8_cache_rejects_legacy_preprocessing(self):
        self.preprocess.qkv_a_proj = SimpleNamespace(quant_method=None)
        with patch.object(self.preprocess, "uses_mlaprolog", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires MLAProlog"):
                self.preprocess.forward(
                    self.positions, self.hidden_states, self.batch, None
                )


class TestRoundUp(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(round_up(16, 16), 16)
        self.assertEqual(round_up(32, 16), 32)

    def test_round_up_to_block(self):
        self.assertEqual(round_up(1, 16), 16)
        self.assertEqual(round_up(10, 16), 16)
        self.assertEqual(round_up(17, 16), 32)

    def test_zero_value(self):
        self.assertEqual(round_up(0, 16), 0)

    def test_zero_align_returns_zero(self):
        self.assertEqual(round_up(100, 0), 0)
        self.assertEqual(round_up(0, 0), 0)

    def test_negative_align(self):
        # Pin current behavior for negative align.
        self.assertEqual(round_up(10, -4), 8)


class TestTransdata(unittest.TestCase):
    def test_aligned_32x32_default_block(self):
        mat = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
        nz = transdata(mat)
        self.assertEqual(nz.shape, (2, 32, 16))

    def test_aligned_element_mapping(self):
        rows, cols = 16, 32
        mat = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        bs = (16, 16)
        nz = transdata(mat, block_size=bs)

        # NZ layout: element (r, c) lands at nz[c // bs1, r, c % bs1].
        for r in range(rows):
            for c in range(cols):
                expected = mat[r, c]
                actual = nz[c // bs[1], r, c % bs[1]]
                self.assertEqual(
                    actual.item(),
                    expected.item(),
                    f"mismatch at (r={r}, c={c})",
                )

    def test_padding_for_non_aligned_shape(self):
        mat = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        nz = transdata(mat, block_size=(16, 16))
        # c_blocks = ceil(10/16) = 1, r_padded = 16, bs1 = 16
        self.assertEqual(nz.shape, (1, 16, 16))

        # Original elements are preserved in the padded NZ layout.
        for r in range(10):
            for c in range(10):
                self.assertEqual(
                    nz[c // 16, r, c % 16].item(),
                    mat[r, c].item(),
                    f"padded mismatch at (r={r}, c={c})",
                )

    def test_non_square_matrix(self):
        mat = torch.arange(16 * 48, dtype=torch.float32).reshape(16, 48)
        nz = transdata(mat, block_size=(16, 16))
        # c_blocks = 48//16 = 3, r = 16, bs1 = 16
        self.assertEqual(nz.shape, (3, 16, 16))

    def test_custom_block_size(self):
        mat = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
        nz = transdata(mat, block_size=(4, 4))
        self.assertEqual(nz.shape, (2, 8, 4))

    def test_3d_input_raises(self):
        mat = torch.arange(2 * 16 * 16, dtype=torch.float32).reshape(2, 16, 16)
        with self.assertRaises(RuntimeError):
            transdata(mat, block_size=(16, 16))


class TestTransRopeWeight(unittest.TestCase):
    def test_basic_reorder(self):
        # 8 rows, rope_dim=4 → last 4 rows are the RoPE region.
        weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=4)

        # RoPE region (rows 4-7) reordered: even indices first → [4,6,5,7].
        rope_region = original[4:8]
        expected_rope = torch.stack(
            [rope_region[0], rope_region[2], rope_region[1], rope_region[3]]
        )
        self.assertTrue(torch.equal(result[4:8], expected_rope))

    def test_non_rope_region_unchanged(self):
        weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4).clone()
        original = weight.clone()
        trans_rope_weight(weight, rope_dim=4)
        self.assertTrue(torch.equal(weight[0:4], original[0:4]))

    def test_rope_dim_zero_reorders_all(self):
        weight = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=0)
        # Rows [0,1,2,3] → [0,2,1,3] (even first, then odd)
        expected = torch.stack([original[0], original[2], original[1], original[3]])
        self.assertTrue(torch.equal(result, expected))

    def test_full_rope_dim(self):
        weight = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=4)
        # rows [0,1,2,3] → [0,2,1,3]
        expected = torch.stack([original[0], original[2], original[1], original[3]])
        self.assertTrue(torch.equal(result, expected))

    def test_3d_weight(self):
        # Shape (2, 8, 4): 8 rows per expert, rope_dim=4
        weight = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=4)

        for e in range(2):
            rope_region = original[e, 4:8]
            expected_rope = torch.stack(
                [rope_region[0], rope_region[2], rope_region[1], rope_region[3]]
            )
            self.assertTrue(torch.equal(result[e, 4:8], expected_rope))


class TestIsFiaNz(unittest.TestCase):
    def setUp(self):
        is_mla_preprocess_enabled.cache_clear()
        is_fia_nz.cache_clear()

    def tearDown(self):
        is_mla_preprocess_enabled.cache_clear()
        is_fia_nz.cache_clear()

    def test_not_set_returns_false(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_USE_FIA_NZ", None)
            os.environ.pop("SGLANG_NPU_USE_MLAPO", None)
            self.assertFalse(is_fia_nz())

    def test_fia_nz_with_mlapo_returns_true(self):
        with patch.dict(
            os.environ,
            {"SGLANG_USE_FIA_NZ": "1", "SGLANG_NPU_USE_MLAPO": "1"},
        ):
            self.assertTrue(is_fia_nz())

    def test_fia_nz_without_mlapo_is_independent(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_NPU_USE_MLAPO", None)
            os.environ["SGLANG_USE_FIA_NZ"] = "1"
            self.assertTrue(is_fia_nz())


if __name__ == "__main__":
    unittest.main()
