"""
Unit tests for sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu.

Tests are ordered to match the source file:
  1. forward_mha_prepare_npu   (L29)
  2. forward_mha_core_npu      (L132)
  3. forward_mla_prepare_npu   (L149)
  4. forward_mla_core_npu      (L282)
  5. forward_dsa_prepare_npu   (L326)
  6. forward_dsa_core_npu      (L464)
  7. npu_mla_preprocess        (L519)
"""

import sys
import types
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE importing sglang.
for _ in (
    "triton",
    "triton.language",
    "triton.runtime",
    "IPython",
    "IPython.display",
    "aiohttp",
    "vllm_ascend",
    "batch_invariant_ops",
):
    sys.modules.setdefault(_, MagicMock())

import unittest
from enum import Enum, auto
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Ensure torch.npu and torch.ops.npu exist for code paths that reference them.
if not hasattr(torch, "npu"):
    torch.npu = MagicMock()
if not hasattr(torch.ops, "npu"):
    torch.ops.npu = MagicMock()
torch.ops.npu.batch_matmul_transpose = MagicMock()

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=6, suite="stage-a-unit-test-npu")

# ---------------------------------------------------------------------------
# Constants for tensor shapes used throughout the tests.
# ---------------------------------------------------------------------------
NT = 4  # num_tokens
NH = 2  # num_local_heads
QK_ND = 6  # qk_nope_head_dim
QK_RD = 4  # qk_rope_head_dim
KV_LR = 6  # kv_lora_rank
VHD = 6  # v_head_dim
QK_HD = QK_ND + QK_RD  # qk_head_dim
QLR = 8  # q_lora_rank


# ---------------------------------------------------------------------------
# Module-level stubs for NPU-only and heavy packages.
# ---------------------------------------------------------------------------
def _make_stub(name):
    """Create a proper ModuleType stub (with __spec__) in sys.modules."""
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, None)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


_torch_npu = _make_stub("torch_npu")
_torch_npu.npu_interleave_rope = MagicMock()
_torch_npu.npu_kv_rmsnorm_rope_cache = MagicMock()
_torch_npu.npu_transpose_batchmatmul = MagicMock()

_sgl_kernel = _make_stub("sgl_kernel_npu")
_sgl_norm = _make_stub("sgl_kernel_npu.norm")
_sgl_fsqn = _make_stub("sgl_kernel_npu.norm.fused_split_qk_norm")
_sgl_norm.fused_split_qk_norm = _sgl_fsqn
_sgl_kernel.norm = _sgl_norm
_sgl_fsqn.fused_split_qk_norm = MagicMock(
    return_value=(
        torch.randn(NT, QLR),
        torch.randn(NT, 1, KV_LR),
        torch.randn(NT, 1, QK_RD),
    )
)

_dsa_indexer = _make_stub("sglang.srt.layers.attention.dsa.dsa_npu_indexer")
_dsa_indexer.scattered_to_tp_attn_full = MagicMock()

_dsa_utils = _make_stub("sglang.srt.layers.attention.dsa.utils")
_dsa_utils.dsa_use_prefill_cp = MagicMock(return_value=False)


class _ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()
    MOE_FULL = auto()


_comm = _make_stub("sglang.srt.layers.communicator")
_comm.ScatterMode = _ScatterMode
_comm.get_attn_tp_context = MagicMock()

from sglang.srt.hardware_backend.npu.modules import deepseek_v2_attention_mla_npu as dsa_mod


# ---------------------------------------------------------------------------
# Helper: set all model attributes on any object.
# ---------------------------------------------------------------------------
def _set_model_attrs(obj, **ov):
    obj.q_lora_rank = ov.get("q_lora_rank", QLR)
    obj.kv_lora_rank = KV_LR
    obj.qk_rope_head_dim = QK_RD
    obj.qk_nope_head_dim = QK_ND
    obj.qk_head_dim = QK_HD
    obj.v_head_dim = VHD
    obj.num_local_heads = NH
    obj.layer_id = ov.get("layer_id", 0)
    obj.use_dsa = ov.get("use_dsa", False)
    obj.use_deepseek_yarn_rope = ov.get("use_deepseek_yarn_rope", False)
    obj.skip_topk = ov.get("skip_topk", False)
    obj.is_nextn = ov.get("is_nextn", False)
    obj.next_skip_topk = ov.get("next_skip_topk", False)
    obj.alt_stream = ov.get("alt_stream", None)
    obj.quant_config = ov.get("quant_config", None)
    obj._disable_npu_fused_split_qk_norm = ov.get(
        "_disable_npu_fused_split_qk_norm", False
    )

    obj.w_kc = torch.randn(NH, QK_ND, KV_LR)
    obj.w_vc = torch.randn(NH, KV_LR, VHD)

    obj.q_proj = MagicMock(return_value=(torch.randn(NT, NH * QK_HD), None))
    obj.q_b_proj = MagicMock(return_value=(torch.randn(NT, NH * QK_HD), None))
    obj.kv_a_proj_with_mqa = MagicMock(
        return_value=(torch.randn(NT, KV_LR + QK_RD), None)
    )
    obj.kv_b_proj = MagicMock(
        return_value=(torch.randn(NT, NH * (QK_ND + VHD)), None)
    )
    obj.fused_qkv_a_proj_with_mqa = MagicMock(
        return_value=(torch.randn(NT, QLR + KV_LR + QK_RD), None)
    )
    obj.o_proj = MagicMock(return_value=(torch.randn(NT, NH * VHD), None))

    obj.q_a_layernorm = MagicMock(side_effect=lambda x: x)
    obj.kv_a_layernorm = MagicMock(side_effect=lambda x: x)
    obj.kv_a_layernorm.weight = torch.ones(KV_LR)
    obj.kv_a_layernorm.variance_epsilon = 1e-6

    obj.rotary_emb = MagicMock(
        return_value=(
            torch.randn(NT, NH, QK_RD),
            torch.randn(NT, 1, QK_RD),
        )
    )
    obj.rotary_emb.is_neox_style = ov.get("is_neox_style", True)
    obj.rotary_emb.cos_sin_cache = torch.randn(100, 2, QK_RD // 2)
    obj.rotary_emb.sin_cos_cache = torch.randn(100, 2, QK_RD // 2)

    obj.attn_mha = MagicMock(return_value=torch.randn(NT, NH, VHD))
    obj.attn_mqa = MagicMock(return_value=torch.randn(NT, NH, KV_LR))

    obj._concat_and_cast_mha_k = MagicMock(
        return_value=torch.randn(NT, NH, QK_ND + QK_RD)
    )
    obj.indexer = MagicMock(return_value=torch.randn(NT, 10))
    obj.rebuild_cp_kv_cache = MagicMock(
        return_value=(
            torch.randn(NT, 1, KV_LR),
            torch.randn(NT, 1, QK_RD),
        )
    )
    return obj


def _make_model(**ov):
    """MagicMock model — hasattr always True for any attribute."""
    return _set_model_attrs(MagicMock(), **ov)


def _make_plain_model(**ov):
    """SimpleNamespace model — hasattr returns False for unset attrs.

    Use this when testing ``hasattr(m, 'mla_preprocess')`` branch logic.
    """
    return _set_model_attrs(SimpleNamespace(), **ov)


def _make_forward_batch(mode="decode"):
    fb = MagicMock()
    fb.out_cache_loc = torch.arange(NT, dtype=torch.int64)
    if mode == "decode":
        fb.forward_mode.is_decode.return_value = True
        fb.forward_mode.is_extend.return_value = False
        fb.forward_mode.is_draft_extend_v2.return_value = False
        fb.forward_mode.is_target_verify.return_value = False
    elif mode == "extend":
        fb.forward_mode.is_decode.return_value = False
        fb.forward_mode.is_extend.return_value = True
        fb.forward_mode.is_draft_extend_v2.return_value = False
        fb.forward_mode.is_target_verify.return_value = False
    return fb


def _make_qkv_latent():
    return torch.randn(NT, QLR + KV_LR + QK_RD)


def _reset_module_mocks():
    """Reset call history on module-level mocks shared across tests."""
    dsa_mod.fused_split_qk_norm.reset_mock()
    dsa_mod.scattered_to_tp_attn_full.reset_mock()
    # scattered_to_tp_attn_full should return its input tensor unchanged
    # so that downstream split/view/bmm operations work on real tensors.
    dsa_mod.scattered_to_tp_attn_full.side_effect = lambda x, fb: x
    _torch_npu.npu_interleave_rope.reset_mock()
    _torch_npu.npu_kv_rmsnorm_rope_cache.reset_mock()
    _torch_npu.npu_transpose_batchmatmul.reset_mock()
    torch.ops.npu.batch_matmul_transpose.reset_mock()
    # Reset torch.npu mocks (Event, current_stream, stream are MagicMocks)
    torch.npu.Event.reset_mock()
    torch.npu.current_stream.reset_mock()


# ===========================================================================
# 1. forward_mha_prepare_npu  (source L29–129)
# ===========================================================================
class TestForwardMhaPrepareNpu(unittest.TestCase):
    def setUp(self):
        dsa_mod._use_ag_after_qlora = False
        _reset_module_mocks()

    def _call(self, m, scatter_modes=None):
        positions = torch.arange(NT)
        hidden_states = torch.randn(NT, 8)
        forward_batch = _make_forward_batch()
        if scatter_modes is None:
            scatter_modes = SimpleNamespace(
                layer_input_mode=_ScatterMode.TP_ATTN_FULL,
                attn_mode=_ScatterMode.TP_ATTN_FULL,
            )
        mock_ctx = MagicMock()
        mock_ctx.fetch_qkv_latent = MagicMock(return_value=_make_qkv_latent())
        mock_pool = MagicMock()
        mock_pool.get_kv_buffer.return_value = (
            torch.randn(100, NH, VHD),
            torch.randn(100, 1, QK_RD),
        )
        with (
            patch.object(dsa_mod, "get_attn_tp_context", return_value=mock_ctx),
            patch.object(dsa_mod, "get_token_to_kv_pool", return_value=mock_pool),
        ):
            return (
                dsa_mod.forward_mha_prepare_npu(
                    m, positions, hidden_states, forward_batch, None, scatter_modes
                ),
                mock_pool,
            )

    # -- L37: q_lora_rank is None → direct q_proj + kv_a_proj_with_mqa --
    def test_q_lora_rank_none_calls_q_proj(self):
        m = _make_model(q_lora_rank=None)
        self._call(m)
        self.assertTrue(m.q_proj.called)
        self.assertTrue(m.kv_a_proj_with_mqa.called)
        self.assertFalse(m.q_b_proj.called)

    # -- L37: q_lora_rank is not None → fetch_qkv_latent + q_b_proj --
    def test_q_lora_rank_not_none_calls_q_b_proj(self):
        m = _make_model(q_lora_rank=QLR)
        self._call(m)
        self.assertTrue(m.q_b_proj.called)
        self.assertFalse(m.q_proj.called)

    # -- L49: use_dsa=True → indexer called --
    def test_use_dsa_calls_indexer(self):
        m = _make_model(q_lora_rank=QLR, use_dsa=True)
        self._call(m)
        self.assertTrue(m.indexer.called)

    # -- L61: use_dsa=False → indexer NOT called --
    def test_no_dsa_does_not_call_indexer(self):
        m = _make_model(q_lora_rank=QLR, use_dsa=False)
        self._call(m)
        self.assertFalse(m.indexer.called)

    # -- L63-69: _use_ag_after_qlora=True + SCATTERED → scattered_to_tp_attn_full --
    def test_ag_after_qlora_calls_scattered_to_tp_attn_full(self):
        dsa_mod._use_ag_after_qlora = True
        m = _make_model(q_lora_rank=QLR, use_dsa=False)
        scatter_modes = SimpleNamespace(
            layer_input_mode=_ScatterMode.SCATTERED,
            attn_mode=_ScatterMode.TP_ATTN_FULL,
        )
        self._call(m, scatter_modes)
        self.assertTrue(dsa_mod.scattered_to_tp_attn_full.called)

    # -- L63: _use_ag_after_qlora=False → scattered NOT called --
    def test_no_ag_after_qlora_no_scattered(self):
        dsa_mod._use_ag_after_qlora = False
        m = _make_model(q_lora_rank=QLR, use_dsa=False)
        self._call(m)
        self.assertFalse(dsa_mod.scattered_to_tp_attn_full.called)

    # -- L80-110: use_deepseek_yarn_rope=True → npu_interleave_rope +
    #    npu_kv_rmsnorm_rope_cache called --
    def test_yarn_rope_calls_npu_interleave_rope(self):
        m = _make_model(use_deepseek_yarn_rope=True)
        m.rotary_emb.get_cos_sin_cache = MagicMock(
            return_value=(torch.randn(NT, QK_RD), torch.randn(NT, QK_RD))
        )
        _torch_npu.npu_interleave_rope.return_value = torch.randn(NT, NH, 1, QK_RD)
        _torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (
            MagicMock(), MagicMock(),
            torch.randn(NT, 1, QK_RD),  # k_pe
            torch.randn(NT, KV_LR),     # kv_a
        )
        mock_pool = MagicMock()
        mock_pool.get_kv_buffer.return_value = (
            torch.randn(100, NH, VHD), torch.randn(100, 1, QK_RD)
        )
        positions = torch.arange(NT)
        hidden_states = torch.randn(NT, 8)
        forward_batch = _make_forward_batch()
        scatter_modes = SimpleNamespace(
            layer_input_mode=_ScatterMode.TP_ATTN_FULL,
            attn_mode=_ScatterMode.TP_ATTN_FULL,
        )
        mock_ctx = MagicMock()
        mock_ctx.fetch_qkv_latent = MagicMock(return_value=_make_qkv_latent())
        with (
            patch.object(dsa_mod, "get_attn_tp_context", return_value=mock_ctx),
            patch.object(dsa_mod, "get_token_to_kv_pool", return_value=mock_pool),
        ):
            dsa_mod.forward_mha_prepare_npu(
                m, positions, hidden_states, forward_batch, None, scatter_modes
            )
        self.assertTrue(_torch_npu.npu_interleave_rope.called)
        self.assertTrue(_torch_npu.npu_kv_rmsnorm_rope_cache.called)

    # -- L80: yarn_rope=True → set_kv_buffer NOT called (fused write instead) --
    def test_yarn_rope_does_not_call_set_kv_buffer(self):
        m = _make_model(use_deepseek_yarn_rope=True)
        m.rotary_emb.get_cos_sin_cache = MagicMock(
            return_value=(torch.randn(NT, QK_RD), torch.randn(NT, QK_RD))
        )
        _torch_npu.npu_interleave_rope.return_value = torch.randn(NT, NH, 1, QK_RD)
        _torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (
            MagicMock(), MagicMock(),
            torch.randn(NT, 1, QK_RD), torch.randn(NT, KV_LR),
        )
        result, mock_pool = self._call(m)
        self.assertFalse(mock_pool.set_kv_buffer.called)

    # -- L111-119: yarn_rope=False, rotary_emb is not None → rotary_emb called --
    def test_no_yarn_rope_calls_rotary_emb(self):
        m = _make_model(use_deepseek_yarn_rope=False)
        self._call(m)
        self.assertTrue(m.rotary_emb.called)

    # -- L114: rotary_emb is None → rotary_emb NOT called, set_kv_buffer still called --
    def test_rotary_emb_none_skips_rotary(self):
        m = _make_model(use_deepseek_yarn_rope=False)
        m.rotary_emb = None
        result, mock_pool = self._call(m)
        self.assertTrue(mock_pool.set_kv_buffer.called)

    # -- L128: return 4-tuple (q, k, v, forward_batch) --
    def test_return_is_four_tuple(self):
        m = _make_model()
        result, _ = self._call(m)
        self.assertEqual(len(result), 4)

    def test_return_contains_forward_batch(self):
        m = _make_model()
        result, _ = self._call(m)
        self.assertIsNotNone(result[3])


# ===========================================================================
# 2. forward_mha_core_npu  (source L132–142)
# ===========================================================================
class TestForwardMhaCoreNpu(unittest.TestCase):
    def setUp(self):
        _reset_module_mocks()

    def test_calls_attn_mha(self):
        m = _make_model()
        dsa_mod.forward_mha_core_npu(
            m, torch.randn(NT, NH, QK_HD), torch.randn(NT, NH, QK_HD),
            torch.randn(NT, NH, VHD), _make_forward_batch(),
        )
        self.assertTrue(m.attn_mha.called)

    def test_calls_o_proj(self):
        m = _make_model()
        dsa_mod.forward_mha_core_npu(
            m, torch.randn(NT, NH, QK_HD), torch.randn(NT, NH, QK_HD),
            torch.randn(NT, NH, VHD), _make_forward_batch(),
        )
        self.assertTrue(m.o_proj.called)

    def test_return_is_tensor(self):
        m = _make_model()
        result = dsa_mod.forward_mha_core_npu(
            m, torch.randn(NT, NH, QK_HD), torch.randn(NT, NH, QK_HD),
            torch.randn(NT, NH, VHD), _make_forward_batch(),
        )
        self.assertIsInstance(result, torch.Tensor)


# ===========================================================================
# 3. forward_mla_prepare_npu  (source L149–279)
# ===========================================================================
class TestForwardMlaPrepareNpu(unittest.TestCase):
    def setUp(self):
        dsa_mod._use_ag_after_qlora = False
        _reset_module_mocks()

    def _call(self, m, scatter_modes=None, mla_enabled=False, use_cp=False):
        positions = torch.arange(NT)
        hidden_states = torch.randn(NT, 8)
        forward_batch = _make_forward_batch()
        if scatter_modes is None:
            scatter_modes = SimpleNamespace(
                layer_input_mode=_ScatterMode.TP_ATTN_FULL,
                attn_mode=_ScatterMode.TP_ATTN_FULL,
            )
        mock_ctx = MagicMock()
        mock_ctx.fetch_qkv_latent = MagicMock(return_value=_make_qkv_latent())
        mock_pool = MagicMock()
        with (
            patch.object(dsa_mod, "get_attn_tp_context", return_value=mock_ctx),
            patch.object(dsa_mod, "get_token_to_kv_pool", return_value=mock_pool),
            patch.object(dsa_mod, "is_mla_preprocess_enabled", return_value=mla_enabled),
            patch.object(dsa_mod, "dsa_use_prefill_cp", return_value=use_cp),
        ):
            return dsa_mod.forward_mla_prepare_npu(
                m, positions, hidden_states, forward_batch, None, scatter_modes
            )

    # -- L157: is_mla_preprocess_enabled=True → mla_preprocess.forward called --
    def test_mla_preprocess_enabled_calls_forward(self):
        m = _make_model()
        m.mla_preprocess = MagicMock()
        m.mla_preprocess.forward = MagicMock(
            return_value=(
                torch.randn(NT, NH, QK_RD), torch.randn(NT, 1, QK_RD),
                torch.randn(NT, NH, KV_LR), torch.randn(NT, 1, KV_LR),
                _make_forward_batch(), None, torch.arange(NT),
            )
        )
        result = self._call(m, mla_enabled=True)
        self.assertTrue(m.mla_preprocess.forward.called)
        self.assertIsNone(result[7])  # topk_indices

    # -- L186: q_lora_rank is None → q_proj + kv_a_proj_with_mqa --
    def test_q_lora_rank_none_calls_q_proj(self):
        m = _make_model(q_lora_rank=None)
        self._call(m)
        self.assertTrue(m.q_proj.called)
        self.assertFalse(m.q_b_proj.called)

    # -- L186: q_lora_rank is not None → q_b_proj --
    def test_q_lora_rank_not_none_calls_q_b_proj(self):
        m = _make_model(q_lora_rank=QLR)
        self._call(m)
        self.assertTrue(m.q_b_proj.called)

    # -- L188-204: _use_ag_after_qlora=True + SCATTERED → scattered_to_tp_attn_full --
    def test_ag_after_qlora_calls_scattered_to_tp_attn_full(self):
        dsa_mod._use_ag_after_qlora = True
        m = _make_model(q_lora_rank=QLR)
        scatter_modes = SimpleNamespace(
            layer_input_mode=_ScatterMode.SCATTERED,
            attn_mode=_ScatterMode.TP_ATTN_FULL,
        )
        self._call(m, scatter_modes)
        self.assertTrue(dsa_mod.scattered_to_tp_attn_full.called)

    # -- L206-219: fused_split_qk_norm path (shape<65536, no cp, not disabled) --
    def test_fused_split_qk_norm_called_when_conditions_met(self):
        m = _make_model(q_lora_rank=QLR)
        self._call(m)
        self.assertTrue(dsa_mod.fused_split_qk_norm.called)

    # -- L220-232: unfused path (_disable_npu_fused_split_qk_norm=True) --
    def test_unfused_path_when_disabled(self):
        m = _make_model(q_lora_rank=QLR, _disable_npu_fused_split_qk_norm=True)
        self._call(m)
        self.assertFalse(dsa_mod.fused_split_qk_norm.called)
        self.assertTrue(m.q_a_layernorm.called)

    # -- L235: use_dsa=True → q_lora=q, indexer called --
    def test_dsa_enabled_calls_indexer(self):
        m = _make_model(q_lora_rank=QLR, use_dsa=True)
        self._call(m)
        self.assertTrue(m.indexer.called)

    # -- L235: use_dsa=False → indexer NOT called --
    def test_dsa_disabled_does_not_call_indexer(self):
        m = _make_model(q_lora_rank=QLR, use_dsa=False)
        self._call(m)
        self.assertFalse(m.indexer.called)

    # -- L252: rotary_emb is not None → rotary_emb called --
    def test_rotary_emb_called(self):
        m = _make_model(q_lora_rank=QLR)
        self._call(m)
        self.assertTrue(m.rotary_emb.called)

    # -- L255-259: dsa_use_prefill_cp=True → rebuild_cp_kv_cache called --
    def test_prefill_cp_calls_rebuild_cp_kv_cache(self):
        m = _make_model(q_lora_rank=QLR)
        self._call(m, use_cp=True)
        self.assertTrue(m.rebuild_cp_kv_cache.called)

    # -- L270: return 8-tuple --
    def test_return_is_eight_tuple(self):
        m = _make_model()
        result = self._call(m)
        self.assertEqual(len(result), 8)


# ===========================================================================
# 4. forward_mla_core_npu  (source L282–319)
# ===========================================================================
class TestForwardMlaCoreNpu(unittest.TestCase):
    def setUp(self):
        _reset_module_mocks()
        _torch_npu.npu_transpose_batchmatmul.return_value = torch.randn(NT, NH, VHD)

    def _args(self):
        return (
            _make_model(),
            torch.randn(NT, NH, QK_RD),   # q_pe
            torch.randn(NT, 1, QK_RD),    # k_pe
            torch.randn(NT, NH, KV_LR),   # q_nope_out
            torch.randn(NT, 1, KV_LR),    # k_nope
            _make_forward_batch(),
        )

    # -- L293: calls attn_mqa --
    def test_calls_attn_mqa(self):
        m, q_pe, k_pe, qno, kn, fb = self._args()
        dsa_mod.forward_mla_core_npu(m, q_pe, k_pe, qno, kn, fb, None, torch.arange(NT), None)
        self.assertTrue(m.attn_mqa.called)

    # -- L300: topk_indices is None → no topk_indices kwarg passed --
    def test_topk_indices_none_no_kwarg(self):
        m, q_pe, k_pe, qno, kn, fb = self._args()
        dsa_mod.forward_mla_core_npu(m, q_pe, k_pe, qno, kn, fb, None, torch.arange(NT), None)
        self.assertNotIn("topk_indices", m.attn_mqa.call_args.kwargs)

    # -- L300: topk_indices is not None → topk_indices kwarg passed --
    def test_topk_indices_not_none_passes_kwarg(self):
        m, q_pe, k_pe, qno, kn, fb = self._args()
        tk = torch.randn(NT, 10)
        dsa_mod.forward_mla_core_npu(m, q_pe, k_pe, qno, kn, fb, None, torch.arange(NT), tk)
        self.assertTrue(torch.equal(m.attn_mqa.call_args.kwargs.get("topk_indices"), tk))

    # -- L308: calls npu_transpose_batchmatmul --
    def test_calls_npu_transpose_batchmatmul(self):
        m, q_pe, k_pe, qno, kn, fb = self._args()
        dsa_mod.forward_mla_core_npu(m, q_pe, k_pe, qno, kn, fb, None, torch.arange(NT), None)
        self.assertTrue(_torch_npu.npu_transpose_batchmatmul.called)

    # -- L317: calls o_proj --
    def test_calls_o_proj(self):
        m, q_pe, k_pe, qno, kn, fb = self._args()
        dsa_mod.forward_mla_core_npu(m, q_pe, k_pe, qno, kn, fb, None, torch.arange(NT), None)
        self.assertTrue(m.o_proj.called)

    # -- L319: return is tensor --
    def test_return_is_tensor(self):
        m, q_pe, k_pe, qno, kn, fb = self._args()
        result = dsa_mod.forward_mla_core_npu(m, q_pe, k_pe, qno, kn, fb, None, torch.arange(NT), None)
        self.assertIsInstance(result, torch.Tensor)


# ===========================================================================
# 5. forward_dsa_prepare_npu  (source L326–461)
# ===========================================================================
class TestForwardDsaPrepareNpu(unittest.TestCase):
    def setUp(self):
        dsa_mod._use_ag_after_qlora = False
        _reset_module_mocks()

    def _call(self, m, **ov):
        positions = torch.arange(NT)
        hidden_states = torch.randn(NT, 8)
        forward_batch = ov.get(
            "forward_batch", _make_forward_batch(ov.get("mode", "extend"))
        )
        scatter_modes = ov.get("scatter_modes", SimpleNamespace(
            layer_input_mode=_ScatterMode.TP_ATTN_FULL,
            attn_mode=_ScatterMode.TP_ATTN_FULL,
        ))
        prev_topk = ov.get("prev_topk_indices", None)
        with (
            patch.object(dsa_mod, "is_mla_preprocess_enabled", return_value=False),
            patch.object(dsa_mod, "dsa_use_prefill_cp", return_value=ov.get("use_cp", False)),
        ):
            return dsa_mod.forward_dsa_prepare_npu(
                m, positions, hidden_states, forward_batch, None, scatter_modes, prev_topk
            )

    # -- L336: is_mla_preprocess_enabled + is_decode → npu_mla_preprocess called --
    @patch.object(dsa_mod, "npu_mla_preprocess")
    def test_mla_preprocess_enabled_decode_calls_npu_mla_preprocess(self, mock_prep):
        mock_prep.return_value = (
            torch.randn(NT, NH, QK_RD), torch.randn(NT, 1, QK_RD),
            torch.randn(NT, NH, KV_LR), torch.randn(NT, 1, KV_LR),
            torch.randn(NT, QLR), _make_forward_batch(), None,
            torch.arange(NT), None,
        )
        m = _make_model()
        fb = _make_forward_batch("decode")
        with patch.object(dsa_mod, "is_mla_preprocess_enabled", return_value=True):
            dsa_mod.forward_dsa_prepare_npu(
                m, torch.arange(NT), torch.randn(NT, 8), fb, None,
                SimpleNamespace(layer_input_mode=_ScatterMode.TP_ATTN_FULL,
                               attn_mode=_ScatterMode.TP_ATTN_FULL),
                None,
            )
        self.assertTrue(mock_prep.called)

    # -- L356: is_neox_style=True → q_a_layernorm called (manual split+norm) --
    def test_neox_style_calls_q_a_layernorm(self):
        m = _make_model(is_neox_style=True)
        self._call(m)
        self.assertTrue(m.q_a_layernorm.called)

    # -- L356: is_neox_style=True → q_b_proj called --
    def test_neox_style_calls_q_b_proj(self):
        m = _make_model(is_neox_style=True)
        self._call(m)
        self.assertTrue(m.q_b_proj.called)

    # -- L362-368: neox + _use_ag_after_qlora=True → scattered_to_tp_attn_full --
    def test_neox_ag_after_qlora_calls_scattered(self):
        dsa_mod._use_ag_after_qlora = True
        m = _make_model(is_neox_style=True)
        self._call(m, scatter_modes=SimpleNamespace(
            layer_input_mode=_ScatterMode.SCATTERED,
            attn_mode=_ScatterMode.TP_ATTN_FULL,
        ))
        self.assertTrue(dsa_mod.scattered_to_tp_attn_full.called)

    # -- L372-378: neox + alt_stream is not None → alt_stream ops --
    @patch.object(torch.Tensor, "record_stream")
    def test_neox_alt_stream_calls_wait_stream(self, _mock_record):
        m = _make_model(is_neox_style=True, alt_stream=MagicMock())
        self._call(m)
        self.assertTrue(m.alt_stream.wait_stream.called)

    @patch.object(torch.Tensor, "record_stream")
    def test_neox_alt_stream_calls_record_stream(self, mock_record):
        m = _make_model(is_neox_style=True, alt_stream=MagicMock())
        self._call(m)
        self.assertTrue(mock_record.called)

    def test_neox_alt_stream_none_no_alt_stream_ops(self):
        m = _make_model(is_neox_style=True, alt_stream=None)
        self._call(m)
        # alt_stream is None so no wait_stream/record_stream
        self.assertIsNone(m.alt_stream)

    # -- L390-403: non-neox + fused path → fused_split_qk_norm called --
    def test_not_neox_fused_path_calls_fused_split_qk_norm(self):
        m = _make_model(is_neox_style=False)
        self._call(m)
        self.assertTrue(dsa_mod.fused_split_qk_norm.called)

    # -- L404-418: non-neox + unfused path (_disable=True) → fused NOT called --
    def test_not_neox_unfused_path_no_fused(self):
        m = _make_model(is_neox_style=False, _disable_npu_fused_split_qk_norm=True)
        self._call(m)
        self.assertFalse(dsa_mod.fused_split_qk_norm.called)
        self.assertTrue(m.q_a_layernorm.called)

    # -- L426: layer_id=0 → sin_cos_cache updated --
    def test_layer_zero_updates_sin_cos_cache(self):
        m = _make_model(is_neox_style=True, layer_id=0)
        original = m.rotary_emb.sin_cos_cache.clone()
        self._call(m)
        self.assertFalse(torch.equal(m.rotary_emb.sin_cos_cache, original))

    # -- L426: layer_id != 0 → sin_cos_cache NOT updated --
    def test_non_zero_layer_no_sin_cos_update(self):
        m = _make_model(is_neox_style=True, layer_id=5)
        original = m.rotary_emb.sin_cos_cache.clone()
        self._call(m)
        self.assertTrue(torch.equal(m.rotary_emb.sin_cos_cache, original))

    # -- L433-437: dsa_use_prefill_cp=True → rebuild_cp_kv_cache called --
    def test_prefill_cp_calls_rebuild_cp_kv_cache(self):
        m = _make_model(is_neox_style=True)
        self._call(m, use_cp=True)
        self.assertTrue(m.rebuild_cp_kv_cache.called)

    # -- L439: skip_topk=False → indexer called --
    def test_not_skip_topk_calls_indexer(self):
        m = _make_model(skip_topk=False)
        self._call(m)
        self.assertTrue(m.indexer.called)

    # -- L439: skip_topk=True, prev not None → indexer NOT called --
    def test_skip_topk_with_prev_no_indexer(self):
        m = _make_model(skip_topk=True)
        prev = torch.tensor([1, 2, 3])
        result = self._call(m, prev_topk_indices=prev)
        self.assertFalse(m.indexer.called)
        self.assertIs(result[4], prev)

    # -- L439: skip_topk=True, is_nextn=True, prev None → indexer called --
    def test_nextn_no_prev_calls_indexer(self):
        m = _make_model(skip_topk=True, is_nextn=True)
        self._call(m, prev_topk_indices=None)
        self.assertTrue(m.indexer.called)

    # -- L452: return 8-tuple --
    def test_return_is_eight_tuple(self):
        m = _make_model()
        result = self._call(m)
        self.assertEqual(len(result), 8)


# ===========================================================================
# 6. forward_dsa_core_npu  (source L464–516)
# ===========================================================================
class TestForwardDsaCoreNpu(unittest.TestCase):
    def setUp(self):
        _reset_module_mocks()

    def _make_args(self, mode="extend"):
        m = _make_model()
        return (
            m,
            torch.randn(NT, NH, QK_RD),   # q_pe
            torch.randn(NT, 1, QK_RD),    # k_pe
            torch.randn(NT, NH, KV_LR),   # q_nope_out
            torch.randn(NT, 1, KV_LR),    # k_nope
            torch.randn(NT, 10),          # topk_indices
            _make_forward_batch(mode),    # forward_batch
        )

    # -- L493-505: extend path → torch.bmm called --
    def test_extend_path_calls_bmm(self):
        m, q_pe, k_pe, qno, kn, tk, fb = self._make_args("extend")
        with patch("torch.bmm") as mock_bmm:
            dsa_mod.forward_dsa_core_npu(m, q_pe, k_pe, qno, kn, tk, fb, None, torch.arange(NT))
        self.assertTrue(mock_bmm.called)

    # -- L506-508: non-extend path → torch.ops.npu.batch_matmul_transpose --
    def test_non_extend_path_calls_batch_matmul_transpose(self):
        m, q_pe, k_pe, qno, kn, tk, fb = self._make_args("decode")
        dsa_mod.forward_dsa_core_npu(m, q_pe, k_pe, qno, kn, tk, fb, None, torch.arange(NT))
        self.assertTrue(torch.ops.npu.batch_matmul_transpose.called)

    # -- L513: next_skip_topk=False → returns (output, None) --
    def test_not_next_skip_topk_returns_none(self):
        m = _make_model(next_skip_topk=False)
        with patch("torch.bmm"):
            result = dsa_mod.forward_dsa_core_npu(
                m, torch.randn(NT, NH, QK_RD), torch.randn(NT, 1, QK_RD),
                torch.randn(NT, NH, KV_LR), torch.randn(NT, 1, KV_LR),
                torch.randn(NT, 10), _make_forward_batch("extend"), None, torch.arange(NT),
            )
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[1])

    # -- L515: next_skip_topk=True → returns (output, topk_indices) --
    def test_next_skip_topk_returns_topk_indices(self):
        m = _make_model(next_skip_topk=True)
        tk = torch.randn(NT, 10)
        with patch("torch.bmm"):
            result = dsa_mod.forward_dsa_core_npu(
                m, torch.randn(NT, NH, QK_RD), torch.randn(NT, 1, QK_RD),
                torch.randn(NT, NH, KV_LR), torch.randn(NT, 1, KV_LR),
                tk, _make_forward_batch("extend"), None, torch.arange(NT),
            )
        self.assertEqual(len(result), 2)
        self.assertIs(result[1], tk)

    # -- L475: calls attn_mqa --
    def test_calls_attn_mqa(self):
        m, q_pe, k_pe, qno, kn, tk, fb = self._make_args("extend")
        with patch("torch.bmm"):
            dsa_mod.forward_dsa_core_npu(m, q_pe, k_pe, qno, kn, tk, fb, None, torch.arange(NT))
        self.assertTrue(m.attn_mqa.called)

    # -- L512: calls o_proj --
    def test_calls_o_proj(self):
        m, q_pe, k_pe, qno, kn, tk, fb = self._make_args("extend")
        with patch("torch.bmm"):
            dsa_mod.forward_dsa_core_npu(m, q_pe, k_pe, qno, kn, tk, fb, None, torch.arange(NT))
        self.assertTrue(m.o_proj.called)


# ===========================================================================
# 7. npu_mla_preprocess  (source L519–612)
# ===========================================================================
class TestNpuMlaPreprocess(unittest.TestCase):
    def setUp(self):
        _reset_module_mocks()

    def _make_model_with_preprocess(self, **ov):
        """Model that already has mla_preprocess (hasattr → True)."""
        m = _make_model(**ov)
        m.mla_preprocess = MagicMock()
        m.mla_preprocess.forward = MagicMock(
            return_value=(
                torch.randn(NT, NH, QK_RD),  # q_pe
                torch.randn(NT, 1, QK_RD),   # k_pe
                torch.randn(NT, NH, KV_LR),   # q_nope_out
                torch.randn(NT, 1, KV_LR),   # k_nope
                _make_forward_batch(),        # forward_batch
                None,                         # zero_allocator
                torch.arange(NT),             # positions
            )
        )
        return m

    def _call(self, m):
        with patch.object(dsa_mod, "is_mla_preprocess_enabled", return_value=True):
            return dsa_mod.npu_mla_preprocess(
                m, torch.randn(NT, 8), torch.arange(NT), _make_forward_batch("decode"), None
            )

    # -- L527: not hasattr(m, "mla_preprocess") → NPUFusedMLAPreprocess constructed --
    @patch.object(dsa_mod, "NPUFusedMLAPreprocess")
    def test_creates_mla_preprocess_when_not_exists(self, mock_cls):
        mock_inst = MagicMock()
        mock_inst.forward = MagicMock(
            return_value=(
                torch.randn(NT, NH, QK_RD), torch.randn(NT, 1, QK_RD),
                torch.randn(NT, NH, KV_LR), torch.randn(NT, 1, KV_LR),
                _make_forward_batch(), None, torch.arange(NT),
            )
        )
        mock_cls.return_value = mock_inst

        # SimpleNamespace: hasattr(m, "mla_preprocess") → False
        m = _make_plain_model()
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        self._call(m)
        self.assertTrue(mock_cls.called)

    # -- L527: hasattr(m, "mla_preprocess") → NPUFusedMLAPreprocess NOT constructed --
    @patch.object(dsa_mod, "NPUFusedMLAPreprocess")
    def test_does_not_create_when_already_exists(self, mock_cls):
        m = self._make_model_with_preprocess()
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        self._call(m)
        self.assertFalse(mock_cls.called)

    # -- L543-546: _is_mlaprolog=True (ignore contains ".*kv_b_proj") --
    #    → forward returns 8 values (with q_lora + dynamic_scale)
    #    → fused_qkv_a_proj_with_mqa NOT called (q_lora comes from forward) --
    def test_mlaprolog_true_forward_returns_eight_and_no_extra_q_lora(self):
        m = self._make_model_with_preprocess()
        m.quant_config = SimpleNamespace(ignore=["model.layers.0.kv_b_proj"])
        m.mla_preprocess.forward = MagicMock(
            return_value=(
                torch.randn(NT, NH, QK_RD), torch.randn(NT, 1, QK_RD),
                torch.randn(NT, NH, KV_LR), torch.randn(NT, 1, KV_LR),
                torch.randn(NT, QLR),  # q_lora from forward
                _make_forward_batch(), torch.arange(NT), None,  # dynamic_scale
            )
        )
        self._call(m)
        self.assertEqual(m.mla_preprocess.forward.call_count, 1)
        self.assertFalse(m.fused_qkv_a_proj_with_mqa.called)
        self.assertFalse(m.q_a_layernorm.called)

    # -- L543-546: _is_mlaprolog=False (ignore doesn't contain kv_b_proj) --
    #    → forward returns 7 values (no q_lora)
    #    → fused_qkv_a_proj_with_mqa + q_a_layernorm called to compute q_lora --
    def test_mlaprolog_false_computes_q_lora_separately(self):
        m = self._make_model_with_preprocess()
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        self._call(m)
        self.assertTrue(m.fused_qkv_a_proj_with_mqa.called)
        self.assertTrue(m.q_a_layernorm.called)

    # -- L543: quant_config has no "ignore" attr → _is_mlaprolog=False --
    def test_quant_config_without_ignore_attr_treats_as_non_mlaprolog(self):
        m = self._make_model_with_preprocess()
        m.quant_config = MagicMock(spec=[])  # no attributes
        self._call(m)
        # non-mlaprolog path → fused_qkv_a_proj_with_mqa called
        self.assertTrue(m.fused_qkv_a_proj_with_mqa.called)

    # -- L560-583: _is_mlaprolog=False, alt_stream is not None --
    #    → torch.npu.Event called (L561), current_stream().wait_event called (L565, L583) --
    def test_alt_stream_not_none_calls_npu_event(self):
        m = self._make_model_with_preprocess(alt_stream=MagicMock())
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        self._call(m)
        self.assertTrue(torch.npu.Event.called)

    def test_alt_stream_not_none_calls_current_stream_wait_event(self):
        """L565 + L583: current_stream().wait_event is called when alt_stream exists."""
        m = self._make_model_with_preprocess(alt_stream=MagicMock())
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        self._call(m)
        self.assertTrue(torch.npu.current_stream.return_value.wait_event.called)

    # -- L584-600: _is_mlaprolog=False, alt_stream is None --
    #    → torch.npu.Event NOT called, current_stream().wait_event NOT called --
    def test_alt_stream_none_no_npu_event(self):
        m = self._make_model_with_preprocess(alt_stream=None)
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        self._call(m)
        self.assertFalse(torch.npu.Event.called)

    def test_alt_stream_none_no_current_stream_wait_event(self):
        """No stream sync when alt_stream is None."""
        m = self._make_model_with_preprocess(alt_stream=None)
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        self._call(m)
        self.assertFalse(torch.npu.current_stream.return_value.wait_event.called)

    # -- L602: return 9-tuple --
    def test_return_is_nine_tuple(self):
        m = self._make_model_with_preprocess()
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        result = self._call(m)
        self.assertEqual(len(result), 9)

    # -- L602: dynamic_scale is None for non-mlaprolog --
    def test_dynamic_scale_none_for_non_mlaprolog(self):
        m = self._make_model_with_preprocess()
        m.quant_config = SimpleNamespace(ignore=["model.q_proj"])
        result = self._call(m)
        self.assertIsNone(result[8])


if __name__ == "__main__":
    unittest.main()
