"""Unit tests for W8A8-INT8 DSpark draft fixes.

Covers the two changes in the current git diff:

1. deepseek_v4_dspark.py – _remap_dspark_weight_name:
   Add the PPU ``weight_scale_inv → weight_scale`` conversion that was
   missing (mirrors the target model's remap_weight_name_to_dpsk_hf_format).
   Without this, W8A8-INT8 weight scales are never loaded on PPU because the
   W8A8Int8LinearMethod parameter name is ``weight_scale``, not
   ``weight_scale_inv``.

2. deepseek_v4_dspark.py & dspark.py – lm_head quantized computation:
   When the shared target lm_head is quantized (e.g. W8A8-INT8), use
   ``quant_method.apply`` instead of a raw ``torch.matmul`` on the int8
   weight tensor.

All tests run on CPU only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Import deepseek_v4 before deepseek_v4_dspark to avoid circular-import
# errors in the MoE triton_utils module (same order as the existing
# test_deepseek_v4_dspark_ppu_q_padding.py).
import sglang.srt.models.deepseek_v4  # noqa: F401  (import side-effect)
import sglang.srt.models.deepseek_v4_dspark as dspark_mod
import sglang.srt.models.dspark as dspark_dense
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeQuantConfig:
    """Minimal quant_config stub with get_name() and is_fp4_experts."""

    def __init__(self, name: str, is_fp4_experts: bool = False):
        self._name = name
        self.is_fp4_experts = is_fp4_experts

    def get_name(self) -> str:
        return self._name


def _make_dspark_model(quant_config):
    """Create a lightweight object that has the attributes
    _remap_dspark_weight_name accesses on self."""
    obj = SimpleNamespace()
    obj.quant_config = quant_config
    obj.confidence_head = None
    return obj


# ---------------------------------------------------------------------------
# Test 1: _remap_dspark_weight_name PPU scale conversion
# ---------------------------------------------------------------------------


class TestRemapDsparkWeightNamePPUScale(unittest.TestCase):
    """Verify that weight-scale names are correctly converted on PPU."""

    # --- attention layer scales ----------------------------------------

    def test_attn_scale_on_ppu_w8a8_int8_becomes_weight_scale(self):
        """On PPU + W8A8-INT8, .scale → .weight_scale (not weight_scale_inv)."""
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        raw = "mtp.0.attn.wq_a.scale"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(mapped, "stages.0.self_attn.wq_a.weight_scale")

    def test_attn_scale_off_ppu_stays_weight_scale_inv(self):
        """Off PPU, .scale → .weight_scale_inv (unchanged by this fix)."""
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        raw = "mtp.0.attn.wq_a.scale"
        with patch.object(dspark_mod, "_is_ppu", False):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(mapped, "stages.0.self_attn.wq_a.weight_scale_inv")

    def test_attn_weight_not_affected(self):
        """Non-scale attention weights (int8 weight) must be unchanged."""
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        raw = "mtp.0.attn.wq_a.weight"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(mapped, "stages.0.self_attn.wq_a.weight")

    # --- multiple attention sub-layers ---------------------------------

    def test_all_attn_sublayer_scales_converted_on_ppu(self):
        """wq_a, wq_b, wkv, wo_b scales must all be converted on PPU."""
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        sublayers = ["wq_a", "wq_b", "wkv", "wo_b"]
        with patch.object(dspark_mod, "_is_ppu", True):
            for sub in sublayers:
                raw = f"mtp.0.attn.{sub}.scale"
                mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
                    model, raw
                )
                self.assertEqual(
                    mapped,
                    f"stages.0.self_attn.{sub}.weight_scale",
                    f"{sub} scale not converted to weight_scale on PPU",
                )

    # --- MoE expert scales --------------------------------------------

    def test_moe_expert_scale_on_ppu_w8a8_int8_becomes_weight_scale(self):
        """MoE expert w1 scale on PPU + W8A8-INT8 → weight_scale."""
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        raw = "mtp.0.ffn.experts.0.w1.scale"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(
            mapped,
            "stages.0.mlp.experts.0.gate_proj.weight_scale",
        )

    def test_moe_expert_scale_off_ppu_stays_weight_scale_inv(self):
        """Off PPU, MoE expert scale stays as weight_scale_inv."""
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        raw = "mtp.0.ffn.experts.0.w1.scale"
        with patch.object(dspark_mod, "_is_ppu", False):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(
            mapped,
            "stages.0.mlp.experts.0.gate_proj.weight_scale_inv",
        )

    # --- FP8 quant_config handling -------------------------------------

    def test_fp8_non_fp4_experts_on_ppu_stays_weight_scale_inv(self):
        """FP8 (non-FP4) on PPU: scales stay as weight_scale_inv."""
        model = _make_dspark_model(_FakeQuantConfig("fp8", is_fp4_experts=False))
        raw = "mtp.0.attn.wq_a.scale"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(mapped, "stages.0.self_attn.wq_a.weight_scale_inv")

    def test_fp8_fp4_experts_moe_scale_on_ppu_becomes_weight_scale(self):
        """FP8 + FP4 experts on PPU: MoE expert scale → weight_scale."""
        model = _make_dspark_model(_FakeQuantConfig("fp8", is_fp4_experts=True))
        raw = "mtp.0.ffn.experts.0.w1.scale"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(
            mapped,
            "stages.0.mlp.experts.0.gate_proj.weight_scale",
        )

    def test_fp8_fp4_experts_attn_scale_on_ppu_stays_inv(self):
        """FP8 + FP4 experts on PPU: non-MoE scale stays weight_scale_inv."""
        model = _make_dspark_model(_FakeQuantConfig("fp8", is_fp4_experts=True))
        raw = "mtp.0.attn.wq_a.scale"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(mapped, "stages.0.self_attn.wq_a.weight_scale_inv")

    # --- None quant_config --------------------------------------------

    def test_none_quant_config_on_ppu_converts_to_weight_scale(self):
        """When quant_config is None on PPU, the else branch is taken
        (quant_config is falsy), so weight_scale_inv -> weight_scale.
        This mirrors the target model's remap behavior."""
        model = _make_dspark_model(None)
        raw = "mtp.0.attn.wq_a.scale"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(mapped, "stages.0.self_attn.wq_a.weight_scale")

    # --- other weight name patterns -----------------------------------

    def test_norm_weights_not_affected(self):
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        for raw, expected_suffix in [
            ("mtp.0.attn_norm.weight", "input_layernorm.weight"),
            ("mtp.0.ffn_norm.weight", "post_attention_layernorm.weight"),
        ]:
            with patch.object(dspark_mod, "_is_ppu", True):
                mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
                    model, raw
                )
            self.assertEqual(mapped, f"stages.0.{expected_suffix}")

    def test_markov_head_passthrough(self):
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        raw = "mtp.0.markov_head.weight"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertEqual(mapped, "markov_head.weight")

    def test_skipped_prefixes_return_none(self):
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        for raw in ["embed.weight", "head.weight", "lm_head.weight"]:
            with patch.object(dspark_mod, "_is_ppu", True):
                mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
                    model, raw
                )
            self.assertIsNone(mapped)

    def test_non_mtp_prefix_returns_none(self):
        model = _make_dspark_model(_FakeQuantConfig("w8a8_int8"))
        raw = "layers.0.attn.wq_a.weight"
        with patch.object(dspark_mod, "_is_ppu", True):
            mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, raw)
        self.assertIsNone(mapped)


# ---------------------------------------------------------------------------
# Test 2: lm_head quantized computation (deepseek_v4_dspark.py)
# ---------------------------------------------------------------------------


class _QuantMethodStub:
    """A quant_method stub whose apply() records the call and returns logits.

    Class name is intentionally not in _UNQUANTIZED_LM_HEAD_METHODS so
    should_apply_lm_head_quant_method returns True.
    """

    def __init__(self):
        self.apply_called = False

    def apply(self, layer, x):
        self.apply_called = True
        return torch.randn(x.shape[0], 4, dtype=torch.float32)


# Create a class with the exact name 'UnquantizedLinearMethod' so that
# should_apply_lm_head_quant_method recognises it and returns False.
_UnquantizedLinearMethodStub = type(
    "UnquantizedLinearMethod",
    (object,),
    {
        "apply": lambda self, layer, x: (_ for _ in ()).throw(
            AssertionError("should not call apply on unquantized method")
        )
    },
)


def _make_quantized_lm_head():
    """A lm_head stub that should_apply_lm_head_quant_method returns True for."""
    lm_head = MagicMock()
    lm_head.weight = torch.zeros(4, 8, dtype=torch.int8)
    lm_head.org_vocab_size = 4
    qm = _QuantMethodStub()
    lm_head.quant_method = qm
    return lm_head, qm


def _make_unquantized_lm_head():
    """A lm_head stub that should_apply_lm_head_quant_method returns False for."""
    lm_head = MagicMock()
    lm_head.weight = torch.randn(4, 8, dtype=torch.float32)
    lm_head.org_vocab_size = 4
    qm = _UnquantizedLinearMethodStub()
    lm_head.quant_method = qm
    return lm_head, qm


class TestDsparkLogitsFromQuantizedLmHead(unittest.TestCase):
    """DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc must use
    quant_method.apply when the shared lm_head is quantized."""

    def _make_model(self, lm_head):
        last_stage = SimpleNamespace(norm=lambda x: x)
        model = SimpleNamespace(
            lm_head=lm_head,
            stages=[None, last_stage],
            _use_fp32_lm_head=False,
            _opt_markov_w2_tp_shard=True,  # skip gather_and_crop_vocab
        )
        return model

    def test_quantized_lm_head_uses_quant_method_apply(self):
        lm_head, qm = _make_quantized_lm_head()
        model = self._make_model(lm_head)
        x = torch.randn(2, 8, dtype=torch.float32)
        logits = DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc(model, x)
        self.assertTrue(qm.apply_called, "quant_method.apply was not called")
        self.assertEqual(logits.shape, (2, 4))

    def test_unquantized_lm_head_uses_matmul(self):
        lm_head, qm = _make_unquantized_lm_head()
        model = self._make_model(lm_head)
        x = torch.randn(2, 8, dtype=torch.float32)
        # With _opt_markov_w2_tp_shard=True, gather_and_crop_vocab is skipped,
        # so the method returns local_logits directly from the matmul path.
        logits = DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc(model, x)
        self.assertFalse(
            getattr(qm, "apply_called", False),
            "should not call apply on unquantized method",
        )
        self.assertEqual(logits.shape, (2, 4))

    def test_none_quant_method_uses_matmul(self):
        lm_head = MagicMock()
        lm_head.weight = torch.randn(4, 8, dtype=torch.float32)
        lm_head.org_vocab_size = 4
        lm_head.quant_method = None
        model = self._make_model(lm_head)
        x = torch.randn(2, 8, dtype=torch.float32)
        logits = DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc(model, x)
        self.assertEqual(logits.shape, (2, 4))


# ---------------------------------------------------------------------------
# Test 3: lm_head quantized computation (dspark.py DSparkDraftMixin)
# ---------------------------------------------------------------------------


class TestDsparkDraftMixinQuantizedLmHead(unittest.TestCase):
    """DSparkDraftMixin.compute_base_logits must use quant_method.apply
    when the shared lm_head is quantized."""

    def _make_mixin(self, lm_head):
        mixin = SimpleNamespace(lm_head=lm_head)
        return mixin

    def test_quantized_lm_head_uses_quant_method_apply(self):
        lm_head, qm = _make_quantized_lm_head()
        mixin = self._make_mixin(lm_head)
        hidden = torch.randn(2, 8, dtype=torch.float32)
        with patch.object(dspark_dense, "gather_and_crop_vocab", lambda x, lm: x):
            logits, _ = DSparkDraftMixin.compute_base_logits(mixin, hidden)
        self.assertTrue(qm.apply_called, "quant_method.apply was not called")
        self.assertEqual(logits.shape, (2, 4))

    def test_unquantized_lm_head_uses_matmul(self):
        lm_head, qm = _make_unquantized_lm_head()
        mixin = self._make_mixin(lm_head)
        hidden = torch.randn(2, 8, dtype=torch.float32)
        with patch.object(dspark_dense, "gather_and_crop_vocab", lambda x, lm: x):
            logits, _ = DSparkDraftMixin.compute_base_logits(mixin, hidden)
        self.assertFalse(
            getattr(qm, "apply_called", False),
            "should not call apply on unquantized method",
        )
        self.assertEqual(logits.shape, (2, 4))


# ---------------------------------------------------------------------------
# Test 4: should_apply_lm_head_quant_method itself
# ---------------------------------------------------------------------------


class TestShouldApplyLmHeadQuantMethod(unittest.TestCase):
    """Sanity-check the gate function used by both model files."""

    def test_none_quant_method_returns_false(self):
        lm_head = MagicMock()
        self.assertFalse(should_apply_lm_head_quant_method(lm_head, None))

    def test_unquantized_method_returns_false(self):
        lm_head = MagicMock()
        lm_head.weight = torch.randn(4, 8)
        qm = _UnquantizedLinearMethodStub()
        self.assertFalse(should_apply_lm_head_quant_method(lm_head, qm))

    def test_quantized_method_returns_true(self):
        lm_head = MagicMock()
        lm_head.weight = torch.zeros(4, 8, dtype=torch.int8)
        qm = _QuantMethodStub()
        self.assertTrue(should_apply_lm_head_quant_method(lm_head, qm))


if __name__ == "__main__":
    unittest.main()
