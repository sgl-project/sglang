"""Unit tests for sgl-project/sglang PR #31220 ("Qwen3.5: quantized attention on
modelopt_fp4 checkpoints").

Written to match the conventions of test/registered/quant/test_quant_config_parsing.py
and test/registered/unit/layers/quantization/test_modelopt_nvfp4.py: unittest +
CustomTestCase, register_cpu_ci (no GPU / no real checkpoint needed), and testing
against the REAL sglang classes (ModelOptFp4Config, RadixAttention) rather than
mocks wherever the real class is CPU-safe.

Covers the three changes in commits d6ee507be7 and 718a48a433 on branch
qwen35-modelopt-fp4-quantized-attention:

  1. Per-prefix quantized/BF16 decision (ModelOptFp4Config.is_layer_excluded) is
     honored for attention instead of being hard-overridden to "always BF16".
  2. RadixAttention registers k_scale/v_scale parameters when constructed with a
     quant_config carrying kv_cache_quant_algo="FP8" (Qwen3_5AttentionDecoderLayer
     now passes quant_config through instead of always passing None).
  3. The checkpoint's baked k_scale/v_scale tensors get remapped from their
     "...k_proj.k_scale" / "...v_proj.v_scale" checkpoint names onto the
     RadixAttention module's "...attn.k_scale" / "...attn.v_scale" parameter names
     and loaded into them, in load_weights().

Change 3 targets `remap_and_load_qwen3_5_kv_scale(name, loaded_weight, params_dict)`,
a module-level helper in sglang.srt.models.qwen3_5 (mirroring the stock
`maybe_remap_kv_scale_name(name, params_dict)` signature/style) that was extracted
from two byte-identical ~26-line inline blocks previously copy-pasted between
Qwen3_5MoeForCausalLM.load_weights and Qwen3_5MoeForConditionalGeneration.load_weights.
Pure dedup + testability extraction; no behavior change.
"""

import unittest

import torch

from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.models.qwen3_5 import remap_and_load_qwen3_5_kv_scale
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Change 1: is_layer_excluded() per-prefix decision (quant vs BF16 attention)
# ---------------------------------------------------------------------------
class TestModelOptFp4AttentionExclusion(CustomTestCase):
    """Reproduces the two checkpoint shapes this PR must tell apart:

    - NVIDIA's released Qwen3.5-397B-A17B-NVFP4 (MoE-only): attention modules are
      listed in exclude_modules, so they must stay BF16 (UnquantizedLinearMethod).
      This is the existing, must-not-regress behavior.
    - A uniform-W4A4 checkpoint (e.g. the PR's Ornith-1.0-35B verification build):
      attention is NOT excluded, so it must be quantized like everything else.
      Before this PR, qwen3_5.py force-overrode this to BF16 regardless of
      exclude_modules, which is the bug being fixed.
    """

    def test_moe_only_checkpoint_excludes_attention(self):
        # Representative of NVIDIA's Qwen3.5 NVFP4 hf_quant_config.json shape:
        # attention (self_attn) and lm_head are excluded, MoE experts are not.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=["*self_attn*", "lm_head"],
        )

        self.assertTrue(
            cfg.is_layer_excluded("model.layers.0.self_attn.qkv_proj"),
            "MoE-only checkpoint: attention must stay excluded (BF16) -- this is "
            "the pre-existing, must-not-regress case.",
        )
        self.assertTrue(cfg.is_layer_excluded("lm_head"))
        self.assertFalse(
            cfg.is_layer_excluded("model.layers.0.mlp.experts.3.gate_up_proj"),
            "MoE experts are not excluded and must be quantized.",
        )

    def test_uniform_w4a4_checkpoint_quantizes_attention(self):
        # Representative of a uniform W4A4 checkpoint (e.g. Ornith-1.0-35B-NVFP4):
        # only lm_head is excluded; attention is deliberately quantized too.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=["lm_head"],
        )

        self.assertFalse(
            cfg.is_layer_excluded("model.layers.0.self_attn.qkv_proj"),
            "Uniform W4A4 checkpoint: attention is NOT excluded and must be "
            "quantized -- this is the bug this PR fixes (previously "
            "hard-overridden to BF16 for any modelopt_fp4 checkpoint).",
        )
        self.assertFalse(
            cfg.is_layer_excluded("model.layers.0.linear_attn.in_proj_qkvz"),
            "Same for the Gated-DeltaNet linear-attention path.",
        )
        self.assertTrue(cfg.is_layer_excluded("lm_head"))


# ---------------------------------------------------------------------------
# Change 2: RadixAttention registers k_scale/v_scale when given quant_config
# ---------------------------------------------------------------------------
class TestRadixAttentionKvScaleRegistration(CustomTestCase):
    """RadixAttention only gets k_scale/v_scale parameters (a place for baked FP8
    KV scales to load into) if it is constructed WITH a quant_config whose
    kv_cache_quant_algo is set -- see ModelOptQuantConfig._get_quant_method's
    `elif self.kv_cache_quant_algo and isinstance(layer, RadixAttention)` branch
    and BaseKVCacheMethod.create_weights. Before this PR,
    Qwen3_5AttentionDecoderLayer always constructed RadixAttention with
    quant_config=None, so this branch never fired and baked KV scales had nowhere
    to load into (silently defaulted to 1.0 at process_weights_after_loading).

    Pure CPU test: RadixAttention.__init__ and create_weights() do no CUDA work.
    """

    def _make_attn(self, quant_config):
        return RadixAttention(
            num_heads=2,
            head_dim=8,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=0,
            quant_config=quant_config,
            prefix="model.layers.0.attn",
        )

    def test_with_fp8_kv_quant_config_registers_scale_params(self):
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=[],
        )
        attn = self._make_attn(cfg)

        self.assertIsInstance(attn.k_scale, torch.nn.Parameter)
        self.assertIsInstance(attn.v_scale, torch.nn.Parameter)
        # BaseKVCacheMethod.create_weights seeds -1.0 (invalid sentinel) so a
        # checkpoint that never emits k_scale/v_scale falls back to 1.0 later
        # in process_weights_after_loading -- distinct from "never registered".
        self.assertEqual(attn.k_scale.item(), -1.0)
        self.assertEqual(attn.v_scale.item(), -1.0)

    def test_without_quant_config_has_no_scale_params(self):
        attn = self._make_attn(None)

        self.assertIsNone(attn.k_scale)
        self.assertIsNone(attn.v_scale)

    def test_quant_config_without_kv_cache_algo_has_no_scale_params(self):
        # A quant_config that quantizes attention linears but does NOT declare a
        # KV-cache quant algo (kv_cache_quant_algo=None) must not register scale
        # params either -- confirms the branch is gated on kv_cache_quant_algo,
        # not merely on quant_config being non-None.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            group_size=16,
            exclude_modules=[],
        )
        attn = self._make_attn(cfg)

        self.assertIsNone(attn.k_scale)
        self.assertIsNone(attn.v_scale)


# ---------------------------------------------------------------------------
# Change 3: baked k_scale/v_scale checkpoint tensors remap onto RadixAttention
# ---------------------------------------------------------------------------
class TestQwen3_5KvScaleRemap(CustomTestCase):
    """load_weights() strips ".self_attn" from parameter names BEFORE the stock
    maybe_remap_kv_scale_name() would see them, so that helper can never match
    (it keys off ".self_attn."/".mixer." still being present). This PR adds an
    explicit remap step right after the strip. These tests target the extracted
    `remap_and_load_qwen3_5_kv_scale` helper (see the report's proposed diff)
    rather than re-deriving the two inline blocks that used to live in
    Qwen3_5MoeForCausalLM.load_weights / Qwen3_5MoeForConditionalGeneration.load_weights.
    """

    def _make_param_dict_with_attn_scale(self, key: str, weight_loader=None):
        scale = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        if weight_loader is not None:
            scale.weight_loader = weight_loader
        return {key: scale}, scale

    def test_remaps_k_proj_scale_to_attn_and_loads_value(self):
        # Name as it arrives in load_weights AFTER ".self_attn" has already been
        # stripped a few lines above the remap block -- see qwen3_5.py L1523-24 /
        # L2015-16 (`if ".self_attn." in name: name = name.replace(".self_attn", "")`).
        name = "model.layers.3.k_proj.k_scale"
        params_dict, scale_param = self._make_param_dict_with_attn_scale(
            "model.layers.3.attn.k_scale"
        )
        loaded_weight = torch.tensor(0.0347, dtype=torch.float32)

        remapped_name = remap_and_load_qwen3_5_kv_scale(
            name, loaded_weight, params_dict
        )

        self.assertEqual(remapped_name, "model.layers.3.attn.k_scale")
        torch.testing.assert_close(
            scale_param.data, torch.tensor(0.0347, dtype=torch.float32)
        )

    def test_remaps_v_proj_scale_to_attn_and_loads_value(self):
        name = "model.layers.3.v_proj.v_scale"
        params_dict, scale_param = self._make_param_dict_with_attn_scale(
            "model.layers.3.attn.v_scale"
        )
        loaded_weight = torch.tensor(0.0128, dtype=torch.float32)

        remapped_name = remap_and_load_qwen3_5_kv_scale(
            name, loaded_weight, params_dict
        )

        self.assertEqual(remapped_name, "model.layers.3.attn.v_scale")
        torch.testing.assert_close(
            scale_param.data, torch.tensor(0.0128, dtype=torch.float32)
        )

    def test_uses_weight_loader_when_target_param_has_one(self):
        # RadixAttention's k_scale/v_scale are plain nn.Parameters with no
        # weight_loader (BaseKVCacheMethod.create_weights sets none), so the
        # .data.copy_ fallback is the common path today -- but other call sites
        # (or a future refactor) may attach a weight_loader, so the loader path
        # must be preferred when present, matching the non-stacked fallback
        # convention used elsewhere in load_weights.
        calls = []

        def spy_loader(param, weight):
            calls.append(weight.item())
            param.data.copy_(weight)

        name = "model.layers.7.k_proj.k_scale"
        params_dict, scale_param = self._make_param_dict_with_attn_scale(
            "model.layers.7.attn.k_scale", weight_loader=spy_loader
        )
        loaded_weight = torch.tensor(0.021, dtype=torch.float32)

        remap_and_load_qwen3_5_kv_scale(name, loaded_weight, params_dict)

        # exactly one call, value compared with float32-precision tolerance
        # (torch.tensor(0.021, float32).item() == 0.020999999716...)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0], 0.021, places=6)

    def test_no_op_when_target_attn_param_absent(self):
        # NVIDIA's MoE-only NVFP4 checkpoint never emits k_scale/v_scale for
        # attention at all (this code path never even runs for it in practice,
        # since load_weights only enters the remap branch when
        # name.endswith(".k_scale"/".v_scale")). This test instead covers the
        # defensive case: a checkpoint DOES emit the scale key, but the running
        # model has no matching RadixAttention param under that prefix (e.g. a
        # pipeline-parallel rank that does not own this layer) -- must not raise,
        # must signal "not consumed" so the caller's generic fallback / warning
        # path can still see it.
        name = "model.layers.99.k_proj.k_scale"
        params_dict = {}  # no "model.layers.99.attn.k_scale" present
        loaded_weight = torch.tensor(0.05, dtype=torch.float32)

        remapped_name = remap_and_load_qwen3_5_kv_scale(
            name, loaded_weight, params_dict
        )

        self.assertIsNone(remapped_name)

    def test_non_kv_scale_name_is_left_untouched(self):
        # Sanity: the helper is only ever invoked from inside the
        # `name.endswith(".k_scale") or name.endswith(".v_scale")` guard in
        # load_weights, but assert its own behavior is a safe no-op if that
        # invariant is ever violated (e.g. future refactor calls it more broadly).
        name = "model.layers.0.mlp.experts.5.down_proj.weight"
        params_dict = {}
        loaded_weight = torch.zeros(4)

        remapped_name = remap_and_load_qwen3_5_kv_scale(
            name, loaded_weight, params_dict
        )

        self.assertIsNone(remapped_name)


if __name__ == "__main__":
    unittest.main()
