"""Unit tests for sgl-project/sglang PR #31220 ("Qwen3.5: quantized attention on
modelopt_fp4 checkpoints").

Written to match the conventions of test/registered/quant/test_quant_config_parsing.py
and test/registered/unit/layers/quantization/test_modelopt_nvfp4.py: unittest +
CustomTestCase, register_cpu_ci (no GPU / no real checkpoint needed), and testing
against the REAL sglang classes (ModelOptFp4Config, RadixAttention) rather than
mocks wherever the real class is CPU-safe.

Covers the three changes on branch qwen35-modelopt-fp4-quantized-attention:

  1. Per-prefix quantized/BF16 decision (ModelOptFp4Config.is_layer_excluded) is
     honored for attention instead of being hard-overridden to "always BF16".
  2. RadixAttention registers k_scale/v_scale parameters when constructed with a
     quant_config carrying kv_cache_quant_algo="FP8" (Qwen3_5AttentionDecoderLayer
     now passes quant_config through instead of always passing None).
  3. The checkpoint's baked k_scale/v_scale tensors get remapped from their
     "...self_attn.k_proj.k_scale" / "...self_attn.v_proj.v_scale" checkpoint
     names onto the RadixAttention module's "...attn.k_scale" / "...attn.v_scale"
     parameter names, in load_weights().

Change 3 targets `QWEN3_5_KV_SCALE_MAPPER`, a module-level WeightsMapper in
sglang.srt.models.qwen3_5 (the same type the loader-facing hf_to_sglang_mapper
class attribute uses) that every Qwen3_5* load_weights() applies to the incoming
weight stream before its name-munging loop.
"""

import unittest

import torch

from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import QWEN3_5_KV_SCALE_MAPPER
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
# Change 3: baked k_scale/v_scale checkpoint names remap onto RadixAttention
# ---------------------------------------------------------------------------
class TestQwen3_5KvScaleMapper(CustomTestCase):
    """Every Qwen3_5* load_weights() strips ".self_attn" out of checkpoint names
    early, so the stock maybe_remap_kv_scale_name() can never match (its modelopt
    branch keys off ".self_attn."/".mixer." still being present, and its
    params_dict membership check fails either way because the sglang module tree
    has no self_attn level). QWEN3_5_KV_SCALE_MAPPER is therefore applied to the
    weight stream BEFORE the loop: the mapped name "...attn.k_scale" no longer
    contains "k_proj", so the stacked-params qkv_proj matching (which used to
    consume and silently drop the scale) never fires, and the name loads through
    the loop's generic fallback branch.
    """

    def test_maps_baked_kv_scale_names_onto_radix_attention(self):
        # Checkpoint-name shape is dictated by ModelOpt's export format (scales
        # baked under the HF attention projections); target-name shape by the
        # RadixAttention attribute path in the sglang module tree. Both sides
        # are external contracts -- a typo in either silently zeroes the scales.
        weights = [
            ("model.layers.3.self_attn.k_proj.k_scale", torch.tensor(0.0347)),
            ("model.layers.3.self_attn.v_proj.v_scale", torch.tensor(0.0128)),
        ]

        mapped = list(QWEN3_5_KV_SCALE_MAPPER.apply(weights))

        self.assertEqual(
            [name for name, _ in mapped],
            ["model.layers.3.attn.k_scale", "model.layers.3.attn.v_scale"],
        )
        torch.testing.assert_close(mapped[0][1], torch.tensor(0.0347))
        torch.testing.assert_close(mapped[1][1], torch.tensor(0.0128))

    def test_all_other_names_pass_through_unchanged(self):
        # Negative-branch contract: the mapper must be a strict no-op for every
        # non-KV-scale weight -- including the self_attn projections themselves
        # (still needed by the qkv_proj stacked matching), the GDN linear-attn
        # projections, and per-projection quant scales like input_scale /
        # weight_scale that modelopt also emits under self_attn. A key that is
        # too broad (or a None mapping, which WeightsMapper.apply DROPS from the
        # stream) would corrupt regular weight loading.
        names = [
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.k_proj.input_scale",
            "model.layers.3.self_attn.k_proj.weight_scale",
            "model.layers.2.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.mlp.experts.5.down_proj.weight",
            "lm_head.weight",
        ]
        weights = [(name, torch.zeros(1)) for name in names]

        mapped = list(QWEN3_5_KV_SCALE_MAPPER.apply(weights))

        self.assertEqual([name for name, _ in mapped], names)

    def test_mapped_scale_loads_via_default_weight_loader(self):
        # After mapping, load_weights' generic fallback branch does
        # `getattr(param, "weight_loader", default_weight_loader)`; RadixAttention
        # scale params carry no weight_loader (BaseKVCacheMethod.create_weights),
        # so default_weight_loader must land the value. Its scalar path
        # (numel()==1 -> fill_) is what tolerates the 0-dim param vs shape-[1]
        # checkpoint tensor -- a future shape-strictness change there would
        # silently break exactly this load.
        scale_param = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        loaded_weight = torch.tensor([0.0347], dtype=torch.float32)

        weight_loader = getattr(scale_param, "weight_loader", default_weight_loader)
        weight_loader(scale_param, loaded_weight)

        self.assertAlmostEqual(scale_param.item(), 0.0347, places=6)


if __name__ == "__main__":
    unittest.main()
