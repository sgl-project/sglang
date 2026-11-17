"""Loading compressed-tensors pack-quantized (WNA16) MLA checkpoints.

Regression tests for #49369: GLM-4.7-Flash W4A16 (cyankiwi/GLM-4.7-Flash-AWQ-4bit)
quantizes q_a_proj, kv_a_proj_with_mqa and kv_b_proj from layer 1 on and keeps
layer 0 and the shared experts in BF16. Kept apart from the per-format DeepSeek
loader tests because the cases span the GLM and DeepSeek loaders, the WNA16
scheme and the compressed-tensors config.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from sglang.kernels.ops.gemm.fused_a_gemm import fused_a_gemm_weight_eligible
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.utils import fuse_qkv_a_proj_tensors
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
from sglang.test.test_utils import CustomTestCase

# The quantization_config of cyankiwi/GLM-4.7-Flash-AWQ-4bit, with its ignore
# list cut to the entries these tests match against.
_GLM47_FLASH_W4A16 = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "dynamic": False,
                "actorder": None,
                "observer": "mse",
            },
            "input_activations": None,
            "output_activations": None,
        }
    },
    "ignore": [
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.shared_experts.gate_proj",
        "model.layers.1.mlp.shared_experts.up_proj",
        "model.layers.1.mlp.shared_experts.down_proj",
        "lm_head",
    ],
}

_QKV_A = "self_attn.fused_qkv_a_proj_with_mqa"


def _packed_layer(out_features, in_features, num_bits=4, group_size=32, seed=0):
    """A layer built by the WNA16 scheme and loaded with random checkpoint
    tensors (packed by compressed-tensors itself), plus the dense weight they
    encode."""
    scheme = CompressedTensorsWNA16(
        strategy="channel" if group_size == -1 else "group",
        num_bits=num_bits,
        group_size=None if group_size == -1 else group_size,
    )
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_size=out_features,
        input_size=in_features,
        output_partition_sizes=[out_features],
        input_size_per_partition=in_features,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    layer.scheme = scheme

    gen = torch.Generator().manual_seed(seed)
    half = 1 << (num_bits - 1)
    codes = torch.randint(-half, half, (out_features, in_features), generator=gen)
    groups = layer.weight_scale.shape[1]
    scale = (torch.rand(out_features, groups, generator=gen) + 0.5).to(torch.bfloat16)
    layer.weight_packed.data.copy_(pack_to_int32(codes.to(torch.int8), num_bits))
    layer.weight_scale.data.copy_(scale)
    expected = (
        codes.float() * scale.float().repeat_interleave(in_features // groups, dim=1)
    ).to(torch.bfloat16)
    return layer, expected


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


def _qkv_a_checkpoint(layer_id):
    """q_a_proj / kv_a_proj_with_mqa tensors as GLM-4.7-Flash W4A16 stores them
    (q_lora_rank 768, kv_lora_rank 512 + rope 64, hidden 2048, group 32)."""
    tensors = {}
    for proj, out in (("q_a_proj", 768), ("kv_a_proj_with_mqa", 576)):
        prefix = f"model.layers.{layer_id}.self_attn.{proj}"
        tensors[f"{prefix}.weight_packed"] = torch.full(
            (out, 256), 7, dtype=torch.int32
        )
        tensors[f"{prefix}.weight_scale"] = torch.ones((out, 64), dtype=torch.bfloat16)
        tensors[f"{prefix}.weight_shape"] = torch.tensor([out, 2048])
    return tensors


class _FakeGlm(Glm4MoeLiteForCausalLM):
    def __init__(self, params, quant_config=None):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            q_lora_rank=768,
            n_routed_experts=0,
            num_hidden_layers=2,
            num_nextn_predict_layers=0,
        )
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self._fake_params = params

    def named_parameters(self, *args, **kwargs):
        return iter(self._fake_params.items())

    def post_load_weights(self, is_nextn=False, weight_names=None):
        pass


class _FakeDeepseekLoader(DeepseekV2WeightLoaderMixin):
    def __init__(self, params):
        self.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=768,
            num_hidden_layers=2,
            num_nextn_predict_layers=0,
        )
        self.model = SimpleNamespace()
        self.quant_config = SimpleNamespace(get_name=lambda: "compressed_tensors")
        self.num_fused_shared_experts = 0
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())

    def _maybe_quant_weights_to_fp8_ue8m0(
        self, weights, attn_quant_modules, nextn_conf
    ):
        return weights

    def post_load_weights(self, is_nextn=False, weight_names=None):
        pass


class TestFusedQkvALoad(CustomTestCase):
    def _params(self, suffixes=("weight_packed", "weight_scale", "weight_shape")):
        return {f"model.layers.1.{_QKV_A}.{s}": _FakeParam() for s in suffixes}

    def _assert_fused(self, params):
        prefix = f"model.layers.1.{_QKV_A}"
        self.assertEqual(params[f"{prefix}.weight_shape"].loaded.tolist(), [1344, 2048])
        self.assertEqual(
            tuple(params[f"{prefix}.weight_packed"].loaded.shape), (1344, 256)
        )
        self.assertEqual(
            tuple(params[f"{prefix}.weight_scale"].loaded.shape), (1344, 64)
        )

    def test_glm_loads_the_fused_weight_shape(self):
        """GLM concatenated the two weight_shape tensors ([out, in] each) into a
        4-element tensor, so loading the fused layer failed its size assert."""
        params = self._params()
        _FakeGlm(params).load_weights(list(_qkv_a_checkpoint(1).items()))
        self._assert_fused(params)

    def test_deepseek_loads_the_fused_weight_shape(self):
        """The DeepSeek loader fuses the pair the same way and hit the same
        weight_shape size assert."""
        params = self._params()
        _FakeDeepseekLoader(params).do_load_weights(list(_qkv_a_checkpoint(1).items()))
        self._assert_fused(params)

    def test_glm_stacks_awq_qweight_on_its_output_dim(self):
        """AWQ stores qweight as [in, out // 8], so the outputs stack on dim 1;
        GLM always used dim 0 and failed on the mismatched widths."""
        params = self._params(("qweight",))
        awq = SimpleNamespace(get_name=lambda: "awq_marlin")
        checkpoint = [
            (f"model.layers.1.self_attn.{proj}.qweight", torch.zeros(2048, cols))
            for proj, cols in (("q_a_proj", 96), ("kv_a_proj_with_mqa", 72))
        ]
        _FakeGlm(params, quant_config=awq).load_weights(checkpoint)
        self.assertEqual(
            tuple(params[f"model.layers.1.{_QKV_A}.qweight"].loaded.shape),
            (2048, 168),
        )

    def test_gptq_packed_tensors_stack_on_their_output_dim(self):
        """GPTQ also stores qweight / qzeros / scales as [in-ish, out], but only
        AWQ formats were recognized, so a GPTQ MLA checkpoint stacked on dim 0.
        GGUF's qweight is row-major and keeps dim 0."""
        q, kv = torch.zeros(256, 768), torch.zeros(256, 576)
        for fmt, dim in (
            ("gptq", 1),
            ("gptq_marlin", 1),
            ("auto-round", 1),
            ("gguf", 0),
        ):
            with self.subTest(fmt=fmt):
                cfg = SimpleNamespace(get_name=lambda fmt=fmt: fmt)
                fused = (
                    fuse_qkv_a_proj_tensors(
                        "x.qweight", q_a=q, kv_a=kv, quant_config=cfg
                    )
                    if dim == 1
                    else fuse_qkv_a_proj_tensors(
                        "x.qweight", q_a=q.T, kv_a=kv.T, quant_config=cfg
                    )
                )
                self.assertEqual(
                    tuple(fused.shape), (256, 1344) if dim == 1 else (1344, 256)
                )

    def test_awq_checkpoints_keep_unconverted_attention_on_dim_0(self):
        """QuantTrio/GLM-4.7-Flash-AWQ leaves self_attn in BF16 under an awq
        config; choosing the dim from the config alone stacked those [out, in]
        weights on dim 1 and broke a checkpoint that loaded on main."""
        params = self._params(("weight",))
        awq = SimpleNamespace(get_name=lambda: "awq_marlin")
        checkpoint = [
            (f"model.layers.1.self_attn.{proj}.weight", torch.zeros(rows, 2048))
            for proj, rows in (("q_a_proj", 768), ("kv_a_proj_with_mqa", 576))
        ]
        _FakeGlm(params, quant_config=awq).load_weights(checkpoint)
        self.assertEqual(
            tuple(params[f"model.layers.1.{_QKV_A}.weight"].loaded.shape),
            (1344, 2048),
        )

    def test_weight_shape_and_act_order_pairs_must_share_the_input(self):
        """Both projections read the same input, so a fused layer has one input
        size and one activation order; a mismatch cannot be fused."""
        with self.assertRaises(AssertionError):
            fuse_qkv_a_proj_tensors(
                "x.weight_shape",
                q_a=torch.tensor([768, 2048]),
                kv_a=torch.tensor([576, 1024]),
            )
        g_idx = torch.arange(64) // 32
        for name in ("x.weight_g_idx", "x.g_idx"):  # compressed-tensors, GPTQ
            with self.subTest(name=name):
                self.assertIs(
                    fuse_qkv_a_proj_tensors(name, q_a=g_idx, kv_a=g_idx.clone()),
                    g_idx,
                )
                with self.assertRaises(ValueError):
                    fuse_qkv_a_proj_tensors(name, q_a=g_idx, kv_a=g_idx.flip(0))


class TestWNA16Dequantize(CustomTestCase):
    def test_matches_the_compressed_tensors_packing(self):
        """The unpack inverts compressed-tensors' pack_to_int32 (natural nibble
        order, +2^(bits-1) offset) exactly, for group and channel scales."""
        for num_bits, group_size in ((4, 32), (4, -1), (8, 32)):
            with self.subTest(num_bits=num_bits, group_size=group_size):
                layer, expected = _packed_layer(
                    24, 64, num_bits=num_bits, group_size=group_size
                )
                weight = layer.scheme.dequantize_weight(layer)
                self.assertEqual(weight.dtype, torch.bfloat16)
                torch.testing.assert_close(weight, expected, rtol=0, atol=0)

    def test_the_kernel_repack_makes_the_weight_unreadable(self):
        """Sharded-state and remote-instance loaders copy kernel-layout tensors,
        then call post_load_weights; dequantizing those would silently build
        garbage absorbed weights, so the repack must mark the layer."""
        layer, _ = _packed_layer(24, 64)

        class _StopAfterMark(Exception):
            pass

        with mock.patch.object(
            compressed_tensors_wNa16,
            "check_marlin_supports_shape",
            side_effect=_StopAfterMark,
        ):
            with self.assertRaises(_StopAfterMark):
                layer.scheme.process_weights_after_loading(layer)
        with self.assertRaises(NotImplementedError):
            layer.scheme.dequantize_weight(layer)

    def test_a_packed_layer_skips_the_bf16_fused_a_gemm(self):
        """The fused-A GEMM eligibility check read layer.weight, which a packed
        W4A16 fused_qkv_a_proj_with_mqa does not have, and crashed the forward."""
        layer, _ = _packed_layer(24, 64)
        layer.bias = None
        self.assertFalse(fused_a_gemm_weight_eligible(layer))


class TestPostLoadPackedKvBProj(CustomTestCase):
    def _post_load(self, nope, v):
        heads, kv_lora_rank = 2, 64
        kv_b_proj, w = _packed_layer(heads * (nope + v), kv_lora_rank)
        kv_b_proj.quant_method = CompressedTensorsLinearMethod(
            CompressedTensorsConfig.from_config(_GLM47_FLASH_W4A16)
        )
        self_attn = SimpleNamespace(
            kv_b_proj=kv_b_proj,
            qk_nope_head_dim=nope,
            v_head_dim=v,
            w_kc=None,
            w_vc=None,
            w_scale=1.0,
        )
        loader = DeepseekV2WeightLoaderMixin.__new__(DeepseekV2WeightLoaderMixin)
        loader.config = SimpleNamespace(num_hidden_layers=1, architectures=["X"])
        loader.quant_config = None
        loader.model = SimpleNamespace(
            start_layer=0,
            end_layer=1,
            layers=[SimpleNamespace(self_attn=self_attn)],
        )

        loader.post_load_weights()
        return self_attn, w

    def test_absorbed_weights_come_from_the_dequantized_kv_b_proj(self):
        """post_load_weights read kv_b_proj.weight, which a packed W4A16 layer
        does not have (the issue's AttributeError)."""
        nope, v = 4, 8
        self_attn, w = self._post_load(nope, v)

        w_kc, w_vc = w.unflatten(0, (-1, nope + v)).split([nope, v], dim=1)
        torch.testing.assert_close(self_attn.w_kc, w_kc, rtol=0, atol=0)
        torch.testing.assert_close(self_attn.w_vc, w_vc.transpose(1, 2), rtol=0, atol=0)
        self.assertEqual(self_attn.w_scale, 1.0)

    def test_online_weight_updates_are_refused_before_writing(self):
        """An online update ends in post_load_weights, which cannot dequantize
        the repacked kv_b_proj again; it raised after the new weights were
        already written. The update must be refused before it starts."""
        self_attn, _ = self._post_load(4, 8)
        model = torch.nn.ModuleDict({"kv_b_proj": self_attn.kv_b_proj})
        error = _unsupported_derived_weight_cache_error(model)
        self.assertIsNotNone(error)
        self.assertIn("compressed-tensors WNA16", error)


class TestGlm47FlashW4A16Config(CustomTestCase):
    def setUp(self):
        self.config = CompressedTensorsConfig.from_config(_GLM47_FLASH_W4A16)

    def test_layer0_fused_layers_follow_their_ignored_shards(self):
        """GLM declared no packed_modules_mapping, so the ignore list's shard
        names never matched the fused layers and BF16 layer 0 (and the shared
        experts' gate_up_proj) was built as W4A16."""
        mapping = Glm4MoeLiteForCausalLM.packed_modules_mapping
        ignore = self.config.ignore
        for name in (
            f"model.layers.0.{_QKV_A}",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.1.mlp.shared_experts.gate_up_proj",
        ):
            self.assertTrue(should_ignore_layer(name, ignore, mapping), name)
        for name in (f"model.layers.1.{_QKV_A}", "model.layers.1.self_attn.kv_b_proj"):
            self.assertFalse(should_ignore_layer(name, ignore, mapping), name)

    def test_fp8_checkpoints_keep_an_ignored_layers_fused_attention_bf16(self):
        """unsloth/GLM-4.7-Flash-FP8-Dynamic ignores layer 1's attention. With no
        mapping its fused_qkv_a_proj_with_mqa was built FP8, the BF16 weights
        were cast in and weight_scale never written: outputs reached 3.4e38."""
        ignore = [
            f"model.layers.1.self_attn.{proj}"
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj")
        ]
        mapping = Glm4MoeLiteForCausalLM.packed_modules_mapping
        self.assertTrue(
            should_ignore_layer(f"model.layers.1.{_QKV_A}", ignore, mapping)
        )
        self.assertFalse(
            should_ignore_layer(f"model.layers.2.{_QKV_A}", ignore, mapping)
        )

    def test_only_a_precision_mismatch_vetoes_shared_expert_fusion(self):
        """GLM fused the BF16 shared expert into the W4A16 routed-expert buffers
        on SM80+. Fusion must stop exactly when any projection of the shared
        expert has a different precision from the same projection of that
        layer's routed experts, in either direction."""
        shared = [
            f"model.layers.1.mlp.shared_experts.{p}"
            for p in ("gate_proj", "up_proj", "down_proj")
        ]
        routed = [
            f"model.layers.1.mlp.experts.0.{p}"
            for p in ("gate_proj", "up_proj", "down_proj")
        ]
        for ignore, can_fuse in (
            (_GLM47_FLASH_W4A16["ignore"], False),
            (["lm_head", "re:.*mlp.shared_expert_gate$"], True),
            (["lm_head", *shared, *routed], True),
            (["lm_head", "re:.*shared_experts.*"], False),
            (["lm_head", "re:.*shared_expert(s|_gate).*"], False),
            # Matches the shared experts without naming "shared_expert".
            (["lm_head", r"re:.*mlp\.shared.*"], False),
            # Anchored to layers that no explicit entry names.
            (["lm_head", r"re:model\.layers\.(3|5)\.mlp\.shared_experts\..*"], False),
            (["lm_head", "shared_experts.gate_proj"], False),
            # Only one shared projection kept BF16 (its down weight was skipped).
            (["lm_head", "model.layers.1.mlp.shared_experts.down_proj"], False),
            (["lm_head", "model.layers.1.mlp.shared_experts.up_proj"], False),
            (["lm_head", "shared_experts.down_proj"], False),
            # The same projection kept BF16 on both sides still fuses.
            (["lm_head", shared[2], routed[2]], True),
            # Routed experts kept BF16 beside quantized shared experts.
            (["lm_head", *routed], False),
            (["lm_head", r"re:.*mlp\.experts\..*"], False),
            (["lm_head", r"re:.*mlp\..*"], True),
        ):
            with self.subTest(ignore=ignore):
                config = CompressedTensorsConfig.from_config(
                    dict(_GLM47_FLASH_W4A16, ignore=ignore)
                )
                self.assertEqual(config.can_fuse_shared_expert(), can_fuse)


if __name__ == "__main__":
    unittest.main()
