import copy
import unittest
from types import SimpleNamespace

import torch
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.ling_v3_dspark import (
    LingV3ForCausalLMDSparkMLA,
    _prepare_mla_weights,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    apply_namespaced_dspark_draft_config,
    checkpoint_bundles_dspark_draft,
    is_namespaced_dspark_bundle,
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def bundle():
    return SimpleNamespace(
        architectures=["BailingMoeV3ForCausalLM"],
        model_type="bailing_hybrid",
        use_kda=True,
        num_hidden_layers=72,
        num_nextn_predict_layers=0,
        dspark_bundle_format="namespaced_safetensors",
        dspark_bundle_version=1,
        dspark_config={
            "architectures": ["LingV3ForCausalLMDSparkMLA"],
            "attention_type": "mla",
            "num_hidden_layers": 3,
            "block_size": 5,
            "target_layer_ids": [69, 70, 71],
        },
    )


class TestLingDSparkBundle(unittest.TestCase):
    def test_bundle_detection_and_projection(self):
        target = bundle()
        draft = copy.deepcopy(target)
        self.assertTrue(checkpoint_bundles_dspark_draft(target))
        self.assertTrue(apply_namespaced_dspark_draft_config(draft))
        self.assertEqual(draft.architectures, ["LingV3ForCausalLMDSparkMLA"])
        self.assertEqual(draft.num_hidden_layers, 3)
        self.assertEqual(draft.full_attention_type, "mla")
        self.assertEqual(draft.attention_type, 1)
        self.assertFalse(draft.use_kda)
        self.assertTrue(target.use_kda)
        self.assertEqual(target.num_hidden_layers, 72)
        self.assertEqual(target.architectures, ["BailingMoeV3ForCausalLM"])

    def test_model_config_does_not_remap_bundle_to_nextn(self):
        config = bundle()
        runner = SimpleNamespace(
            is_draft_model=True,
            speculative_algorithm="DSPARK",
            hf_config=config,
            hf_text_config=config,
        )
        ModelConfig._config_draft_model(runner)
        self.assertEqual(config.architectures, ["LingV3ForCausalLMDSparkMLA"])
        self.assertEqual(config.num_hidden_layers, 3)

    def test_target_config_is_not_modified(self):
        config = bundle()
        runner = SimpleNamespace(
            is_draft_model=False,
            speculative_algorithm="DSPARK",
            hf_config=config,
            hf_text_config=config,
        )
        ModelConfig._config_draft_model(runner)
        self.assertEqual(config.architectures, ["BailingMoeV3ForCausalLM"])
        self.assertEqual(config.num_hidden_layers, 72)

    def test_other_namespaced_draft_family_is_not_projected(self):
        config = bundle()
        config.model_type = "deepseek_v4"
        config.dspark_config["architectures"] = ["DeepseekV4ForCausalLMDSpark"]
        self.assertFalse(is_namespaced_dspark_bundle(config))
        self.assertFalse(apply_namespaced_dspark_draft_config(config))
        self.assertEqual(config.num_hidden_layers, 72)

    def test_unsupported_version_is_not_projected(self):
        config = bundle()
        config.dspark_bundle_version = 2
        self.assertFalse(is_namespaced_dspark_bundle(config))
        self.assertFalse(apply_namespaced_dspark_draft_config(config))
        self.assertEqual(config.num_hidden_layers, 72)

    def test_invalid_architecture_fails_before_mutation(self):
        config = bundle()
        config.dspark_config["architectures"] = []
        with self.assertRaises(ValueError):
            apply_namespaced_dspark_draft_config(config)
        self.assertEqual(config.num_hidden_layers, 72)

    def test_parser_uses_nested_gamma_and_mla_without_mutation(self):
        config = bundle()
        parsed = parse_dspark_draft_config(draft_hf_config=config)
        self.assertEqual(parsed.gamma, 5)
        self.assertEqual(parsed.num_hidden_layers, 3)
        self.assertTrue(parsed.is_mla)
        self.assertEqual(parsed.target_layer_ids, [69, 70, 71])
        self.assertEqual(config.num_hidden_layers, 72)

    def test_native_mla_draft_does_not_get_a_linear_backend(self):
        config = bundle()
        apply_namespaced_dspark_draft_config(config)
        self.assertIsNone(mambaish_config(SimpleNamespace(hf_config=config)))

    def test_unsupported_bundle_fields_fail_before_mutation(self):
        for field, value in (
            ("architectures", ["BailingMoeV3ForCausalLM"]),
            ("architectures", [123]),
            ("attention_type", "gqa"),
        ):
            with self.subTest(field=field, value=value):
                config = bundle()
                config.dspark_config[field] = value
                with self.assertRaises(ValueError):
                    apply_namespaced_dspark_draft_config(config)
                self.assertEqual(config.num_hidden_layers, 72)
                self.assertTrue(config.use_kda)

    def test_shared_mla_projection_views_and_reentry(self):
        weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
        attention = SimpleNamespace(
            kv_b_proj=SimpleNamespace(weight=weight),
            qk_nope_head_dim=2,
            v_head_dim=2,
            w_kc=None,
            w_vc=None,
        )
        layers = [SimpleNamespace(attention=attention)]
        _prepare_mla_weights(layers)
        expected_k, expected_v = weight.reshape(2, 4, 4).split(2, dim=1)
        torch.testing.assert_close(attention.w_kc, expected_k)
        torch.testing.assert_close(attention.w_vc, expected_v.transpose(1, 2))
        pointers = attention.w_kc.data_ptr(), attention.w_vc.data_ptr()
        _prepare_mla_weights(layers)
        self.assertEqual(
            pointers, (attention.w_kc.data_ptr(), attention.w_vc.data_ptr())
        )

    def test_weight_loader_maps_bf16_and_int4_expert_scales(self):
        for dtype in (torch.bfloat16, torch.int32):
            with self.subTest(dtype=dtype):
                params = {
                    "layers.0.attention.fused_qkv_a_proj_with_mqa.weight": torch.nn.Parameter(
                        torch.empty(3, 2), requires_grad=False
                    ),
                    "layers.0.mlp.experts.w13_weight": torch.nn.Parameter(
                        torch.empty(1, dtype=dtype), requires_grad=False
                    ),
                    "norm.weight": torch.nn.Parameter(
                        torch.empty(2), requires_grad=False
                    ),
                }
                calls = []

                def expert_loader(
                    param, value, name, *, shard_id, expert_id, calls=calls
                ):
                    calls.append((name, shard_id, expert_id, value.dtype))

                params["layers.0.mlp.experts.w13_weight"].weight_loader = expert_loader
                weights = [
                    ("dspark.layers.0.attention.q_a_proj.weight", torch.ones(2, 2)),
                    (
                        "dspark.layers.0.attention.kv_a_proj_with_mqa.weight",
                        torch.full((1, 2), 2.0),
                    ),
                    (
                        "dspark.layers.0.mlp.experts.0.gate_proj.weight",
                        torch.ones(1, dtype=dtype),
                    ),
                    (
                        "dspark.layers.0.mlp.experts.0.up_proj.weight",
                        torch.ones(1, dtype=dtype),
                    ),
                    ("dspark.final_layernorm.weight", torch.ones(2)),
                ]
                if dtype == torch.int32:
                    params["layers.0.mlp.experts.w13_weight_scale"] = (
                        torch.nn.Parameter(torch.empty(1), requires_grad=False)
                    )
                    params[
                        "layers.0.mlp.experts.w13_weight_scale"
                    ].weight_loader = expert_loader
                    weights.append(
                        (
                            "dspark.layers.0.mlp.experts.0.gate_proj.weight_scale",
                            torch.ones(1),
                        )
                    )
                model = SimpleNamespace(
                    config=SimpleNamespace(num_experts=1, q_lora_rank=2),
                    is_bundled=True,
                    confidence_head=None,
                    layers=[],
                    named_parameters=lambda params=params: params.items(),
                )
                loaded = LingV3ForCausalLMDSparkMLA.load_weights(model, iter(weights))
                self.assertEqual(loaded, set(params))
                torch.testing.assert_close(
                    params["layers.0.attention.fused_qkv_a_proj_with_mqa.weight"],
                    torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]),
                )
                self.assertEqual([x[1] for x in calls[:2]], ["w1", "w3"])
                self.assertTrue(all(x[2] == 0 for x in calls))
                if dtype == torch.int32:
                    self.assertTrue(calls[-1][0].endswith("w13_weight_scale"))

    def test_weight_loader_rejects_missing_or_unexpected_tensors(self):
        for weights in ([], [("dspark.unknown.weight", torch.ones(1))]):
            model = SimpleNamespace(
                config=SimpleNamespace(num_experts=1, q_lora_rank=None),
                is_bundled=True,
                confidence_head=None,
                layers=[],
                named_parameters=lambda: [
                    (
                        "norm.weight",
                        torch.nn.Parameter(torch.ones(1), requires_grad=False),
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                LingV3ForCausalLMDSparkMLA.load_weights(model, weights)

    def test_legacy_bundle_still_detected(self):
        config = SimpleNamespace(dspark_block_size=5)
        self.assertTrue(checkpoint_bundles_dspark_draft(config))
        self.assertFalse(apply_namespaced_dspark_draft_config(config))


if __name__ == "__main__":
    unittest.main()
