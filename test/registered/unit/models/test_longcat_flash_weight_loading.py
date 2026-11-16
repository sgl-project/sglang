"""LongCat target/MTP loading and NPU block-FP8 derived weights."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models import longcat_flash, longcat_flash_nextn  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLongcatFlashNpuWeightLoading(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = self.enterContext(ExitStack())
        # Exercise NPU post-load logic with CPU tensors; no accelerator kernels.
        for module in (longcat_flash, longcat_flash_nextn):
            for name, value in (
                ("_is_npu", True),
                ("_is_cuda", False),
                ("_is_hip", False),
                ("_is_fp8_fnuz", False),
            ):
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(
                patch.object(
                    module,
                    "should_deepgemm_weight_requant_ue8m0",
                    return_value=False,
                )
            )

    def _make_model(self, *, num_heads=2, num_layers=1, fp8=True):
        generator = torch.Generator().manual_seed(17)
        layers = []
        for _ in range(num_layers):
            attentions = []
            for _ in range(2):
                weight = torch.randn(num_heads * 256, 512, generator=generator)
                proj = SimpleNamespace(
                    weight=torch.nn.Parameter(
                        weight.to(torch.float8_e4m3fn if fp8 else torch.bfloat16),
                        requires_grad=False,
                    )
                )
                if fp8:
                    # Real FP32 dequantization scales, deliberately not UE8M0.
                    proj.weight_scale_inv = torch.nn.Parameter(
                        torch.rand(num_heads * 2, 4, generator=generator) * 0.003
                        + 0.000201634,
                        requires_grad=False,
                    )
                attentions.append(
                    SimpleNamespace(
                        kv_b_proj=proj,
                        qk_nope_head_dim=128,
                        v_head_dim=128,
                        w_kc=None,
                        w_vc=None,
                        w_scale=None,
                    )
                )
            layers.append(SimpleNamespace(self_attn=attentions))
        return SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=num_layers,
                mla_scale_q_lora=False,
                mla_scale_kv_lora=False,
            ),
            quant_config=SimpleNamespace(weight_block_size=[128, 128]) if fp8 else None,
            model=SimpleNamespace(layers=layers),
        )

    def _assert_derived_weights(self, attention):
        proj = attention.kv_b_proj
        reference = proj.weight.float()
        if hasattr(proj, "weight_scale_inv"):
            # Index the scale for each element independently of block expansion.
            rows = torch.arange(reference.shape[0]) // 128
            cols = torch.arange(reference.shape[1]) // 128
            reference = reference * proj.weight_scale_inv[rows[:, None], cols[None, :]]
        reference = reference.to(torch.bfloat16).reshape(-1, 256, 512)
        torch.testing.assert_close(attention.w_kc, reference[:, :128], rtol=0, atol=0)
        torch.testing.assert_close(
            attention.w_vc, reference[:, 128:].transpose(1, 2), rtol=0, atol=0
        )
        self.assertEqual(attention.w_kc.dtype, torch.bfloat16)
        self.assertEqual(attention.w_vc.dtype, torch.bfloat16)
        self.assertTrue(attention.w_kc.transpose(1, 2).is_contiguous())
        self.assertTrue(attention.w_vc.transpose(1, 2).is_contiguous())
        self.assertIsNone(attention.w_scale)

    def test_block_fp8_dequantizes_derived_weights_without_changing_checkpoint(self):
        for tp_size in (1, 8, 32):
            with self.subTest(tp_size=tp_size):
                model = self._make_model(num_heads=64 // tp_size)
                originals = [
                    (
                        a.kv_b_proj.weight,
                        a.kv_b_proj.weight.view(torch.uint8).clone(),
                        a.kv_b_proj.weight_scale_inv,
                        a.kv_b_proj.weight_scale_inv.clone(),
                    )
                    for a in model.model.layers[0].self_attn
                ]
                longcat_flash.LongcatFlashForCausalLM.post_load_weights(model)
                for attention, (weight, payload, scale, scales) in zip(
                    model.model.layers[0].self_attn, originals
                ):
                    self._assert_derived_weights(attention)
                    self.assertIs(attention.kv_b_proj.weight, weight)
                    self.assertIs(attention.kv_b_proj.weight_scale_inv, scale)
                    torch.testing.assert_close(weight.view(torch.uint8), payload)
                    torch.testing.assert_close(scale, scales, rtol=0, atol=0)

    def test_partial_reload_refreshes_bf16_weights_in_place(self):
        model = self._make_model(num_layers=2)
        longcat_flash.LongcatFlashForCausalLM.post_load_weights(model)
        original_buffers = [
            (a.w_kc, a.w_vc, a.w_kc.clone(), a.w_vc.clone())
            for layer in model.model.layers
            for a in layer.self_attn
        ]
        with torch.no_grad():
            for attention in model.model.layers[1].self_attn:
                attention.kv_b_proj.weight_scale_inv.mul_(1.3)
        longcat_flash.LongcatFlashForCausalLM.post_load_weights(
            model,
            weight_names=["model.layers.1.self_attn.0.kv_b_proj.weight_scale_inv"],
        )
        for layer_id, layer in enumerate(model.model.layers):
            for index, attention in enumerate(layer.self_attn):
                w_kc, w_vc, old_kc, old_vc = original_buffers[layer_id * 2 + index]
                self.assertIs(attention.w_kc, w_kc)
                self.assertIs(attention.w_vc, w_vc)
                self._assert_derived_weights(attention)
                if layer_id == 0:
                    torch.testing.assert_close(attention.w_kc, old_kc, rtol=0, atol=0)
                    torch.testing.assert_close(attention.w_vc, old_vc, rtol=0, atol=0)
                else:
                    self.assertFalse(torch.equal(attention.w_kc, old_kc))
                    self.assertFalse(torch.equal(attention.w_vc, old_vc))

    def test_unquantized_bf16_still_uses_original_weights(self):
        model = self._make_model(fp8=False)
        longcat_flash.LongcatFlashForCausalLM.post_load_weights(model)
        for attention in model.model.layers[0].self_attn:
            self._assert_derived_weights(attention)

    def test_mtp_derived_weights_and_reload(self):
        for tp_size in (1, 8, 32):
            for fp8 in (True, False):
                with self.subTest(tp_size=tp_size, fp8=fp8):
                    target = self._make_model(num_heads=64 // tp_size, fp8=fp8)
                    attention = target.model.layers[0].self_attn[0]
                    model = SimpleNamespace(
                        config=target.config,
                        quant_config=target.quant_config,
                        model=SimpleNamespace(
                            decoder=SimpleNamespace(self_attn=attention)
                        ),
                    )
                    projection = attention.kv_b_proj
                    payload = projection.weight.view(torch.uint8).clone()
                    method = longcat_flash_nextn.LongcatFlashForCausalLMNextN.post_load_weights
                    method(model)
                    self._assert_derived_weights(attention)
                    w_kc, w_vc = attention.w_kc, attention.w_vc
                    if fp8:
                        projection.weight_scale_inv.data.mul_(1.3)
                    method(model)
                    self._assert_derived_weights(attention)
                    self.assertIs(attention.w_kc, w_kc)
                    self.assertIs(attention.w_vc, w_vc)
                    torch.testing.assert_close(
                        projection.weight.view(torch.uint8), payload
                    )

    def test_target_reload_filters_checkpoint_layers_by_resolved_depth(self):
        params = {
            "model.embed_tokens.weight": torch.nn.Parameter(torch.zeros(1)),
            "lm_head.weight": torch.nn.Parameter(torch.zeros(1)),
            **{
                f"model.layers.{i}.input_layernorm.0.weight": torch.nn.Parameter(
                    torch.zeros(1)
                )
                for i in range(12)
            },
        }
        for param in params.values():
            param.weight_loader = longcat_flash.default_weight_loader
        post_names = []
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_layers=38, num_hidden_layers=12, n_routed_experts=1, q_lora_rank=2
            ),
            quant_config=None,
            use_ngram_embedding=False,
            named_parameters=lambda: params.items(),
            post_load_weights=lambda weight_names: post_names.extend(weight_names),
        )
        extra_names = [
            "model.layers.12.mlps.0.gate_proj.weight",
            "model.layers.37.mlp.experts.0.gate_proj.weight",
            "model.layers.12.self_attn.0.q_a_proj.weight",
            "model.layers.12.self_attn.0.kv_a_proj_with_mqa.weight",
            "model.layers.37.self_attn.0.kv_b_proj.weight_scale_inv",
            "model.mtp.layers.0.eh_proj.weight",
            *[f"model.layers.{i}.input_layernorm.0.weight" for i in range(12, 38)],
        ]
        for value in (1.0, 2.0):
            post_names.clear()
            weights = [
                (name, torch.full((1,), value)) for name in (*extra_names, *params)
            ]
            longcat_flash.LongcatFlashForCausalLM.load_weights(model, weights)
            self.assertEqual(set(post_names), set(params))
            for param in params.values():
                torch.testing.assert_close(param, torch.full((1,), value))

    def _make_mtp_indexer_loader(self, *, fp8=True):
        prefix = "model.decoder.self_attn.indexer."
        params = {
            prefix + "weights_proj.weight": torch.nn.Parameter(
                torch.zeros(32, 256), requires_grad=False
            ),
            prefix + "wk.weight": torch.nn.Parameter(
                torch.zeros(128, 256, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            prefix + "wk.weight_scale_inv": torch.nn.Parameter(
                torch.zeros(1, 2), requires_grad=False
            ),
            prefix + "wq_b.weight": torch.nn.Parameter(
                torch.zeros(128, 128, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            prefix + "wq_b.weight_scale_inv": torch.nn.Parameter(
                torch.zeros(1, 1), requires_grad=False
            ),
        }
        for param in params.values():
            param.weight_loader = longcat_flash.default_weight_loader
        return SimpleNamespace(
            config=SimpleNamespace(q_lora_rank=None),
            quant_config=SimpleNamespace(weight_block_size=[128, 128]) if fp8 else None,
            named_parameters=lambda: params.items(),
            post_load_weights=lambda: None,
        ), params

    def test_target_full_depth_loads_last_checkpoint_layer(self):
        name = "model.layers.37.input_layernorm.0.weight"
        param = torch.nn.Parameter(torch.zeros(1))
        param.weight_loader = longcat_flash.default_weight_loader
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=38, n_routed_experts=1),
            quant_config=None,
            use_ngram_embedding=False,
            named_parameters=lambda: [(name, param)],
            post_load_weights=lambda weight_names: None,
        )
        longcat_flash.LongcatFlashForCausalLM.load_weights(
            model, [(name, torch.ones(1))]
        )
        torch.testing.assert_close(param, torch.ones(1))

    def test_mtp_fp8_fused_qkv_loads_payload_and_tail_block_scales(self):
        source = "model.mtp.layers.0.self_attn."
        destination = "model.decoder.self_attn.fused_qkv_a_proj_with_mqa."
        weights = {
            source + "q_a_proj.weight_scale_inv": torch.tensor([[0.0031]]),
            source + "kv_a_proj_with_mqa.weight_scale_inv": torch.tensor([[0.0017]]),
            source + "q_a_proj.weight": torch.ones(128, 128).to(torch.float8_e4m3fn),
            source + "kv_a_proj_with_mqa.weight": torch.full((64, 128), 2.0).to(
                torch.float8_e4m3fn
            ),
        }
        params = {
            destination + "weight": torch.nn.Parameter(
                torch.zeros(192, 128, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            destination + "weight_scale_inv": torch.nn.Parameter(
                torch.zeros(2, 1), requires_grad=False
            ),
        }
        for param in params.values():
            param.weight_loader = longcat_flash.default_weight_loader
        model = SimpleNamespace(
            config=SimpleNamespace(q_lora_rank=128),
            quant_config=SimpleNamespace(get_name=lambda: "fp8"),
            named_parameters=lambda: params.items(),
            post_load_weights=lambda: None,
        )
        longcat_flash_nextn.LongcatFlashForCausalLMNextN.load_weights(
            model, weights.items()
        )
        expected = torch.cat(
            (
                weights[source + "q_a_proj.weight"],
                weights[source + "kv_a_proj_with_mqa.weight"],
            )
        )
        torch.testing.assert_close(
            params[destination + "weight"].view(torch.uint8), expected.view(torch.uint8)
        )
        torch.testing.assert_close(
            params[destination + "weight_scale_inv"],
            torch.tensor([[0.0031], [0.0017]]),
            rtol=0,
            atol=0,
        )

    def test_mtp_fp8_indexer_loads_scales_and_dequantizes_weights_proj(self):
        generator = torch.Generator().manual_seed(37)
        source = "model.mtp.layers.0.self_attn.indexer."
        destination = "model.decoder.self_attn.indexer."
        weights = {
            source + "wq_b.weight": torch.randn(128, 128, generator=generator).to(
                torch.float8_e4m3fn
            ),
            source + "wq_b.weight_scale_inv": torch.tensor([[0.0031]]),
            source + "wk.weight": torch.randn(128, 256, generator=generator).to(
                torch.float8_e4m3fn
            ),
            source + "wk.weight_scale_inv": torch.tensor([[0.0021, 0.0053]]),
            source + "weights_proj.weight": torch.randn(
                32, 256, generator=generator
            ).to(torch.float8_e4m3fn),
            source + "weights_proj.weight_scale_inv": torch.tensor(
                [[0.000201634, 0.0037]]
            ),
        }
        for reverse in (False, True):
            with self.subTest(scale_first=reverse):
                model, params = self._make_mtp_indexer_loader()
                checkpoint = list(weights.items())
                if reverse:
                    checkpoint.reverse()
                checkpoint.append(
                    ("model.layers.37.self_attn.0.indexer.wk.weight", torch.zeros(1))
                )
                longcat_flash_nextn.LongcatFlashForCausalLMNextN.load_weights(
                    model, checkpoint
                )
                for name, value in weights.items():
                    suffix = name.removeprefix(source)
                    if suffix.startswith("weights_proj."):
                        continue
                    actual = params[destination + suffix]
                    if value.dtype == torch.float8_e4m3fn:
                        actual, value = (
                            actual.view(torch.uint8),
                            value.view(torch.uint8),
                        )
                    torch.testing.assert_close(actual, value, rtol=0, atol=0)
                weight = weights[source + "weights_proj.weight"].float()
                scales = weights[source + "weights_proj.weight_scale_inv"][0]
                expected = torch.cat(
                    (weight[:, :128] * scales[0], weight[:, 128:] * scales[1]), dim=1
                )
                torch.testing.assert_close(
                    params[destination + "weights_proj.weight"],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_mtp_fp32_indexer_weights_proj_loads_without_scale(self):
        model, params = self._make_mtp_indexer_loader(fp8=False)
        weight = torch.randn(32, 256)
        longcat_flash_nextn.LongcatFlashForCausalLMNextN.load_weights(
            model,
            [("model.mtp.layers.0.self_attn.indexer.weights_proj.weight", weight)],
        )
        torch.testing.assert_close(
            params["model.decoder.self_attn.indexer.weights_proj.weight"],
            weight,
            rtol=0,
            atol=0,
        )

    def test_mtp_fp8_indexer_rejects_incomplete_weight_scale_pair(self):
        for suffix, value in (
            ("weight", torch.ones(32, 256).to(torch.float8_e4m3fn)),
            ("weight_scale_inv", torch.ones(1, 2)),
        ):
            with self.subTest(suffix=suffix):
                model, _ = self._make_mtp_indexer_loader()
                with self.assertRaisesRegex(ValueError, "Incomplete FP8 MTP indexer"):
                    longcat_flash_nextn.LongcatFlashForCausalLMNextN.load_weights(
                        model,
                        [
                            (
                                "model.mtp.layers.0.self_attn.indexer.weights_proj."
                                + suffix,
                                value,
                            )
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
