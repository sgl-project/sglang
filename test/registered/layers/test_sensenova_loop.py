import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.sensenova_loop import (
    SenseNovaLoopForConditionalGeneration,
    SenseNovaLoopModel,
    SenseNovaMHCRouter,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestSenseNovaMHCRouter(CustomTestCase):
    def test_visits_share_parameters_and_own_cache_identities(self):
        for linear in (False, True):
            with self.subTest(linear=linear):
                physical = torch.nn.Module()
                physical.layer_id = 4
                physical.mlp = torch.nn.Linear(16, 16)
                physical.mhc_attn = SenseNovaMHCRouter(
                    hidden_size=16,
                    settings=dict(streams=2, iterations=20, fixed_identity=False),
                )
                if linear:
                    physical.linear_attn = torch.nn.Module()
                    physical.linear_attn.layer_id = 4
                    physical.linear_attn.proj = torch.nn.Linear(16, 16)
                    physical.linear_attn.attn = RadixLinearAttention(
                        layer_id=4,
                        num_q_heads=1,
                        num_k_heads=1,
                        num_v_heads=1,
                        head_q_dim=16,
                        head_k_dim=16,
                        head_v_dim=16,
                    )
                else:
                    physical.attn = RadixAttention(
                        num_heads=1,
                        head_dim=16,
                        scaling=0.25,
                        num_kv_heads=1,
                        layer_id=4,
                    )
                visit = SenseNovaLoopModel._new_visit(
                    physical, execution_layer_id=28, is_linear_attention=linear
                )
                source_params = dict(physical.named_parameters())
                for name, parameter in visit.named_parameters():
                    self.assertIs(parameter, source_params[name])
                source = physical.linear_attn if linear else physical
                target = visit.linear_attn if linear else visit
                self.assertIsNot(source.attn, target.attn)
                self.assertEqual(source.attn.layer_id, 4)
                self.assertEqual(target.attn.layer_id, 28)
                self.assertEqual(physical.layer_id, 4)
                self.assertEqual(visit.layer_id, 28)

    def test_uniform_routing_has_the_expected_residual_update(self):
        for streams in (2, 4):
            for fixed in (False, True):
                for scale in (0.5, 1.0):
                    with self.subTest(streams=streams, fixed=fixed, scale=scale):
                        settings = dict(
                            streams=streams,
                            iterations=20,
                            fixed_identity=fixed,
                            pre_scale=scale,
                        )
                        router = SenseNovaMHCRouter(
                            hidden_size=16, settings=settings
                        ).cuda()
                        with torch.no_grad():
                            router.weight.zero_()
                            router.bias.zero_()
                        x = torch.arange(
                            1, streams + 1, device="cuda", dtype=torch.bfloat16
                        )
                        x = x.view(1, streams, 1).expand(3, streams, 16).contiguous()
                        branch, mix, post, norm_fused = router(x.flatten(-2))
                        expected_branch = torch.full_like(
                            branch, streams * (streams + 1) / 4 * scale
                        )
                        torch.testing.assert_close(
                            branch, expected_branch, rtol=0, atol=0
                        )
                        expected = x if fixed else torch.full_like(x, (streams + 1) / 2)
                        expected = expected + expected_branch.unsqueeze(-2)
                        output = router.merge(branch, x.flatten(-2), mix, post)
                        torch.testing.assert_close(
                            output, expected.flatten(-2), rtol=0, atol=0
                        )
                        self.assertFalse(norm_fused)

    def test_sinkhorn_is_doubly_stochastic(self):
        torch.manual_seed(42)
        router = SenseNovaMHCRouter(
            hidden_size=64,
            settings=dict(streams=4, iterations=20, fixed_identity=False),
        ).cuda()
        with torch.no_grad():
            router.weight.normal_(std=0.03)
            router.bias.normal_(std=0.03)
        x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
        branch, mix, post, _ = router(x)
        torch.testing.assert_close(
            mix.sum(-1), torch.ones_like(mix.sum(-1)), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            mix.sum(-2), torch.ones_like(mix.sum(-2)), rtol=1e-5, atol=1e-6
        )
        self.assertTrue(torch.all(mix >= 0))
        self.assertEqual(branch.dtype, torch.bfloat16)
        self.assertEqual(mix.dtype, torch.float32)
        self.assertEqual(post.dtype, torch.float32)

    def test_router_parameters_stay_fp32_under_the_model_default_dtype(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            router = SenseNovaMHCRouter(
                hidden_size=16,
                settings=dict(streams=2, iterations=20, fixed_identity=False),
            )
        finally:
            torch.set_default_dtype(previous)
        self.assertEqual(router.weight.dtype, torch.float32)
        self.assertEqual(router.bias.dtype, torch.float32)

    def test_empty_batch_keeps_the_stream_shape(self):
        router = SenseNovaMHCRouter(
            hidden_size=16,
            settings=dict(streams=2, iterations=20, fixed_identity=False),
        ).cuda()
        x = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16)
        branch, mix, post, _ = router(x)
        self.assertEqual(router.merge(branch, x, mix, post).shape, x.shape)


class TestSenseNovaLoopForward(CustomTestCase):
    def test_forward_rejects_unsupported_inputs(self):
        """Unsupported PP/DeepStack inputs must not be silently discarded."""
        model = SenseNovaLoopModel.__new__(SenseNovaLoopModel)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList()
        model.norm = torch.nn.Identity()
        hidden_states = torch.ones(1, 16)
        for name, value, message in (
            (
                "pp_proxy_tensors",
                PPProxyTensors({"hidden_states": hidden_states}),
                "pipeline parallel",
            ),
            ("input_deepstack_embeds", hidden_states, "DeepStack"),
        ):
            with self.subTest(input=name), self.assertRaisesRegex(ValueError, message):
                model(
                    input_ids=None,
                    positions=torch.zeros(1, dtype=torch.long),
                    forward_batch=SimpleNamespace(),
                    input_embeds=hidden_states,
                    **{name: value},
                )


class TestSenseNovaLoopWeights(CustomTestCase):
    def test_canonical_weights_load_vision_and_late_shared_layers(self):
        # Exercise the real multimodal loader with tiny parameters, without a
        # private checkpoint or allocation of the full serving model.
        model = SenseNovaLoopForConditionalGeneration.__new__(
            SenseNovaLoopForConditionalGeneration
        )
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(tie_word_embeddings=False)
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.pooler = None
        body = torch.nn.Module()
        body.start_layer, body.end_layer = 0, 56
        plan = list(range(4)) + list(range(4, 28)) * 2 + list(range(28, 32))
        layers = []
        for _ in range(32):
            layer = torch.nn.Module()
            layer.input_layernorm = torch.nn.LayerNorm(2)
            layers.append(layer)
        body.layers = torch.nn.ModuleList([layers[i] for i in plan])
        body._weight_layer_indices = {i: plan.index(i) for i in range(32)}
        model.model = body
        model.visual = torch.nn.Module()
        model.visual.patch_embed = torch.nn.Linear(2, 2, bias=False)
        model.lm_head = torch.nn.Linear(2, 2, bias=False)
        loaded = model.load_weights(
            iter(
                [
                    ("model.visual.patch_embed.weight", torch.full((2, 2), 3.0)),
                    (
                        "model.language_model.layers.4.input_layernorm.weight",
                        torch.full((2,), 4.0),
                    ),
                    (
                        "model.language_model.layers.31.input_layernorm.weight",
                        torch.full((2,), 5.0),
                    ),
                    ("lm_head.weight", torch.full((2, 2), 6.0)),
                    ("model.language_model.loop_gate.logits", torch.ones(1)),
                ]
            )
        )
        torch.testing.assert_close(
            model.visual.patch_embed.weight, torch.full((2, 2), 3.0)
        )
        torch.testing.assert_close(
            body.layers[4].input_layernorm.weight, torch.full((2,), 4.0)
        )
        self.assertIs(
            body.layers[4].input_layernorm.weight,
            body.layers[28].input_layernorm.weight,
        )
        torch.testing.assert_close(
            body.layers[55].input_layernorm.weight, torch.full((2,), 5.0)
        )
        torch.testing.assert_close(model.lm_head.weight, torch.full((2, 2), 6.0))
        self.assertEqual(
            loaded,
            {
                "visual.patch_embed.weight",
                "model.layers.4.input_layernorm.weight",
                "model.layers.55.input_layernorm.weight",
                "lm_head.weight",
            },
        )


if __name__ == "__main__":
    unittest.main()
