import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMarlinConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.sensenova_loop import (
    SenseNovaLoopForConditionalGeneration,
    SenseNovaLoopModel,
    SenseNovaMHCRouter,
    _SenseNovaLoopAWQEmbeddingConfig,
)
from sglang.srt.runtime_context import get_parallel
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


class TestSenseNovaLoopAWQEmbedding(CustomTestCase):
    def setUp(self):
        parallel = get_parallel().override(tp_group=None)
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)
        self.config = {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "lm_head": True,
            "embedding": {
                "bits": 4,
                "group_size": 128,
                "layout": "row_major",
                "zero_point": True,
            },
        }
        self.values = torch.arange(17 * 1024, device="cuda").reshape(17, 1024) % 16
        self.zeros = torch.arange(17 * 8, device="cuda").reshape(17, 8) % 16
        self.scales = (
            torch.linspace(0.001, 0.03, 17 * 8, device="cuda").reshape(17, 8).half()
        )
        self.reference = (
            (
                (self.values.unflatten(-1, (8, 128)).float() - self.zeros.unsqueeze(-1))
                * self.scales.float().unsqueeze(-1)
            )
            .flatten(-2)
            .bfloat16()
        )

    def _embedding(self):
        layer = VocabParallelEmbedding(
            17,
            1024,
            params_dtype=torch.bfloat16,
            quant_config=_SenseNovaLoopAWQEmbeddingConfig.from_config(
                self.config["embedding"]
            ),
            prefix="model.embed_tokens",
            enable_tp=False,
        ).cuda()
        for name, source in (
            ("qweight", self.values),
            ("qzeros", self.zeros),
            ("scales", self.scales),
        ):
            if name != "scales":
                # Row-major INT4 uses consecutive nibbles, unlike AWQ GEMM packing.
                source = sum(
                    source[:, offset::8].int() << (4 * offset) for offset in range(8)
                )
            param = getattr(layer, name)
            param.weight_loader(param, source)
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    @torch.inference_mode()
    def test_row_major_lookup_preserves_fp16_scales_and_vocab_padding(self):
        layer = self._embedding()
        ids = torch.tensor([[0, 16, 3], [16, 1, 0]], device="cuda")
        torch.testing.assert_close(layer(ids), self.reference[ids], rtol=0, atol=0)
        self.assertEqual(layer.scales.dtype, torch.float16)
        self.assertEqual(
            layer(torch.empty(0, dtype=torch.long, device="cuda")).shape,
            (0, 1024),
        )

    @torch.inference_mode()
    def test_lm_head_uses_linear_awq_layout(self):
        layer = ParallelLMHead(
            64,
            256,
            enable_tp=False,
            params_dtype=torch.bfloat16,
            quant_config=AWQMarlinConfig.from_config(self.config),
            prefix="lm_head",
        ).cuda()
        self.assertEqual(layer.qweight.shape, (256, 8))
        self.assertEqual(layer.qzeros.shape, (2, 8))
        self.assertEqual(layer.scales.shape, (2, 64))

    @torch.inference_mode()
    def test_standard_awq_keeps_dense_embedding_lookup(self):
        for config_class in (AWQConfig, AWQMarlinConfig):
            with self.subTest(config=config_class.__name__):
                layer = VocabParallelEmbedding(
                    17,
                    1024,
                    enable_tp=False,
                    params_dtype=torch.bfloat16,
                    quant_config=config_class.from_config(self.config),
                    prefix="model.embed_tokens",
                ).cuda()
                layer.weight_loader(layer.weight, self.reference)
                ids = torch.tensor([0, 16, 3], device="cuda")
                torch.testing.assert_close(
                    layer(ids), self.reference[ids], rtol=0, atol=0
                )

    @torch.inference_mode()
    def test_lookup_replays_with_new_tokens_in_cuda_graph(self):
        layer = self._embedding()
        ids = torch.tensor([0, 16, 3], device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer(ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(ids)
        ids.copy_(torch.tensor([16, 1, 0], device="cuda"))
        graph.replay()
        torch.testing.assert_close(output, self.reference[ids], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
