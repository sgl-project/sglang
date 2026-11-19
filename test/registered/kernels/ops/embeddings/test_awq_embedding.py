import unittest

import torch

from sglang.srt.layers.quantization.awq import AWQConfig, AWQMarlinConfig
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestAWQEmbedding(CustomTestCase):
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

    def _embedding(self, config_class=AWQMarlinConfig):
        layer = VocabParallelEmbedding(
            17,
            1024,
            params_dtype=torch.bfloat16,
            quant_config=config_class.from_config(self.config),
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
        for config_class in (AWQConfig, AWQMarlinConfig):
            with self.subTest(config=config_class.__name__):
                layer = self._embedding(config_class)
                ids = torch.tensor([[0, 16, 3], [16, 1, 0]], device="cuda")
                torch.testing.assert_close(
                    layer(ids), self.reference[ids], rtol=0, atol=0
                )
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
    def test_dense_embedding_without_metadata_or_when_excluded(self):
        for config_class in (AWQConfig, AWQMarlinConfig):
            for excluded in (False, True):
                with self.subTest(config=config_class.__name__, excluded=excluded):
                    config = dict(self.config)
                    if excluded:
                        config["modules_to_not_convert"] = ["embed_tokens"]
                    else:
                        del config["embedding"]
                    layer = VocabParallelEmbedding(
                        17,
                        1024,
                        enable_tp=False,
                        params_dtype=torch.bfloat16,
                        quant_config=config_class.from_config(config),
                        prefix="model.embed_tokens",
                    ).cuda()
                    layer.weight_loader(layer.weight, self.reference)
                    ids = torch.tensor([0, 16, 3], device="cuda")
                    torch.testing.assert_close(
                        layer(ids), self.reference[ids], rtol=0, atol=0
                    )

    def test_embedding_metadata_does_not_quantize_disabled_lm_head(self):
        config = {**self.config, "lm_head": False}
        layer = ParallelLMHead(
            64,
            256,
            enable_tp=False,
            params_dtype=torch.bfloat16,
            quant_config=AWQMarlinConfig.from_config(config),
            prefix="lm_head",
        )
        self.assertEqual(layer.weight.shape, (64, 256))
        self.assertIsInstance(layer.quant_method, UnquantizedEmbeddingMethod)

    def test_unsupported_embedding_formats_fail_before_loading(self):
        for config_class in (AWQConfig, AWQMarlinConfig):
            for key, value in (("bits", 8), ("layout", "gemm"), ("zero_point", False)):
                with self.subTest(config=config_class.__name__, key=key):
                    config = {
                        **self.config,
                        "embedding": {**self.config["embedding"], key: value},
                    }
                    with self.assertRaisesRegex(ValueError, "row_major INT4"):
                        VocabParallelEmbedding(
                            17,
                            1024,
                            enable_tp=False,
                            quant_config=config_class.from_config(config),
                            prefix="model.embed_tokens",
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
