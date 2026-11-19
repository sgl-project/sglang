import copy
import unittest

import torch

from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMarlinConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


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

    def _embedding(self, config_class):
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
        for config_class, dtype in (
            (AWQConfig, torch.float16),
            (AWQMarlinConfig, torch.bfloat16),
        ):
            with self.subTest(config=config_class.__name__):
                layer = ParallelLMHead(
                    64,
                    256,
                    enable_tp=False,
                    params_dtype=dtype,
                    quant_config=config_class.from_config(self.config),
                    prefix="lm_head",
                ).cuda()
                for name, value in (
                    ("qweight", -1145324613),
                    ("qzeros", 572662306),
                    ("scales", 0.001),
                ):
                    param = getattr(layer, name)
                    param.weight_loader(param, torch.full_like(param, value))
                self.assertEqual(layer.qweight.shape, (256, 8))

    @torch.inference_mode()
    def test_unquantized_or_excluded_embedding_keeps_dense_lookup(self):
        for excluded in (False, True):
            with self.subTest(excluded=excluded):
                config = copy.deepcopy(self.config)
                if excluded:
                    config["modules_to_not_convert"] = ["embed_tokens"]
                else:
                    config.pop("embedding")
                layer = VocabParallelEmbedding(
                    17,
                    1024,
                    enable_tp=False,
                    params_dtype=torch.bfloat16,
                    quant_config=AWQConfig.from_config(config),
                    prefix="model.embed_tokens",
                ).cuda()
                layer.weight_loader(layer.weight, self.reference)
                ids = torch.tensor([0, 16, 3], device="cuda")
                torch.testing.assert_close(
                    layer(ids), self.reference[ids], rtol=0, atol=0
                )

    @torch.inference_mode()
    def test_lookup_replays_with_new_tokens_in_cuda_graph(self):
        layer = self._embedding(AWQMarlinConfig)
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
