import unittest

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.models.grok import Grok1Model
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGrokQuantizationPrefix(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def make_model(self, ignored_layers):
        reset_context()
        publish(
            ServerArgs(
                model_path="dummy", moe_runner_backend="triton", moe_a2a_backend="none"
            ),
            role="test",
        )
        config = PretrainedConfig(
            num_local_experts=2,
            num_experts_per_tok=1,
            hidden_size=128,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=2,
            vocab_size=128,
            pad_token_id=0,
            max_position_embeddings=128,
            rms_norm_eps=1e-5,
            residual_moe=True,
        )
        quant = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            ignored_layers=ignored_layers,
        )
        with (
            get_parallel().override(
                tp_rank=0, moe_tp_rank=0, moe_ep_rank=0, attn_tp_rank=0
            ),
            torch.device("cuda"),
        ):
            return Grok1Model(config, quant_config=quant, prefix="model")

    def test_expert_exclusion_is_layer_specific(self):
        model = self.make_model(["model.layers.0.block_sparse_moe.experts"])
        self.assertIsInstance(
            model.layers[0].block_sparse_moe.experts.quant_method,
            UnquantizedFusedMoEMethod,
        )
        self.assertIsInstance(
            model.layers[1].block_sparse_moe.experts.quant_method, Fp8MoEMethod
        )

    def test_attention_exclusion_uses_checkpoint_name(self):
        model = self.make_model(["model.layers.0.self_attn.o_proj"])
        self.assertIsInstance(
            model.layers[0].self_attn.o_proj.quant_method, UnquantizedLinearMethod
        )
        self.assertIsInstance(
            model.layers[1].self_attn.o_proj.quant_method, Fp8LinearMethod
        )

    def test_residual_mlp_exclusion_is_layer_specific(self):
        model = self.make_model(["model.layers.0.mlp.down_proj"])
        self.assertIsInstance(
            model.layers[0].mlp.down_proj.quant_method, UnquantizedLinearMethod
        )
        self.assertIsInstance(
            model.layers[1].mlp.down_proj.quant_method, Fp8LinearMethod
        )

    def test_no_exclusions_keeps_fp8(self):
        model = self.make_model([])
        for layer in model.layers:
            self.assertIsInstance(
                layer.block_sparse_moe.experts.quant_method, Fp8MoEMethod
            )
            self.assertIsInstance(layer.self_attn.o_proj.quant_method, Fp8LinearMethod)
            self.assertIsInstance(layer.mlp.down_proj.quant_method, Fp8LinearMethod)


if __name__ == "__main__":
    unittest.main()
