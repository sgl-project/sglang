"""TorchNativeLlamaForCausalLM and TorchNativePhi3ForCausalLM load checkpoint
tensors and FP8 KV-cache scales.

`load_weights` used to collect only the root module's own parameters, so it
loaded no checkpoint tensor, and the attention was built without the
quantization config, so it had no k_scale/v_scale parameters.
"""

import unittest

import torch
from transformers import LlamaConfig, Phi3Config

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.models.torch_native_llama import (
    TorchNativeLlamaForCausalLM,
    TorchNativePhi3ForCausalLM,
)
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

K_SCALE, V_SCALE = 0.25, 0.0625

# A compressed-tensors checkpoint that only declares a static FP8 KV cache.
KV_CACHE_ONLY = {
    "quant_method": "compressed-tensors",
    "config_groups": {},
    "kv_cache_scheme": {
        "num_bits": 8,
        "type": "float",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    },
}

SIZES = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=128,
    pad_token_id=0,
    max_position_embeddings=64,
)
MODELS = {
    TorchNativeLlamaForCausalLM: LlamaConfig,
    TorchNativePhi3ForCausalLM: Phi3Config,
}

# compressed-tensors (LLM Compressor) and ModelOpt names of the same scales.
NAMINGS = {
    "compressed-tensors": ("self_attn.k_scale", "self_attn.v_scale"),
    "modelopt": ("self_attn.k_proj.k_scale", "self_attn.v_proj.v_scale"),
}


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestTorchNativeLlamaWeightLoading(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        self.quant_config = CompressedTensorsConfig.from_config(KV_CACHE_ONLY)

    def build(self, cls):
        with torch.device("cuda"):
            return cls(MODELS[cls](**SIZES), self.quant_config)

    def load_scales(self, cls, weights):
        model = self.build(cls)
        attn = model.model.layers[1].self_attn.attn
        self.assertIsInstance(attn.quant_method, BaseKVCacheMethod)
        model.load_weights(weights)
        attn.quant_method.process_weights_after_loading(attn)
        return attn.k_scale_float, attn.v_scale_float

    def test_checkpoint_scales_reach_attention(self):
        for cls in MODELS:
            for naming, (k_name, v_name) in NAMINGS.items():
                with self.subTest(model=cls.__name__, naming=naming):
                    scales = self.load_scales(
                        cls,
                        [
                            (f"model.layers.1.{k_name}", torch.tensor(K_SCALE)),
                            (f"model.layers.1.{v_name}", torch.tensor(V_SCALE)),
                        ],
                    )
                    self.assertEqual(scales, (K_SCALE, V_SCALE))

    def test_single_kv_scale_serves_keys_and_values(self):
        for cls in MODELS:
            with self.subTest(model=cls.__name__):
                scales = self.load_scales(
                    cls, [("model.layers.1.self_attn.kv_scale", torch.tensor(K_SCALE))]
                )
                self.assertEqual(scales, (K_SCALE, K_SCALE))

    def test_checkpoint_tensors_load(self):
        # Llama checkpoints store q/k/v_proj and gate/up_proj, Phi-3 checkpoints
        # store the fused qkv_proj and gate_up_proj.
        q, k, v = torch.rand(64, 64), torch.rand(32, 64), torch.rand(32, 64)
        gate, up, norm = torch.rand(128, 64), torch.rand(128, 64), torch.rand(64)
        prefix = "model.layers.1"
        layouts = {
            TorchNativeLlamaForCausalLM: [
                (f"{prefix}.self_attn.q_proj.weight", q),
                (f"{prefix}.self_attn.k_proj.weight", k),
                (f"{prefix}.self_attn.v_proj.weight", v),
                (f"{prefix}.mlp.gate_proj.weight", gate),
                (f"{prefix}.mlp.up_proj.weight", up),
            ],
            TorchNativePhi3ForCausalLM: [
                (f"{prefix}.self_attn.qkv_proj.weight", torch.cat([q, k, v])),
                (f"{prefix}.mlp.gate_up_proj.weight", torch.cat([gate, up])),
            ],
        }
        for cls, weights in layouts.items():
            with self.subTest(model=cls.__name__):
                model = self.build(cls)
                model.load_weights(
                    weights + [(f"{prefix}.input_layernorm.weight", norm)]
                )
                layer = model.model.layers[1]
                for param, expected in (
                    (layer.self_attn.qkv_proj.weight, torch.cat([q, k, v])),
                    (layer.mlp.gate_up_proj.weight, torch.cat([gate, up])),
                    (layer.input_layernorm.weight, norm),
                ):
                    torch.testing.assert_close(param.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
