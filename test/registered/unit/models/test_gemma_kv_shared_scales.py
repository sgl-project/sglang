"""KV-shared layers of Gemma 3n and Gemma 4 use their owner's KV-cache scales.

A KV-shared layer writes no KV cache: it reads the cache of an earlier layer,
the owner, which holds K/V divided by the owner's k_scale/v_scale. The
KV-shared layers kept their own scales, which stay 1.0 because checkpoints
have none for them, so they read the owner's FP8 cache with the wrong scales.
"""

import socket
import unittest

import torch
from transformers import Gemma3nTextConfig, Gemma4TextConfig

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.models.gemma3n_causal import Gemma3nTextModel
from sglang.srt.models.gemma4_causal import Gemma4TextModel
from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

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

# Four layers; layers 2 and 3 read the caches of layers 0 and 1.
LAYER_TYPES = ["sliding_attention", "full_attention"] * 2
# (k_scale, v_scale) the checkpoint has for layers 0 and 1.
OWNER_SCALES = {0: (0.25, 0.0625), 1: (0.5, 0.125)}
SHARED = {2: 0, 3: 1}


def gemma4_model(quant_config):
    config = Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=8,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=LAYER_TYPES,
    )
    return Gemma4TextModel(config, quant_config=quant_config)


def gemma3n_model(quant_config):
    config = Gemma3nTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=8,
        laurel_rank=8,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=LAYER_TYPES,
        activation_sparsity_pattern=[0.0] * 4,
    )
    return Gemma3nTextModel(config, quant_config=quant_config)


def attentions(model):
    return {
        int(name.split(".")[1]): module
        for name, module in model.named_modules()
        if isinstance(module, RadixAttention)
    }


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestGemmaKvSharedScales(CustomTestCase):
    def setUp(self):
        if torch.distributed.is_initialized():
            self.skipTest("requires an isolated distributed test process")
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)
        server = ServerArgs(model_path="dummy", device="cuda", tp_size=1)
        publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        self.addCleanup(destroy_distributed_environment)
        initialize_model_parallel()
        self.addCleanup(destroy_model_parallel)
        initialize_dp_attention(server)
        self.quant_config = CompressedTensorsConfig.from_config(KV_CACHE_ONLY)

    def test_kv_shared_layers_use_owner_scales(self):
        for build in (gemma4_model, gemma3n_model):
            # The loader processes the layers in module order; the result must
            # not depend on it.
            for order in ("owner first", "reader first"):
                with self.subTest(model=build.__name__, order=order):
                    with torch.device("cuda"):
                        attns = attentions(build(self.quant_config))
                    for reader, owner in SHARED.items():
                        self.assertEqual(attns[reader].layer_id, owner)
                    # the checkpoint's scales for the owners, as loaded
                    for layer, (k, v) in OWNER_SCALES.items():
                        attns[layer].k_scale.data.fill_(k)
                        attns[layer].v_scale.data.fill_(v)
                    layers = sorted(attns, reverse=order == "reader first")
                    for layer in layers:
                        attn = attns[layer]
                        attn.quant_method.process_weights_after_loading(attn)
                    for layer in attns:
                        expected = OWNER_SCALES[SHARED.get(layer, layer)]
                        attn = attns[layer]
                        self.assertEqual(
                            (attn.k_scale_float, attn.v_scale_float), expected
                        )
                        self.assertEqual(
                            (attn.k_scale.item(), attn.v_scale.item()), expected
                        )

    def test_without_kv_cache_quantization(self):
        for build in (gemma4_model, gemma3n_model):
            with self.subTest(model=build.__name__), torch.device("cuda"):
                attns = attentions(build(None))
                for attn in attns.values():
                    self.assertIsNone(attn.quant_method)
                    self.assertIsNone(attn.k_scale)


if __name__ == "__main__":
    unittest.main()
