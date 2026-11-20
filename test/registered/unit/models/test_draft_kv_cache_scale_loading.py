"""Checkpoint FP8 KV-cache scales reach the attention layers of speculative drafts.

The DFlash loader (which the DSpark drafts also use) and the LLaMA EAGLE-3
loader had no rule for KV-cache scale names. They dropped `<attn>.k_scale`
(compressed-tensors) and `<attn>.k_proj.k_scale` (ModelOpt), and the draft's
FP8 KV cache ran with a scale of 1.0 whatever the checkpoint stored.
"""

import unittest
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, Qwen3Config

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import Qwen3DSparkModel
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

# (k_scale, v_scale) of layer 0 and layer 1.
SCALES = ((0.25, 0.0625), (0.5, 0.125))

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

ATTENTION = dict(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=64,
    rms_norm_eps=1e-6,
    max_position_embeddings=64,
)
DFLASH = dict(
    ATTENTION,
    num_hidden_layers=2,
    num_target_layers=2,
    dflash_config={"block_size": 4, "target_layer_ids": [0, 1]},
)

# compressed-tensors (LLM Compressor) and ModelOpt names of the same scales.
NAMINGS = {
    "compressed-tensors": ("k_scale", "v_scale"),
    "modelopt": ("k_proj.k_scale", "v_proj.v_scale"),
}


def build_dflash(quant_config):
    return DFlashDraftModel(Qwen3Config(**DFLASH), quant_config=quant_config)


def build_dspark(quant_config):
    config = Qwen3Config(**DFLASH, markov_rank=4, markov_head_type="vanilla")
    return Qwen3DSparkModel(config, quant_config=quant_config)


def build_eagle3(quant_config):
    config = LlamaConfig(**ATTENTION, num_hidden_layers=1, draft_vocab_size=64)
    return LlamaForCausalLMEagle3(config, quant_config=quant_config)


# name: (builder, attention module paths of the draft, checkpoint attention
#        paths per layer under each naming the checkpoints use)
DRAFTS = {
    "dflash": (
        build_dflash,
        ("layers.0.self_attn", "layers.1.self_attn"),
        (
            ("layers.0.self_attn", "layers.1.self_attn"),
            ("model.layers.0.self_attn", "model.layers.1.self_attn"),
        ),
    ),
    "dspark": (
        build_dspark,
        ("layers.0.self_attn", "layers.1.self_attn"),
        (("layers.0.self_attn", "layers.1.self_attn"),),
    ),
    "eagle3": (
        build_eagle3,
        ("model.layers.0.self_attn",),
        (("midlayer.self_attn",), ("layers.0.self_attn",)),
    ),
}


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestDraftKvCacheScaleLoading(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        # The EAGLE-3 draft reads the pipeline group; one rank holds every layer.
        pp_group = SimpleNamespace(
            world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
        )
        override = get_parallel().override(pp_group=pp_group)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        self.quant_config = CompressedTensorsConfig.from_config(KV_CACHE_ONLY)

    def test_checkpoint_scales_reach_attention(self):
        for draft, (build, attn_paths, ckpt_layouts) in DRAFTS.items():
            for ckpt_paths in ckpt_layouts:
                for naming, (k_name, v_name) in NAMINGS.items():
                    with self.subTest(draft=draft, ckpt=ckpt_paths[0], naming=naming):
                        with torch.device("cuda"):
                            model = build(self.quant_config)
                        modules = dict(model.named_modules())
                        attns = [modules[f"{path}.attn"] for path in attn_paths]
                        weights = []
                        for path, (k, v) in zip(ckpt_paths, SCALES):
                            weights.append((f"{path}.{k_name}", torch.tensor(k)))
                            weights.append((f"{path}.{v_name}", torch.tensor(v)))
                        model.load_weights(weights)
                        for attn, expected in zip(attns, SCALES):
                            self.assertIsNotNone(attn.quant_method)
                            attn.quant_method.process_weights_after_loading(attn)
                            self.assertEqual(
                                (attn.k_scale_float, attn.v_scale_float), expected
                            )

    def test_scales_without_kv_cache_method_are_skipped(self):
        """An unquantized draft has no KV-cache scale parameters to load into."""
        for draft, (build, _, ckpt_layouts) in DRAFTS.items():
            for naming, (k_name, v_name) in NAMINGS.items():
                with self.subTest(draft=draft, naming=naming):
                    with torch.device("cuda"):
                        model = build(None)
                    path = ckpt_layouts[0][0]
                    model.load_weights(
                        [
                            (f"{path}.{k_name}", torch.tensor(0.25)),
                            (f"{path}.{v_name}", torch.tensor(0.0625)),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
