"""Checkpoint FP8 KV-cache scales reach the attention layers of Qwen3 models.

Several Qwen3-family loaders had no rule for KV-scale names, or renamed
k_proj.k_scale to qkv_proj.k_scale and then skipped it, so an FP8 KV cache ran
with a scale of 1.0 whatever the checkpoint stored.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.models.cosmos3 import Cosmos3ForConditionalGeneration
from sglang.srt.models.lightonocr import LightOnOCRForConditionalGeneration
from sglang.srt.models.qwen3 import Qwen3Attention, Qwen3ForCausalLM
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.models.qwen3_classification import Qwen3ForSequenceClassification
from sglang.srt.models.qwen3_embedding import Qwen3Model
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.srt.models.qwen3_rm import Qwen3ForRewardModel
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
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

CONFIG = dict(
    tie_word_embeddings=False,
    num_experts=2,
    encoder_only=False,
    language_only=False,
)


def build_attention(kind, quant_config, prefix):
    args = dict(head_dim=16, quant_config=quant_config, prefix=prefix)
    if kind == "qwen3":
        return Qwen3Attention(64, 4, 2, rms_norm_eps=1e-6, **args)
    if kind == "qwen3_moe":
        return Qwen3MoeAttention(64, 4, 2, config=SimpleNamespace(), **args)
    raise ValueError(kind)


# model class: (attention kind, module path of the layer, checkpoint path of the
#               layer)
LOADERS = {
    Qwen3ForCausalLM: ("qwen3", "model.layers.0", "model.layers.0"),
    Qwen3Model: ("qwen3", "model.layers.0", "model.layers.0"),
    Qwen3ForSequenceClassification: ("qwen3", "model.layers.0", "model.layers.0"),
    Qwen3ForRewardModel: ("qwen3", "model.layers.0", "model.layers.0"),
    Qwen3VLForConditionalGeneration: (
        "qwen3",
        "model.layers.0",
        "model.language_model.layers.0",
    ),
    # diffusers layout, which its WeightsMapper rewrites to the Qwen3-VL names
    Cosmos3ForConditionalGeneration: ("qwen3", "model.layers.0", "layers.0"),
    LightOnOCRForConditionalGeneration: (
        "qwen3",
        "language_model.model.layers.0",
        "model.language_model.layers.0",
    ),
    Qwen3ASRForConditionalGeneration: (
        "qwen3",
        "language_model.model.layers.0",
        "thinker.model.layers.0",
    ),
    Qwen3VLMoeForConditionalGeneration: (
        "qwen3_moe",
        "model.layers.0",
        "model.language_model.layers.0",
    ),
    Qwen3OmniMoeForConditionalGeneration: (
        "qwen3_moe",
        "thinker.model.layers.0",
        "thinker.model.layers.0",
    ),
}

# compressed-tensors (LLM Compressor) and ModelOpt names of the same scales.
NAMINGS = {
    "compressed-tensors": ("self_attn.k_scale", "self_attn.v_scale"),
    "modelopt": ("self_attn.k_proj.k_scale", "self_attn.v_proj.v_scale"),
}


def place(root, path, module):
    node = root
    parts = path.split(".")
    for part, nxt in zip(parts, parts[1:]):
        if part.isdigit():
            node = node[int(part)]
            continue
        if not hasattr(node, part):
            setattr(
                node,
                part,
                nn.ModuleList([nn.Module()]) if nxt.isdigit() else nn.Module(),
            )
        node = getattr(node, part)
    setattr(node, parts[-1], module)


def bare(cls, config, quant_config):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = config
    model.quant_config = quant_config
    model.pp_group = SimpleNamespace(is_last_rank=True)
    return model


def build_model(cls, quant_config):
    kind, attn_path, _ = LOADERS[cls]
    config = SimpleNamespace(**CONFIG)
    attn = build_attention(kind, quant_config, f"{attn_path}.self_attn")
    model = bare(cls, config, quant_config)
    if cls is Qwen3ASRForConditionalGeneration:
        model.config = SimpleNamespace(
            thinker_config=SimpleNamespace(text_config=config)
        )
    if cls is Qwen3OmniMoeForConditionalGeneration:
        model.enable_talker = False
    if cls is LightOnOCRForConditionalGeneration:
        for part in (
            "vision_encoder",
            "patch_merger",
            "vision_projection_norm",
            "vision_language_adapter",
        ):
            setattr(model, part, nn.Module())
        model.language_model = bare(Qwen3ForCausalLM, config, quant_config)
    place(model, f"{attn_path}.self_attn", attn)
    return model, attn.attn


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestQwen3KvCacheScaleLoading(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        self.quant_config = CompressedTensorsConfig.from_config(KV_CACHE_ONLY)

    def test_attention_gets_kv_cache_scales(self):
        for kind in ("qwen3", "qwen3_moe"):
            with self.subTest(kind=kind), torch.device("cuda"):
                attention = build_attention(
                    kind, self.quant_config, "model.layers.0.self_attn"
                )
                self.assertIsInstance(attention.attn.quant_method, BaseKVCacheMethod)
                self.assertIsInstance(attention.attn.k_scale, nn.Parameter)
                self.assertIsInstance(attention.attn.v_scale, nn.Parameter)

    def test_checkpoint_scales_reach_attention(self):
        for cls, (_, _, ckpt_path) in LOADERS.items():
            for naming, (k_name, v_name) in NAMINGS.items():
                with self.subTest(model=cls.__name__, naming=naming):
                    with torch.device("cuda"):
                        model, attn = build_model(cls, self.quant_config)
                    self.assertIsNotNone(attn.quant_method)
                    model.load_weights(
                        [
                            (f"{ckpt_path}.{k_name}", torch.tensor(K_SCALE)),
                            (f"{ckpt_path}.{v_name}", torch.tensor(V_SCALE)),
                        ]
                    )
                    attn.quant_method.process_weights_after_loading(attn)
                    self.assertEqual(
                        (attn.k_scale_float, attn.v_scale_float), (K_SCALE, V_SCALE)
                    )


if __name__ == "__main__":
    unittest.main()
