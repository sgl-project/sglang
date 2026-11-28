"""Checkpoint FP8 KV-cache scales reach the attention layers of these models.

These models built RadixAttention without the quantization config, so the
attention had no k_scale/v_scale parameters, and most of their loaders had no
rule for KV-scale names. An FP8 KV cache then ran with a scale of 1.0 whatever
the checkpoint stored.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.models import utils as model_utils
from sglang.srt.models.exaone_moe import ExaoneMoEAttention, ExaoneMoEForCausalLM
from sglang.srt.models.exaone_moe_mtp import ExaoneMoEForCausalLMMTP
from sglang.srt.models.glm4_moe import Glm4MoeAttention, Glm4MoeForCausalLM
from sglang.srt.models.glm4_moe_nextn import Glm4MoeForCausalLMNextN
from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration
from sglang.srt.models.hunyuan import HunYuanAttention, HunYuanMoEV1ForCausalLM
from sglang.srt.models.hunyuan_v3 import HYV3Attention, HYV3ForCausalLM
from sglang.srt.models.hunyuan_v3_nextn import HYV3ForCausalLMNextN
from sglang.srt.models.lfm2 import (
    Lfm2Attention,
    Lfm2BidirectionalModel,
    Lfm2ForCausalLM,
)
from sglang.srt.models.lfm2_moe import Lfm2MoeAttention, Lfm2MoeForCausalLM
from sglang.srt.models.llama4 import Llama4Attention, Llama4ForCausalLM
from sglang.srt.models.mellum import MellumAttention
from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
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
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    norm_eps=1e-6,
    max_position_embeddings=64,
    rope_theta=10000.0,
    rope_scaling=None,
    num_hidden_layers=2,
    num_nextn_predict_layers=0,
    num_experts=2,
    n_routed_experts=2,
    num_local_experts=2,
    tie_word_embeddings=False,
    layer_types=["full_attention", "full_attention"],
    sliding_window=None,
    use_qk_norm=False,
    attn_temperature_tuning=False,
    floor_scale=8192,
    attn_scale=0.1,
)


def build_attention(kind, config, quant_config, prefix):
    args = dict(quant_config=quant_config, prefix=prefix)
    if kind == "glm4_moe":
        return Glm4MoeAttention(64, 4, 2, head_dim=16, **args)
    if kind == "hunyuan":
        return HunYuanAttention(config, 64, 4, 2, layer_id=0, **args)
    if kind == "hunyuan_v3":
        return HYV3Attention(config, 64, 4, 2, **args)
    if kind == "exaone_moe":
        return ExaoneMoEAttention(config, 64, 4, 2, **args)
    if kind == "lfm2":
        return Lfm2Attention(config, 0, **args)
    if kind == "lfm2_moe":
        return Lfm2MoeAttention(config, 0, **args)
    if kind == "llama4":
        return Llama4Attention(config, 0, 64, 4, 2, **args)
    if kind == "qwen3_moe":
        return Qwen3MoeAttention(64, 4, 2, head_dim=16, config=config, **args)
    if kind == "mellum":
        return MellumAttention(64, 4, 2, head_dim=16, config=config, **args)
    raise ValueError(kind)


# model class: (attention kind, module path of the layer, checkpoint path of the
#               layer, config overrides)
LOADERS = {
    Glm4MoeForCausalLM: ("glm4_moe", "model.layers.0", "model.layers.0", {}),
    Glm4MoeForCausalLMNextN: (
        "glm4_moe",
        "model.decoder",
        "model.layers.2",
        {"num_nextn_predict_layers": 1},
    ),
    Glm4vMoeForConditionalGeneration: (
        "glm4_moe",
        "model.layers.0",
        "model.language_model.layers.0",
        {},
    ),
    HunYuanMoEV1ForCausalLM: ("hunyuan", "model.layers.0", "model.layers.0", {}),
    HYV3ForCausalLM: ("hunyuan_v3", "model.layers.0", "model.layers.0", {}),
    HYV3ForCausalLMNextN: ("hunyuan_v3", "model.decoder", "model.layers.2", {}),
    ExaoneMoEForCausalLM: ("exaone_moe", "model.layers.0", "model.layers.0", {}),
    ExaoneMoEForCausalLMMTP: ("exaone_moe", "model.layers.0", "mtp.layers.0", {}),
    Lfm2ForCausalLM: ("lfm2", "model.layers.0", "model.layers.0", {}),
    Lfm2BidirectionalModel: ("lfm2", "layers.0", "layers.0", {}),
    Lfm2MoeForCausalLM: ("lfm2_moe", "model.layers.0", "model.layers.0", {}),
    Llama4ForCausalLM: ("llama4", "model.layers.0", "model.layers.0", {}),
    Llama4ForConditionalGeneration: (
        "llama4",
        "language_model.model.layers.0",
        "language_model.model.layers.0",
        {},
    ),
}

# compressed-tensors (LLM Compressor) and ModelOpt names of the same scales.
NAMINGS = {
    "compressed-tensors": ("self_attn.k_scale", "self_attn.v_scale"),
    "modelopt": ("self_attn.k_proj.k_scale", "self_attn.v_proj.v_scale"),
}
# HunYuan V1 still raises on the ModelOpt names, before its remap is reached.
SKIP = {(HunYuanMoEV1ForCausalLM, "modelopt")}


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


def build_model(cls, quant_config):
    kind, attn_path, _, overrides = LOADERS[cls]
    config = SimpleNamespace(**{**CONFIG, **overrides})
    attn = build_attention(kind, config, quant_config, f"{attn_path}.self_attn")
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = config
    if cls is Llama4ForConditionalGeneration:
        model.config = SimpleNamespace(text_config=config)
        model.has_vision = False
    model.quant_config = quant_config
    model.num_fused_shared_experts = 0
    place(model, f"{attn_path}.self_attn", attn)
    return model, attn.attn


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestAttentionKvCacheScaleLoading(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        self.quant_config = CompressedTensorsConfig.from_config(KV_CACHE_ONLY)

    def test_attention_gets_kv_cache_scales(self):
        config = SimpleNamespace(**CONFIG)
        kinds = (
            "glm4_moe",
            "hunyuan",
            "hunyuan_v3",
            "exaone_moe",
            "lfm2",
            "lfm2_moe",
            "llama4",
            "qwen3_moe",
            "mellum",
        )
        for kind in kinds:
            with self.subTest(kind=kind), torch.device("cuda"):
                attention = build_attention(
                    kind, config, self.quant_config, "model.layers.0.self_attn"
                )
                self.assertIsInstance(attention.attn.quant_method, BaseKVCacheMethod)
                self.assertIsInstance(attention.attn.k_scale, nn.Parameter)
                self.assertIsInstance(attention.attn.v_scale, nn.Parameter)

    def test_checkpoint_scales_reach_attention(self):
        for cls, (_, _, ckpt_path, _) in LOADERS.items():
            for naming, (k_name, v_name) in NAMINGS.items():
                if (cls, naming) in SKIP:
                    continue
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


class TestFusedKvCacheWriteWithScales(CustomTestCase):
    """A layer with KV-cache scales may take the CUDA fused RoPE + KV write,
    which only runs on a bf16 pool, where the scales do not apply."""

    def fused_arg(self, pool_dtype):
        buffer = torch.zeros(4, 2, 16)
        kv = SimpleNamespace(key=lambda: buffer, value=lambda: buffer)
        layer = SimpleNamespace(
            layer_id=0, k_scale=torch.tensor(K_SCALE), v_scale=torch.tensor(V_SCALE)
        )
        with (
            patch.object(model_utils, "_is_hip", False),
            patch.object(
                model_utils,
                "get_token_to_kv_pool",
                return_value=SimpleNamespace(dtype=pool_dtype),
            ),
            patch.object(
                model_utils,
                "get_kv_layers",
                return_value=SimpleNamespace(read=lambda layer_id: kv),
            ),
        ):
            return model_utils.create_fused_set_kv_buffer_arg(
                torch.zeros(2, 32),
                layer,
                SimpleNamespace(out_cache_loc=torch.arange(2)),
            )

    def test_bf16_pool_accepts_kv_scales(self):
        self.assertIsNotNone(self.fused_arg(torch.bfloat16))

    def test_quantized_pool_rejects_kv_scales(self):
        with self.assertRaises(AssertionError):
            self.fused_arg(torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
