"""Checkpoint FP8 KV-cache scales reach the attention layers of these models.

Their loaders had no rule for KV-scale names, applied the rule only after the
q/k/v_proj -> qkv_proj rename, or had a rule that only knew the self_attn and
mixer attention names. The scales were then skipped and an FP8 KV cache ran
with a scale of 1.0 whatever the checkpoint stored.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_loader.weight_utils import maybe_remap_kv_scale_name
from sglang.srt.models.afmoe import AfmoeForCausalLM
from sglang.srt.models.bailing_moe_linear import BailingMoeV2_5ForCausalLM
from sglang.srt.models.cohere2_moe import Cohere2MoeForCausalLM
from sglang.srt.models.dbrx import DbrxForCausalLM
from sglang.srt.models.embedding_gemma2 import EmbeddingGemma2Model
from sglang.srt.models.gemma3n_causal import Gemma3nForCausalLM
from sglang.srt.models.gemma3n_mm import Gemma3nForConditionalGeneration
from sglang.srt.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration
from sglang.srt.models.glm4 import Glm4ForCausalLM
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
from sglang.srt.models.glm_ocr_nextn import GlmOcrForConditionalGenerationNextN
from sglang.srt.models.grok import Grok1ForCausalLM
from sglang.srt.models.hrm_text import HrmTextForCausalLM
from sglang.srt.models.locate_anything import LocateAnythingForConditionalGeneration
from sglang.srt.models.mixtral_quant import QuantMixtralForCausalLM
from sglang.srt.models.nanbeige import (
    NanbeigeForCausalLM,
    NanbeigeForSequenceClassification,
)
from sglang.srt.models.opt import OPTForCausalLM
from sglang.srt.models.persimmon import PersimmonForCausalLM
from sglang.srt.models.phi import PhiForCausalLM
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_moe import Qwen2MoeForCausalLM
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.srt.models.qwen3_next_mtp import Qwen3NextForCausalLMMTP
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.models.spark2_5 import Spark2_5ForCausalLM
from sglang.srt.models.starcoder2 import Starcoder2ForCausalLM
from sglang.srt.models.zaya import ZayaForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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

# model class: (module paths of the RadixAttention layers that hold the
#               scales, checkpoint path of the attention module)
CASES = {
    AfmoeForCausalLM: (["model.layers.0.self_attn.attn"], "model.layers.0.self_attn"),
    BailingMoeV2_5ForCausalLM: (
        ["model.layers.0.attention.attn"],
        "model.layers.0.attention",
    ),
    Cohere2MoeForCausalLM: (
        ["model.layers.0.self_attn.attn"],
        "model.layers.0.self_attn",
    ),
    DbrxForCausalLM: (
        ["transformer.blocks.0.norm_attn_norm.attn.attn"],
        "transformer.blocks.0.norm_attn_norm.attn",
    ),
    EmbeddingGemma2Model: (
        ["language_model.layers.0.self_attn.attn"],
        "language_model.layers.0.self_attn",
    ),
    Gemma3nForCausalLM: (
        ["model.layers.0.self_attn.attn"],
        "model.language_model.layers.0.self_attn",
    ),
    Gemma3nForConditionalGeneration: (
        ["language_model.layers.0.self_attn.attn"],
        "model.language_model.layers.0.self_attn",
    ),
    Gemma4UnifiedForConditionalGeneration: (
        ["language_model.layers.0.self_attn.attn"],
        "model.language_model.layers.0.self_attn",
    ),
    Glm4ForCausalLM: (["model.layers.0.self_attn.attn"], "model.layers.0.self_attn"),
    Glm4vForConditionalGeneration: (
        ["model.layers.0.self_attn.attn"],
        "model.language_model.layers.0.self_attn",
    ),
    GlmOcrForConditionalGeneration: (
        ["model.layers.0.self_attn.attn"],
        "model.language_model.layers.0.self_attn",
    ),
    GlmOcrForConditionalGenerationNextN: (
        ["model.decoder.self_attn.attn"],
        "model.language_model.layers.2.self_attn",
    ),
    Grok1ForCausalLM: (["model.layers.0.self_attn.attn"], "model.layers.0.self_attn"),
    # One RadixAttention per recurrence step shares the layer's checkpoint scale.
    HrmTextForCausalLM: (
        [
            "model.L_module.layers.0.self_attn.attn.0",
            "model.L_module.layers.0.self_attn.attn.1",
        ],
        "model.L_module.layers.0.attn",
    ),
    LocateAnythingForConditionalGeneration: (
        ["language_model.model.layers.0.self_attn.attn"],
        "language_model.model.layers.0.self_attn",
    ),
    # One RadixAttention per loop shares the layer's checkpoint scale.
    NanbeigeForCausalLM: (
        ["model.layers.0.self_attn.attn.0", "model.layers.0.self_attn.attn.1"],
        "model.layers.0.self_attn",
    ),
    NanbeigeForSequenceClassification: (
        ["model.layers.0.self_attn.attn.0", "model.layers.0.self_attn.attn.1"],
        "model.layers.0.self_attn",
    ),
    OPTForCausalLM: (
        ["model.decoder.layers.0.self_attn.attn"],
        "model.decoder.layers.0.self_attn",
    ),
    PersimmonForCausalLM: (
        ["model.layers.0.self_attn.attn"],
        "model.layers.0.self_attn",
    ),
    PhiForCausalLM: (["model.layers.0.self_attn.attn"], "model.layers.0.self_attn"),
    QuantMixtralForCausalLM: (
        ["model.layers.0.self_attn.attn"],
        "model.layers.0.self_attn",
    ),
    Qwen2MoeForCausalLM: (
        ["model.layers.0.self_attn.attn"],
        "model.layers.0.self_attn",
    ),
    Qwen2_5_VLForConditionalGeneration: (
        ["model.layers.0.self_attn.attn"],
        "model.layers.0.self_attn",
    ),
    Qwen3NextForCausalLM: (["model.layers.0.attn"], "model.layers.0.self_attn"),
    Qwen3NextForCausalLMMTP: (["model.layers.0.attn"], "mtp.layers.0.self_attn"),
    Qwen4ExpForConditionalGeneration: (
        ["model.layers.0.attn"],
        "model.language_model.layers.0.self_attn",
    ),
    Spark2_5ForCausalLM: (
        ["model.layers.0.self_attn.attn"],
        "model.layers.0.self_attn",
    ),
    Starcoder2ForCausalLM: (
        ["model.layers.0.self_attn.attn"],
        "model.layers.0.self_attn",
    ),
    ZayaForCausalLM: (["model.layers.0.self_attn.attn"], "model.layers.0.self_attn"),
}

# compressed-tensors (LLM Compressor) and ModelOpt names of the same scales.
NAMINGS = {
    "compressed-tensors": ("k_scale", "v_scale"),
    "modelopt": ("k_proj.k_scale", "v_proj.v_scale"),
}

CONFIG = dict(
    tie_word_embeddings=False,
    num_hidden_layers=2,
    num_nextn_predict_layers=1,
    num_attention_heads=4,
    num_experts=2,
    num_local_experts=2,
    q_lora_rank=None,
    encoder_only=False,
    language_only=False,
)


def place(root, path, module):
    node = root
    parts = path.split(".")
    for part, nxt in zip(parts, parts[1:]):
        container = nn.ModuleList if nxt.isdigit() else nn.Module
        if part.isdigit():
            while len(node) <= int(part):
                node.append(container())
            node = node[int(part)]
            continue
        if not hasattr(node, part):
            setattr(node, part, container())
        node = getattr(node, part)
    if parts[-1].isdigit():
        while len(node) <= int(parts[-1]):
            node.append(nn.Module())
        node[int(parts[-1])] = module
    else:
        setattr(node, parts[-1], module)


def build_model(cls, quant_config):
    """The model with only its attention layers, which is all its loader needs
    for these names."""
    attn_paths, _ = CASES[cls]
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    config = SimpleNamespace(**CONFIG)
    config.text_config = config
    model.config = config
    model.quant_config = quant_config
    model.pp_group = SimpleNamespace(
        world_size=1, is_first_rank=True, is_last_rank=True
    )
    attns = []
    for i, path in enumerate(attn_paths):
        attn = RadixAttention(
            4, 16, 1.0, 2, layer_id=i, quant_config=quant_config, prefix=path
        )
        place(model, path, attn)
        attns.append(attn)
    for stack in ("model", "language_model"):
        if hasattr(model, stack):
            getattr(model, stack).start_layer = 0
            getattr(model, stack).end_layer = 2
    # model-specific attributes their loaders read
    if not isinstance(getattr(cls, "start_layer", None), property):
        model.start_layer, model.end_layer = 0, 2
    model.loaded_param_names = set()
    model.language_model_only = True
    model.audio_tower = model.vision_tower = None
    if cls is EmbeddingGemma2Model:
        model.language_model.device = "cpu"
        model.language_model.dtype = lambda: torch.float32
    if cls is Gemma4UnifiedForConditionalGeneration:
        model.stacked_params_mapping = []
        model._get_k_eq_v_layers = lambda: set()
    if cls is BailingMoeV2_5ForCausalLM:
        model.model.layer_group_size = 1
        model.post_load_weights = lambda **kwargs: None
    if cls in (Qwen3NextForCausalLM, Qwen3NextForCausalLMMTP):
        model.enable_shared_expert_fusion = False
        model.num_fused_shared_experts = 0
    if cls is Qwen4ExpForConditionalGeneration:
        model._load_qwen4_exp_ple_buffer = lambda *args: False
    return model, attns


class TestKvCacheScaleNameRules(CustomTestCase):
    def setUp(self):
        self.quant_config = CompressedTensorsConfig.from_config(KV_CACHE_ONLY)

    def test_checkpoint_scales_reach_attention(self):
        for cls, (_, ckpt_path) in CASES.items():
            for naming, (k_name, v_name) in NAMINGS.items():
                with self.subTest(model=cls.__name__, naming=naming):
                    model, attns = build_model(cls, self.quant_config)
                    model.load_weights(
                        [
                            (f"{ckpt_path}.{k_name}", torch.tensor(K_SCALE)),
                            (f"{ckpt_path}.{v_name}", torch.tensor(V_SCALE)),
                        ]
                    )
                    for attn in attns:
                        attn.quant_method.process_weights_after_loading(attn)
                        self.assertEqual(
                            (attn.k_scale_float, attn.v_scale_float),
                            (K_SCALE, V_SCALE),
                        )


class TestRemapKvScaleName(CustomTestCase):
    PARAMS = {
        "model.layers.0.self_attn.attn.k_scale": None,
        "model.layers.0.attention.attn.k_scale": None,
        "blocks.0.norm_attn_norm.attn.attn.v_scale": None,
    }

    def test_modelopt_names_under_any_attention_module(self):
        for name, expected in [
            (
                "model.layers.0.self_attn.k_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.attention.k_proj.k_scale",
                "model.layers.0.attention.attn.k_scale",
            ),
            (
                "blocks.0.norm_attn_norm.attn.v_proj.v_scale",
                "blocks.0.norm_attn_norm.attn.attn.v_scale",
            ),
        ]:
            with self.subTest(name=name):
                self.assertEqual(maybe_remap_kv_scale_name(name, self.PARAMS), expected)

    def test_compressed_tensors_and_own_names(self):
        for name in (
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.attn.k_scale",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    maybe_remap_kv_scale_name(name, self.PARAMS),
                    "model.layers.0.self_attn.attn.k_scale",
                )

    def test_unknown_target_is_skipped(self):
        self.assertIsNone(
            maybe_remap_kv_scale_name("model.layers.0.mlp.k_proj.k_scale", self.PARAMS)
        )


if __name__ == "__main__":
    unittest.main()
