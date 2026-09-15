"""Legacy checkpoint tensors must populate each family's registered FFN.

These fixtures exercise production loaders and real parameter storage; they do
not construct full models or substitute the loaders with mocks.
"""

import importlib
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _model_load_weights_direct,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    map_state_dict_names,
)
from sglang.srt.models import qwen3
from sglang.srt.models.interns1 import InternS1ForConditionalGeneration
from sglang.srt.models.internvl import InternVLChatModel
from sglang.srt.models.minicpmv import MiniCPMBaseModel, MiniCPMV
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.models.qwen3_classification import Qwen3ForPooledOutput
from sglang.srt.models.qwen3_embedding import Qwen3Model as Qwen3Embedding
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _packed_loader(param, tensor, shard_id):
    default_weight_loader(param.chunk(2, dim=0)[shard_id], tensor)


def _fixture(module_name, class_name, member="ffn"):
    cls = getattr(
        importlib.import_module(f"sglang.srt.models.{module_name}"), class_name
    )
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        tie_word_embeddings=False,
        num_hidden_layers=1,
        hidden_size=4,
        intermediate_size=6,
        num_experts=2,
        n_routed_experts=2,
        num_local_experts=2,
        first_k_dense_replace=1,
    )
    model.quant_config = None
    model.num_fused_shared_experts = 0
    model.pp_group = SimpleNamespace(
        is_first_rank=True, is_last_rank=True, world_size=1
    )
    model.model = nn.Module()
    model.model.start_layer, model.model.end_layer = 0, 1
    for name, value in (("start_layer", 0), ("end_layer", 1)):
        if not isinstance(getattr(cls, name, None), property):
            setattr(model, name, value)
    module = importlib.import_module(f"sglang.srt.models.{module_name}")
    layer = nn.Module()
    ffn = nn.Module()
    ffn.gate_up_proj = nn.Linear(4, 12, bias=False)
    ffn.gate_up_proj.weight.weight_loader = _packed_loader
    ffn.down_proj = nn.Linear(6, 4, bias=False)
    if module_name == "exaone":
        ffn.c_proj = ffn.down_proj
        del ffn.down_proj
    layer.add_module(member, ffn)
    model.model.layers = nn.ModuleList([layer])
    return model, ffn


class TestFFNModelLoaders(unittest.TestCase):
    def test_shared_experts_load_into_fused_expert_slot(self):
        from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
            DeepseekV2WeightLoaderMixin,
        )

        cases = [
            ("glm4_moe", "Glm4MoeForCausalLM"),
            ("glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
            ("glm5_next", "Glm5NextForConditionalGeneration"),
            ("bailing_moe_v3", "BailingMoeV3ForCausalLM"),
            ("minimax_m3", "MiniMaxM3SparseForCausalLM"),
            ("qwen3_next", "Qwen3NextForCausalLM"),
        ]
        for module, cls in cases:
            with self.subTest(model=module), ExitStack() as stack:
                model, ffn = _fixture(module, cls)
                model.num_fused_shared_experts = 1
                model.enable_shared_expert_fusion = True
                model.config.num_nextn_predict_layers = 0
                ffn.experts = _experts(3)
                # Attention post-processing needs a full attention model; the
                # production expert name routing and tensor writes run here.
                if hasattr(model, "post_load_weights"):
                    stack.enter_context(patch.object(model, "post_load_weights"))
                stack.enter_context(
                    patch.object(DeepseekV2WeightLoaderMixin, "post_load_weights")
                )
                stack.enter_context(torch.no_grad())
                shared = "shared_expert" if module == "qwen3_next" else "shared_experts"
                prefix = f"model.layers.0.mlp.{shared}"
                gate = torch.arange(24.0).reshape(6, 4)
                up, down = gate + 30, gate.T.contiguous() + 60
                model.load_weights(
                    [
                        (f"{prefix}.gate_proj.weight", gate),
                        (f"{prefix}.up_proj.weight", up),
                        (f"{prefix}.down_proj.weight", down),
                    ]
                )
                torch.testing.assert_close(
                    ffn.experts.w13_weight[2], torch.cat([gate, up])
                )
                torch.testing.assert_close(ffn.experts.w2_weight[2], down)
                self.assertEqual(ffn.experts.w13_weight[:2].count_nonzero(), 0)
                self.assertEqual(ffn.experts.w2_weight[:2].count_nonzero(), 0)

    def test_mllama4_loads_stacked_expert_and_default_weights(self):
        from sglang.srt.models import mllama4
        from sglang.srt.models.llama4 import Llama4DecoderLayer

        model = _empty(mllama4.Llama4ForConditionalGeneration)
        model.has_vision = False
        model.config = SimpleNamespace(num_local_experts=2)
        model.language_model = nn.Module()
        model.language_model.model = nn.Module()
        layer = _empty(Llama4DecoderLayer)
        layer.ffn = _ffn()
        del layer.ffn.gate_up_proj.weight_scale
        layer.ffn.experts = nn.Module()
        layer.ffn.experts.w2_scale = nn.Parameter(torch.zeros(2, 1))
        model.language_model.model.layers = nn.ModuleList([layer])
        prefix = "model.layers.0.feed_forward"
        weights = [
            (f"{prefix}.gate_proj.weight", torch.full((6, 4), 2.0)),
            (f"{prefix}.up_proj.weight", torch.full((6, 4), 3.0)),
            (f"{prefix}.down_proj.weight", torch.full((4, 6), 4.0)),
            (f"{prefix}.experts.down_proj_scale", torch.tensor([5.0])),
        ]
        with (
            torch.no_grad(),
        ):
            model.load_weights(weights)
        torch.testing.assert_close(
            layer.ffn.gate_up_proj.weight, torch.cat([weights[0][1], weights[1][1]])
        )
        torch.testing.assert_close(layer.ffn.down_proj.weight, weights[2][1])
        torch.testing.assert_close(layer.ffn.experts.w2_scale, torch.full((2, 1), 5.0))

    def test_inkling_loads_weights_through_its_helpers(self):
        from sglang.srt.models import inkling

        model = _empty(inkling.InklingForConditionalGeneration)
        model.text_config = SimpleNamespace(inference_moe_w13_interleaved=False)
        model.llm = nn.Module()
        layer = _empty(inkling.InklingDecoderLayer)
        layer.ffn = _ffn()
        del layer.ffn.gate_up_proj.weight_scale
        layer.ffn.experts = nn.Module()
        layer.ffn.experts.w2_weight = nn.Parameter(torch.zeros(2, 4, 6))
        layer.ffn.experts.w2_weight_scale = nn.Parameter(torch.zeros(2, 4, 1))
        model.llm.layers = nn.ModuleList([layer])
        prefix = "llm.layers.0.mlp"
        weights = [
            (f"{prefix}.gate_proj.weight", torch.full((6, 4), 2.0)),
            (f"{prefix}.up_proj.weight", torch.full((6, 4), 3.0)),
            (f"{prefix}.down_proj.weight", torch.full((4, 6), 4.0)),
            (f"{prefix}.experts.w2_weight", torch.full((2, 4, 6), 5.0)),
            (f"{prefix}.experts.w2_weight.scale", torch.full((2, 4, 1), 6.0)),
            (f"{prefix}.experts.1.down_proj.weight", torch.full((4, 6), 7.0)),
        ]
        with (
            patch.object(
                inkling, "get_parallel", return_value=SimpleNamespace(moe_ep_size=1)
            ),
            torch.no_grad(),
        ):
            loaded = model.load_weights(weights)
        self.assertEqual(loaded, set(dict(model.named_parameters())))
        torch.testing.assert_close(
            layer.ffn.gate_up_proj.weight, torch.cat([weights[0][1], weights[1][1]])
        )
        torch.testing.assert_close(layer.ffn.down_proj.weight, weights[2][1])
        torch.testing.assert_close(layer.ffn.experts.w2_weight[0], weights[3][1][0])
        torch.testing.assert_close(layer.ffn.experts.w2_weight[1], weights[5][1])
        torch.testing.assert_close(layer.ffn.experts.w2_weight_scale, weights[4][1])

    def test_kimi_wrapper_maps_dense_and_expert_checkpoint_names(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        for checkpoint_member in ("mlp", "ffn", "block_sparse_moe"):
            with self.subTest(member=checkpoint_member):
                model, ffn = _fixture("kimi_k3", "KimiK3LinearForCausalLM")
                model.config.linear_attn_config = None
                model.config.is_moe = True
                model.config.is_linear_attn = False
                ffn.experts = _experts(2)
                wrapper = _empty(KimiK3ForConditionalGeneration)
                wrapper.config = SimpleNamespace(language_only=True)
                wrapper.language_model = model
                prefix = f"language_model.layers.0.{checkpoint_member}"
                gate = torch.arange(24.0).reshape(6, 4)
                up, down = gate + 30, gate.T.contiguous() + 60
                with torch.no_grad(), patch.object(model, "post_load_weights"):
                    wrapper.load_weights(
                        [
                            (f"{prefix}.gate_proj.weight", gate),
                            (f"{prefix}.up_proj.weight", up),
                            (f"{prefix}.down_proj.weight", down),
                            (f"{prefix}.experts.1.w1.weight", gate),
                            (f"{prefix}.experts.1.w3.weight", up),
                            (f"{prefix}.experts.1.w2.weight", down),
                        ]
                    )
                torch.testing.assert_close(
                    ffn.gate_up_proj.weight, torch.cat([gate, up])
                )
                torch.testing.assert_close(ffn.down_proj.weight, down)
                torch.testing.assert_close(
                    ffn.experts.w13_weight[1], torch.cat([gate, up])
                )
                torch.testing.assert_close(ffn.experts.w2_weight[1], down)
                self.assertEqual(ffn.experts.w13_weight[0].count_nonzero(), 0)

    def test_gated_dense_family_loaders(self):
        cases = [
            ("llama", "LlamaForCausalLM"),
            ("qwen2", "Qwen2ForCausalLM"),
            ("gemma", "GemmaForCausalLM"),
            ("gemma2", "Gemma2ForCausalLM"),
            ("gemma3_causal", "Gemma3ForCausalLM"),
            ("gemma4_causal", "Gemma4ForCausalLM"),
            ("glm4", "Glm4ForCausalLM"),
            ("baichuan", "BaichuanForCausalLM"),
            ("xverse", "XverseForCausalLM"),
            ("nanbeige", "NanbeigeForCausalLM"),
            ("olmo", "OlmoForCausalLM"),
            ("olmo2", "Olmo2ForCausalLM"),
            ("exaone", "ExaoneForCausalLM"),
            ("exaone4", "Exaone4ForCausalLM"),
            ("solar", "SolarForCausalLM"),
            ("stablelm", "StableLmForCausalLM"),
            ("commandr", "CohereForCausalLM"),
        ]
        for module, cls in cases:
            with self.subTest(module=module):
                model, ffn = _fixture(module, cls)
                pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
                for old_name, value in (("mlp", 2.0), ("ffn", 5.0)):
                    weights = [
                        (
                            f"model.layers.0.{old_name}.gate_proj.weight",
                            torch.full((6, 4), value),
                        ),
                        (
                            f"model.layers.0.{old_name}.up_proj.weight",
                            torch.full((6, 4), value + 1),
                        ),
                        (
                            f"model.layers.0.{old_name}.down_proj.weight",
                            torch.full((4, 6), value + 2),
                        ),
                    ]
                    if module == "exaone":
                        weights = [
                            (
                                name.replace("gate_proj", "c_fc_0")
                                .replace("up_proj", "c_fc_1")
                                .replace("down_proj", "c_proj"),
                                value,
                            )
                            for name, value in weights
                        ]
                    with torch.no_grad():
                        loaded = model.load_weights(weights)
                    self.assertTrue(torch.all(ffn.gate_up_proj.weight[:6] == value))
                    self.assertTrue(torch.all(ffn.gate_up_proj.weight[6:] == value + 1))
                    down_proj = ffn.c_proj if module == "exaone" else ffn.down_proj
                    self.assertTrue(torch.all(down_proj.weight == value + 2))
                    if isinstance(loaded, set):
                        self.assertEqual(loaded, set(pointers))
                self.assertEqual(
                    pointers, {n: p.data_ptr() for n, p in model.named_parameters()}
                )

    def test_ungated_and_reverse_lookup_loaders(self):
        for module, cls, up_name, down_name in (
            ("apertus", "ApertusForCausalLM", "up_proj", "down_proj"),
            ("arcee", "ArceeForCausalLM", "up_proj", "down_proj"),
            ("phi", "PhiForCausalLM", "fc1", "fc2"),
        ):
            with self.subTest(module=module):
                model, ffn = _fixture(module, cls)
                del ffn.gate_up_proj, ffn.down_proj
                up, down = nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)
                ffn.add_module(up_name, up)
                ffn.add_module(down_name, down)
                model.stacked_params_mapping = [
                    (".qkv_proj", ".q_proj", "q"),
                    (".qkv_proj", ".k_proj", "k"),
                    (".qkv_proj", ".v_proj", "v"),
                ]
                with torch.no_grad():
                    model.load_weights(
                        [
                            (
                                f"model.layers.0.mlp.{up_name}.weight",
                                torch.full((6, 4), 2.0),
                            ),
                            (
                                f"model.layers.0.mlp.{down_name}.weight",
                                torch.full((4, 6), 3.0),
                            ),
                        ]
                    )
                self.assertTrue(torch.all(up.weight == 2))
                self.assertTrue(torch.all(down.weight == 3))

    def test_moe_expert_shards_use_ffn_registration(self):
        for module, cls in (
            ("qwen2_moe", "Qwen2MoeForCausalLM"),
            ("qwen3_moe", "Qwen3MoeForCausalLM"),
        ):
            with self.subTest(module=module):
                model, ffn = _fixture(module, cls)
                del ffn.gate_up_proj, ffn.down_proj
                ffn.experts = nn.Module()
                ffn.experts.w13_weight = nn.Parameter(torch.zeros(2, 12, 4))
                ffn.experts.w2_weight = nn.Parameter(torch.zeros(2, 4, 6))

                def load_expert(param, value, name, *, shard_id, expert_id):
                    target = param[expert_id]
                    if shard_id in ("w1", "w3"):
                        target = target.chunk(2, dim=0)[shard_id == "w3"]
                    default_weight_loader(target, value)

                for param in ffn.experts.parameters():
                    param.weight_loader = load_expert
                weights = []
                for expert in range(2):
                    for proj, shape, value in (
                        ("gate_proj", (6, 4), 1.0 + expert),
                        ("up_proj", (6, 4), 3.0 + expert),
                        ("down_proj", (4, 6), 5.0 + expert),
                    ):
                        weights.append(
                            (
                                f"model.layers.0.mlp.experts.{expert}.{proj}.weight",
                                torch.full(shape, value),
                            )
                        )
                with torch.no_grad():
                    model.load_weights(weights)
                for expert in range(2):
                    self.assertTrue(
                        torch.all(ffn.experts.w13_weight[expert, :6] == 1 + expert)
                    )
                    self.assertTrue(
                        torch.all(ffn.experts.w13_weight[expert, 6:] == 3 + expert)
                    )
                    self.assertTrue(
                        torch.all(ffn.experts.w2_weight[expert] == 5 + expert)
                    )

    def test_state_dict_metadata_and_projector_are_preserved(self):
        model = nn.Module()
        model.ffn = nn.Linear(3, 2)
        model.projector = nn.Module()
        model.projector.mlp = nn.Linear(2, 1)
        old = nn.Module()
        old.mlp = nn.Linear(3, 2)
        old.projector = model.projector
        state = old.state_dict()
        mapped = map_state_dict_names(
            state,
            lambda name: (
                name.replace("mlp", "ffn", 1)
                if name == "mlp" or name.startswith("mlp.")
                else name
            ),
        )
        self.assertEqual(set(mapped), set(model.state_dict()))
        self.assertIn("ffn", mapped._metadata)
        self.assertIn("projector.mlp", mapped._metadata)
        model.load_state_dict(mapped, strict=True)
        self.assertTrue(torch.equal(model.ffn.weight, old.mlp.weight))


def _empty(cls=nn.Module):
    obj = cls.__new__(cls)
    if isinstance(obj, nn.Module):
        nn.Module.__init__(obj)
    return obj


def _ffn():
    ffn = nn.Module()
    ffn.gate_up_proj = nn.Linear(4, 12, bias=False)
    ffn.gate_up_proj.weight.weight_loader = _packed_loader
    ffn.gate_up_proj.weight_scale = nn.Parameter(torch.zeros(2))
    ffn.gate_up_proj.weight_scale.weight_loader = _packed_loader
    ffn.down_proj = nn.Linear(6, 4, bias=False)
    return ffn


def _experts(num_experts):
    def load(param, tensor, name, shard_id, expert_id):
        target = param[expert_id]
        if shard_id in ("w1", "w3"):
            target = target.chunk(2, dim=0)[0 if shard_id == "w1" else 1]
        default_weight_loader(target, tensor)

    experts = nn.Module()
    experts.w13_weight = nn.Parameter(torch.zeros(num_experts, 12, 4))
    experts.w2_weight = nn.Parameter(torch.zeros(num_experts, 4, 6))
    for param in experts.parameters():
        param.weight_loader = load
    return experts


def _backbone(dense=True):
    backbone = _empty(qwen3.Qwen3Model if dense else nn.Module)
    layer = _empty(qwen3.Qwen3DecoderLayer) if dense else nn.Module()
    layer.add_module("ffn" if dense else "mlp", _ffn())
    backbone.layers = nn.ModuleList([layer])
    backbone.start_layer, backbone.end_layer = 0, 1
    return backbone


def _qwen3_fixture(kind, dense=True):
    backbone = _backbone(dense)
    lm = _empty(qwen3.Qwen3ForCausalLM if dense else nn.Module)
    lm.model = backbone
    config = SimpleNamespace(tie_word_embeddings=False)
    classes = {
        "causal": qwen3.Qwen3ForCausalLM,
        "embedding": Qwen3Embedding,
        "classification": Qwen3ForPooledOutput,
        "vl": Qwen3VLForConditionalGeneration,
        "asr": Qwen3ASRForConditionalGeneration,
        "internvl": InternVLChatModel,
        "interns1": InternS1ForConditionalGeneration,
        "minicpmv": MiniCPMV,
    }
    model = _empty(classes[kind])
    model.config = config
    model.pp_group = SimpleNamespace(is_last_rank=True)
    checkpoint_prefix = "model"
    if kind in ("causal", "embedding", "classification", "vl"):
        model.model = backbone
        if kind == "vl":
            checkpoint_prefix = "model.language_model"
    elif kind in ("asr", "internvl", "interns1"):
        model.language_model = lm
        if kind == "asr":
            config.thinker_config = SimpleNamespace(text_config=config)
            checkpoint_prefix = "thinker.model"
        else:
            text_config = SimpleNamespace(
                architectures=["Qwen3ForCausalLM" if dense else "Qwen2ForCausalLM"]
            )
            config.text_config = config.llm_config = text_config
            checkpoint_prefix = (
                "model.language_model" if kind == "interns1" else "language_model.model"
            )
    else:
        model.minicpmv = _empty(MiniCPMBaseModel)
        model.minicpmv.llm = lm
        checkpoint_prefix = "llm.model"
    return model, backbone, checkpoint_prefix


class TestMultimodalFFNLoading(unittest.TestCase):
    def test_qwen3_multimodal_fused_expert_checkpoints(self):
        for module, cls in (
            ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
            ("qwen3_omni_moe", "Qwen3OmniMoeForConditionalGeneration"),
        ):
            with self.subTest(model=module):
                model, ffn = _fixture(module, cls)
                model.config.encoder_only = False
                prefix = "model.language_model.layers.0.mlp.experts"
                if module == "qwen3_omni_moe":
                    model.enable_talker = False
                    model.thinker = nn.Module()
                    model.thinker.model = model.model
                    del model.model
                    prefix = "thinker.model.layers.0.mlp.experts"
                ffn.experts = _experts(2)
                gate_up = torch.arange(96.0).reshape(2, 4, 12)
                down = torch.arange(48.0).reshape(2, 6, 4) + 100
                with torch.no_grad():
                    model.load_weights(
                        [
                            (f"{prefix}.gate_up_proj", gate_up),
                            (f"{prefix}.down_proj", down),
                        ]
                    )
                torch.testing.assert_close(
                    ffn.experts.w13_weight, gate_up.transpose(-1, -2)
                )
                torch.testing.assert_close(
                    ffn.experts.w2_weight, down.transpose(-1, -2)
                )

    def test_llava_keeps_transformers_vision_mlp(self):
        from transformers import (
            CLIPVisionConfig,
            CLIPVisionModel,
            SiglipVisionConfig,
            SiglipVisionModel,
        )

        from sglang.srt.models import llava

        for name, config_cls, vision_cls in (
            ("clip", CLIPVisionConfig, CLIPVisionModel),
            ("siglip", SiglipVisionConfig, SiglipVisionModel),
        ):
            with self.subTest(vision=name):
                model = _empty(llava.LlavaBaseForCausalLM)
                model.config = SimpleNamespace(
                    mm_vision_tower=name,
                    mm_vision_select_layer=-1,
                    mm_vision_select_feature="patch",
                )
                model.language_model, ffn = _fixture("qwen3", "Qwen3ForCausalLM")
                vision = vision_cls(
                    config_cls(
                        hidden_size=8,
                        intermediate_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        image_size=8,
                        patch_size=4,
                    )
                )
                param = dict(vision.named_parameters())[
                    "encoder.layers.0.mlp.fc1.weight"
                ]
                value = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param)
                down = torch.full((4, 6), 5.0)
                with (
                    patch.object(vision_cls, "from_pretrained", return_value=vision),
                    torch.no_grad(),
                ):
                    model.load_weights(
                        [
                            (
                                "model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
                                value,
                            ),
                            ("model.layers.0.mlp.down_proj.weight", down),
                        ]
                    )
                torch.testing.assert_close(param, value)
                torch.testing.assert_close(ffn.down_proj.weight, down)

    def test_mimo_v2_vision_blocks_and_dense_merger(self):
        model, ffn = _fixture("mimo_v2", "MiMoV2ForCausalLM")
        model.config.encoder_only = False
        model._is_multimodal = True
        model.visual = nn.Module()
        block = nn.Module()
        block.ffn = _ffn()
        model.visual.blocks = nn.ModuleList([block])
        model.visual.merger = nn.Module()
        model.visual.merger.mlp = nn.Sequential(nn.Linear(4, 4))
        gate = torch.arange(24.0).reshape(6, 4)
        merger = torch.arange(16.0).reshape(4, 4)
        with torch.no_grad():
            model.load_weights(
                [
                    ("vision_model.visual.blocks.0.mlp.gate_proj.weight", gate),
                    ("vision_model.visual.blocks.0.mlp.up_proj.weight", gate + 30),
                    ("vision_model.visual.merger.mlp.0.weight", merger),
                    ("model.layers.0.mlp.down_proj.weight", gate.T.contiguous()),
                ]
            )
        torch.testing.assert_close(
            block.ffn.gate_up_proj.weight, torch.cat([gate, gate + 30])
        )
        torch.testing.assert_close(model.visual.merger.mlp[0].weight, merger)
        torch.testing.assert_close(ffn.down_proj.weight, gate.T)

    def test_shared_loaders_assign_legacy_weights_and_scales(self):
        for kind in (
            "causal",
            "embedding",
            "classification",
            "vl",
            "asr",
            "internvl",
            "interns1",
            "minicpmv",
        ):
            with self.subTest(loader=kind):
                model, backbone, prefix = _qwen3_fixture(kind)
                ffn = backbone.layers[0].ffn
                gate = torch.arange(24.0).reshape(6, 4)
                up = gate + 30
                down = torch.arange(24.0).reshape(4, 6) + 60
                model.load_weights(
                    [
                        (f"{prefix}.layers.0.mlp.gate_proj.weight", gate),
                        (f"{prefix}.layers.0.mlp.up_proj.weight", up),
                        (f"{prefix}.layers.0.mlp.down_proj.weight", down),
                        (
                            f"{prefix}.layers.0.mlp.gate_proj.weight_scale",
                            torch.tensor([2.0]),
                        ),
                        (
                            f"{prefix}.layers.0.mlp.up_proj.weight_scale",
                            torch.tensor([3.0]),
                        ),
                    ]
                )
                torch.testing.assert_close(
                    ffn.gate_up_proj.weight, torch.cat((gate, up))
                )
                torch.testing.assert_close(ffn.down_proj.weight, down)
                torch.testing.assert_close(
                    ffn.gate_up_proj.weight_scale, torch.tensor([2.0, 3.0])
                )
                # Runtime updates through load_weights also accept renamed paths.
                model.load_weights(
                    [(f"{prefix}.layers.0.ffn.down_proj.weight", down + 1)]
                )
                torch.testing.assert_close(ffn.down_proj.weight, down + 1)

    def test_vision_merger_keeps_its_dense_mlp_name(self):
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

        model = _empty(Qwen3VLMoeVisionModel)
        block = nn.Module()
        block.ffn = nn.Linear(2, 2, bias=False)
        model.blocks = nn.ModuleList([block])
        model.merger = nn.Module()
        model.merger.mlp = nn.Linear(2, 2, bias=False)
        loaded = model.load_weights(
            [
                ("blocks.0.mlp.weight", torch.full((2, 2), 3.0)),
                ("merger.mlp.weight", torch.full((2, 2), 4.0)),
            ]
        )
        self.assertEqual(loaded, {"blocks.0.ffn.weight", "merger.mlp.weight"})
        torch.testing.assert_close(block.ffn.weight, torch.full((2, 2), 3.0))
        torch.testing.assert_close(model.merger.mlp.weight, torch.full((2, 2), 4.0))

    def test_bare_checkpoint_and_direct_internal_update(self):
        for kind in ("causal", "embedding"):
            with self.subTest(loader=kind):
                model, backbone, _ = _qwen3_fixture(kind)
                value = torch.full((4, 6), 11.0)
                model.load_weights([("layers.0.mlp.down_proj.weight", value)])
                param = backbone.layers[0].ffn.down_proj.weight
                torch.testing.assert_close(param, value)
                pointer = param.data_ptr()
                _model_load_weights_direct(
                    model, [("model.layers.0.ffn.down_proj.weight", value + 1)]
                )
                self.assertEqual(param.data_ptr(), pointer)
                torch.testing.assert_close(param, value + 1)


class TestFFNCheckpointQuantization(unittest.TestCase):
    def _load_config(self, model_class, checkpoint_config, method="modelopt_fp4"):
        from transformers import PretrainedConfig

        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader import loader

        config = ModelConfig.__new__(ModelConfig)
        config.hf_config = PretrainedConfig(quantization_config=checkpoint_config)
        config.quantization = method
        config.dtype = torch.bfloat16
        config.is_draft_model = False
        with (
            patch.object(
                loader, "get_model_architecture", return_value=(model_class, "")
            ),
            patch.object(loader, "get_device_capability", return_value=(None, None)),
        ):
            return loader._get_quantization_config(config, LoadConfig())

    def _assert_unquantized(self, config, prefix):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        layer = LinearBase.__new__(LinearBase)
        nn.Module.__init__(layer)
        self.assertIsInstance(
            config.get_quant_method(layer, prefix), UnquantizedLinearMethod
        )

    def test_modelopt_checkpoint_exclusions_before_model_construction(self):
        from sglang.srt.models.glm4_moe import (
            GlmMoeDsaForCausalLM,
            GlmMoeDsaForCausalLMNextN,
        )
        from sglang.srt.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        )
        from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
        from sglang.srt.models.qwen3_5_text import (
            Qwen3_5ForCausalLM,
            Qwen3_5MoeForCausalLM,
        )

        for cls in (
            GlmMoeDsaForCausalLM,
            GlmMoeDsaForCausalLMNextN,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
            Qwen3_5ForCausalLM,
            Qwen3_5MoeForCausalLM,
            Qwen3_5ForCausalLMMTP,
        ):
            with self.subTest(model=cls.__name__):
                ignored = [
                    "model.layers.10.mlp.shared_experts*",
                    "*.mlp.shared_expert.*",
                    "lm_head",
                ]
                config = self._load_config(
                    cls,
                    {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                        "ignore": ignored,
                    },
                )
                self._assert_unquantized(
                    config, "model.layers.10.ffn.shared_experts.gate_up_proj"
                )
                self._assert_unquantized(
                    config, "model.layers.0.ffn.shared_expert.down_proj"
                )
                self.assertFalse(
                    config.is_layer_excluded(
                        "model.layers.10.ffn.experts.0.gate_up_proj"
                    )
                )
                self.assertEqual(ignored[0], "model.layers.10.mlp.shared_experts*")
                self.assertIn("lm_head", config.exclude_modules)

    def test_kimi_checkpoint_regex_keeps_dense_ffn_unquantized(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        ignored = [r"re:.*mlp\.(gate|up|gate_up|down)_proj.*", r"re:.*vision_tower.*"]
        config = self._load_config(
            KimiK3ForConditionalGeneration,
            {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "config_groups": {},
                "ignore": ignored,
            },
            method="compressed-tensors",
        )
        self._assert_unquantized(config, "model.layers.0.ffn.gate_up_proj")
        self.assertEqual(config.ignore[0], r"re:.*ffn\.(gate|up|gate_up|down)_proj.*")
        self.assertEqual(config.ignore[1], ignored[1])
        self.assertEqual(ignored[0], r"re:.*mlp\.(gate|up|gate_up|down)_proj.*")

    def test_inkling_checkpoint_exclusions_keep_bf16_experts(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.models.inkling_common.quantization.config import (
            InklingModelOptNvfp4Config,
        )

        ignored = [
            "model.llm.layers.0.mlp.w13_dn",
            "model.llm.layers.0.mlp.w2_md",
            "model.llm.layers.2.mlp.experts",
            "model.llm.layers.2.mlp_norm",
            "model.llm.layers.2.mlp_sconv",
            "model.visual.mlp",
            "model.unembed",
        ]
        config = InklingModelOptNvfp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "exclude_modules": ignored,
                }
            }
        )
        self._assert_unquantized(config, "llm.layers.0.ffn.gate_up_proj")
        self._assert_unquantized(config, "llm.layers.0.ffn.down_proj")
        experts = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(experts)
        experts.is_shared_fused_moe = False
        self.assertIsInstance(
            config.get_quant_method(experts, "llm.layers.2.ffn.experts"),
            UnquantizedFusedMoEMethod,
        )
        self.assertFalse(config.exclude_layer("llm.layers.3.ffn.experts"))
        for name in (
            "llm.layers.2.ffn_norm",
            "llm.layers.2.ffn_sconv",
            "model.visual.mlp",
            "lm_head",
        ):
            self.assertIn(name, config.exclude_modules)
        self.assertEqual(ignored[2], "model.llm.layers.2.mlp.experts")


if __name__ == "__main__":
    unittest.main()
