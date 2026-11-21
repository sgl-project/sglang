"""Configuration for Hunyuan-4-Flash (``model_type = "hy_v4_flash"``).

The checkpoint is a VL wrapper: the language model fields and the FP8
quantization config live under ``text_config``. The text fields are mirrored to
the top level so the engine-wide helpers that read ``hf_config`` directly (DSA
detection, MLA geometry, quantization) see them.
"""

from copy import deepcopy
from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.hy_v4 import HYV4VLVisionConfig
from sglang.srt.configs.hy_v4_vl_processing import HYV4VLProcessor
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)

SPARSE_ATTENTION = "deepseek_sparse_attention"
SLIDING_ATTENTION = "sliding_attention"
_SPARSE_ALIASES = {
    "sparse",
    "sparse_attention",
    "indexed_attention",
    "deepseek_sparse_attention",
}

_TOP_LEVEL_TEXT_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "routed_scaling_factor",
    "norm_topk_prob",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "qk_head_dim",
    "head_dim",
    "v_head_dim",
    "index_topk",
    "index_head_dim",
    "index_n_heads",
    "layer_types",
    "mlp_layer_types",
    "indexer_types",
    "rope_parameters",
    "rope_theta",
    "rope_interleave",
    "indexer_rope_interleave",
    "first_k_dense_replace",
    "moe_layer_freq",
    "scoring_func",
    "topk_method",
    "n_group",
    "topk_group",
    "num_nextn_predict_layers",
    "enable_lm_head_fp32",
    "router_fp32",
    "swiglu_limit",
    "sliding_window",
    "hc_mult",
    "tie_word_embeddings",
    "learnable_sink",
    "gated_mla",
    "gating_type",
    "enable_ihc",
    "engram_layer_ids",
)


def _normalize_ignored_module(name: str) -> list[str]:
    name = name.removeprefix("language_model.model.")
    name = name.replace(".compressed_indexer.", ".indexer.")
    names = [name]
    if name.endswith((".self_attn.q_a_proj", ".self_attn.kv_a_proj_with_mqa")):
        names.append(
            name.rsplit(".", 1)[0] + ".fused_qkv_a_proj_with_mqa",
        )
    if name.endswith((".mlp.gate_proj", ".mlp.up_proj")):
        names.append(name.rsplit(".", 1)[0] + ".gate_up_proj")
    if name.endswith((".shared_experts.gate_proj", ".shared_experts.up_proj")):
        names.append(name.rsplit(".", 1)[0] + ".gate_up_proj")
    return names


def _normalize_quantization_config(quant: Optional[dict]) -> Optional[dict]:
    if not quant:
        return quant
    quant = dict(quant)
    ignored = quant.get("modules_to_not_convert")
    if ignored:
        normalized = []
        for entry in ignored:
            if entry.startswith("vit."):
                continue
            normalized.extend(_normalize_ignored_module(entry))
        quant["modules_to_not_convert"] = list(dict.fromkeys(normalized))
    return quant


class HYV4FlashTextConfig(PretrainedConfig):
    model_type = "hy_v4_flash_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 3072,
        intermediate_size: int = 13824,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-5,
        n_routed_experts: int = 512,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 16,
        routed_scaling_factor: float = 3.998,
        norm_topk_prob: bool = True,
        q_lora_rank: int = 1024,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        mlp_layer_types: Optional[list] = None,
        layer_types: Optional[list] = None,
        indexer_types: Optional[list] = None,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 16,
        enable_lm_head_fp32: bool = True,
        enable_ihc: bool = True,
        hc_mult: int = 4,
        hc_magnitude: float = 2.0,
        hc_eps: float = 1e-6,
        gated_mla: bool = True,
        gating_type: str = "elementwise",
        learnable_sink: bool = True,
        swiglu_limit: float = 10.0,
        rope_parameters: Optional[dict] = None,
        num_nextn_predict_layers: int = 0,
        routed_expert_hidden_size: Optional[int] = None,
        sandwich_norm: bool = True,
        dsa_skip_attn_post_norm: bool = False,
        swa_num_attention_heads: Optional[int] = None,
        swa_qk_nope_head_dim: Optional[int] = None,
        swa_qk_rope_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        sliding_window: int = 512,
        use_compressed_indexer: bool = True,
        compressed_indexer_compress_ratio: int = 4,
        compressed_indexer_num_attention_heads: int = 16,
        compressed_indexer_head_dim: int = 128,
        compressed_indexer_group_topk: int = 512,
        compressed_indexer_candidate_topk: int = 2048,
        compressed_indexer_reuse_fine_indexer: bool = False,
        compressed_indexer_skip_fine_select: bool = True,
        compressed_indexer_key_pooling: str = "gated",
        engram_layer_ids: Optional[list] = None,
        engram_max_ngram_size: int = 3,
        engram_n_embed_per_ngram: int = 768,
        engram_n_head_per_ngram: int = 8,
        engram_pad_id: int = 120002,
        engram_seed: int = 0,
        engram_fp8_embedding: bool = False,
        engram_layer_head_vocab_sizes: Optional[dict] = None,
        quantization_config: Optional[dict] = None,
        **kwargs,
    ):
        kwargs.pop("engram_enable", None)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.enable_ihc = enable_ihc
        self.hc_mult = hc_mult
        self.hc_magnitude = hc_magnitude
        self.hc_eps = hc_eps
        self.gated_mla = gated_mla
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.swiglu_limit = swiglu_limit
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.sandwich_norm = sandwich_norm
        self.dsa_skip_attn_post_norm = dsa_skip_attn_post_norm
        self.swa_num_attention_heads = swa_num_attention_heads or num_attention_heads
        self.swa_qk_nope_head_dim = swa_qk_nope_head_dim or qk_nope_head_dim
        self.swa_qk_rope_head_dim = swa_qk_rope_head_dim or qk_rope_head_dim
        self.swa_v_head_dim = swa_v_head_dim or v_head_dim
        self.sliding_window = sliding_window
        self.use_compressed_indexer = use_compressed_indexer
        self.compressed_indexer_compress_ratio = compressed_indexer_compress_ratio
        self.compressed_indexer_num_attention_heads = (
            compressed_indexer_num_attention_heads
        )
        self.compressed_indexer_head_dim = compressed_indexer_head_dim
        self.compressed_indexer_group_topk = compressed_indexer_group_topk
        self.compressed_indexer_candidate_topk = compressed_indexer_candidate_topk
        self.compressed_indexer_reuse_fine_indexer = (
            compressed_indexer_reuse_fine_indexer
        )
        self.compressed_indexer_skip_fine_select = compressed_indexer_skip_fine_select
        self.compressed_indexer_key_pooling = compressed_indexer_key_pooling
        self.engram_layer_ids = list(engram_layer_ids or [])
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        self.engram_n_head_per_ngram = engram_n_head_per_ngram
        self.engram_pad_id = engram_pad_id
        self.engram_seed = engram_seed
        self.engram_fp8_embedding = engram_fp8_embedding
        self.engram_layer_head_vocab_sizes = engram_layer_head_vocab_sizes or {}
        if quantization_config is not None:
            self.quantization_config = _normalize_quantization_config(
                quantization_config
            )

        self.mlp_layer_types = (
            mlp_layer_types
            if mlp_layer_types is not None
            else ["dense"] + ["sparse"] * (num_hidden_layers - 1)
        )
        self.layer_types = [
            SPARSE_ATTENTION if kind in _SPARSE_ALIASES else kind
            for kind in (layer_types or [SPARSE_ATTENTION] * num_hidden_layers)
        ]
        self.indexer_types = indexer_types or ["full"] * num_hidden_layers
        # Per-layer-type rope; ``rope_parameters`` stays flat (the sparse layers'
        # values) for engine helpers that expect a single rope config.
        self.mrope_section, self.mrope_interleaved = self._mrope_settings(
            rope_parameters
        )
        self.layer_rope_parameters = self._normalize_rope(rope_parameters)
        self.rope_parameters = dict(self.rope_for_layer_type(SPARSE_ATTENTION))
        if self.mrope_section is not None:
            self.rope_parameters["mrope_section"] = list(self.mrope_section)
            self.rope_parameters["mrope_interleaved"] = self.mrope_interleaved
        self.rope_theta = self.rope_parameters["rope_theta"]
        self.rope_interleave = True
        self.indexer_rope_interleave = True
        self.router_fp32 = True
        self.first_k_dense_replace = 1
        self.moe_layer_freq = 1
        self.scoring_func = "sigmoid"
        self.topk_method = "noaux_tc"
        self.n_group = 1
        self.topk_group = 1
        self._validate()

    def _normalize_rope(self, rope: Optional[dict]) -> dict:
        rope = deepcopy(rope or {})
        for alias in _SPARSE_ALIASES - {SPARSE_ATTENTION}:
            if alias in rope:
                rope[SPARSE_ATTENTION] = rope.pop(alias)
        nested = {k: v for k, v in rope.items() if isinstance(v, dict)}
        if nested and set(nested) >= set(self.layer_types):
            pass
        elif "deepseek_sparse_attention" in nested or "sliding_attention" in nested:
            raise ValueError(f"rope_parameters miss a layer type: {rope}")
        else:
            flat = {"rope_theta": 10000.0, "rope_type": "default", **rope}
            nested = {kind: dict(flat) for kind in set(self.layer_types)}
        for kind, value in nested.items():
            value.setdefault("rope_type", "default")
            if value["rope_type"] != "default":
                raise ValueError(f"Unsupported rope_type for {kind}: {value}")
            # Text positions are equal on all MRoPE axes, which reduces the
            # interleaved MRoPE to plain GPT-J style rope.
            value.pop("mrope_section", None)
            value.pop("mrope_interleaved", None)
        return nested

    def _mrope_settings(self, rope: Optional[dict]):
        rope = rope or {}
        values = [rope, *(v for v in rope.values() if isinstance(v, dict))]
        sections = {tuple(v["mrope_section"]) for v in values if "mrope_section" in v}
        interleaved = {
            v["mrope_interleaved"] for v in values if "mrope_interleaved" in v
        }
        if len(sections) > 1 or len(interleaved) > 1:
            raise ValueError("MRoPE sections and interleaving must agree across layers")
        if not sections:
            return None, False
        section = next(iter(sections))
        if len(section) != 3 or 2 * sum(section) != self.qk_rope_head_dim:
            raise ValueError(f"mrope_section must cover the rope dimension: {section}")
        if False in interleaved:
            raise ValueError("Hy4-Flash supports only interleaved MRoPE")
        return section, True

    def rope_for_layer_type(self, layer_type: str) -> dict:
        return self.layer_rope_parameters[layer_type]

    def is_sliding_layer(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == SLIDING_ATTENTION

    def _validate(self):
        n = self.num_hidden_layers
        for name in ("mlp_layer_types", "layer_types", "indexer_types"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must contain {n} entries")
        if set(self.layer_types) - {SPARSE_ATTENTION, SLIDING_ATTENTION}:
            raise ValueError(f"Unsupported layer_types: {set(self.layer_types)}")
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("mlp_layer_types only supports dense and sparse")
        if not self.enable_ihc or self.hc_mult <= 0:
            raise ValueError("Hy4-Flash requires the iHC residual stream")
        if not self.gated_mla or self.gating_type != "elementwise":
            raise ValueError("Hy4-Flash requires elementwise gated MLA")
        if not self.learnable_sink:
            raise ValueError("Hy4-Flash requires learnable attention sinks")
        if (
            not self.use_compressed_indexer
            or not self.compressed_indexer_skip_fine_select
            or self.compressed_indexer_reuse_fine_indexer
            or self.compressed_indexer_key_pooling != "gated"
        ):
            raise ValueError("Hy4-Flash supports the gated coarse-only IndexPool")
        ratio = self.compressed_indexer_compress_ratio
        if self.index_topk != self.compressed_indexer_group_topk * ratio:
            raise ValueError("IndexPool requires index_topk = group_topk * ratio")
        if self.compressed_indexer_head_dim != 128 or self.index_head_dim != 128:
            raise ValueError("IndexPool requires 128-dimensional keys")
        if self.swa_qk_rope_head_dim != self.qk_rope_head_dim:
            raise ValueError("SWA and DSA layers must share the rope dimension")
        for layer in self.engram_layer_ids:
            if not 1 <= layer <= n:
                raise ValueError("engram_layer_ids are 1-based decoder layer numbers")


@register_customized_processor(HYV4VLProcessor)
class HYV4FlashConfig(PretrainedConfig):
    model_type = "hy_v4_flash"
    sub_configs = {
        "text_config": HYV4FlashTextConfig,
        "vision_config": HYV4VLVisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Any = None,
        vision_config: Optional[dict] = None,
        image_start_token_id: int = 120118,
        image_end_token_id: int = 120119,
        image_token_id: int = 120120,
        image_newline_token_id: int = 120121,
        video_start_token_id: int = 120122,
        video_end_token_id: int = 120123,
        video_token_id: int = 120124,
        language_model_only: bool = False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            text_config.pop("architectures", None)
            self.text_config = HYV4FlashTextConfig(**text_config)
        elif text_config is None:
            top_level = {
                k: v for k, v in kwargs.items() if k in HYV4FlashTextConfig().__dict__
            }
            self.text_config = HYV4FlashTextConfig(**top_level)
        else:
            self.text_config = text_config
        self.vision_config = self._build_vision_config(vision_config)
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.image_newline_token_id = image_newline_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id
        self.language_model_only = language_model_only
        quant = getattr(self.text_config, "quantization_config", None)
        if quant is not None:
            kwargs["quantization_config"] = quant
        super().__init__(**kwargs)
        for key in _TOP_LEVEL_TEXT_KEYS:
            if hasattr(self.text_config, key):
                setattr(self, key, getattr(self.text_config, key))
        if self.architectures:
            self.text_config.architectures = list(self.architectures)

    def _build_vision_config(self, vision_config) -> Optional[HYV4VLVisionConfig]:
        if vision_config is None:
            return None
        if not isinstance(vision_config, HYV4VLVisionConfig):
            vision_config = HYV4VLVisionConfig(**vision_config)
        hidden_size = self.text_config.hidden_size
        if vision_config.out_hidden_size is None:
            vision_config.out_hidden_size = hidden_size
        if vision_config.out_hidden_size != hidden_size:
            raise ValueError("Hy4-Flash projector output must match text hidden_size")
        if vision_config.cat_extra_token:
            raise ValueError("Hy4-Flash vision does not support cat_extra_token")
        vision_config.text_hidden_size = hidden_size
        return vision_config

    def get_text_config(self, *args, **kwargs):
        return self.text_config


__all__ = ["HYV4FlashConfig", "HYV4FlashTextConfig"]
