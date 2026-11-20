"""K/V-only DFlash2 prefill encoder. No Q/O, MLP, selector or draft forward."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.hf_transformers import get_rope_config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.dflash import (
    _get_dflash_layer_attention_params as get_dflash_layer_attention_params,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import (
    parse_dflash_draft_config,
)


class DFlashContextAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        parallel = get_parallel()
        self.tp_size, self.tp_rank = parallel.tp_size, parallel.tp_rank
        self.head_dim = int(config.head_dim)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if (
            self.total_num_kv_heads % self.tp_size
            or int(config.num_attention_heads) % self.tp_size
        ):
            raise ValueError("Context encoder head geometry must divide its TP group")
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        self.num_heads = int(config.num_attention_heads) // self.tp_size
        self.kv_size = self.num_kv_heads * self.head_dim
        # Keep baseline's combined [K,V] GEMM shape/layout; no Q allocation.
        self.kv_proj = nn.Linear(int(config.hidden_size), 2 * self.kv_size, bias=False)
        self.k_norm = RMSNorm(
            self.head_dim, eps=float(getattr(config, "rms_norm_eps", 1e-6))
        )
        theta, scaling = get_rope_config(config)
        rotary_dim = self.head_dim
        partial = float(getattr(config, "partial_rotary_factor", 1.0))
        if scaling is not None and "rope_dim" in scaling:
            rotary_dim = int(scaling["rope_dim"])
            partial = 1.0
            scaling = {
                k: v
                for k, v in scaling.items()
                if k
                not in {
                    "rope_dim",
                    "partial_rotary_factor",
                    "mrope_section",
                    "mrope_interleaved",
                }
            }
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=theta,
            rope_scaling=scaling,
            is_neox_style=bool(
                getattr(
                    config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
                )
            ),
            partial_rotary_factor=partial,
        )
        self.sliding_window_size, self.attn_type = get_dflash_layer_attention_params(
            config, layer_id
        )
        self.v_scale = parse_dflash_draft_config(
            draft_hf_config=config
        ).attention_value_scale
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
        )

    def kv_proj_only(self, hidden):
        k, v = F.linear(hidden, self.kv_proj.weight).split(
            [self.kv_size, self.kv_size], dim=-1
        )
        return k, v if self.v_scale is None else v * self.v_scale

    def apply_k_norm(self, k):
        return self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)

    def apply_k_rope(self, positions, k):
        _, result = self.rotary_emb(positions, k.new_empty(k.shape), k)
        return result


class DFlashContextLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.self_attn = DFlashContextAttention(config, layer_id)
        self.attention_conv = self.mlp_conv = None


def validate_context_encoder_config(config):
    if (
        getattr(config, "quantization_config", None)
        or parse_dflash_draft_config(draft_hf_config=config).is_mla
        or getattr(config, "use_gemma_norm", False)
        or getattr(config, "attention_bias", False)
    ):
        raise ValueError(
            "Slim context encoder currently supports unquantized, unbiased RMSNorm GQA DFlash2"
        )


def initialize_context_encoder(model, config, quant_config):
    # ModelConfig may supply the target's quantization policy to an ordinary
    # BF16 draft. Validate the draft checkpoint itself, as the full DFlash2
    # model does, rather than rejecting an inherited target policy.
    validate_context_encoder_config(config)
    model.context_only = True
    model.config = config
    model.draft_config = parse_dflash_draft_config(draft_hf_config=config)
    model.block_size = model.draft_config.resolve_block_size(default=16)
    target_layers = model.draft_config.resolve_target_layer_ids(
        target_num_layers=int(
            model.draft_config.num_target_layers
            or max(model.draft_config.target_layer_ids) + 1
        ),
        draft_num_layers=int(config.num_hidden_layers),
    )
    model.num_context_features = len(target_layers)
    model.rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
    model.fc = nn.Linear(
        model.num_context_features * int(config.hidden_size),
        int(config.hidden_size),
        bias=False,
    )
    model.hidden_norm = RMSNorm(int(config.hidden_size), eps=model.rms_norm_eps)
    model.layers = nn.ModuleList(
        [DFlashContextLayer(config, i) for i in range(int(config.num_hidden_layers))]
    )
    model.quantizable = False
    model.fc_tp_columns = None
    model.supports_fused_context_kv = False
    model.supports_fc_tp_split = False
    model.is_nemotron_35_draft = False
    model.candidate_selector = model.lilicorr = model.embed_tokens = model.lm_head = (
        None
    )
    model.projector_type = model.draft_config.projector_type


@torch.no_grad()
def load_context_encoder_weights(model, weights):
    required = {"fc.weight", "hidden_norm.weight"}
    for i in range(len(model.layers)):
        required.update(
            f"layers.{i}.self_attn.{name}.weight"
            for name in ("k_proj", "v_proj", "k_norm")
        )
    seen = set()
    for name, tensor in weights:
        name = name.removeprefix("model.")
        name = {
            "encoder.fc.weight": "fc.weight",
            "encoder.output_norm_enc.weight": "hidden_norm.weight",
        }.get(name, name)
        if name not in required:
            continue
        if name in seen:
            raise ValueError(f"Duplicate context encoder checkpoint tensor: {name}")
        if name == "fc.weight":
            target = model.fc.weight
        elif name == "hidden_norm.weight":
            target = model.hidden_norm.weight
        else:
            parts = name.split(".")
            attn = model.layers[int(parts[1])].self_attn
            if parts[3] == "k_norm":
                target = attn.k_norm.weight
            else:
                expected = (
                    attn.total_num_kv_heads * attn.head_dim,
                    int(model.config.hidden_size),
                )
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"Context encoder global head shape mismatch: {name}"
                    )
                tensor = tensor[
                    attn.tp_rank * attn.kv_size : (attn.tp_rank + 1) * attn.kv_size
                ]
                start = 0 if parts[3] == "k_proj" else attn.kv_size
                target = attn.kv_proj.weight[start : start + attn.kv_size]
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(f"Context encoder tensor shape mismatch: {name}")
        target.copy_(tensor)
        seen.add(name)
    if seen != required:
        raise ValueError(f"Missing context encoder weights: {sorted(required - seen)}")
    model.context_loaded_tensors = sorted(seen)
