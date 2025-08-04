# Adapted from llama.py
"""Inference-only TeleChat2 model."""

import contextlib
import functools
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig

from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel

from .llama import LlamaDecoderLayer


class TeleChat2Model(LlamaModel):
    def __init__(self, config: LlamaConfig, prefix: str = ""):
        config.attention_bias = True
        config.mlp_bias = True
        super().__init__(config=config, prefix=prefix)

        self._configure_layer_biases()

    def _configure_layer_biases(self):
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            self._configure_attention_bias(layer)
            self._configure_mlp_bias(layer)

    def _configure_attention_bias(self, layer):
        layer.self_attn.qkv_proj.bias = None
        layer.self_attn.qkv_proj.skip_bias_add = True

    def _configure_mlp_bias(self, layer):
        layer.mlp.gate_up_proj.bias = None
        layer.mlp.gate_up_proj.skip_bias_add = True

        out_features = layer.mlp.down_proj.weight.shape[0]
        layer.mlp.down_proj.bias = nn.Parameter(
            torch.zeros(out_features), requires_grad=True
        )
        layer.mlp.down_proj.skip_bias_add = False


class TeleChat2ForCausalLM(LlamaForCausalLM):
    PREFIX_MAPPING = {
        "transformer.": "model.",
    }

    SUBSTRING_MAPPING = {
        ".mlp.": ".mlp.",
        ".down_proj.": ".down_proj.",
        ".h.": ".layers.",
        ".self_attention.": ".self_attn.",
        ".word_embeddings.": ".embed_tokens.",
        ".dense.": ".o_proj.",
        ".ln_f.": ".norm.",
    }

    STACKED_PARAMS_MAPPING = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    @staticmethod
    def translate_hf_to_sglang(name: str) -> str:
        transformations = [
            lambda n: next(
                (
                    n.replace(old, new, 1)
                    for old, new in TeleChat2ForCausalLM.PREFIX_MAPPING.items()
                    if n.startswith(old)
                ),
                n,
            ),
            lambda n: n.replace(".h.", ".layers."),
            lambda n: functools.reduce(
                lambda acc, mapping: acc.replace(*mapping),
                TeleChat2ForCausalLM.SUBSTRING_MAPPING.items(),
                n,
            ),
        ]

        return functools.reduce(
            lambda acc, transform: transform(acc), transformations, name
        )

    @staticmethod
    def harmonize_config(config: LlamaConfig):
        config.mlp_bias = True

        harmonization_rules = {
            "intermediate_size": ("ffn_hidden_size", None),
            "hidden_act": ("activation_function", "silu"),
            "rms_norm_eps": ("layer_norm_epsilon", 1e-6),
            "initializer_range": ("init_std", 0.02),
            "num_key_value_heads": ("num_attention_heads", None),
        }

        config_dict = vars(config)
        for target, (source, fallback) in harmonization_rules.items():
            if target not in config_dict:
                config_dict[target] = config_dict.get(source, fallback)

        with contextlib.suppress(AttributeError):
            defaults = {
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
            }
            for attr, default in defaults.items():
                setattr(config, attr, getattr(config, attr, default))

        if not hasattr(config, "head_dim"):
            config.head_dim = config.hidden_size // config.num_attention_heads

    def _init_model(
        self,
        config: LlamaConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        self.harmonize_config(config)
        return TeleChat2Model(config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        total_num_heads = self.config.n_head
        head_dim = self.config.hidden_size // total_num_heads

        for original_name, loaded_weight in weights:
            name = self.translate_hf_to_sglang(original_name)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "self_attn.key_value" in name:
                k_weight = []
                v_weight = []

                for i in range(total_num_heads):
                    start_idx = i * head_dim * 2
                    k_start = start_idx
                    k_end = start_idx + head_dim
                    v_start = start_idx + head_dim
                    v_end = start_idx + head_dim * 2

                    k_weight.append(loaded_weight[k_start:k_end, :])
                    v_weight.append(loaded_weight[v_start:v_end, :])

                k_weight = torch.cat(k_weight, dim=0)
                v_weight = torch.cat(v_weight, dim=0)

                qkv_name = name.replace("key_value", "qkv_proj")
                if qkv_name in params_dict:
                    param = params_dict[qkv_name]
                    param.weight_loader(param, k_weight, "k")
                    param.weight_loader(param, v_weight, "v")
                    loaded_params.add(qkv_name)

            elif "query" in name:
                qkv_name = name.replace("query", "qkv_proj")
                if qkv_name in params_dict:
                    param = params_dict[qkv_name]
                    param.weight_loader(param, loaded_weight, "q")
                    loaded_params.add(qkv_name)

            elif any(
                weight_name in name for _, weight_name, _ in self.STACKED_PARAMS_MAPPING
            ):
                for param_name, weight_name, shard_id in self.STACKED_PARAMS_MAPPING:
                    if weight_name in name:
                        stacked_name = name.replace(weight_name, param_name)
                        if stacked_name in params_dict:
                            param = params_dict[stacked_name]
                            param.weight_loader(param, loaded_weight, shard_id)
                            loaded_params.add(stacked_name)
                        break

            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


EntryClass = [TeleChat2ForCausalLM]
