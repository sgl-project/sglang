# SPDX-License-Identifier: Apache-2.0
"""Audited native encoder state, separate from pipeline and transport policy."""

import importlib
import json
from collections.abc import Callable

import msgspec

from sglang.multimodal_gen.runtime.loader.utils import finalize_loaded_model


def _validate_h3_architecture(recipe):
    from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
        MiniMaxH3Qwen3VLArchConfig,
        MiniMaxH3Qwen3VLConfig,
    )

    native = recipe.config
    arch = native.arch_config
    if (
        type(native) is not MiniMaxH3Qwen3VLConfig
        or type(arch) is not MiniMaxH3Qwen3VLArchConfig
        or arch.conditioning_projection_path is not None
        or arch.checkpoint_num_hidden_layers != 64
        or arch.num_hidden_layers != 50
        or arch.text_config.num_hidden_layers != 50
        or arch.hidden_size != 5120
        or arch.architectures != ["MiniMaxH3Qwen3VLEncoder"]
        or arch.text_config.use_cache
        or arch.text_config.output_hidden_states
        or native.enable_image_understanding
        or native.honor_cache_free_padding_mask
    ):
        raise ValueError("Unverified resolved H3 text encoder representation")


def _finalize_h3(model):
    # Request state is process-local, never part of the shared weight bundle.
    if model.model.rope_deltas is not None or model.conditioning_projection is not None:
        raise ValueError("Unexpected H3 text encoder derived state")
    return finalize_loaded_model(model)


class NativeEncoderStateContract(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    contract_id: str
    model_module: str
    model_name: str
    config_json: str
    validate_architecture: Callable
    finalize_after_import: Callable = finalize_loaded_model

    @property
    def expected_config(self):
        return json.loads(self.config_json)

    def matches_model(self, model_cls):
        return (model_cls.__module__, model_cls.__name__) == (
            self.model_module,
            self.model_name,
        ) and model_cls is getattr(
            importlib.import_module(self.model_module), self.model_name
        )

    def validate_supported(self, frozen, *, attention):
        recipe = frozen.thaw()
        if not self.matches_model(recipe.model_cls):
            raise ValueError("No audited state contract for this native encoder")
        config = {
            key: value
            for key, value in recipe.hf_config.items()
            if not key.startswith("_") and key != "transformers_version"
        }
        expected = self.expected_config
        # The shared native config reader removes HF's dispatch-only model_type.
        expected.pop("model_type")
        if config != expected:
            raise ValueError(
                "Unverified text encoder architecture/configuration extension"
            )
        args, native = recipe.server_args, recipe.config
        self.validate_architecture(recipe)
        if (
            native.quant_config is not None
            or native.lora_config is not None
            or native.parallel_folding_mode is not None
        ):
            raise ValueError(
                "Unverified quantized/LoRA/folded text encoder representation"
            )
        if (
            recipe.dtype != "bf16"
            or attention != "fa"
            or args.attention_backend not in (None, "fa")
        ):
            raise ValueError("Text encoder cache requires bf16 / FA")
        if (
            recipe.component_starts_on_cpu
            or args.residency_mode(recipe.component_name) != "resident"
            or args.should_use_fsdp_for_component(recipe.component_name)
            or args.enable_torch_compile
            or args.enable_breakable_cuda_graph
            or args.lora_path is not None
            or args.attention_backend_config
        ):
            raise ValueError(
                "Text encoder cache requires resident non-FSDP eager inference without LoRA"
            )
        for name in (
            "num_gpus",
            "nnodes",
            "tp_size",
            "sp_degree",
            "cfg_parallel_degree",
            "dp_size",
        ):
            if getattr(args, name) != 1:
                raise ValueError(f"Text encoder cache requires {name}=1")

    def adapt_meta_schema(self, model):
        # Registered RoPE buffers (including non-persistent buffers) already match
        # the ordinary loader. No assignment/repacking or extra tensor reads.
        return model


MINIMAX_H3_TEXT_ENCODER = NativeEncoderStateContract(
    "native.minimax_h3_qwen3vl_32b.layer50.bf16.fa.v1",
    "sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl",
    "MiniMaxH3Qwen3VLEncoder",
    """
{
  "architectures": [
    "Qwen3VLForConditionalGeneration"
  ],
  "image_token_id": 151655,
  "model_type": "qwen3_vl",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0,
    "bos_token_id": 151643,
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 25600,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 64,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "rms_norm_eps": 0.000001,
    "rope_scaling": {
      "mrope_interleaved": true,
      "mrope_section": [
        24,
        20,
        20
      ],
      "rope_type": "default"
    },
    "rope_theta": 5000000,
    "use_cache": true,
    "vocab_size": 151936
  },
  "tie_word_embeddings": false,
  "video_token_id": 151656,
  "vision_config": {
    "deepstack_visual_indexes": [
      8,
      16,
      24
    ],
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "model_type": "qwen3_vl",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652
}
    """,
    validate_architecture=_validate_h3_architecture,
    finalize_after_import=_finalize_h3,
)


def for_model(model_cls):
    for contract in (MINIMAX_H3_TEXT_ENCODER,):
        if contract.matches_model(model_cls):
            return contract
    raise ValueError("No audited native encoder weight-cache state contract")
