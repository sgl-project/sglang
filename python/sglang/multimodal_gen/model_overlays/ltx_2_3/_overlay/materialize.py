import json
import os

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.utils.model_overlay import (
    _copytree_link_or_copy,
    _ensure_dir,
    _link_or_copy_file,
)

AUXILIARY_MODEL_ID = "Lightricks/LTX-2"
CONFIG_DONOR_MODEL_ID = "FastVideo/LTX-2.3-Distilled-Diffusers"

AUXILIARY_PATTERNS = [
    "audio_vae/**",
    "scheduler/**",
    "text_encoder/**",
    "tokenizer/**",
    "vae/**",
    "vocoder/**",
]

CONFIG_DONOR_PATTERNS = [
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/**",
]

MONOLITH_PREFIX = "model.diffusion_model."
VIDEO_CONNECTOR_PREFIX = f"{MONOLITH_PREFIX}video_embeddings_connector."
AUDIO_CONNECTOR_PREFIX = f"{MONOLITH_PREFIX}audio_embeddings_connector."
TEXT_PROJ_IN_PREFIX = f"{MONOLITH_PREFIX}text_proj_in."
VIDEO_AGGREGATE_PREFIX = "text_embedding_projection.video_aggregate_embed."
AUDIO_AGGREGATE_PREFIX = "text_embedding_projection.audio_aggregate_embed."


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _rename_connector_key(key: str) -> str | None:
    if key.startswith(VIDEO_CONNECTOR_PREFIX):
        suffix = key[len(VIDEO_CONNECTOR_PREFIX) :]
        suffix = suffix.replace("transformer_1d_blocks", "transformer_blocks")
        suffix = suffix.replace(".attn1.q_norm.", ".attn1.norm_q.")
        suffix = suffix.replace(".attn1.k_norm.", ".attn1.norm_k.")
        return f"video_connector.{suffix}"
    if key.startswith(AUDIO_CONNECTOR_PREFIX):
        suffix = key[len(AUDIO_CONNECTOR_PREFIX) :]
        suffix = suffix.replace("transformer_1d_blocks", "transformer_blocks")
        suffix = suffix.replace(".attn1.q_norm.", ".attn1.norm_q.")
        suffix = suffix.replace(".attn1.k_norm.", ".attn1.norm_k.")
        return f"audio_connector.{suffix}"
    if key.startswith(TEXT_PROJ_IN_PREFIX):
        return key[len(MONOLITH_PREFIX) :]
    if key.startswith(VIDEO_AGGREGATE_PREFIX):
        return f"video_aggregate_embed.{key[len(VIDEO_AGGREGATE_PREFIX):]}"
    if key.startswith(AUDIO_AGGREGATE_PREFIX):
        return f"audio_aggregate_embed.{key[len(AUDIO_AGGREGATE_PREFIX):]}"
    return None


def _repack_transformer_weights(source_path: str, output_path: str) -> None:
    tensors = {}
    with safe_open(source_path, framework="pt") as f:
        for key in f.keys():
            if not key.startswith(MONOLITH_PREFIX):
                continue
            if key.startswith(VIDEO_CONNECTOR_PREFIX):
                continue
            if key.startswith(AUDIO_CONNECTOR_PREFIX):
                continue
            if key.startswith(TEXT_PROJ_IN_PREFIX):
                continue
            tensors[key[len(MONOLITH_PREFIX) :]] = f.get_tensor(key)
    if not tensors:
        raise ValueError("No transformer tensors found in LTX-2.3 source checkpoint.")
    save_file(tensors, output_path)


def _repack_connectors_weights(source_path: str, output_path: str) -> None:
    tensors = {}
    with safe_open(source_path, framework="pt") as f:
        for key in f.keys():
            renamed = _rename_connector_key(key)
            if renamed is None:
                continue
            tensors[renamed] = f.get_tensor(key)
    if not tensors:
        raise ValueError("No connector tensors found in LTX-2.3 source checkpoint.")
    save_file(tensors, output_path)


def _build_transformer_config(config_donor_dir: str) -> dict:
    config = _load_json(os.path.join(config_donor_dir, "transformer", "config.json"))
    config["_class_name"] = "LTX2VideoTransformer3DModel"
    return config


def _build_connectors_config(config_donor_dir: str) -> dict:
    text_encoder_config = _load_json(
        os.path.join(config_donor_dir, "text_encoder", "config.json")
    )
    return {
        "_class_name": "LTX2TextConnectors",
        "_diffusers_version": "0.37.0.dev0",
        "audio_connector_attention_head_dim": text_encoder_config[
            "audio_connector_attention_head_dim"
        ],
        "audio_connector_num_attention_heads": text_encoder_config[
            "audio_connector_num_attention_heads"
        ],
        "audio_connector_num_layers": text_encoder_config["audio_connector_num_layers"],
        "audio_connector_num_learnable_registers": text_encoder_config[
            "connector_num_learnable_registers"
        ],
        "audio_feature_extractor_out_features": text_encoder_config[
            "audio_feature_extractor_out_features"
        ],
        "caption_channels": text_encoder_config["hidden_size"],
        "causal_temporal_positioning": False,
        "connector_apply_gated_attention": text_encoder_config[
            "connector_apply_gated_attention"
        ],
        "feature_extractor_in_features": text_encoder_config[
            "feature_extractor_in_features"
        ],
        "connector_rope_base_seq_len": text_encoder_config[
            "connector_positional_embedding_max_pos"
        ][0],
        "rope_double_precision": text_encoder_config[
            "connector_double_precision_rope"
        ],
        "rope_theta": text_encoder_config["connector_positional_embedding_theta"],
        "rope_type": text_encoder_config["connector_rope_type"],
        "text_proj_in_factor": text_encoder_config["feature_extractor_in_features"]
        // text_encoder_config["hidden_size"],
        "video_feature_extractor_out_features": text_encoder_config[
            "video_feature_extractor_out_features"
        ],
        "video_connector_attention_head_dim": text_encoder_config[
            "connector_attention_head_dim"
        ],
        "video_connector_num_attention_heads": text_encoder_config[
            "connector_num_attention_heads"
        ],
        "video_connector_num_layers": text_encoder_config["connector_num_layers"],
        "video_connector_num_learnable_registers": text_encoder_config[
            "connector_num_learnable_registers"
        ],
    }


def _build_vae_config(auxiliary_dir: str) -> dict:
    config = _load_json(os.path.join(auxiliary_dir, "vae", "config.json"))
    config["use_official_image_encoder"] = True
    config["official_image_encoder_subdir"] = "ltx23_image_encoder"
    return config


def _repack_ltx23_image_encoder_weights(source_path: str, output_path: str) -> None:
    tensors = {}
    with safe_open(source_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("encoder.") or key.startswith("per_channel_statistics."):
                tensors[key] = f.get_tensor(key)
    if not tensors:
        raise ValueError("No LTX-2.3 image-encoder tensors found in donor checkpoint.")
    save_file(tensors, output_path)


def materialize(
    *,
    overlay_dir: str,
    source_dir: str,
    output_dir: str,
    manifest: dict,
) -> None:
    _ = overlay_dir, manifest

    auxiliary_dir = snapshot_download(
        repo_id=AUXILIARY_MODEL_ID,
        allow_patterns=AUXILIARY_PATTERNS,
        max_workers=8,
    )
    config_donor_dir = snapshot_download(
        repo_id=CONFIG_DONOR_MODEL_ID,
        allow_patterns=CONFIG_DONOR_PATTERNS,
        max_workers=8,
    )

    for component_name in (
        "audio_vae",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "vae",
        "vocoder",
    ):
        _copytree_link_or_copy(
            os.path.join(auxiliary_dir, component_name),
            os.path.join(output_dir, component_name),
        )

    source_checkpoint = os.path.join(source_dir, "ltx-2.3-22b-dev.safetensors")

    transformer_dir = os.path.join(output_dir, "transformer")
    _ensure_dir(transformer_dir)
    _write_json(
        os.path.join(transformer_dir, "config.json"),
        _build_transformer_config(config_donor_dir),
    )
    _repack_transformer_weights(
        source_checkpoint, os.path.join(transformer_dir, "model.safetensors")
    )

    connectors_dir = os.path.join(output_dir, "connectors")
    _ensure_dir(connectors_dir)
    _write_json(
        os.path.join(connectors_dir, "config.json"),
        _build_connectors_config(config_donor_dir),
    )
    _repack_connectors_weights(
        source_checkpoint, os.path.join(connectors_dir, "model.safetensors")
    )

    vae_dir = os.path.join(output_dir, "vae")
    _write_json(os.path.join(vae_dir, "config.json"), _build_vae_config(auxiliary_dir))

    image_encoder_dir = os.path.join(vae_dir, "ltx23_image_encoder")
    _ensure_dir(image_encoder_dir)
    _link_or_copy_file(
        os.path.join(config_donor_dir, "vae", "config.json"),
        os.path.join(image_encoder_dir, "config.json"),
    )
    _repack_ltx23_image_encoder_weights(
        os.path.join(config_donor_dir, "vae", "model.safetensors"),
        os.path.join(image_encoder_dir, "model.safetensors"),
    )

    _link_or_copy_file(
        os.path.join(source_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"),
        os.path.join(output_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"),
    )
    _link_or_copy_file(
        os.path.join(source_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        os.path.join(output_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    )
