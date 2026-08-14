#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert VoiceChat's EarTTS subtree into an SGLang-loadable checkpoint.

Requires the NVIDIA NeMo Speech VoiceChat environment because the trained
character-aware subword encoder must be instantiated once to bake its
deterministic full-vocabulary lookup table.
"""

import argparse
import json
from pathlib import Path

import torch
import tqdm
from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


def _load_tts_weights(model_path: Path):
    result = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("tts_model."):
                result[key[len("tts_model.") :]] = handle.get_tensor(key)
    if not result:
        raise ValueError(f"No tts_model tensors found in {model_path}")
    return result


def _precompute_subwords(model, batch_size):
    module = model.tts_model.embed_subword.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    mapping = module.subword_id_to_char_ids
    vocab_size = max(int(key) for key in mapping) + 1
    hidden_size = module.proj_embedding.out_features
    dtype = next(module.parameters()).dtype
    table = torch.zeros(vocab_size, hidden_size, dtype=dtype, device=device)
    with torch.no_grad():
        for start in tqdm.tqdm(range(0, vocab_size, batch_size)):
            end = min(start + batch_size, vocab_size)
            ids = torch.arange(start, end, device=device).unsqueeze(0)
            mask = torch.ones_like(ids, dtype=torch.bool)
            table[start:end] = module(ids, mask).squeeze(0).to(dtype)
    return table.cpu()


def _runtime_config(cfg, vocab_size):
    backbone_dict = OmegaConf.to_container(
        cfg.model.tts_config.backbone_config, resolve=True
    )
    backbone_type = cfg.model.tts_config.get("backbone_type")
    backbone = AutoConfig.for_model(backbone_type, **backbone_dict)
    output = {
        "architectures": ["EarTTSForCausalLM"],
        "model_type": "eartts",
        "vocab_size": 2,
        "emb_vocab_size": vocab_size,
        "backbone_type": backbone_type,
    }
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rope_local_base_freq",
        "sliding_window",
        "layer_types",
        "query_pre_attn_scalar",
        "attention_bias",
        "rms_norm_eps",
        "hidden_activation",
    ):
        if hasattr(backbone, key):
            value = getattr(backbone, key)
            output[key] = list(value) if isinstance(value, tuple) else value
    for key in ("latent_size", "codebook_size", "num_quantizers", "exponent"):
        output[key] = cfg.model.tts_config[key]
    for key in ("num_layers", "low_rank", "num_predictions", "min_log_std", "eps"):
        output[f"mog_{key}"] = cfg.model.tts_config.mog_head_config[key]
    output.update(
        num_iter=8,
        noise_scale=cfg.model.get("inference_noise_scale", 0.8),
        top_p_or_k=cfg.model.get("inference_top_p_or_k", 0.8),
        use_gated_fusion_for_text_audio=cfg.model.tts_config.use_gated_fusion_for_text_audio,
        use_audio_prompt_frozen_projection=cfg.model.tts_config.use_audio_prompt_frozen_projection,
        dtype="float32",
    )
    return output


def convert(
    config_path: Path,
    model_path: Path,
    output: Path,
    batch_size: int,
    base_model: str | None = None,
):
    output.mkdir(parents=True, exist_ok=True)
    full_config = json.loads(config_path.read_text())
    cfg = DictConfig(full_config["model"]["speech_generation"])
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.data.add_audio_prompt_after_description = True
    cfg.model.subword_mask_exactly_as_eartts = False
    cfg.model.context_hidden_mask_exactly_as_eartts = False
    cfg.model.tts_config.disable_eos_prediction = True
    cfg.model.inference_force_speech_silence_on_eos = True
    cfg.model.use_word_sep_tokenizer = False
    cfg.model.num_delay_speech_tokens = 0
    cfg.data.source_sample_rate = 22050
    cfg.data.target_sample_rate = 22050
    cfg.model.pretrained_model = None
    if base_model is not None:
        cfg.model.pretrained_lm_name = base_model
        cfg.model.tts_config.cas_config.pretrained_tokenizer_name = base_model

    model = DuplexEARTTS(OmegaConf.to_container(cfg, resolve=True)).eval()
    outer_weights = _load_tts_weights(model_path)
    model.load_state_dict(outer_weights, strict=False)
    subword_table = _precompute_subwords(model, batch_size)
    silence = model.codec_silence_tokens.detach().cpu().to(torch.int32)

    weights = {
        key[len("tts_model.") :]: value
        for key, value in outer_weights.items()
        if key.startswith("tts_model.")
    }
    rvq = torch.nn.functional.pad(weights["rvq_embs"], [0, 0, 0, 1])
    converted = {
        "model.total_emb.bos_emb": weights["bos_emb"],
        "model.total_emb.embed_subword.embed_subwords.weight": subword_table,
        # EarTTS uses the trained projection in both input fusion and MaskGIT.
        # Safetensors rejects two keys backed by the same storage, so give the
        # input-side copy independent storage while preserving identical data.
        "model.total_emb.embed_code.weight": weights["embed_code.weight"].clone(),
        "model.sil_tokens": silence,
    }
    for index, table in enumerate(rvq):
        converted[f"model.total_emb.rvq_embs.{index}.weight"] = table
    for key, value in weights.items():
        if (
            key.startswith("gated_fusion_audio_text.")
            or key == "audio_prompt_projection_W"
        ):
            converted[f"model.total_emb.{key}"] = value
        if key.startswith("backbone."):
            converted[f"model.{key}"] = value
        if key.startswith(("rvq_embs", "embed_code", "mog_head")):
            converted[f"model.sampler.{key}"] = value
    # The backbone input embedding is unused because EarTTS always supplies
    # fused embeddings; do not synthesize a shape-incompatible placeholder.
    save_file(converted, output / "model.safetensors")
    (output / "config.json").write_text(
        json.dumps(_runtime_config(cfg, subword_table.shape[0]), indent=2)
    )
    speaker_dir = output / "speaker_latents"
    for key, value in outer_weights.items():
        if "audio_prompt_latents." in key:
            speaker_dir.mkdir(exist_ok=True)
            torch.save(
                value, speaker_dir / f"{key.split('audio_prompt_latents.')[-1]}.pt"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--precompute-batch-size", type=int, default=256)
    args = parser.parse_args()
    convert(
        args.config,
        args.model,
        args.output,
        args.precompute_batch_size,
        args.base_model,
    )


if __name__ == "__main__":
    main()
