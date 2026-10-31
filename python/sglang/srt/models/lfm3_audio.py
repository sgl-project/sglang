# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LFM3 audio understanding using the standard SGLang multimodal embed path.

The LFM3-Audio frontend uses three stride-2 convolutions, a causal sliding-window
LFM2 encoder, and an MLP projection. Audio generation (depthformer/codebooks)
is deliberately not instantiated by this text-output model.
"""

from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from transformers import Lfm2Model
from transformers.masking_utils import create_sliding_window_causal_mask

from sglang.srt.configs.lfm3_audio import Lfm3AudioConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.lfm3 import Lfm3ForCausalLM
from sglang.srt.utils import add_prefix


class Lfm3AudioDownsampler(nn.Module):
    def __init__(self, dim_in, dim_conv, dim_out):
        super().__init__()
        if dim_in % 8:
            raise ValueError("Audio feature dimension must be divisible by 8")
        self.conv_in = nn.Conv2d(1, dim_conv, 3, stride=2, padding=1)
        self.conv_mid = nn.Conv2d(
            dim_conv, dim_conv, 3, stride=2, padding=1, groups=dim_conv
        )
        self.lin_mid = nn.Conv2d(dim_conv, dim_conv, 1)
        self.conv_end = nn.Conv2d(
            dim_conv, dim_conv, 3, stride=2, padding=1, groups=dim_conv
        )
        self.lin_end = nn.Conv2d(dim_conv, dim_conv, 1)
        self.lin_out = nn.Conv1d(dim_in * dim_conv // 8, dim_out, 1)
        self.act = nn.SiLU()

    def forward(self, features):
        x = self.act(self.conv_in(features.unsqueeze(1)))
        x = self.act(self.lin_mid(self.conv_mid(x)))
        x = self.act(self.lin_end(self.conv_end(x)))
        return self.lin_out(x.flatten(1, 2))


class Lfm3AudioLfmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Preserve the encoder dimensions already normalized by HF.
        self.config = deepcopy(config.text_config)
        self.config.sliding_window = config.sliding_window
        self.config._attn_implementation = "sdpa"
        self.lfm = Lfm2Model(self.config)
        del self.lfm.embed_tokens

    def forward(self, embeddings):
        mask = None
        if self.config.sliding_window is not None:
            mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=embeddings,
                attention_mask=None,
                past_key_values=None,
            )
        return self.lfm(
            inputs_embeds=embeddings,
            attention_mask=mask,
            use_cache=False,
            sliding_window=self.config.sliding_window,
        ).last_hidden_state


class Lfm3AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample = Lfm3AudioDownsampler(
            config.num_mel_bins, config.conv_channels, config.hidden_size
        )
        self.lfm = Lfm3AudioLfmEncoder(config)

    def forward(self, features):
        return self.lfm(self.downsample(features).mT)


class Lfm3AudioProjector(nn.Module):
    def __init__(self, dim_in, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, features):
        return self.model(features)


class Lfm3AudioForConditionalGeneration(nn.Module):
    def __init__(self, config: Lfm3AudioConfig, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.encoder = Lfm3AudioEncoder(config.audio_encoder_config)
        self.mlp = Lfm3AudioProjector(
            config.audio_encoder_config.hidden_size, config.text_config.hidden_size
        )
        self.language_model = Lfm3ForCausalLM(
            config.text_config,
            quant_config,
            prefix=add_prefix("lfm", prefix),
        )

    def get_num_kv_cache_layers(self):
        return self.language_model.get_num_kv_cache_layers()

    def get_attention_sliding_window_size(self):
        return self.language_model.get_attention_sliding_window_size()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def pad_input_ids(self, input_ids, mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def get_audio_feature(self, items: list[MultimodalDataItem]):
        parameter = next(self.encoder.parameters())
        downsampled_features = []
        # Downsample each clip independently to preserve convolution boundaries.
        for item in items:
            features = torch.as_tensor(item.feature).to(
                device=parameter.device, dtype=parameter.dtype
            )
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            if features.ndim != 2:
                raise ValueError(
                    "Expected one [mel_bins, frames] tensor per audio item"
                )
            embedding = self.encoder.downsample(features.unsqueeze(0))
            downsampled_features.append(embedding.mT.squeeze(0))

        lengths = [embedding.shape[0] for embedding in downsampled_features]
        padded = nn.utils.rnn.pad_sequence(downsampled_features, batch_first=True)
        # LFM2's attention and convolutions are causal: right padding cannot
        # affect valid positions. Discard padded outputs before projection.
        encoded = self.encoder.lfm(padded)
        return self.mlp(
            torch.cat(
                [row[:length] for row, length in zip(encoded, lengths, strict=True)]
            )
        )

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch, **kwargs):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = {
            name: parameter
            for name, parameter in self.named_parameters()
            if name.startswith(("encoder.", "mlp."))
        }
        loaded = set()
        # Only known audio-output components may be skipped. Typos or unknown
        # audio-input weights must fail instead of leaving random parameters.
        output_prefixes = (
            "depthformer.",
            "depth_embeddings.",
            "depth_linear.",
            "codebook_embedding.",
            "model.depth_decoder.",
            "model.codebook_embedding.",
        )

        def language_weights():
            for name, weight in weights:
                if name.startswith("model.audio_encoder."):
                    name = name.removeprefix("model.audio_encoder.")
                    if name.startswith("lfm."):
                        name = "lfm." + name
                    name = "encoder." + name
                elif name.startswith("model.audio_projector."):
                    layer, suffix = name.removeprefix("model.audio_projector.").split(
                        ".", 1
                    )
                    index = {"layer_norm": 0, "linear_1": 1, "linear_2": 3}[layer]
                    name = f"mlp.model.{index}.{suffix}"
                elif name.startswith("model.language_model."):
                    name = "lfm.model." + name.removeprefix("model.language_model.")
                elif name.startswith("lm_head."):
                    name = "lfm." + name

                if name.startswith("lfm."):
                    yield name.removeprefix("lfm."), weight
                elif name in params:
                    default_weight_loader(params[name], weight)
                    loaded.add(name)
                elif name == "mask_embedding" or name.startswith(output_prefixes):
                    continue
                else:
                    raise ValueError(f"Unexpected LFM3 audio weight: {name}")

        language_loaded = self.language_model.load_weights(language_weights())
        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"Missing LFM3 audio weights: {sorted(missing)}")
        missing_language = (
            dict(self.language_model.named_parameters()).keys() - language_loaded
        )
        if missing_language:
            raise ValueError(
                f"Missing LFM3 language weights: {sorted(missing_language)}"
            )
        return loaded | {f"language_model.{name}" for name in language_loaded}


EntryClass = Lfm3AudioForConditionalGeneration
