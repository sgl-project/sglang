# SPDX-License-Identifier: Apache-2.0
# Adapted from https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b
"""Inference-only Ultravox (speech-to-text LLM) model."""

import glob
import os
from array import array
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    safetensors_weights_iterator,
)
from sglang.srt.models.llama import LlamaForCausalLM

# Ultravox's audio tower is an unmodified HF Whisper encoder, which is what
# Voxtral already implements (plain SDPA, no KV cache, HF weight names).
from sglang.srt.models.voxtral import VoxtralWhisperEncoder
from sglang.srt.utils.hf_transformers_utils import download_from_hf


class UltravoxProjector(nn.Module):
    """Stack `stack_factor` encoder frames, then RMSNorm -> Linear -> SwiGLU -> Linear.

    Stacking is what makes the audio cheap: it trades `stack_factor` frames of
    sequence length for the same factor of channel width.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.stack_factor = config.stack_factor
        dim_in = config.audio_config.d_model * self.stack_factor
        # config.hidden_size is the projector's width, not the text model's.
        dim_hidden = config.hidden_size
        self.swiglu = config.projector_act == "swiglu"
        dim_mid = dim_hidden // 2 if self.swiglu else dim_hidden
        dim_out = config.text_config.hidden_size

        self.ln_pre = RMSNorm(dim_in)
        self.linear_1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.linear_2 = nn.Linear(dim_mid, dim_out, bias=False)
        # v0.5.0 and above norm after the first linear; v0.4.1 and below after the second.
        if config.projector_ln_mid:
            self.ln_mid: nn.Module = RMSNorm(dim_mid)
            self.ln_post: nn.Module = nn.Identity()
        else:
            self.ln_mid = nn.Identity()
            self.ln_post = RMSNorm(dim_out)

    def _pad_and_stack(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        b, t, c = audio_embeds.shape
        t_pad = (t + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, t_pad - t))
        return audio_embeds.view(b, t_pad // self.stack_factor, c * self.stack_factor)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if not self.swiglu:
            return F.gelu(x)
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """[batch, frames, d_model] -> [batch, ceil(frames / stack_factor), text_hidden]"""
        hidden_states = self._pad_and_stack(audio_features)
        batch, seq_len, _ = hidden_states.shape
        # RMSNorm's fused kernels take 2D input.
        hidden_states = hidden_states.reshape(batch * seq_len, -1)
        hidden_states = self.ln_pre(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self._act(hidden_states)
        hidden_states = self.ln_mid(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states.reshape(batch, seq_len, -1)


class UltravoxModel(nn.Module):
    """Ultravox: Whisper encoder + stacking projector + Llama decoder.

    HF weight prefixes:
        audio_tower.*           -> self.audio_tower
        multi_modal_projector.* -> self.multi_modal_projector
    The text backbone is not in this checkpoint; see `_text_model_weights`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = VoxtralWhisperEncoder(config.audio_config, quant_config)
        self.multi_modal_projector = UltravoxProjector(config)
        self.language_model = LlamaForCausalLM(
            config.text_config, quant_config=quant_config
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        weight = self.audio_tower.conv1.weight
        embeddings = []
        for item in items:
            feature = item.feature
            if feature.dim() == 2:
                feature = feature.unsqueeze(0)
            feature = feature.to(device=weight.device, dtype=weight.dtype)
            frame_lens = item.audio_lens
            # One chunk per encoder call, trimmed to its true frame count: the
            # processor right-pads every chunk out to the 30s window, and those
            # padding frames would otherwise become extra audio tokens. Trimming
            # is equivalent to HF's padding mask because the padding is zeros.
            for index, chunk in enumerate(feature):
                num_frames = int(frame_lens[index])
                encoder_out = self.audio_tower(chunk[..., :num_frames].unsqueeze(0))
                embeddings.append(self.multi_modal_projector(encoder_out).squeeze(0))
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
            positions=positions,
        )

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _text_model_weights(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """Ultravox ships only the audio tower and projector; the frozen text
        backbone lives in the separate repo named by `text_model_id`."""
        text_model_id = self.config.text_model_id
        if text_model_id is None:
            raise ValueError(
                "Ultravox checkpoint has no language_model weights and no "
                "text_model_id to load them from."
            )
        local_dir = download_from_hf(
            text_model_id, allow_patterns=["*.safetensors", "*.json"]
        )
        files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
        if not files:
            raise ValueError(f"No safetensors found for text_model_id={text_model_id}")
        return safetensors_weights_iterator(files)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        encoder_dict = dict(self.audio_tower.named_parameters())
        projector_dict = dict(self.multi_modal_projector.named_parameters())

        weights_list = list(weights)
        # Whisper omits k_proj bias, but the fused QKVParallelLinear allocates one.
        weights_list += [
            (name.replace(".weight", ".bias"), torch.zeros(w.shape[0], dtype=w.dtype))
            for name, w in weights_list
            if name.startswith("audio_tower.")
            and name.endswith(".self_attn.k_proj.weight")
        ]

        llm_weights = []
        for name, loaded_weight in weights_list:
            if name.startswith("audio_tower."):
                trimmed = name[len("audio_tower.") :]
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if f".{weight_name}." not in trimmed:
                        continue
                    stacked = trimmed.replace(weight_name, param_name)
                    if stacked in encoder_dict:
                        param = encoder_dict[stacked]
                        param.weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    param = encoder_dict[trimmed]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            elif name.startswith("multi_modal_projector."):
                param = projector_dict[name[len("multi_modal_projector.") :]]
                default_weight_loader(param, loaded_weight)
            elif name.startswith("language_model."):
                llm_weights.append((name[len("language_model.") :], loaded_weight))
            else:
                llm_weights.append((name, loaded_weight))

        self.language_model.load_weights(llm_weights or self._text_model_weights())


EntryClass = [UltravoxModel]
