# Copied and adapted from: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only MiniCPM-o model compatible with HuggingFace weights."""

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import torch.types
from torch import nn
from torch.nn.utils import weight_norm
from tqdm import tqdm
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.whisper.modeling_whisper import (
    WHISPER_ATTENTION_CLASSES,
    WhisperConfig,
    WhisperEncoder,
)

from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.minicpmv import (
    Idefics2VisionTransformer,
    MiniCPMBaseModel,
    Resampler2_5,
)
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import logger

try:
    from transformers import LogitsWarper
    from vector_quantize_pytorch import GroupedResidualFSQ
    from vocos import Vocos
    from vocos.pretrained import instantiate_class

    _tts_deps = True
except:
    LogitsWarper = None
    _tts_deps = False


def apply_spk_emb(
    input_ids: torch.Tensor = None,
    spk_emb: torch.Tensor = None,
    input_embeds: torch.Tensor = None,
    spk_emb_token_id: int = 0,
    num_spk_embs: int = 1,
):
    """
    Replace consecutive `num_spk_embs` speaker embedding placeholders in input_embeds with pre-prepared speaker embeddings. This is an in-place replacement, no new tensor is created, so no value is returned.

    Args:
        input_ids (torch.Tensor): Input ID tensor, shape [batch_size, seq_len_max]
        spk_emb (torch.Tensor): Speaker embedding tensor, shape [batch_size, num_spk_emb, hidden_dim]
        input_embeds (torch.Tensor): Input embedding tensor, shape [batch_size, seq_len_max, hidden_dim]
        spk_emb_token_id (int): ID of the speaker embedding token
        num_spk_embs (int): Number of speaker embeddings

    Returns:
        None
    """

    batch_size = input_ids.shape[0]

    for idx in range(batch_size):
        input_ids_ = input_ids[idx]  # [seq_len_max]
        spk_emb_ = spk_emb[idx]  # [num_spk_emb]
        mask_ = input_ids_ == spk_emb_token_id  # [batch_size, seq_len_max]
        nonzero_position_idx = mask_.nonzero(as_tuple=False)  # [num_spk_emb, 1]
        assert nonzero_position_idx.shape[0] == num_spk_embs
        begin_idx = nonzero_position_idx.min()
        end_idx = nonzero_position_idx.max()
        input_embeds[idx, begin_idx : end_idx + 1, :] = spk_emb_

    return


@dataclass
class ConditionalChatTTSGenerationOutput(ModelOutput):
    """
    Output class for ConditionalChatTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.

    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    finished: bool = None


def make_streaming_chunk_mask_generation(
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int,
    streaming_tts_text_mask: torch.Tensor,
    streaming_reserved_length: int = 300,
    streaming_audio_chunk_size: int = 50,
    streaming_text_chunk_size: int = 10,
    num_spk_emb: int = 1,
    use_spk_emb: bool = True,
) -> torch.Tensor:
    """
    In streaming audio generation, determine which `text` positions the TTS model can attend to when generating each chunk of `audio` tokens.

    This function creates a mask that allows the model to attend to a specific chunk of text
    tokens when generating each chunk of audio tokens, enabling streaming TTS generation.

    Args:
        inputs_embeds (torch.Tensor): Input embeddings tensor.
        past_seen_tokens (int): Number of tokens already seen by the model.
        streaming_tts_text_mask (torch.Tensor): Mask for the text tokens.
        streaming_reserved_length (int, optional): Number of reserved tokens for streaming. Defaults to 300.
        streaming_text_chunk_size (int, optional): Size of each text chunk. Defaults to 7.

    Returns:
        torch.Tensor: Causal mask for streaming TTS generation, shape is [batch_size=1, 1, seq_len=1, past_seen_tokens+1]

    Raises:
        AssertionError: If the batch size is not 1 (only supports batch size of 1 for inference).
    """
    assert inputs_embeds.shape[0] == 1

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    min_dtype = torch.finfo(dtype).min

    # Add `1` to the past seen tokens to account for new `tokens` during `generate`
    causal_mask = torch.full(
        (1, past_seen_tokens + inputs_embeds.shape[1]),
        fill_value=0,
        dtype=dtype,
        device=device,
    )

    # Calculate the start of invisible text tokens
    invisible_text_tokens_start = (
        min(
            math.ceil(
                (past_seen_tokens - streaming_reserved_length)
                / streaming_audio_chunk_size
            )
            * streaming_text_chunk_size,
            streaming_reserved_length,
        )
        + 1
        + num_spk_emb * use_spk_emb
    )  # Add 1 for [Stts] and N for [spk_emb] tokens if `use_spk_emb` is True

    invisible_text_tokens_end = (
        streaming_reserved_length + 1 + num_spk_emb * use_spk_emb + 1
    )  # Add 1 for [Ptts] (aka `audio_bos_token_id`)

    # Set invisible text tokens to min_dtype (effectively -inf)
    causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype

    # Mask padding positions in the text mask
    causal_mask[
        0, 0 : 1 + num_spk_emb * use_spk_emb + streaming_reserved_length + 1
    ].masked_fill_(streaming_tts_text_mask == 0, min_dtype)

    # Add extra dimensions for batch and heads
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.coef = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        y.transpose_(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(y)
        del y
        y = self.pwconv1(x)
        del x
        x = self.act(y)
        del y
        y = self.pwconv2(x)
        del x
        if self.coef is not None:
            y *= self.coef
        y.transpose_(1, 2)  # (B, T, C) -> (B, C, T)

        x = y + residual
        del y

        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class DVAEDecoder(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer=12,
        bn_dim=64,
        hidden=256,
        kernel=7,
        dilation=2,
        up=False,
    ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),
        )
        self.decoder_block = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden,
                    hidden * 4,
                    kernel,
                    dilation,
                )
                for _ in range(n_layer)
            ]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # B, C, T
        y = self.conv_in(x)
        del x
        for f in self.decoder_block:
            y = f(y, conditioning)

        x = self.conv_out(y)
        del y
        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class GFSQ(nn.Module):
    def __init__(
        self,
        dim: int,
        levels: List[int],
        G: int,
        R: int,
        eps=1e-5,
        transpose=True,
    ):
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim,
            levels=list(levels),
            num_quantizers=R,
            groups=G,
        )
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose_(1, 2) if self.transpose else feat

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            x.transpose_(1, 2)
        _, ind = self.quantizer(x)
        ind = ind.permute(1, 2, 0, 3).contiguous()
        ind = ind.view(ind.size(0), ind.size(1), -1)
        return ind.transpose_(1, 2) if self.transpose else ind


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class DVAE(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        coef = torch.rand(100)
        self.coef = nn.Parameter(coef.unsqueeze(0).unsqueeze_(2))

        self.downsample_conv = nn.Sequential(
            nn.Conv1d(100, 512, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(512, 512, 4, 2, 1),
            nn.GELU(),
        )

        self.encoder = DVAEDecoder(
            idim=512,
            odim=1024,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.decoder = DVAEDecoder(
            idim=512,
            odim=512,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.out_conv = nn.Conv1d(512, 100, 3, 1, 1, bias=False)

        self.vq_layer = GFSQ(
            dim=1024,
            levels=(5, 5, 5, 5),
            G=2,
            R=2,
        )

    @torch.inference_mode()
    def forward(
        self, inp: torch.Tensor, mode: Literal["encode", "decode"] = "decode"
    ) -> torch.Tensor:
        if mode == "encode" and hasattr(self, "encoder") and self.vq_layer is not None:
            mel = inp.clone()
            x: torch.Tensor = self.downsample_conv(
                torch.div(mel, self.coef.view(100, 1).expand(mel.shape), out=mel),
            ).unsqueeze_(0)
            del mel
            x = self.encoder(x)
            ind = self.vq_layer(x)
            del x
            return ind

        if self.vq_layer is not None:
            vq_feats = self.vq_layer._embed(inp)
        else:
            vq_feats = inp

        vq_feats = (
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
            )
            .permute(0, 2, 3, 1)
            .flatten(2)
        )

        dec_out = self.out_conv(
            self.decoder(
                x=vq_feats,
            ),
        )

        del vq_feats

        return torch.mul(dec_out, self.coef, out=dec_out)


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/processors.py`
class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(
                0, self.max_input_ids, freq.size(0) - self.max_input_ids
            ).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


class ConditionalChatTTS(PreTrainedModel):
    """A conditional text-to-speech model that can generate speech from text with speaker conditioning.

    This model extends PreTrainedModel to provide text-to-speech capabilities with:
    - LLM hidden state conditioning
    - Streaming generation

    The model uses a transformer architecture with LLM hidden states and can operate in both
    streaming and non-streaming modes for flexible deployment.

    The model process sequence in the following format:
    | text bos token | LLM embedding projected to tts embedding space | text tokens (fixed length, reserved for future tokens) | audio bos token | audio tokens (audio token length is not fixed)| audio eos token |

    The format is designed to support LLM-conditioned streaming audio generation.

    Usage:
    To support streaming generation, two global variables should be maintained outside of the model.
        1. `audio_input_ids`: stores *discrete* audio codes. It is a tensor with shape [1, sequence length+1, num_vq].
        2. `past_key_values`: stores the KV cache for both text tokens and audio codes. It is a list of tuples, each tuple contains two tensors with shape [1, num_attention_heads, sequence length, hidden_size // num_attention_heads]

    where `num_vq` is the number of audio codebooks, in default setting, it is `4`.

    1. Create an empty `past_key_values` with
    ```python
    initial_kv_cache_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len # where `1` denotes the `bos` token
    dtype = model.emb_text.weight.dtype
    device = model.emb_text.weight.device
    past_key_values = [
        (
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device),
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device)
        )
        for _ in range(model.config.num_hidden_layers)
    ]

    2. At the same time, create an empty `audio_input_ids` with shape [1, sequence length, num_vq], `num_vq` denotes multiple layer audio codebooks. But here we also include text tokens in the sequence, but they will be zeros, and will not be used, just a placeholder.

    ```python
    initial_audio_input_ids_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len + 1
    # [bos token, speaker embeddings, text tokens, audio bos token]
    audio_input_ids = torch.zeros(batch_size=1, initial_audio_input_ids_length, model.num_vq)
    ```

    2. Prefill some text tokens to TTS model (for example, 10 tokens) using `prefill_text` method.

    ```python
    outputs = llm.generate(**kwargs)
    llm_tokens = some_function_to_extract_llm_tokens(outputs)
    lm_spk_emb_last_hidden_states = some_function_to_extract_lm_spk_emb_last_hidden_states(outputs)
    tts_text_input_ids = tts_tokenizer.encode(llm_tokenizer.decode(llm_tokens))
    # here assume we are prefilling text token 0 to text token 9 (included), totally 10 tokens.
    begin = 0
    end = 9+1
    position_ids = torch.arange(begin, end, dtype=torch.long, device=device)

    past_key_values = model.prefill_text(
        input_ids=tts_text_input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
    )
    ```

    3. Make a `streaming_tts_text_mask` to denote which position contains valid text tokens, similar to `attention_mask` in standard causal attention.

    ```python
    streaming_tts_text_mask = torch.zeros(model.streaming_reserved_length)
    streaming_tts_text_mask[0:end] = 1 # denotes these post
    ```

    3. Generate audio codes using `generate` method.

    ```python
    outputs = model.generate(
        input_ids=audio_input_ids,
        past_key_values=past_key_values,
        streaming_tts_text_mask=streaming_tts_text_mask,
        max_new_token=50,
    )

    # update past_key_values and input_ids
    past_key_values = outputs.past_key_values
    audio_input_ids = outputs.input_ids
    ```

    The `past_key_values` is extended by `max_new_token=50`, and `audio_input_ids` is also extended by `max_new_token=50` after `generate` calling.

    4. Notice that after prefilling `10` text tokens, the model can generate up to `50` audio tokens, if you want to generate more audio tokens, you need to prefill next `10` text tokens. And it is okay to only generate `25` audio tokens for faster initial response.

    5. Repeat steps `2,3,4` as needed in your streaming audio generation cases, but ensure usage complies with the following guidelines discussed above.
    """

    config_class = PretrainedConfig
    _no_split_modules = []

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.use_speaker_embedding = config.use_speaker_embedding
        self.use_llm_hidden_state = config.use_llm_hidden_state
        self.num_spk_embs = config.num_spk_embs
        self.spk_emb_token_id = config.spk_emb_token_id

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_size = config.streaming_text_chunk_size
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        if self.config.use_mlp:
            self.projector = MultiModalProjector(config.llm_dim, config.hidden_size)
        else:
            self.projector = nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        self.emb_code = nn.ModuleList(
            [
                nn.Embedding(config.num_audio_tokens, config.hidden_size)
                for _ in range(config.num_vq)
            ]
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )

        dvae = DVAE()
        self.dvae = dvae

        model_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            attn_implementation=config.attn_implementation,
        )

        model = LlamaModel(model_config)
        self.model = model

    @torch.inference_mode()
    def merge_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Merge `input_ids` and `lm_spk_emb_last_hidden_states` to `inputs_embeds`.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            lm_spk_emb_last_hidden_states (Optional[torch.Tensor], optional): Last hidden states of speaker embeddings from the language model. Defaults to None.

        Raises:
            NotImplementedError: If speaker embedding is not used and language model hidden states are not implemented.

        Returns:
            torch.Tensor: Prepared input embeddings for the model.
        """
        assert input_ids.shape[0] == 1

        # Embed input_ids to input_embeds
        inputs_embeds = self.emb_text(input_ids)

        # Inject speaker embedding to input_embeds if it exists
        if self.use_speaker_embedding:
            spk_emb_mask = input_ids == self.spk_emb_token_id
            if spk_emb_mask.any():
                assert lm_spk_emb_last_hidden_states is not None
                # Project spk emb to tts hidden size first, [batch_size, num_spk_emb, llm_dim] -> [batch_size, num_spk_emb, self.hidden_size]
                lm_spk_emb_last_hidden_states = lm_spk_emb_last_hidden_states.to(
                    self.projector.linear1.weight.dtype
                )
                projected_spk_emb = self.projector(lm_spk_emb_last_hidden_states)
                projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)
                apply_spk_emb(
                    input_ids=input_ids,
                    spk_emb=projected_spk_emb,
                    input_embeds=inputs_embeds,
                    spk_emb_token_id=self.spk_emb_token_id,
                    num_spk_embs=self.num_spk_embs,
                )
        else:
            raise NotImplementedError

        return inputs_embeds

    @torch.inference_mode()
    def prefill_text(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Prefill a chunk of new text tokens in streaming setting.
        Specifically speaking, update `past_key_values` using new text tokens, then the model will read the new text tokens.

        Args:
            input_ids (Tensor): Tensor of shape [batch_size, seq_len]
            position_ids (LongTensor): Tensor of shape [batch_size, seq_len]
            past_key_values (List[Tuple[Tensor]]): KV Cache of all layers, each layer is a tuple (Tensor, Tensor) denoting keys and values. Each tensor is of seq_len = `self.streaming_text_reserved_len`. `past_key_values` will be updated.
            lm_spk_emb_last_hidden_states (Tensor, optional): Tensor of shape [batch_size, num_spk_emb, llm_dim]. Defaults to None.

        Note that all `batch_size` should be `1`.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        # Merge text and LLM embeddings
        inputs_embeds = self.merge_inputs_embeds(
            input_ids=input_ids,
            lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
        )

        # Clone KV Cache
        past_key_values_for_prefill = []
        for i in range(len(past_key_values)):
            past_key_values_for_prefill.append(
                (
                    past_key_values[i][0][:, :, : position_ids[:, 0], :].clone(),
                    past_key_values[i][1][:, :, : position_ids[:, 0], :].clone(),
                )
            )

        # ModelMiniCPMVBaseModel
        outputs_prefill: BaseModelOutputWithPast = self.model(
            attention_mask=None,  # because for text, it is standard causal attention mask, do nothing
            position_ids=position_ids,  # position_ids denotes the position of new text tokens in the sequence
            past_key_values=past_key_values_for_prefill,  # `past_key_values` will be updated by the model
            inputs_embeds=inputs_embeds,  # contains text and language model embedding
            use_cache=True,
            output_attentions=False,
            cache_position=position_ids,  # which new positions will use this cache, basically the same as position_ids
        )

        # Get model updated KV Cache
        past_key_values_for_prefill_updated = outputs_prefill.past_key_values

        # Update generated KV Cache to input `past_key_values`
        for layer_idx in range(len(past_key_values)):
            # Update keys
            past_key_values[layer_idx][0][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1, :
            ] = past_key_values_for_prefill_updated[layer_idx][0][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1
            ].clone()
            # Update values
            past_key_values[layer_idx][1][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1, :
            ] = past_key_values_for_prefill_updated[layer_idx][1][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1
            ].clone()

        # TODO: del past_key_values_for_prefill_updated recursively
        # TODO: del outputs_prefill recursively

        return past_key_values

    @torch.inference_mode()
    def prefill_audio_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        streaming_tts_text_mask=None,
        add_audio_bos: bool = True,
    ):
        """Prefill a chunk of audio ids to the model. Used in sliding-window long audio generation.
        Specifically, prefill many audio ids (typically from last window) to the model in the new window.

        Args:
            input_ids (torch.Tensor): (1, seq_len, num_vq) Audio input token ids.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        code_emb = [self.emb_code[i](input_ids[:, :, i]) for i in range(self.num_vq)]
        inputs_embeds = torch.stack(code_emb, 3).sum(3)  # [1,seq_len,768]
        input_len = input_ids.shape[1]

        if add_audio_bos:
            narrowed_input_ids = torch.tensor(
                [[self.audio_bos_token_id]], dtype=torch.long, device=self.device
            )
            bos_inputs_embeds = self.emb_text(narrowed_input_ids)
            inputs_embeds = torch.cat([bos_inputs_embeds, inputs_embeds], dim=1)
            input_len += 1

        past_key_values_length = past_key_values[0][0].shape[2]
        position_ids = torch.arange(
            past_key_values_length,
            past_key_values_length + input_len,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        cache_position = position_ids.clone()
        causal_mask = make_streaming_chunk_mask_generation(
            inputs_embeds=inputs_embeds,
            past_seen_tokens=past_key_values[0][0].shape[2],
            streaming_tts_text_mask=streaming_tts_text_mask,
            streaming_reserved_length=self.streaming_text_reserved_len,
            streaming_text_chunk_size=self.streaming_text_chunk_size,
        )  # [1, 1, 1, past_key_values_length + input_len]

        # Model forward
        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=cache_position,
        )
        past_key_values = outputs.past_key_values
        return past_key_values

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        streaming_tts_text_mask=None,
        force_no_stop=False,
        min_new_token=10,
        max_new_token=50,
        logits_warpers: List[LogitsWarper] = [],
        logits_processors: List[CustomRepetitionPenaltyLogitsProcessorRepeat] = [],
        show_tqdm=False,
    ):
        """Generate audio codes in streaming setting or non-streaming setting.
        Specifically speaking, generate audio codes when not all text tokens are prefilled.

        Always pass a valid `past_key_values` to the method. The method does not do `prefill` by itself. It relies on `prefill_text` method to provide valid `past_key_values`. Please refer to docstring of this class for more details.

        In this method, we borrowed a lot of codes from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/gpt.py`.

        Args:
            input_ids (torch.Tensor): Input token ids.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
            temperature (torch.Tensor): Temperature for sampling.
            eos_token (Union[int, torch.Tensor]): End of sequence token.
            streaming_tts_text_mask (Optional[torch.Tensor], optional): Mask for streaming TTS text. Defaults to None.
            max_new_token (int, optional): Maximum number of new tokens to generate. Defaults to 50.
            logits_warpers (List[LogitsWarper], optional): List of logits warpers. Defaults to [].
            logits_processors (List[CustomRepetitionPenaltyLogitsProcessorRepeat], optional): List of logits processors. Defaults to [].
            show_tqdm (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            GenerationOutputs: Generation outputs.
        """

        # We only support batch size `1` for now
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        # fix: this should not be `input_ids.shape[1]`
        # start_idx = input_ids.shape[1]
        start_idx = (
            1
            + self.num_spk_embs * self.use_speaker_embedding
            + self.streaming_text_reserved_len
            + 1
        )

        finish = torch.zeros(input_ids.shape[0], device=input_ids.device).bool()

        temperature = (
            temperature.unsqueeze(0)
            .expand(input_ids.shape[0], -1)
            .contiguous()
            .view(-1, 1)
        )

        progress = input_ids.shape[1]

        # Pre-allocate input_ids, shape is [batch_size=1, max_possible_seq_len, self.num_vqs]
        input_ids_buf = torch.zeros(
            input_ids.shape[0],  # batch_size
            progress
            + max_new_token,  # max_possible_seq_len = input_ids.shape[1] + max_new_token
            input_ids.shape[2],  # self.num_vqs
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        # Copy existing `input_ids` to `input_ids_buf`
        input_ids_buf.narrow(1, 0, progress).copy_(input_ids)

        del input_ids
        input_ids = input_ids_buf.narrow(1, 0, progress)

        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        condition_length = (
            1
            + self.num_spk_embs * self.use_speaker_embedding
            + self.streaming_text_reserved_len
            + 1
        )

        for i in range(max_new_token):
            # Prepare generation inputs
            audio_bos = False

            # If this is the first audio token, the case is SPECIAL
            if progress == condition_length:
                audio_bos = True

            assert progress == (
                past_key_values[0][0].shape[2] + 1
            )  # If you are using according to the guidelines, this should be passed.

            if audio_bos:
                # Generate the first token, activate the model with `self.audio_bos_token_id`, the model will predict
                # a new audio token. This is a special case because without the `audio bos token`, it is impossible
                # to generate the first audio token in our streaming setting.
                narrowed_input_ids = torch.tensor(
                    [[self.audio_bos_token_id]], dtype=torch.long, device=self.device
                )
                inputs_embeds = self.emb_text(narrowed_input_ids)
                del narrowed_input_ids
            else:
                # Generate the following audio tokens, it is applicable to all other cases, including second and the
                # following calling of `generate`.
                narrowed_input_ids = input_ids.narrow(
                    dim=1, start=input_ids.shape[1] - 1, length=1
                )
                code_emb = [
                    self.emb_code[i](narrowed_input_ids[:, :, i])
                    for i in range(self.num_vq)
                ]
                inputs_embeds = torch.stack(code_emb, 3).sum(3)

            position_ids = torch.tensor(
                [past_key_values[0][0].shape[2]], dtype=torch.long, device=self.device
            ).unsqueeze(0)

            cache_position = position_ids.clone()

            # Make causal mask
            causal_mask = make_streaming_chunk_mask_generation(
                inputs_embeds=inputs_embeds,
                past_seen_tokens=past_key_values[0][0].shape[2],
                streaming_tts_text_mask=streaming_tts_text_mask,
                streaming_reserved_length=self.streaming_text_reserved_len,
                streaming_text_chunk_size=self.streaming_text_chunk_size,
            )

            # Model forward
            outputs: BaseModelOutputWithPast = self.model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                cache_position=cache_position,
            )

            del position_ids
            del inputs_embeds
            del cache_position
            del causal_mask

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            # logits = logits[:, -1].float()
            logits = logits.narrow(1, -1, 1).squeeze_(1).float()

            # logits = rearrange(logits, "b c n -> (b n) c")
            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))
            # logits_token = rearrange(input_ids[:, start_idx:], "b c n -> (b n) c")
            input_ids_sliced = input_ids.narrow(
                1,
                start_idx,
                input_ids.size(1) - start_idx,
            ).permute(0, 2, 1)
            logits_token = input_ids_sliced.reshape(
                input_ids_sliced.size(0) * input_ids_sliced.size(1),
                -1,
            ).to(self.device)
            del input_ids_sliced

            logits /= temperature

            if not audio_bos:
                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)
            if not audio_bos:
                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

            del logits_token

            if i < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1)  # .to(finish.device)

            del scores

            # idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            # Store new `token` into `input_ids_buf`
            input_ids_buf.narrow(1, progress, 1).copy_(idx_next.unsqueeze_(1))

            if i == 0 and finish.any():
                # raise Exception
                break

            del idx_next
            progress += 1
            input_ids = input_ids_buf.narrow(1, 0, progress)

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            if show_tqdm:
                logger.info(f"incomplete result. hit max_new_token: {max_new_token}")

        del input_ids_buf

        if finish.all():
            # the last may contains eos token
            genrated_input_ids = input_ids[:, condition_length:-1, :]
        else:
            # there is no eos token
            genrated_input_ids = input_ids[:, condition_length:, :]

        return ConditionalChatTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=input_ids,  # for update purpose
            past_key_values=past_key_values,  # for update purpose
            finished=finish.all(),
        )

    @torch.inference_mode()
    def decode_to_mel_specs(
        self,
        result_list: List[torch.Tensor],
    ):
        """Decode discrete audio codes to mel spectrograms.

        Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/core.py`

        Args:
            result_list (List[torch.Tensor]): Audio codes output from `generate`.

        Returns:
            torch.Tensor: Mel spectrograms.
        """

        decoder = self.dvae
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
            del src

        mel_specs = decoder(batch_result)
        del batch_result
        return mel_specs


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer and add use_cache for streaming inference
class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, embed_dim)`):
                Hidden states to be fed into the encoder layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, tgt_len, src_len)`):
                Attention mask where padding elements are indicated by large negative values.
            layer_head_mask (`torch.FloatTensor` of shape `(encoder_attention_heads,)`):
                Mask to nullify selected heads of the attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention weights.
            past_key_values (`EncoderDecoderCache`, *optional*):
                Past key-value pairs used for incremental decoding.
            use_cache (`bool`, *optional*):
                Whether or not to return updated `past_key_values` for caching.

        Returns:
            A tuple of shape `(hidden_states, optional(attn_weights), optional(past_key_values))`.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=False
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                MiniCPMWhisperEncoderLayer(config, layer_idx=i)
                for i in range(config.encoder_layers)
            ]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
    ):
        r"""
        Forward pass of the Whisper encoder.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of log-mel features extracted from the raw audio waveform. Typically generated
                by a feature extractor (e.g., `WhisperFeatureExtractor`) that processes `.flac` or `.wav`
                files into padded 2D mel spectrogram frames. These features are projected via convolution layers
                (`conv1` and `conv2`) and then transformed into embeddings for the encoder.

            attention_mask (`torch.Tensor`, *optional*):
                Not used by Whisper for masking `input_features`, but included for API compatibility with
                other models. If provided, it is simply ignored within the model. By default, Whisper
                effectively ignores silence in the input log-mel spectrogram.

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected attention heads. The elements should be either 1 or 0, where:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked** (i.e., the attention head is dropped).

            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors of all encoder layers. If set to `True`, the
                returned tuple (or `BaseModelOutputWithPast`) will contain an additional element with
                attention weights for each encoder layer.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. If set to `True`, the returned
                tuple (or `BaseModelOutputWithPast`) will contain a tuple of hidden states, including the
                initial embedding output as well as the outputs of each layer.

            return_dict (`bool`, *optional*):
                Whether or not to return a `BaseModelOutputWithPast` (a subclass of `ModelOutput`) instead
                of a plain tuple. If set to `True`, the output will be a `BaseModelOutputWithPast` object,
                otherwise it will be a tuple.

            past_key_values (`EncoderDecoderCache`, *optional*):
                When using caching for faster inference, this is an object that stores the key-value pairs
                for attention states. If provided, the model will append new states to the existing cache
                and return the updated cache. This speeds up sequential decoding or chunked inference.

                - If `past_key_values` is `None`, no past states are used or returned.
                - If `past_key_values` is not `None` and `use_cache=True`, the model will use the provided
                cache and return the updated cache (as `next_encoder_cache`).

            use_cache (`bool`, *optional*):
                Whether or not the model should use caching (`past_key_values`) to speed up processing
                during inference. When set to `True`, the model will:
                - Inspect and use `past_key_values` if provided.
                - Return updated `past_key_values` (under the name `next_encoder_cache` in
                    `BaseModelOutputWithPast`).

        Returns:
            `BaseModelOutputWithPast` or `tuple` (depending on `return_dict`):
                If `return_dict=True`, a `BaseModelOutputWithPast` is returned, which contains:
                - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The output of the final encoder layer.
                - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True`):
                Hidden states of the model at each layer (including the initial projection).
                - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_attentions=True`):
                Attention weights from each encoder layer.
                - **past_key_values** (an object of type `EncoderDecoderCache` or `None`, *optional*):
                Updated cache of key-value pairs if `use_cache=True`.

                If `return_dict=False`, a tuple is returned, where the format is:
                `(last_hidden_state, hidden_states, attentions)`, with `hidden_states` and `attentions`
                only present if their respective `output_*` arguments are set to `True`.

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Ignore copy
        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(
                    DynamicCache.from_legacy_cache(past_key_values), DynamicCache()
                )
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:
                pass
            past_key_values_length = (
                past_key_values.self_attention_cache.get_usable_length(
                    inputs_embeds.shape[1]
                )
            )
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning(
                    "seems the audio is longer than 30s. repeating the last part of the audio"
                )
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1]
                            - embed_pos.shape[0]
                            + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[
                    past_key_values_length : inputs_embeds.shape[1]
                    + past_key_values_length,
                    :,
                ]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMO(MiniCPMBaseModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config, quant_config=quant_config)

        self.llm = self.init_llm(config=config, quant_config=quant_config)

        self.embed_dim = self.llm.config.hidden_size

        # init vision module
        if self.config.init_vision:
            # print("vision-understanding enabled")
            self.vpm = self.init_vision_module(config=config, quant_config=quant_config)
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # init audio module
        self.config.init_audio = True
        if self.config.init_audio:
            # print("audio-understanding enabled")
            self.apm = self.init_audio_module()
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
            self.audio_avg_pooler = nn.AvgPool1d(
                self.config.audio_pool_step, stride=self.config.audio_pool_step
            )
            self.audio_projection_layer = MultiModalProjector(
                in_dim=audio_output_dim, out_dim=self.embed_dim
            )
            self.audio_encoder_layer = -1

        # init tts module
        self.config.init_tts = False
        logger.info("TTS is disabled for now")
        if self.config.init_tts:
            # print("tts enabled")
            assert (
                _tts_deps
            ), "please make sure vector_quantize_pytorch and vocos are installed."
            self.tts = self.init_tts_module()

    def init_tts_module(self):
        model = ConditionalChatTTS(self.config.tts_config)
        return model

    def init_audio_module(self):
        model = MiniCPMWhisperEncoder(self.config.audio_config)
        return model

    def init_llm(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        return Qwen2ForCausalLM(config=config, quant_config=quant_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def pad_input_ids(self, input_ids: List[int], mm_input: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = mm_input.im_start_id
        im_end_id: int = mm_input.im_end_id
        slice_start_id: int = mm_input.slice_start_id
        slice_end_id: int = mm_input.slice_end_id

        media_token_pairs = [
            (im_start_id, im_end_id),
            (slice_start_id, slice_end_id),
            (mm_input.audio_start_id, mm_input.audio_end_id),
        ]
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return pattern.pad_input_tokens(input_ids, mm_input)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_audio_embedding_streaming(self, items: List[MultimodalDataItem]):
        r"""
        Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        wavforms = flatten_nested_list(
            [item.audio_features for item in items if item.audio_features]
        )
        # list, [[x1, x2], [y1], [z1]]
        audio_feature_lens_raw = flatten_nested_list(
            [item.audio_feature_lens for item in items if item.audio_feature_lens]
        )

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values[0][0].shape[2]
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            audio_outputs = self.apm(
                wavforms, past_key_values=self.audio_past_key_values, use_cache=True
            )
            audio_states = (
                audio_outputs.last_hidden_state
            )  # [:, :audio_feat_lengths, :]
            self.audio_past_key_values = audio_outputs.past_key_values

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(
                audio_feature_lens
            )

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(
                        audio_embeds[idx, : num_audio_tokens[idx], :]
                    )
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device

        Returns:
            torch.Tensor: mask

        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def get_audio_embedding(self, items: List[MultimodalDataItem], chunk_length=-1):
        r"""
        Extract full audio embeddings with optional chunk-based attention.

        This method computes embeddings for all audio frames at once, either using full attention (when
        `chunk_length` is -1) or chunk-based attention (when `chunk_length` is a positive number). It does
        not use key-value caching and is suitable for non-streaming inference.

        Args:
            chunk_length (int, optional): Determines whether to use full attention (-1) or chunk-based
                attention (>0) during embedding computation.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        # (bs, 80, frames) or [], multi audios need filled in advance
        wavforms = flatten_nested_list(
            [item.audio_features for item in items if item.audio_features]
        )
        # list, [[x1, x2], [y1], [z1]]
        audio_feature_lens_raw = flatten_nested_list(
            [item.audio_feature_lens for item in items if item.audio_feature_lens]
        )

        final_audio_embeds = []

        assert isinstance(wavforms, list)
        assert isinstance(wavforms[0], torch.Tensor)
        # exist audio
        for wavform in wavforms:
            if len(wavform) > 0:
                audio_feature_lens = torch.hstack(audio_feature_lens_raw)
                batch_size, _, max_mel_seq_len = wavform.shape
                max_seq_len = (max_mel_seq_len - 1) // 2 + 1

                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(
                        0,
                        max_seq_len,
                        dtype=audio_feature_lens.dtype,
                        device=audio_feature_lens.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feature_lens.unsqueeze(1).expand(
                    batch_size, max_seq_len
                )
                # Create mask
                padding_mask = seq_range >= lengths_expand  # 1 for padded values

                audio_attention_mask_ = padding_mask.view(
                    batch_size, 1, 1, max_seq_len
                ).expand(batch_size, 1, max_seq_len, max_seq_len)
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.apm.conv1.weight.dtype,
                    device=self.apm.conv1.weight.device,
                )

                if chunk_length > 0:
                    chunk_num_frame = int(chunk_length * 50)
                    chunk_mask = self.subsequent_chunk_mask(
                        size=max_seq_len,
                        chunk_size=chunk_num_frame,
                        num_left_chunks=-1,
                        device=audio_attention_mask_.device,
                    )
                    audio_attention_mask_ = torch.logical_or(
                        audio_attention_mask_, torch.logical_not(chunk_mask)
                    )

                audio_attention_mask[audio_attention_mask_] = float("-inf")
                audio_states = self.apm(
                    wavform,
                    output_hidden_states=True,
                    attention_mask=audio_attention_mask,
                ).hidden_states[self.audio_encoder_layer]
                audio_embeds = self.audio_projection_layer(audio_states)

                audio_embeds = audio_embeds.transpose(1, 2)
                audio_embeds = self.audio_avg_pooler(audio_embeds)
                audio_embeds = audio_embeds.transpose(1, 2)

                _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(
                    audio_feature_lens
                )

                num_audio_tokens = feature_lens_after_pooling

                idx = 0
                for i in range(len(audio_feature_lens_raw)):
                    target_audio_embeds = []
                    for _ in range(len(audio_feature_lens_raw[i])):
                        target_audio_embeds.append(
                            audio_embeds[idx, : num_audio_tokens[idx], :]
                        )
                        idx += 1
                    final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        embedding = self.get_omni_embedding(
            items=items,
            chunk_length=self.config.audio_chunk_length,
            stream_input=False,
        )
        return embedding

    def get_omni_embedding(
        self,
        items: List[MultimodalDataItem],
        chunk_length=-1,
        stream_input=False,
    ):
        """
        Args:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding
        Returns:
            final embeddings with audio feature
        """

        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(items)
        else:
            audio_embeddings = self.get_audio_embedding(items, chunk_length)
        bs = len(audio_embeddings)
        # batch size
        audio_embs = torch.cat(flatten_nested_list(audio_embeddings), dim=0)

        return audio_embs

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # list of tensors
        pixel_values = flatten_nested_list([item.pixel_values for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return self.resampler(vision_embedding, tgt_sizes)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:

        mm_input = forward_batch.merge_mm_inputs()
        placeholder_token_ids = (
            ([mm_input.im_token_id] + [item.pad_value for item in mm_input.mm_items])
            if forward_batch.contains_mm_inputs()
            else []
        )
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm,
            image_data_embedding_func=self.get_image_feature,
            audio_data_embedding_func=self.get_audio_feature,
            placeholder_token_ids=placeholder_token_ids,
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq~" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # adapt to parametrization
            if self.config.init_tts and "tts" in name:
                name = name.replace(".parametrizations", "")
                name = name.replace(".weight.original0", ".weight_g")
                name = name.replace(".weight.original1", ".weight_v")

            # adapt to VisionAttention
            if "vpm" in name:
                name = name.replace(r"self_attn.out_proj", r"self_attn.proj")

            if not self.config.init_tts and "tts" in name:
                continue
            if not self.config.init_audio and ("apm" in name or "audio" in name):
                continue
            if not self.config.init_vision and "vpm" in name:
                continue

            if (
                "sampler" in name
                or "apm" in name
                or ("tts" in name and "self_attn" in name)
                or ("tts.model.layers" in name and ".mlp" in name)
            ):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # replace the name and load with customized loader
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [MiniCPMO]
