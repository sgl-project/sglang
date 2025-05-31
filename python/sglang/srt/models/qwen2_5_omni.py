# Copied and adapted from: https://github.com/huggingface/transformers/blob/5efaed689114030ffaf51c02f6f82adcbfc72389/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
from typing import Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLMLP,
)

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Attention, Qwen2MLP
from sglang.srt.utils import add_prefix
from sglang.utils import logger

############################
#      Start Thinker       #
############################


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2)
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniVisionBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            qkv_backend="sdpa",
            softmax_in_single_precision=True,
            flatten_batch=True,
            quant_config=quant_config,
            proj_bias=True,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, position_embeddings) -> torch.Tensor:
        seq_len, _ = hidden_states.size()

        normed_hs = self.norm1(hidden_states)
        normed_hs = normed_hs.unsqueeze(0)
        attn = self.attn(
            normed_hs, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + attn
        hidden_states = hidden_states.view(seq_len, -1)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5OmniPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5OmniVisionEncoder(nn.Module):
    _no_split_modules = ["Qwen2_5OmniVisionBlock"]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5OmniVisionBlock(config, quant_config=quant_config)
                for _ in range(config.depth)
            ]
        )
        self.merger = Qwen2_5OmniPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Modification here
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.gate_proj.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.gate_proj.weight.device


class Qwen2_5OmniDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        if (
            config.use_sliding_window
            and config._attn_implementation != "flash_attention_2"
        ):
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        self.rope_scaling = config.rope_scaling
        hidden_size = config.hidden_size
        head_num = config.num_attention_heads
        kv_head_num = config.num_key_value_heads
        head_dim = hidden_size // head_num

        self.q_size = head_num * head_dim
        self.kv_size = kv_head_num * head_dim

        text_config = config
        self.mlp = Qwen2MLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        rope_scaling = config.rope_scaling
        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_heads=head_num,
            num_kv_heads=kv_head_num,
            layer_id=layer_idx,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        text_config = config
        self.mlp = Qwen2MLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=position_ids,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2_5OmniThinkerModel(nn.Module):
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )

        self.layers = nn.ModuleList(
            [
                Qwen2_5OmniDecoderLayer(config, layer_idx, quant_config=quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_output = decoder_layer(
                positions=positions,
                position_ids=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states = layer_output

        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        assert hidden_states.dim() == 2, hidden_states.shape
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen2_5OmniThinkerForConditionalGeneration(nn.Module):
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config
        self.text_config = config.text_config

        self.audio_tower = Qwen2_5OmniAudioEncoder(config.audio_config)

        self.visual = Qwen2_5OmniVisionEncoder(
            config.vision_config, quant_config=quant_config
        )

        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerModel(
            config.text_config, quant_config=quant_config
        )
        text_config = config.text_config
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.is_mrope_enabled = "mrope_section" in self.config.text_config.rope_scaling

        self.logits_processor = LogitsProcessor(config.text_config)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.concat([item.pixel_values for item in items], dim=0).type(
            self.visual.dtype
        )

        image_grid_thws = torch.concat([item.image_grid_thws for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thws.dim() == 2, image_grid_thws.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thws)

        return image_embeds

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        input_features = (
            torch.cat([item.audio_features for item in items])
            .type(self.audio_tower.dtype)
            .to(next(self.audio_tower.parameters()).device)
        )
        feature_attention_mask = torch.cat(
            [item.feature_attention_mask for item in items]
        )
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_attention_mask = torch.sum(feature_attention_mask, dim=1)

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                audio_feature_lengths
                if audio_feature_lengths is not None
                else feature_attention_mask.sum(-1)
            )
        )
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            image_data_embedding_func=self.get_image_feature,
            audio_data_embedding_func=self.get_audio_feature,
            positions=positions,
        )
        return self.logits_processor(input_ids, hs, self.lm_head, forward_batch)


############################
#    Start Qwen2.5Omni     #
############################


class Qwen2_5OmniModel(nn.Module):
    _no_split_modules = [
        "Qwen2_5OmniTalkerForConditionalGeneration",
        "Qwen2_5OmniToken2WavModel",
    ]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(
            config.thinker_config,
            quant_config=quant_config,
            prefix=add_prefix("thinker", prefix),
        )
        self.has_talker = config.enable_audio_output
        self.speaker_map = {}

        config.enable_audio_output = False
        logger.info(f"Talker is not yet supported.")
        if config.enable_audio_output:
            self.enable_talker()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        media_token_ids = [mm_inputs.im_token_id, mm_inputs.audio_token_id]
        pattern = MultiModalityDataPaddingPatternMultimodalTokens(media_token_ids)
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def load_speakers(self, path):
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value
        logger.info("Speaker {} loaded".format(list(self.speaker_map.keys())))

    @classmethod
    def can_generate(cls) -> bool:
        return True

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # 1. Generate from thinker module
        thinker_result = self.thinker.forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
        )

        return thinker_result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # VisionAttention
            (".qkv_proj.", ".q.", "q"),
            (".qkv_proj.", ".k.", "k"),
            (".qkv_proj.", ".v.", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "audio_tower" in name or "talker" in name:
                    continue

                if "visual" in name:
                    # mlp
                    if "gate_proj" in name or "up_proj" in name:
                        continue
                    ...

                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    print(f"skipping {name}")
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

                if "talker" in name or "token2wav" in name:
                    continue
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            logger.warn(
                f"Some weights are not initialized from checkpoints: {sorted(unloaded_params)}"
            )


EntryClass = Qwen2_5OmniModel
