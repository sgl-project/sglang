import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.runtime.models.dits.ltx_2_layers import (
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerArgs,
    AdaLayerNormSingle,
    PixArtAlphaCombinedTimestepSizeEmbeddings,
    rms_norm,
)

class LTX2Transformer(BaseDiT):
    """
    LTX-2 Asymmetric Dual-Stream Transformer implementation for SGLang.
    """
    
    param_names_mapping = LTX2Config().arch_config.param_names_mapping
    _fsdp_shard_conditions = [
        lambda name, module: "transformer_blocks" in name,
    ]
    _compile_conditions: list = []
    
    def __init__(
        self,
        config: LTX2Config,
        hf_config: dict[str, Any],
        **kwargs
    ):
        super().__init__(config, hf_config)
        self.config = config.arch_config
        
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        
        self.audio_in_channels = self.config.audio_in_channels
        self.audio_out_channels = self.config.audio_out_channels
        self.audio_inner_dim = self.config.audio_num_attention_heads * self.config.audio_attention_head_dim
        
        # --- Video Stream Components ---
        self.patchify_proj = nn.Linear(self.in_channels, self.inner_dim)
        self.adaln_single = AdaLayerNormSingle(self.inner_dim)
        self.caption_projection = nn.Linear(self.config.caption_channels, self.inner_dim)
        
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels)
        
        # --- Audio Stream Components ---
        self.audio_patchify_proj = nn.Linear(self.audio_in_channels, self.audio_inner_dim)
        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim)
        self.audio_caption_projection = nn.Linear(self.config.caption_channels, self.audio_inner_dim)
        
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, self.audio_out_channels)
        
        # --- Audio-Video Cross Components ---
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=4)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=4)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=1)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=1)

        # --- Blocks ---
        video_config = TransformerConfig(
            dim=self.inner_dim,
            heads=self.config.num_attention_heads,
            d_head=self.config.attention_head_dim,
            context_dim=self.config.cross_attention_dim,
        )
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.config.audio_num_attention_heads,
            d_head=self.config.audio_attention_head_dim,
            context_dim=self.config.audio_cross_attention_dim,
        )
        
        self.transformer_blocks = nn.ModuleList([
            BasicAVTransformerBlock(
                idx=i,
                video=video_config,
                audio=audio_config,
            ) for i in range(self.config.num_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor, # Video latents
        audio_hidden_states: Optional[torch.Tensor] = None, # Audio latents
        encoder_hidden_states: Optional[torch.Tensor] = None, # Text embeddings
        timestep: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # 1. Patchify
        if hidden_states.ndim == 5:
            b, c, f, h, w = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3) # [B, L, C]
            
        x = self.patchify_proj(hidden_states)
        
        # Audio Path
        x_audio = None
        if audio_hidden_states is not None:
            if audio_hidden_states.ndim == 4:  # [B, C, T, F] (channels, frames, mel_bins)
                b, c, t, f = audio_hidden_states.shape
                audio_hidden_states = (
                    audio_hidden_states.permute(0, 2, 1, 3).reshape(b, t, c * f)
                )  # [B, T, C*F] == [B, frames, audio_in_channels]
            x_audio = self.audio_patchify_proj(audio_hidden_states)
            
        # 2. AdaLN & Timestep
        # Prepare timestep embeddings
        # Video
        v_emb, v_ts = self.adaln_single(timestep)
        # Audio
        a_emb, a_ts = self.audio_adaln_single(timestep)
        
        # Cross Attention Timesteps
        _, av_scale_shift_ts = self.av_ca_video_scale_shift_adaln_single(timestep)
        _, av_gate_ts = self.av_ca_a2v_gate_adaln_single(timestep)
        
        # Prepare Args
        video_args = TransformerArgs(
            x=x,
            timesteps=v_ts,
            context=encoder_hidden_states, # Assume text embeddings match
            cross_scale_shift_timestep=av_scale_shift_ts,
            cross_gate_timestep=av_gate_ts,
        )
        
        audio_args = TransformerArgs(
            x=x_audio,
            timesteps=a_ts,
            context=encoder_hidden_states, # Assume text embeddings match
            cross_scale_shift_timestep=av_scale_shift_ts, # Reuse for simplicity or compute audio specific
            cross_gate_timestep=av_gate_ts,
        )
        
        # 3. Blocks
        for block in self.transformer_blocks:
            video_args, audio_args = block(video_args, audio_args)
        
        x = video_args.x
        x_audio = audio_args.x
        
        # 4. Output Projection
        # Final Norm & Proj
        # Video
        shift, scale = self.scale_shift_table.chunk(2, dim=0)
        x = rms_norm(x) * (1 + scale) + shift
        x = self.proj_out(x)
        
        # Audio
        if x_audio is not None:
            shift_a, scale_a = self.audio_scale_shift_table.chunk(2, dim=0)
            x_audio = rms_norm(x_audio) * (1 + scale_a) + shift_a
            x_audio = self.audio_proj_out(x_audio)
            
        return x, x_audio

EntryClass = LTX2Transformer
