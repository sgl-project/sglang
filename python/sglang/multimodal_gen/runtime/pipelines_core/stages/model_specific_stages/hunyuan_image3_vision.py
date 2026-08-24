"""Minimal SigLIP2 Vision Transformer and LightProjector for HunyuanImage-3.

Ported from vllm-omni's ``siglip2.py`` without tensor-parallelism support.
Used by the AR stage to encode conditional images for TI2I / I2I tasks.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class _Config:
    """Convert dict config to object with attribute access."""

    def __init__(self, config):
        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


# ---------------------------------------------------------------------------
# SigLIP2 Vision Embeddings
# ---------------------------------------------------------------------------

class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Resize position embeddings per image and concatenate (packed)
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        pe_for_resize = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        if pe_for_resize.device.type == "cpu":
            pe_for_resize = pe_for_resize.to(torch.float32)

        position_embs: list[torch.Tensor] = []
        for i in range(spatial_shapes.shape[0]):
            height, width = int(spatial_shapes[i, 0]), int(spatial_shapes[i, 1])
            resized = F.interpolate(
                pe_for_resize,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape(self.embed_dim, height * width).transpose(0, 1)
            position_embs.append(resized.to(target_dtype))

        packed_position_embs = torch.cat(position_embs, dim=0)
        return patch_embeds + packed_position_embs


# ---------------------------------------------------------------------------
# SigLIP2 Attention (plain nn.Linear, no TP)
# ---------------------------------------------------------------------------

class Siglip2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Packed attention with cumulative sequence lengths."""
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        # Build attention mask from cu_seqlens for packed sequence
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        batch_size = cu_seqlens.shape[0] - 1

        # Use SDPA with proper masking for packed sequences
        # Expand to batch format (B, H, max_seqlen, head_dim)
        attn_output = self._packed_sdpa(q, k, v, cu_seqlens, batch_size, max_seqlen)
        attn_output = attn_output.reshape(seq_length, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)

    def _packed_sdpa(self, q, k, v, cu_seqlens, batch_size, max_seqlen):
        """Run scaled-dot-product attention on packed sequences."""
        head_dim = q.shape[-1]
        num_heads = q.shape[1]

        # Create output buffer
        output = torch.zeros_like(q)

        for i in range(batch_size):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start

            qi = q[start:end]  # (seq_len, H, D)
            ki = k[start:end]
            vi = v[start:end]

            # Transpose to (H, seq_len, D) for SDPA
            qi = qi.transpose(0, 1)
            ki = ki.transpose(0, 1)
            vi = vi.transpose(0, 1)

            out_i = F.scaled_dot_product_attention(
                qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
                is_causal=False,
            ).squeeze(0)  # (H, seq_len, D)

            output[start:end] = out_i.transpose(0, 1)

        return output


# ---------------------------------------------------------------------------
# SigLIP2 MLP
# ---------------------------------------------------------------------------

class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# SigLIP2 Encoder Layer / Encoder
# ---------------------------------------------------------------------------

class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, cu_seqlens):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            Siglip2EncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states, cu_seqlens):
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens)
        return hidden_states


# ---------------------------------------------------------------------------
# SigLIP2 Vision Transformer
# ---------------------------------------------------------------------------

class Siglip2VisionTransformer(nn.Module):
    """Minimal SigLIP2 Vision Transformer (no TP).

    Input/output format matches vllm-omni's version:
        pixel_values:   (B, max_patches, C*P*P)
        attention_mask:  (B, max_patches) 1=real, 0=pad
        spatial_shapes:  (B, 2) (h, w) per image
    Returns:
        (B, max_patches, hidden_size) with zeros at padding positions
    """

    def __init__(self, config):
        super().__init__()
        config = _Config(config)
        self.config = config
        self.embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Batched pixel values
                (B, max_num_patches, num_channels * patch_size * patch_size)
            attention_mask: (B, max_num_patches) with 1 for real, 0 for padding
            spatial_shapes: (B, 2) with (height, width) per image

        Returns:
            (B, max_num_patches, hidden_size) with zeros at padding positions
        """
        batch_size, max_patches, _ = pixel_values.shape

        # Ensure attention_mask is 2D (B, max_patches)
        if attention_mask.ndim > 2:
            attention_mask = attention_mask.reshape(batch_size, max_patches)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Ensure spatial_shapes is 2D (B, 2)
        spatial_shapes = spatial_shapes.reshape(-1, 2)

        # Pack: extract real tokens using attention_mask
        mask_bool = attention_mask.bool()
        packed_pixels = pixel_values[mask_bool]

        # Compute cu_seqlens from spatial_shapes
        seq_lens = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(torch.int32)
        cu_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=pixel_values.device
        )
        cu_seqlens[1:] = seq_lens.cumsum(0)

        # Embeddings (packed)
        hidden_states = self.embeddings(packed_pixels, spatial_shapes)

        # Encoder (packed)
        hidden_states = self.encoder(hidden_states, cu_seqlens)

        # Post layernorm
        hidden_states = self.post_layernorm(hidden_states)

        # Unpack: scatter back to (B, max_patches, hidden_size)
        output = torch.zeros(
            batch_size, max_patches, self.embed_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        output[mask_bool] = hidden_states

        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with qkv stacking (matches vllm-omni's load_weights)."""
        stacked_params_mapping = [
            ("q_proj", "q_proj", None),
            ("k_proj", "k_proj", None),
            ("v_proj", "v_proj", None),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                param.data.copy_(loaded_weight)
                loaded_params.add(name)
        return loaded_params


# ---------------------------------------------------------------------------
# LightProjector (vision aligner)
# ---------------------------------------------------------------------------

class LightProjector(nn.Module):
    """Simple projection layer for aligning ViT embeddings to transformer dim."""

    def __init__(self, config):
        config = _Config(config)
        super().__init__()

        if config.projector_type == "linear":
            self.layers = nn.Linear(config.input_dim, config.n_embed)
        elif config.projector_type == "mlp_gelu":
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x):
        return self.layers(x)
