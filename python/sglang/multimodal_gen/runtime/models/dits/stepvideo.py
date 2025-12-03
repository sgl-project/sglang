# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import Any

import torch
from einops import rearrange, repeat
from torch import nn

from sglang.multimodal_gen.configs.models.dits import StepVideoConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import TimestepEmbedder
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert patch_size to 2-tuple
        if isinstance(patch_size, list | tuple):
            if len(patch_size) == 1:
                patch_size = (patch_size[0], patch_size[0])
        else:
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            dtype=dtype,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class StepVideoRMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class SelfAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        head_dim,
        rope_split: tuple[int, int, int] = (64, 32, 32),
        bias: bool = False,
        with_rope: bool = True,
        with_qk_norm: bool = True,
        attn_type: str = "torch",
        supported_attention_backends=(
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        ),
    ):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.rope_split = list(rope_split)
        self.n_heads = hidden_dim // head_dim

        self.wqkv = ReplicatedLinear(hidden_dim, hidden_dim * 3, bias=bias)
        self.wo = ReplicatedLinear(hidden_dim, hidden_dim, bias=bias)

        self.with_rope = with_rope
        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = StepVideoRMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = StepVideoRMSNorm(head_dim, elementwise_affine=True)

        # self.core_attention = self.attn_processor(attn_type=attn_type)
        self.parallel = attn_type == "parallel"
        self.attn = USPAttention(
            num_heads=self.n_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        x:   [B, S, H, D]
        cos: [S, D/2]  where D = head_dim = sum(self.rope_split)
        sin: [S, D/2]
        returns x with rotary applied exactly as v0 did
        """
        B, S, H, D = x.shape
        # 1) split cos/sin per chunk
        half_splits = [c // 2 for c in self.rope_split]  # [32,16,16] for [64,32,32]
        cos_splits = cos.split(half_splits, dim=1)
        sin_splits = sin.split(half_splits, dim=1)

        outs = []
        idx = 0
        for chunk_size, cos_i, sin_i in zip(
            self.rope_split, cos_splits, sin_splits, strict=True
        ):
            # slice the corresponding channels
            x_chunk = x[..., idx : idx + chunk_size]  # [B,S,H,chunk_size]
            idx += chunk_size

            # flatten to [S, B*H, chunk_size]
            x_flat = rearrange(x_chunk, "b s h d -> s (b h) d")

            # apply rotary on *that* chunk
            out_flat = _apply_rotary_emb(x_flat, cos_i, sin_i, is_neox_style=True)

            # restore [B,S,H,chunk_size]
            out = rearrange(out_flat, "s (b h) d -> b s h d", b=B, h=H)
            outs.append(out)

        # concatenate back to [B,S,H,D]
        return torch.cat(outs, dim=-1)

    def forward(
        self,
        x,
        cu_seqlens=None,
        max_seqlen=None,
        rope_positions=None,
        cos_sin=None,
        attn_mask=None,
        mask_strategy=None,
    ):

        B, S, _ = x.shape
        xqkv, _ = self.wqkv(x)
        xqkv = xqkv.view(*x.shape[:-1], self.n_heads, 3 * self.head_dim)
        q, k, v = torch.split(xqkv, [self.head_dim] * 3, dim=-1)  # [B,S,H,D]

        if self.with_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.with_rope:
            if rope_positions is not None:
                F, Ht, W = rope_positions
            assert F * Ht * W == S, "rope_positions mismatches sequence length"

            cos, sin = cos_sin
            cos = cos.to(x.device, dtype=x.dtype)
            sin = sin.to(x.device, dtype=x.dtype)

            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        output = self.attn(q, k, v)  # [B,heads,S,D]

        output = rearrange(output, "b s h d -> b s (h d)")
        output, _ = self.wo(output)

        return output


class CrossAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        head_dim,
        bias=False,
        with_qk_norm=True,
        supported_attention_backends=(
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        ),
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = hidden_dim // head_dim

        self.wq = ReplicatedLinear(hidden_dim, hidden_dim, bias=bias)
        self.wkv = ReplicatedLinear(hidden_dim, hidden_dim * 2, bias=bias)
        self.wo = ReplicatedLinear(hidden_dim, hidden_dim, bias=bias)

        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = StepVideoRMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = StepVideoRMSNorm(head_dim, elementwise_affine=True)

        self.attn = LocalAttention(
            num_heads=self.n_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, attn_mask=None
    ) -> torch.Tensor:

        xq, _ = self.wq(x)
        xq = xq.view(*xq.shape[:-1], self.n_heads, self.head_dim)

        xkv, _ = self.wkv(encoder_hidden_states)
        xkv = xkv.view(*xkv.shape[:-1], self.n_heads, 2 * self.head_dim)

        xk, xv = torch.split(xkv, [self.head_dim] * 2, dim=-1)  ## seq_len, n, dim

        if self.with_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        output = self.attn(xq, xk, xv)

        output = rearrange(output, "b s h d -> b s (h d)")
        output, _ = self.wo(output)

        return output


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, time_step_rescale=1000):
        super().__init__()

        self.emb = TimestepEmbedder(embedding_dim)

        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(embedding_dim, 6 * embedding_dim, bias=True)

        self.time_step_rescale = time_step_rescale  ## timestep usually in [0, 1], we rescale it to [0,1000] for stability

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep * self.time_step_rescale)

        out, _ = self.linear(self.silu(embedded_timestep))

        return out, embedded_timestep


class StepVideoTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        attention_head_dim: int,
        norm_eps: float = 1e-5,
        ff_inner_dim: int | None = None,
        ff_bias: bool = False,
        attention_type: str = "torch",
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNormScaleShift(
            dim, norm_type="layer", elementwise_affine=True, eps=norm_eps
        )
        self.attn1 = SelfAttention(
            dim,
            attention_head_dim,
            bias=False,
            with_rope=True,
            with_qk_norm=True,
        )

        self.norm2 = LayerNormScaleShift(
            dim, norm_type="layer", elementwise_affine=True, eps=norm_eps
        )
        self.attn2 = CrossAttention(
            dim, attention_head_dim, bias=False, with_qk_norm=True
        )

        self.ff = MLP(
            input_dim=dim,
            mlp_hidden_dim=dim * 4 if ff_inner_dim is None else ff_inner_dim,
            act_type="gelu_pytorch_tanh",
            bias=ff_bias,
        )

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    @torch.no_grad()
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        t_expand: torch.LongTensor,
        attn_mask=None,
        rope_positions: list | None = None,
        cos_sin=None,
        mask_strategy=None,
    ) -> torch.Tensor:

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            torch.clone(chunk)
            for chunk in (
                self.scale_shift_table[None] + t_expand.reshape(-1, 6, self.dim)
            ).chunk(6, dim=1)
        )

        scale_shift_q = self.norm1(
            q, scale=scale_msa.squeeze(1), shift=shift_msa.squeeze(1)
        )

        attn_q = self.attn1(
            scale_shift_q,
            rope_positions=rope_positions,
            cos_sin=cos_sin,
            mask_strategy=mask_strategy,
        )

        q = attn_q * gate_msa + q

        attn_q = self.attn2(q, kv, attn_mask)

        q = attn_q + q

        scale_shift_q = self.norm2(
            q, scale=scale_mlp.squeeze(1), shift=shift_mlp.squeeze(1)
        )

        ff_output = self.ff(scale_shift_q)

        q = ff_output * gate_mlp + q

        return q


class StepVideoModel(BaseDiT):
    # (Optional) Keep the same attribute for compatibility with splitting, etc.
    _fsdp_shard_conditions = [
        lambda n, m: "transformer_blocks" in n and n.split(".")[-1].isdigit(),
        # lambda n, m: "pos_embed" in n  # If needed for the patch embedding.
    ]
    param_names_mapping = StepVideoConfig().param_names_mapping
    reverse_param_names_mapping = StepVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = StepVideoConfig().lora_param_names_mapping
    _supported_attention_backends = StepVideoConfig()._supported_attention_backends

    def __init__(self, config: StepVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.patch_size = config.patch_size
        self.norm_type = config.norm_type
        self.norm_elementwise_affine = config.norm_elementwise_affine
        self.norm_eps = config.norm_eps
        self.use_additional_conditions = config.use_additional_conditions
        self.caption_channels = config.caption_channels
        self.attention_type = config.attention_type
        self.num_channels_latents = config.num_channels_latents
        # Compute inner dimension.
        self.hidden_size = config.hidden_size

        # Image/video patch embedding.
        self.pos_embed = PatchEmbed2D(
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
        )

        self._rope_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        # Transformer blocks.
        self.transformer_blocks = nn.ModuleList(
            [
                StepVideoTransformerBlock(
                    dim=self.hidden_size,
                    attention_head_dim=self.attention_head_dim,
                    attention_type=self.attention_type,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output blocks.
        self.norm_out = LayerNormScaleShift(
            self.hidden_size,
            norm_type="layer",
            eps=self.norm_eps,
            elementwise_affine=self.norm_elementwise_affine,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.hidden_size) / (self.hidden_size**0.5)
        )
        self.proj_out = ReplicatedLinear(
            self.hidden_size, self.patch_size * self.patch_size * self.out_channels
        )
        # Time modulation via adaptive layer norm.
        self.adaln_single = AdaLayerNormSingle(self.hidden_size)

        # Set up caption conditioning.
        if isinstance(self.caption_channels, int):
            caption_channel = self.caption_channels
        else:
            caption_channel, clip_channel = self.caption_channels
            self.clip_projection = ReplicatedLinear(clip_channel, self.hidden_size)
        self.caption_norm = nn.LayerNorm(
            caption_channel,
            eps=self.norm_eps,
            elementwise_affine=self.norm_elementwise_affine,
        )
        self.caption_projection = MLP(
            input_dim=caption_channel,
            mlp_hidden_dim=self.hidden_size,
            act_type="gelu_pytorch_tanh",
        )

        # Flag to indicate if using parallel attention.
        self.parallel = self.attention_type == "parallel"

        self.__post_init__()

    def patchfy(self, hidden_states) -> torch.Tensor:
        hidden_states = rearrange(hidden_states, "b f c h w -> (b f) c h w")
        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    def prepare_attn_mask(
        self, encoder_attention_mask, encoder_hidden_states, q_seqlen
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_seqlens = encoder_attention_mask.sum(dim=1).int()
        mask = torch.zeros(
            [len(kv_seqlens), q_seqlen, max(kv_seqlens)],
            dtype=torch.bool,
            device=encoder_attention_mask.device,
        )
        encoder_hidden_states = encoder_hidden_states[:, : max(kv_seqlens)]
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 1
        return encoder_hidden_states, mask

    def block_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        t_expand=None,
        rope_positions=None,
        cos_sin=None,
        attn_mask=None,
        parallel=True,
        mask_strategy=None,
    ) -> torch.Tensor:

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                t_expand=t_expand,
                attn_mask=attn_mask,
                rope_positions=rope_positions,
                cos_sin=cos_sin,
                mask_strategy=mask_strategy[i],
            )

        return hidden_states

    def _get_rope(
        self,
        rope_positions: tuple[int, int, int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        F, Ht, W = rope_positions
        key = (F, Ht, W, dtype)
        if key not in self._rope_cache:
            cos, sin = get_rotary_pos_embed(
                rope_sizes=(F * get_sp_world_size(), Ht, W),
                hidden_size=self.hidden_size,
                heads_num=self.hidden_size // self.attention_head_dim,
                rope_dim_list=(64, 32, 32),  # same split you used
                rope_theta=1.0e4,
                dtype=torch.float32,  # build once in fp32
            )
            # move & cast once
            self._rope_cache[key] = (
                cos.to(device, dtype=dtype),
                sin.to(device, dtype=dtype),
            )
        return self._rope_cache[key]

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        t_expand: torch.LongTensor | None = None,
        encoder_hidden_states_2: torch.Tensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        return_dict: bool = True,
        mask_strategy=None,
        guidance=None,
    ):
        assert hidden_states.ndim == 5
        "hidden_states's shape should be (bsz, f, ch, h ,w)"
        frame = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> b f c h w", f=frame)
        if mask_strategy is None:
            mask_strategy = [None, None]
        bsz, frame, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        hidden_states = self.patchfy(hidden_states)
        len_frame = hidden_states.shape[1]

        t_expand, embedded_timestep = self.adaln_single(t_expand)
        encoder_hidden_states = self.caption_projection(
            self.caption_norm(encoder_hidden_states)
        )

        if encoder_hidden_states_2 is not None and hasattr(self, "clip_projection"):
            clip_embedding, _ = self.clip_projection(encoder_hidden_states_2)
            encoder_hidden_states = torch.cat(
                [clip_embedding, encoder_hidden_states], dim=1
            )

        hidden_states = rearrange(
            hidden_states, "(b f) l d->  b (f l) d", b=bsz, f=frame, l=len_frame
        ).contiguous()
        encoder_hidden_states, attn_mask = self.prepare_attn_mask(
            encoder_attention_mask, encoder_hidden_states, q_seqlen=frame * len_frame
        )

        cos_sin = self._get_rope(
            (frame, height, width), hidden_states.dtype, hidden_states.device
        )

        hidden_states = self.block_forward(
            hidden_states,
            encoder_hidden_states,
            t_expand=t_expand,
            rope_positions=[frame, height, width],
            cos_sin=cos_sin,
            attn_mask=attn_mask,
            parallel=self.parallel,
            mask_strategy=mask_strategy,
        )

        hidden_states = rearrange(
            hidden_states, "b (f l) d -> (b f) l d", b=bsz, f=frame, l=len_frame
        )

        embedded_timestep = repeat(
            embedded_timestep, "b d -> (b f) d", f=frame
        ).contiguous()

        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None]
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(
            hidden_states, shift=shift.squeeze(1), scale=scale.squeeze(1)
        )
        # Modulation
        hidden_states, _ = self.proj_out(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            )
        )

        hidden_states = rearrange(hidden_states, "n h w p q c -> n c h p w q")
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.patch_size,
                width * self.patch_size,
            )
        )

        output = rearrange(output, "(b f) c h w -> b c f h w", f=frame)
        return output


EntryClass = StepVideoModel
