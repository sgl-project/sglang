# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, Literal, List, Callable
from einops import rearrange
from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention, AttnProcessor


class AttnUtils:
    """
    Shared utility functions for attention processing.

    This class provides common operations used across different attention processors
    to eliminate code duplication and improve maintainability.
    """

    @staticmethod
    def check_pytorch_compatibility():
        """
        Check PyTorch compatibility for scaled_dot_product_attention.

        Raises:
            ImportError: If PyTorch version doesn't support scaled_dot_product_attention
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    @staticmethod
    def handle_deprecation_warning(args, kwargs):
        """
        Handle deprecation warning for the 'scale' argument.

        Args:
            args: Positional arguments passed to attention processor
            kwargs: Keyword arguments passed to attention processor
        """
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored."
                "Please remove it, as passing it will raise an error in the future."
                "`scale` should directly be passed while calling the underlying pipeline component"
                "i.e., via `cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)

    @staticmethod
    def prepare_hidden_states(
        hidden_states, attn, temb, spatial_norm_attr="spatial_norm", group_norm_attr="group_norm"
    ):
        """
        Common preprocessing of hidden states for attention computation.

        Args:
            hidden_states: Input hidden states tensor
            attn: Attention module instance
            temb: Optional temporal embedding tensor
            spatial_norm_attr: Attribute name for spatial normalization
            group_norm_attr: Attribute name for group normalization

        Returns:
            Tuple of (processed_hidden_states, residual, input_ndim, shape_info)
        """
        residual = hidden_states

        spatial_norm = getattr(attn, spatial_norm_attr, None)
        if spatial_norm is not None:
            hidden_states = spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        group_norm = getattr(attn, group_norm_attr, None)
        if group_norm is not None:
            hidden_states = group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        return hidden_states, residual, input_ndim, (batch_size, channel, height, width)

    @staticmethod
    def prepare_attention_mask(attention_mask, attn, sequence_length, batch_size):
        """
        Prepare attention mask for scaled_dot_product_attention.

        Args:
            attention_mask: Input attention mask tensor or None
            attn: Attention module instance
            sequence_length: Length of the sequence
            batch_size: Batch size

        Returns:
            Prepared attention mask tensor reshaped for multi-head attention
        """
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        return attention_mask

    @staticmethod
    def reshape_qkv_for_attention(tensor, batch_size, attn_heads, head_dim):
        """
        Reshape Q/K/V tensors for multi-head attention computation.

        Args:
            tensor: Input tensor to reshape
            batch_size: Batch size
            attn_heads: Number of attention heads
            head_dim: Dimension per attention head

        Returns:
            Reshaped tensor with shape [batch_size, attn_heads, seq_len, head_dim]
        """
        return tensor.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

    @staticmethod
    def apply_norms(query, key, norm_q, norm_k):
        """
        Apply Q/K normalization layers if available.

        Args:
            query: Query tensor
            key: Key tensor
            norm_q: Query normalization layer (optional)
            norm_k: Key normalization layer (optional)

        Returns:
            Tuple of (normalized_query, normalized_key)
        """
        if norm_q is not None:
            query = norm_q(query)
        if norm_k is not None:
            key = norm_k(key)
        return query, key

    @staticmethod
    def finalize_output(hidden_states, input_ndim, shape_info, attn, residual, to_out):
        """
        Common output processing including projection, dropout, reshaping, and residual connection.

        Args:
            hidden_states: Processed hidden states from attention
            input_ndim: Original input tensor dimensions
            shape_info: Tuple containing original shape information
            attn: Attention module instance
            residual: Residual connection tensor
            to_out: Output projection layers [linear, dropout]

        Returns:
            Final output tensor after all processing steps
        """
        batch_size, channel, height, width = shape_info

        # Apply output projection and dropout
        hidden_states = to_out[0](hidden_states)
        hidden_states = to_out[1](hidden_states)

        # Reshape back if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # Apply residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # Apply rescaling
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Base class for attention processors (eliminating initialization duplication)
class BaseAttnProcessor(nn.Module):
    """
    Base class for attention processors with common initialization.

    This base class provides shared parameter initialization and module registration
    functionality to reduce code duplication across different attention processor types.
    """

    def __init__(
        self,
        query_dim: int,
        pbr_setting: List[str] = ["albedo", "mr"],
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
        **kwargs,
    ):
        """
        Initialize base attention processor with common parameters.

        Args:
            query_dim: Dimension of query features
            pbr_setting: List of PBR material types to process (e.g., ["albedo", "mr"])
            cross_attention_dim: Dimension of cross-attention features (optional)
            heads: Number of attention heads
            kv_heads: Number of key-value heads for grouped query attention (optional)
            dim_head: Dimension per attention head
            dropout: Dropout rate
            bias: Whether to use bias in linear projections
            upcast_attention: Whether to upcast attention computation to float32
            upcast_softmax: Whether to upcast softmax computation to float32
            cross_attention_norm: Type of cross-attention normalization (optional)
            cross_attention_norm_num_groups: Number of groups for cross-attention norm
            qk_norm: Type of query-key normalization (optional)
            added_kv_proj_dim: Dimension for additional key-value projections (optional)
            added_proj_bias: Whether to use bias in additional projections
            norm_num_groups: Number of groups for normalization (optional)
            spatial_norm_dim: Dimension for spatial normalization (optional)
            out_bias: Whether to use bias in output projection
            scale_qk: Whether to scale query-key products
            only_cross_attention: Whether to only perform cross-attention
            eps: Small epsilon value for numerical stability
            rescale_output_factor: Factor to rescale output values
            residual_connection: Whether to use residual connections
            _from_deprecated_attn_block: Flag for deprecated attention blocks
            processor: Optional attention processor instance
            out_dim: Output dimension (optional)
            out_context_dim: Output context dimension (optional)
            context_pre_only: Whether to only process context in pre-processing
            pre_only: Whether to only perform pre-processing
            elementwise_affine: Whether to use element-wise affine transformations
            is_causal: Whether to use causal attention masking
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        AttnUtils.check_pytorch_compatibility()

        # Store common attributes
        self.pbr_setting = pbr_setting
        self.n_pbr_tokens = len(self.pbr_setting)
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention
        self.added_proj_bias = added_proj_bias

        # Validation
        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None."
                "Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

    def register_pbr_modules(self, module_types: List[str], **kwargs):
        """
        Generic PBR module registration to eliminate code repetition.

        Dynamically registers PyTorch modules for different PBR material types
        based on the specified module types and PBR settings.

        Args:
            module_types: List of module types to register ("qkv", "v_only", "out", "add_kv")
            **kwargs: Additional arguments for module configuration
        """
        for pbr_token in self.pbr_setting:
            if pbr_token == "albedo":
                continue

            for module_type in module_types:
                if module_type == "qkv":
                    self.register_module(
                        f"to_q_{pbr_token}", nn.Linear(self.query_dim, self.inner_dim, bias=self.use_bias)
                    )
                    self.register_module(
                        f"to_k_{pbr_token}", nn.Linear(self.cross_attention_dim, self.inner_dim, bias=self.use_bias)
                    )
                    self.register_module(
                        f"to_v_{pbr_token}", nn.Linear(self.cross_attention_dim, self.inner_dim, bias=self.use_bias)
                    )
                elif module_type == "v_only":
                    self.register_module(
                        f"to_v_{pbr_token}", nn.Linear(self.cross_attention_dim, self.inner_dim, bias=self.use_bias)
                    )
                elif module_type == "out":
                    if not self.pre_only:
                        self.register_module(
                            f"to_out_{pbr_token}",
                            nn.ModuleList(
                                [
                                    nn.Linear(self.inner_dim, self.out_dim, bias=kwargs.get("out_bias", True)),
                                    nn.Dropout(self.dropout),
                                ]
                            ),
                        )
                    else:
                        self.register_module(f"to_out_{pbr_token}", None)
                elif module_type == "add_kv":
                    if self.added_kv_proj_dim is not None:
                        self.register_module(
                            f"add_k_proj_{pbr_token}",
                            nn.Linear(self.added_kv_proj_dim, self.inner_kv_dim, bias=self.added_proj_bias),
                        )
                        self.register_module(
                            f"add_v_proj_{pbr_token}",
                            nn.Linear(self.added_kv_proj_dim, self.inner_kv_dim, bias=self.added_proj_bias),
                        )
                    else:
                        self.register_module(f"add_k_proj_{pbr_token}", None)
                        self.register_module(f"add_v_proj_{pbr_token}", None)


# Rotary Position Embedding utilities (specialized for PoseRoPE)
class RotaryEmbedding:
    """
    Rotary position embedding utilities for 3D spatial attention.

    Provides functions to compute and apply rotary position embeddings (RoPE)
    for 1D, 3D spatial coordinates used in 3D-aware attention mechanisms.
    """

    @staticmethod
    def get_1d_rotary_pos_embed(dim: int, pos: torch.Tensor, theta: float = 10000.0, linear_factor=1.0, ntk_factor=1.0):
        """
        Compute 1D rotary position embeddings.

        Args:
            dim: Embedding dimension (must be even)
            pos: Position tensor
            theta: Base frequency for rotary embeddings
            linear_factor: Linear scaling factor
            ntk_factor: NTK (Neural Tangent Kernel) scaling factor

        Returns:
            Tuple of (cos_embeddings, sin_embeddings)
        """
        assert dim % 2 == 0
        theta = theta * ntk_factor
        freqs = (
            1.0
            / (theta ** (torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device)[: (dim // 2)] / dim))
            / linear_factor
        )
        freqs = torch.outer(pos, freqs)
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
        return freqs_cos, freqs_sin

    @staticmethod
    def get_3d_rotary_pos_embed(position, embed_dim, voxel_resolution, theta: int = 10000):
        """
        Compute 3D rotary position embeddings for spatial coordinates.

        Args:
            position: 3D position tensor with shape [..., 3]
            embed_dim: Embedding dimension
            voxel_resolution: Resolution of the voxel grid
            theta: Base frequency for rotary embeddings

        Returns:
            Tuple of (cos_embeddings, sin_embeddings) for 3D positions
        """
        assert position.shape[-1] == 3
        dim_xy = embed_dim // 8 * 3
        dim_z = embed_dim // 8 * 2

        grid = torch.arange(voxel_resolution, dtype=torch.float32, device=position.device)
        freqs_xy = RotaryEmbedding.get_1d_rotary_pos_embed(dim_xy, grid, theta=theta)
        freqs_z = RotaryEmbedding.get_1d_rotary_pos_embed(dim_z, grid, theta=theta)

        xy_cos, xy_sin = freqs_xy
        z_cos, z_sin = freqs_z

        embed_flattn = position.view(-1, position.shape[-1])
        x_cos = xy_cos[embed_flattn[:, 0], :]
        x_sin = xy_sin[embed_flattn[:, 0], :]
        y_cos = xy_cos[embed_flattn[:, 1], :]
        y_sin = xy_sin[embed_flattn[:, 1], :]
        z_cos = z_cos[embed_flattn[:, 2], :]
        z_sin = z_sin[embed_flattn[:, 2], :]

        cos = torch.cat((x_cos, y_cos, z_cos), dim=-1)
        sin = torch.cat((x_sin, y_sin, z_sin), dim=-1)

        cos = cos.view(*position.shape[:-1], embed_dim)
        sin = sin.view(*position.shape[:-1], embed_dim)
        return cos, sin

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]):
        """
        Apply rotary position embeddings to input tensor.

        Args:
            x: Input tensor to apply rotary embeddings to
            freqs_cis: Tuple of (cos_embeddings, sin_embeddings) or single tensor

        Returns:
            Tensor with rotary position embeddings applied
        """
        cos, sin = freqs_cis
        cos, sin = cos.to(x.device), sin.to(x.device)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out


# Core attention processing logic (eliminating major duplication)
class AttnCore:
    """
    Core attention processing logic shared across processors.

    This class provides the fundamental attention computation pipeline
    that can be reused across different attention processor implementations.
    """

    @staticmethod
    def process_attention_base(
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        get_qkv_fn: Callable = None,
        apply_rope_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Generic attention processing core shared across different processors.

        This function implements the common attention computation pipeline including:
        1. Hidden state preprocessing
        2. Attention mask preparation
        3. Q/K/V computation via provided function
        4. Tensor reshaping for multi-head attention
        5. Optional normalization and RoPE application
        6. Scaled dot-product attention computation

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            get_qkv_fn: Function to compute Q, K, V tensors
            apply_rope_fn: Optional function to apply rotary position embeddings
            **kwargs: Additional keyword arguments passed to subfunctions

        Returns:
            Tuple containing (attention_output, residual, input_ndim, shape_info,
            batch_size, num_heads, head_dim)
        """
        # Prepare hidden states
        hidden_states, residual, input_ndim, shape_info = AttnUtils.prepare_hidden_states(hidden_states, attn, temb)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # Prepare attention mask
        attention_mask = AttnUtils.prepare_attention_mask(attention_mask, attn, sequence_length, batch_size)

        # Get Q, K, V
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query, key, value = get_qkv_fn(attn, hidden_states, encoder_hidden_states, **kwargs)

        # Reshape for attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = AttnUtils.reshape_qkv_for_attention(query, batch_size, attn.heads, head_dim)
        key = AttnUtils.reshape_qkv_for_attention(key, batch_size, attn.heads, head_dim)
        value = AttnUtils.reshape_qkv_for_attention(value, batch_size, attn.heads, value.shape[-1] // attn.heads)

        # Apply normalization
        query, key = AttnUtils.apply_norms(query, key, getattr(attn, "norm_q", None), getattr(attn, "norm_k", None))

        # Apply RoPE if provided
        if apply_rope_fn is not None:
            query, key = apply_rope_fn(query, key, head_dim, **kwargs)

        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        return hidden_states, residual, input_ndim, shape_info, batch_size, attn.heads, head_dim


# Specific processor implementations (minimal unique code)
class PoseRoPEAttnProcessor2_0:
    """
    Attention processor with Rotary Position Encoding (RoPE) for 3D spatial awareness.

    This processor extends standard attention with 3D rotary position embeddings
    to provide spatial awareness for 3D scene understanding tasks.
    """

    def __init__(self):
        """Initialize the RoPE attention processor."""
        AttnUtils.check_pytorch_compatibility()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_indices: Dict = None,
        temb: Optional[torch.Tensor] = None,
        n_pbrs=1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply RoPE-enhanced attention computation.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            position_indices: Dictionary containing 3D position information for RoPE
            temb: Optional temporal embedding tensor
            n_pbrs: Number of PBR material types
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Attention output tensor with applied rotary position encodings
        """
        AttnUtils.handle_deprecation_warning(args, kwargs)

        def get_qkv(attn, hidden_states, encoder_hidden_states, **kwargs):
            return attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

        def apply_rope(query, key, head_dim, **kwargs):
            if position_indices is not None:
                if head_dim in position_indices:
                    image_rotary_emb = position_indices[head_dim]
                else:
                    image_rotary_emb = RotaryEmbedding.get_3d_rotary_pos_embed(
                        rearrange(
                            position_indices["voxel_indices"].unsqueeze(1).repeat(1, n_pbrs, 1, 1),
                            "b n_pbrs l c -> (b n_pbrs) l c",
                        ),
                        head_dim,
                        voxel_resolution=position_indices["voxel_resolution"],
                    )
                    position_indices[head_dim] = image_rotary_emb

                query = RotaryEmbedding.apply_rotary_emb(query, image_rotary_emb)
                key = RotaryEmbedding.apply_rotary_emb(key, image_rotary_emb)
            return query, key

        # Core attention processing
        hidden_states, residual, input_ndim, shape_info, batch_size, heads, head_dim = AttnCore.process_attention_base(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            get_qkv_fn=get_qkv,
            apply_rope_fn=apply_rope,
            position_indices=position_indices,
            n_pbrs=n_pbrs,
        )

        # Finalize output
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        hidden_states = hidden_states.to(hidden_states.dtype)

        return AttnUtils.finalize_output(hidden_states, input_ndim, shape_info, attn, residual, attn.to_out)


class SelfAttnProcessor2_0(BaseAttnProcessor):
    """
    Self-attention processor with PBR (Physically Based Rendering) material support.

    This processor handles multiple PBR material types (e.g., albedo, metallic-roughness)
    with separate attention computation paths for each material type.
    """

    def __init__(self, **kwargs):
        """
        Initialize self-attention processor with PBR support.

        Args:
            **kwargs: Arguments passed to BaseAttnProcessor initialization
        """
        super().__init__(**kwargs)
        self.register_pbr_modules(["qkv", "out", "add_kv"], **kwargs)

    def process_single(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        token: Literal["albedo", "mr"] = "albedo",
        multiple_devices=False,
        *args,
        **kwargs,
    ):
        """
        Process attention for a single PBR material type.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            token: PBR material type to process ("albedo", "mr", etc.)
            multiple_devices: Whether to use multiple GPU devices
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed attention output for the specified PBR material type
        """
        target = attn if token == "albedo" else attn.processor
        token_suffix = "" if token == "albedo" else "_" + token

        # Device management (if needed)
        if multiple_devices:
            device = torch.device("cuda:0") if token == "albedo" else torch.device("cuda:1")
            for attr in [f"to_q{token_suffix}", f"to_k{token_suffix}", f"to_v{token_suffix}", f"to_out{token_suffix}"]:
                getattr(target, attr).to(device)

        def get_qkv(attn, hidden_states, encoder_hidden_states, **kwargs):
            return (
                getattr(target, f"to_q{token_suffix}")(hidden_states),
                getattr(target, f"to_k{token_suffix}")(encoder_hidden_states),
                getattr(target, f"to_v{token_suffix}")(encoder_hidden_states),
            )

        # Core processing using shared logic
        hidden_states, residual, input_ndim, shape_info, batch_size, heads, head_dim = AttnCore.process_attention_base(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, get_qkv_fn=get_qkv
        )

        # Finalize
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        hidden_states = hidden_states.to(hidden_states.dtype)

        return AttnUtils.finalize_output(
            hidden_states, input_ndim, shape_info, attn, residual, getattr(target, f"to_out{token_suffix}")
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply self-attention with PBR material processing.

        Processes multiple PBR material types sequentially, applying attention
        computation for each material type separately and combining results.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor with PBR dimension
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Combined attention output for all PBR material types
        """
        AttnUtils.handle_deprecation_warning(args, kwargs)

        B = hidden_states.size(0)
        pbr_hidden_states = torch.split(hidden_states, 1, dim=1)

        # Process each PBR setting
        results = []
        for token, pbr_hs in zip(self.pbr_setting, pbr_hidden_states):
            processed_hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c").to("cuda:0")
            result = self.process_single(attn, processed_hs, None, attention_mask, temb, token, False)
            results.append(result)

        outputs = [rearrange(result, "(b n_pbrs n) l c -> b n_pbrs n l c", b=B, n_pbrs=1) for result in results]
        return torch.cat(outputs, dim=1)


class RefAttnProcessor2_0(BaseAttnProcessor):
    """
    Reference attention processor with shared value computation across PBR materials.

    This processor computes query and key once, but uses separate value projections
    for different PBR material types, enabling efficient multi-material processing.
    """

    def __init__(self, **kwargs):
        """
        Initialize reference attention processor.

        Args:
            **kwargs: Arguments passed to BaseAttnProcessor initialization
        """
        super().__init__(**kwargs)
        self.pbr_settings = self.pbr_setting  # Alias for compatibility
        self.register_pbr_modules(["v_only", "out"], **kwargs)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply reference attention with shared Q/K and separate V projections.

        This method computes query and key tensors once and reuses them across
        all PBR material types, while using separate value projections for each
        material type to maintain material-specific information.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Stacked attention output for all PBR material types
        """
        AttnUtils.handle_deprecation_warning(args, kwargs)

        def get_qkv(attn, hidden_states, encoder_hidden_states, **kwargs):
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            # Concatenate values from all PBR settings
            value_list = [attn.to_v(encoder_hidden_states)]
            for token in ["_" + token for token in self.pbr_settings if token != "albedo"]:
                value_list.append(getattr(attn.processor, f"to_v{token}")(encoder_hidden_states))
            value = torch.cat(value_list, dim=-1)

            return query, key, value

        # Core processing
        hidden_states, residual, input_ndim, shape_info, batch_size, heads, head_dim = AttnCore.process_attention_base(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, get_qkv_fn=get_qkv
        )

        # Split and process each PBR setting output
        hidden_states_list = torch.split(hidden_states, head_dim, dim=-1)
        output_hidden_states_list = []

        for i, hs in enumerate(hidden_states_list):
            hs = hs.transpose(1, 2).reshape(batch_size, -1, heads * head_dim).to(hs.dtype)
            token_suffix = "_" + self.pbr_settings[i] if self.pbr_settings[i] != "albedo" else ""
            target = attn if self.pbr_settings[i] == "albedo" else attn.processor

            hs = AttnUtils.finalize_output(
                hs, input_ndim, shape_info, attn, residual, getattr(target, f"to_out{token_suffix}")
            )
            output_hidden_states_list.append(hs)

        return torch.stack(output_hidden_states_list, dim=1)
