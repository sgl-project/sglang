# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention.vision import (
    BATCH_BUCKETS,
    FLASHINFER_MAX_SEQLEN_BUCKETS,
    FLASHINFER_WORKSPACE_SIZE_BYTES,
    VisionAttention,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding.utils import rotate_half
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import add_prefix, get_compiler_backend, round_up

logger = logging.getLogger(__name__)


@dataclass
class CLIPVisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    hidden_act: str
    layer_norm_eps: float
    img_token_compression_config: dict
    position_embedding_type: str
    rope_mode: str
    rope_theta: float
    vision_segment_max_frames: Optional[int]

    @classmethod
    def from_dict(cls, d: dict) -> "CLIPVisionConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        if "rope_theta" not in filtered and isinstance(d.get("rope_parameters"), dict):
            rope_theta = d["rope_parameters"].get("rope_theta")
            if rope_theta is not None:
                filtered["rope_theta"] = rope_theta
        return cls(**filtered)


class MiniMaxVLMultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str,
        multimodal_projector_bias: bool,
        projector_hidden_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()

        mid_size = (
            projector_hidden_size
            if projector_hidden_size is not None
            else text_hidden_size
        )

        tp_size = 1 if use_data_parallel else get_parallel().attn_tp_size
        tp_rank = 0 if use_data_parallel else get_parallel().attn_tp_rank

        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            mid_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        assert (
            projector_hidden_act == "gelu"
        ), f"Only gelu activation is supported, got {projector_hidden_act}"
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            mid_size,
            text_hidden_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
            tp_size=tp_size,
            tp_rank=tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniMaxVLPatchMerger(nn.Module):
    def __init__(
        self,
        spatial_merge_size: int,
        text_hidden_size: int,
        projector_hidden_act: str,
        patch_merge_bias: bool,
        projector_hidden_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size

        mid_size = (
            projector_hidden_size
            if projector_hidden_size is not None
            else text_hidden_size
        )

        tp_size = 1 if use_data_parallel else get_parallel().attn_tp_size
        tp_rank = 0 if use_data_parallel else get_parallel().attn_tp_rank

        self.linear_1 = ColumnParallelLinear(
            text_hidden_size * spatial_merge_size**2,
            mid_size,
            bias=patch_merge_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        assert (
            projector_hidden_act == "gelu"
        ), f"Only gelu activation is supported, got {projector_hidden_act}"
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            mid_size,
            text_hidden_size,
            bias=patch_merge_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
            tp_size=tp_size,
            tp_rank=tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features.reshape(
            image_features.shape[0] // (self.spatial_merge_size**2), -1
        )
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


def _prepare_rotary_cos_sin(
    freqs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = freqs.cos().repeat(1, 2).unsqueeze(-2).float()
    sin = freqs.sin().repeat(1, 2).unsqueeze(-2).float()
    return cos, sin


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _minimax_rope_applier(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    x_shape=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE uses rope_dim=60 < head_dim=64; trailing dims pass through unrotated."""
    cos, sin = position_embeddings
    rot_dim = cos.shape[-1]

    q_rot = q[..., :rot_dim].float()
    q_pass = q[..., rot_dim:]
    k_rot = k[..., :rot_dim].float()
    k_pass = k[..., rot_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q = torch.cat((q_rot.to(q_pass.dtype), q_pass), dim=-1)
    k = torch.cat((k_rot.to(k_pass.dtype), k_pass), dim=-1)
    return q, k


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.input_num_channels = config.num_channels

        self.temporal_patch_size = config.img_token_compression_config.get(
            "temporal_patch_size", 2
        )

        self.patch_embedding = nn.Conv3d(
            in_channels=self.input_num_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if self.patch_embedding.weight.dtype != pixel_values.dtype:
            self.patch_embedding = self.patch_embedding.to(pixel_values.dtype)

        assert (
            pixel_values.dim() == 2
        ), f"pixel_values must be 2D, got {pixel_values.dim()}D"
        pixel_values = pixel_values.reshape(
            pixel_values.shape[0],
            self.input_num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.reshape(patch_embeds.shape[0], -1)
        return patch_embeds


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        workspace_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.use_data_parallel = use_data_parallel
        tp_size = 1 if use_data_parallel else get_parallel().attn_tp_size
        tp_rank = 0 if use_data_parallel else get_parallel().attn_tp_rank

        self.self_attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=is_dp_attention_enabled(),
            customized_position_embedding_applier=_minimax_rope_applier,
            workspace_buffer=workspace_buffer,
        )

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.fc1",
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        hidden_act = getattr(config, "hidden_act", "gelu")
        assert (
            hidden_act == "gelu"
        ), f"Only gelu activation is supported, got {hidden_act}"
        self.act = get_act_fn(hidden_act)
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.fc2",
            tp_size=tp_size,
            tp_rank=tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seq_len: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: Optional[int] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            x=hidden_states,
            cu_seqlens=cu_seq_len,
            position_embeddings=rotary_pos_emb,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        workspace_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.use_data_parallel = use_data_parallel

        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    use_data_parallel=use_data_parallel,
                    workspace_buffer=workspace_buffer,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        cu_seq_len: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: Optional[int] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        cos_sin = _prepare_rotary_cos_sin(rotary_pos_emb)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seq_len,
                cos_sin,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )

        return hidden_states


class MiniMaxVLVisionTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.use_data_parallel = use_data_parallel
        embed_dim = config.hidden_size

        self.temporal_patch_size = config.img_token_compression_config.get(
            "temporal_patch_size", 2
        )
        self.spatial_merge_size = config.img_token_compression_config.get(
            "spatial_merge_size", 2
        )

        self.embeddings = CLIPVisionEmbeddings(config)
        # NOTE: Typo "layrnorm" matches the original transformers code and the
        # weight names used in the published checkpoints; do not "fix" it.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        workspace_buffer: Optional[torch.Tensor] = None
        if (
            get_server_args().mm_attention_backend == "flashinfer_cudnn"
            and torch.cuda.is_available()
        ):
            workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_SIZE_BYTES,
                dtype=torch.uint8,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        self.encoder = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
            use_data_parallel=use_data_parallel,
            workspace_buffer=workspace_buffer,
        )

        assert (
            self.config.position_embedding_type == "rope"
        ), "Only rope position embedding is supported"
        assert self.config.rope_mode == "3d", "Only 3D RoPE is supported"
        rope_theta = getattr(config, "rope_theta")
        assert rope_theta is not None, "rope_theta must be set"
        self.vision_segment_max_frames = getattr(config, "vision_segment_max_frames")

        head_dim = embed_dim // config.num_attention_heads
        rope_dims = 2 * (head_dim // 2)

        self.t_dim = int(2 * ((rope_dims // 3) // 2))
        self.h_dim = int(2 * ((rope_dims // 3) // 2))
        self.w_dim = int(2 * ((rope_dims // 3) // 2))

        inv_freq_t = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.t_dim, 2, dtype=torch.float32) / self.t_dim)
        )
        inv_freq_h = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.h_dim, 2, dtype=torch.float32) / self.h_dim)
        )
        inv_freq_w = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.w_dim, 2, dtype=torch.float32) / self.w_dim)
        )

        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers
        self.post_layernorm = (
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            if require_post_norm
            else None
        )

    def _get_3d_rope_embed(
        self, grid_t: int, grid_h: int, grid_w: int, spatial_merge_size: int
    ) -> torch.Tensor:
        tokens_per_frame = grid_h * grid_w

        tpos_ids = (
            torch.arange(grid_t, device=self.inv_freq_t.device)
            .unsqueeze(1)
            .expand(-1, tokens_per_frame)
            .flatten()
        )

        hpos_ids = (
            torch.arange(grid_h, device=self.inv_freq_h.device)
            .unsqueeze(1)
            .expand(-1, grid_w)
        )
        hpos_ids = hpos_ids.reshape(
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

        wpos_ids = (
            torch.arange(grid_w, device=self.inv_freq_w.device)
            .unsqueeze(0)
            .expand(grid_h, -1)
        )
        wpos_ids = wpos_ids.reshape(
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

        max_t = max(grid_t, 1)
        max_hw = max(grid_h, grid_w)

        seq_t = torch.arange(
            max_t, device=self.inv_freq_t.device, dtype=self.inv_freq_t.dtype
        )
        seq_hw = torch.arange(
            max_hw, device=self.inv_freq_h.device, dtype=self.inv_freq_h.dtype
        )

        freqs_t = torch.outer(seq_t, self.inv_freq_t)
        freqs_h = torch.outer(seq_hw, self.inv_freq_h)
        freqs_w = torch.outer(seq_hw, self.inv_freq_w)

        emb_t = freqs_t[tpos_ids]
        emb_h = freqs_h[hpos_ids]
        emb_w = freqs_w[wpos_ids]

        return torch.cat([emb_t, emb_h, emb_w], dim=-1)

    def _get_rope_embed_3d(self, grid_thw, spatial_merge_size: int) -> torch.Tensor:
        all_rope_embeds = [
            self._get_3d_rope_embed(grid_t, grid_h, grid_w, spatial_merge_size)
            for grid_t, grid_h, grid_w in grid_thw
        ]
        return torch.cat(all_rope_embeds, dim=0)

    def _apply_max_frames_limit(
        self, origin_grid_thw: list[list[int]]
    ) -> List[List[int]]:
        if self.vision_segment_max_frames is None:
            return origin_grid_thw
        max_frames = self.vision_segment_max_frames
        ret_grid_thw = []
        for grid_t, grid_h, grid_w in origin_grid_thw:
            if grid_t <= max_frames:
                ret_grid_thw.append([grid_t, grid_h, grid_w])
            else:
                for i in range(0, grid_t, max_frames):
                    sub_grid_t = min(max_frames, grid_t - i)
                    ret_grid_thw.append([sub_grid_t, grid_h, grid_w])
        return ret_grid_thw

    def _compute_cu_seq_len(
        self,
        grid_thw: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        grid_thw = self._apply_max_frames_limit(grid_thw)
        cu_seq_len = [0]
        for grid_t, grid_h, grid_w in grid_thw:
            cu_seq_len.append(grid_t * grid_h * grid_w)
        cu_seq_len = torch.tensor(cu_seq_len, device=device).to(torch.int32)
        cu_seq_len = torch.cumsum(cu_seq_len, dim=0).to(torch.int32)
        return cu_seq_len

    @staticmethod
    def _bucket_flashinfer_batch_size(batch_size: int) -> int:
        return next(
            (b for b in BATCH_BUCKETS if b >= batch_size),
            round_up(batch_size, BATCH_BUCKETS[0]),
        )

    @staticmethod
    def _bucket_flashinfer_max_seqlen(real_max_seqlen: int) -> int:
        if real_max_seqlen <= 0:
            return FLASHINFER_MAX_SEQLEN_BUCKETS[0]
        return next(
            (s for s in FLASHINFER_MAX_SEQLEN_BUCKETS if s >= real_max_seqlen),
            round_up(real_max_seqlen, FLASHINFER_MAX_SEQLEN_BUCKETS[-1]),
        )

    @classmethod
    def _compute_flashinfer_sequence_lengths_padded(
        cls,
        token_cu_seqlens: np.ndarray,
    ) -> np.ndarray:
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)
        seq_lens = (token_cu_seqlens[1:] - token_cu_seqlens[:-1]).astype(np.int32)
        B_padded = cls._bucket_flashinfer_batch_size(B)
        if B_padded != B:
            pad = np.zeros((B_padded - B,), dtype=np.int32)
            seq_lens = np.concatenate([seq_lens, pad], axis=0)
        return seq_lens

    @classmethod
    def _compute_flashinfer_batch_offsets_packed(
        cls,
        token_cu_seqlens: np.ndarray,
        *,
        elem_per_token: int,
    ) -> np.ndarray:
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)
        B_padded = cls._bucket_flashinfer_batch_size(B)
        token_indptr = token_cu_seqlens.astype(np.int64, copy=False)
        if B_padded != B:
            pad = np.full((B_padded - B,), token_indptr[-1], dtype=token_indptr.dtype)
            token_indptr = np.concatenate([token_indptr, pad], axis=0)
        elem_indptr = (token_indptr * int(elem_per_token)).astype(np.int32)
        return np.concatenate([elem_indptr, elem_indptr, elem_indptr], axis=0)

    def _build_flashinfer_cudnn_inputs(
        self,
        cu_seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        device = cu_seq_len.device
        token_cu_seqlens_np = cu_seq_len.detach().cpu().numpy().astype(np.int32)

        real_seq_lens = token_cu_seqlens_np[1:] - token_cu_seqlens_np[:-1]
        max_seqlen = self._bucket_flashinfer_max_seqlen(
            int(real_seq_lens.max()) if real_seq_lens.size > 0 else 0
        )

        seq_lens_padded = self._compute_flashinfer_sequence_lengths_padded(
            token_cu_seqlens_np
        )

        attn_tp_size = 1 if self.use_data_parallel else get_parallel().attn_tp_size
        elem_per_token = self.config.hidden_size // attn_tp_size

        offsets_packed = self._compute_flashinfer_batch_offsets_packed(
            token_cu_seqlens_np,
            elem_per_token=elem_per_token,
        )

        sequence_lengths = (
            torch.from_numpy(seq_lens_padded)
            .to(device=device, dtype=torch.int32, non_blocking=True)
            .view(-1, 1, 1, 1)
        )
        cu_seqlens_packed = torch.from_numpy(offsets_packed).to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        return cu_seqlens_packed, sequence_lengths, int(max_seqlen)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        assert pixel_values.dtype == torch.bfloat16, "pixel_values must be bfloat16"

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        grid_thw = self._apply_max_frames_limit(grid_thw)

        cu_seq_len = self._compute_cu_seq_len(grid_thw, hidden_states.device)
        rotary_pos_emb = self._get_rope_embed_3d(grid_thw, self.spatial_merge_size)

        assert (
            rotary_pos_emb.device == hidden_states.device
        ), "rotary_pos_emb and hidden_states must be on the same device"

        max_seqlen: Optional[int] = None
        sequence_lengths: Optional[torch.Tensor] = None
        encoder_cu_seq_len = cu_seq_len
        if get_server_args().mm_attention_backend == "flashinfer_cudnn":
            (
                encoder_cu_seq_len,
                sequence_lengths,
                max_seqlen,
            ) = self._build_flashinfer_cudnn_inputs(cu_seq_len)

        return self.encoder(
            inputs_embeds=hidden_states,
            cu_seq_len=encoder_cu_seq_len,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )


class MiniMaxVLVisionModel(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        text_hidden_size: int,
        projector_hidden_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        multimodal_projector_bias: bool = True,
        patch_merge_bias: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.use_data_parallel = get_server_args().mm_enable_dp_encoder
        self.vision_config = config

        self.vision_model = MiniMaxVLVisionTransformer(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        self.multi_modal_projector = MiniMaxVLMultiModalProjector(
            vision_hidden_size=config.hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            multimodal_projector_bias=multimodal_projector_bias,
            projector_hidden_size=projector_hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("multi_modal_projector", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        spatial_merge_size = config.img_token_compression_config.get(
            "spatial_merge_size", 2
        )
        self.spatial_merge_size = spatial_merge_size
        self.patch_merge_mlp = MiniMaxVLPatchMerger(
            spatial_merge_size=spatial_merge_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            patch_merge_bias=patch_merge_bias,
            projector_hidden_size=projector_hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("patch_merge_mlp", prefix),
            use_data_parallel=self.use_data_parallel,
        )
        self.dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        # Required by run_dp_sharded_mrope_vision_model when input is empty.
        self.out_hidden_size = text_hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden_states = self.vision_model(pixel_values=pixel_values, grid_thw=grid_thw)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        hidden_states = self.multi_modal_projector(hidden_states)
        hidden_states = self.patch_merge_mlp(hidden_states)
        return hidden_states


def _run_vision_tower(
    vision_tower: MiniMaxVLVisionModel,
    pixel_values: torch.Tensor,
    grid_thw: list[list[int]],
    use_data_parallel: bool,
) -> torch.Tensor:
    if use_data_parallel:
        return run_dp_sharded_mrope_vision_model(
            vision_tower,
            pixel_values,
            grid_thw,
            rope_type="rope_3d",
        )
    return vision_tower(pixel_values, grid_thw=grid_thw)


def get_image_feature(
    vision_tower: MiniMaxVLVisionModel,
    items: List[MultimodalDataItem],
    use_data_parallel: bool,
) -> torch.Tensor:
    pixel_values = torch.cat([item.feature for item in items], dim=0).type(
        vision_tower.dtype
    )
    image_grid_thw: list[list[int]] = []
    for item in items:
        image_grid_thw.extend(item.image_grid_thw.tolist())
    return _run_vision_tower(
        vision_tower, pixel_values, image_grid_thw, use_data_parallel
    )


def get_video_feature(
    vision_tower: MiniMaxVLVisionModel,
    items: List[MultimodalDataItem],
    use_data_parallel: bool,
) -> torch.Tensor:
    pixel_values = torch.cat([item.feature for item in items], dim=0).type(
        vision_tower.dtype
    )
    video_grid_thw: list[list[int]] = []
    for item in items:
        video_grid_thw.extend(item.video_grid_thw.tolist())
    assert pixel_values.dim() == 2, pixel_values.dim()
    return _run_vision_tower(
        vision_tower, pixel_values, video_grid_thw, use_data_parallel
    )


def _parse_vit_layer_idx(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return None


def load_vision_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict,
    vit_qkv_weights: dict,
    vit_qkv_biases: dict,
) -> None:
    if (
        "self_attn.q_proj" in name
        or "self_attn.k_proj" in name
        or "self_attn.v_proj" in name
    ):
        if name.endswith(".weight"):
            target = vit_qkv_weights
        elif name.endswith(".bias"):
            target = vit_qkv_biases
        else:
            return
        layer_idx = _parse_vit_layer_idx(name)
        if layer_idx is None:
            return
        qkv_type = "q" if "q_proj" in name else ("k" if "k_proj" in name else "v")
        target.setdefault(layer_idx, {})[qkv_type] = loaded_weight
        return

    param_name = name
    if "vision_tower.vision_model." in param_name:
        param_name = param_name.replace(".mlp.fc1.", ".fc1.")
        param_name = param_name.replace(".mlp.fc2.", ".fc2.")
        param_name = param_name.replace(".self_attn.out_proj.", ".self_attn.proj.")
    if name.startswith("patch_merge_mlp.") or name.startswith("multi_modal_projector."):
        param_name = "vision_tower." + param_name

    if param_name in params_dict:
        param = params_dict[param_name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def merge_vit_qkv_weights(
    vit_qkv_weights: dict,
    vit_qkv_biases: dict,
    params_dict: dict,
) -> None:
    for layer_idx, qkv_dict in vit_qkv_weights.items():
        if {"q", "k", "v"} <= qkv_dict.keys():
            merged = torch.cat([qkv_dict["q"], qkv_dict["k"], qkv_dict["v"]], dim=0)
            param_name = (
                f"vision_tower.vision_model.encoder.layers.{layer_idx}"
                ".self_attn.qkv_proj.weight"
            )
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, merged)

    for layer_idx, qkv_dict in vit_qkv_biases.items():
        if {"q", "k", "v"} <= qkv_dict.keys():
            merged = torch.cat([qkv_dict["q"], qkv_dict["k"], qkv_dict["v"]], dim=0)
            param_name = (
                f"vision_tower.vision_model.encoder.layers.{layer_idx}"
                ".self_attn.qkv_proj.bias"
            )
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, merged)
