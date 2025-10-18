import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel

from sglang.srt.configs.dots_vlm import DotsVisionConfig
from sglang.srt.distributed import parallel_state
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        pre_norm="layernorm",
        init_merger_std=None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.pre_norm = pre_norm
        if self.pre_norm == "layernorm":
            self.ln_q = LayerNorm(context_dim, eps=1e-6)
        elif self.pre_norm == "rmsnorm":
            self.ln_q = RMSNorm(context_dim, eps=1e-6)
        else:
            logger.warning(f"no norm in patch merger: {self.pre_norm}")

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

        if init_merger_std is not None:
            nn.init.normal_(self.mlp[0].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.normal_(self.mlp[2].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        else:
            x = self.mlp(x.view(-1, self.hidden_size))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


class DotsPatchEmbed(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        x = x.view(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


class DotsViTPreprocessor(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.patch_h = config.patch_size
        self.patch_w = config.patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.patchifier = DotsPatchEmbed(config, quant_config)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        tokens = self.patchifier(x, grid_thw)
        return tokens


class DotsVisionBlock(nn.Module):
    def __init__(
        self,
        config: DotsVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        if attn_implementation == "flash_attention_2":
            qkv_backend = "fa3"
            softmax_in_single_precision = False
        else:
            raise RuntimeError("Unimplemented")
        self.attn = VisionAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_attention_heads,
            projection_size=config.embed_dim,
            use_qkv_parallel=True,
            qkv_backend=qkv_backend,
            softmax_in_single_precision=softmax_in_single_precision,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=config.num_dummy_heads,
            qkv_bias=config.use_bias,
            proj_bias=config.use_bias,
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config, quant_config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionTransformer(PreTrainedModel):
    def __init__(
        self,
        config: DotsVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._update_vision_config()
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config, quant_config)
        self._init_weights(self.patch_embed.patchifier.proj)

        head_dim = config.embed_dim // config.num_attention_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        _num_hidden_layers = config.num_hidden_layers
        self.blocks = nn.ModuleList(
            [
                DotsVisionBlock(
                    config, quant_config, f"blocks.{i}", config.attn_implementation
                )
                for i in range(_num_hidden_layers)
            ]
        )

        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            init_merger_std=self.config.init_merger_std,
            quant_config=quant_config,
        )

        self.gradient_checkpointing = False

    def _update_vision_config(self):
        """update vision config to support tp"""
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        num_heads = self.config.num_attention_heads
        head_dim = self.config.embed_dim // num_heads
        num_dummy_heads = 0

        if num_heads % world_size != 0:
            num_dummy_heads = (
                (num_heads + world_size) // world_size
            ) * world_size - num_heads

        setattr(self.config, "head_dim", head_dim)
        setattr(self.config, "num_dummy_heads", num_dummy_heads)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def get_pos_ids_by_grid(self, grid_thw):
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

        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def calc_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        rotary_pos_emb = (cos, sin)
        return rotary_pos_emb

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, bf16=True
    ) -> torch.Tensor:
        if bf16:
            hidden_states = hidden_states.bfloat16()
        hidden_states = self.patch_embed(hidden_states, grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = self.calc_cos_sin(rotary_pos_emb)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states
