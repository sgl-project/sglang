import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm
from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class SelectFirstElement(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size

        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    frequency_embedding_size, mid_size, bias=True, gather_output=False
                ),
                nn.SiLU(),
                RowParallelLinear(
                    mid_size, out_size, bias=True, input_is_parallel=True
                ),
            ]
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast(current_platform.device_type, enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
                / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat(
                    [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                )
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
        t_emb, _ = self.mlp[0](t_freq)
        t_emb = self.mlp[1](t_emb)
        t_emb, _ = self.mlp[2](t_emb)
        return t_emb


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    Fused value_noisy and value_clean in to a single tensor.
    where if noise_mask == 1, fill value_noisy
    else fill value_clean(t_embedder(ones))

    Args:
        value_noisy (tensor): t_embedder(t)
        value_clean (tensor): t_embedder(ones_like(t))
        noise_mask (tensor): 0/1
        seqlen (int): seqlen
    Returns
        (tensor): tensor fill with value_noisy and value_clean.
            shape was expand to (batch, seqlen, dim)
    """
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Use MergedColumnParallelLinear for gate and up projection (fused)
        self.w13 = MergedColumnParallelLinear(
            dim, [hidden_dim, hidden_dim], bias=False, gather_output=False
        )
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)
        self.act = SiluAndMul()

    def forward(self, x):
        x13, _ = self.w13(x)
        x = self.act(x13)
        out, _ = self.w2(x)
        return out


class ZImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_norm = qk_norm

        tp_size = get_tp_world_size()
        assert (
            num_heads % tp_size == 0
        ), f"num_heads {num_heads} must be divisible by tp world size {tp_size}"
        assert (
            num_kv_heads % tp_size == 0
        ), f"num_kv_heads {num_kv_heads} must be divisible by tp world size {tp_size}"
        self.local_num_heads = num_heads // tp_size
        self.local_num_kv_heads = num_kv_heads // tp_size

        self.to_q = ColumnParallelLinear(dim, dim, bias=False, gather_output=False)
        self.to_k = ColumnParallelLinear(
            dim, self.head_dim * num_kv_heads, bias=False, gather_output=False
        )
        self.to_v = ColumnParallelLinear(
            dim, self.head_dim * num_kv_heads, bias=False, gather_output=False
        )

        if self.qk_norm:
            self.norm_q = RMSNorm(self.head_dim, eps=eps)
            self.norm_k = RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.ModuleList(
            [RowParallelLinear(dim, dim, bias=False, input_is_parallel=True)]
        )

        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.local_num_kv_heads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)
        q = q.view(*q.shape[:-1], self.local_num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.local_num_kv_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.local_num_kv_heads, self.head_dim)

        if self.qk_norm:
            if (
                q.is_cuda
                and (self.norm_q.variance_epsilon == self.norm_k.variance_epsilon)
                and can_use_fused_inplace_qknorm(self.head_dim, q.dtype)
            ):
                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.norm_q,
                    k_norm=self.norm_k,
                    head_dim=self.head_dim,
                    allow_inplace=True,
                )
            else:
                q = self.norm_q(q)
                k = self.norm_k(k)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            if q.is_cuda and q.shape == k.shape:
                cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
                q, k = apply_flashinfer_rope_qk_inplace(
                    q, k, cos_sin_cache, is_neox=False
                )
            else:
                q = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
                k = _apply_rotary_emb(k, cos, sin, is_neox_style=False)

        hidden_states = self.attn(q, k, v)
        hidden_states = hidden_states.flatten(2)

        hidden_states, _ = self.to_out[0](hidden_states)

        return hidden_states


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(
            dim=dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            eps=1e-5,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.Sequential(
                ReplicatedLinear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            seq_len = x.shape[1]
            if noise_mask is not None:
                # Per-token modulation: different modulation for noisy/clean tokens
                assert adaln_noisy is not None
                assert adaln_clean is not None
                mod_noisy, _ = self.adaLN_modulation(adaln_noisy)
                mod_clean, _ = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = (
                    mod_noisy.chunk(4, dim=1)
                )
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = (
                    mod_clean.chunk(4, dim=1)
                )

                gate_msa_noisy, gate_mlp_noisy = (
                    gate_msa_noisy.tanh(),
                    gate_mlp_noisy.tanh(),
                )
                gate_msa_clean, gate_mlp_clean = (
                    gate_msa_clean.tanh(),
                    gate_mlp_clean.tanh(),
                )

                scale_msa_noisy, scale_mlp_noisy = (
                    1.0 + scale_msa_noisy,
                    1.0 + scale_mlp_noisy,
                )
                scale_msa_clean, scale_mlp_clean = (
                    1.0 + scale_msa_clean,
                    1.0 + scale_mlp_clean,
                )

                scale_msa = select_per_token(
                    scale_msa_noisy, scale_msa_clean, noise_mask, seq_len
                )
                scale_mlp = select_per_token(
                    scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len
                )
                gate_msa = select_per_token(
                    gate_msa_noisy, gate_msa_clean, noise_mask, seq_len
                )
                gate_mlp = select_per_token(
                    gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len
                )

            else:
                # Global modulation: same modulation for all tokens (avoid double select)
                assert adaln_input is not None
                scale_msa_gate, _ = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = scale_msa_gate.unsqueeze(
                    1
                ).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x) * scale_mlp,
                )
            )
        else:
            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x),
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = ColumnParallelLinear(
            hidden_size, out_channels, bias=True, gather_output=True
        )

        self.act = nn.SiLU()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)[0]
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)[0]
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            # Original global modulation
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale, _ = self.adaLN_modulation(c)
            scale = 1.0 + scale
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
        x, _ = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(
            axes_lens
        ), "axes_dims and axes_lens must have the same length"

        self.cos_cached = None
        self.sin_cached = None

    @staticmethod
    def precompute_freqs(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            cos_list = []
            sin_list = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
                )
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()

                cos_list.append(torch.cos(freqs))
                sin_list.append(torch.sin(freqs))

            return cos_list, sin_list

    def __call__(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ids: [batch, len(axes_dims)] or [seq_len, len(axes_dims)]
        Returns:
            cos: [batch/seq, head_dim // 2]
            sin: [batch/seq, head_dim // 2]
        """
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.cos_cached is None:
            self.cos_cached, self.sin_cached = self.precompute_freqs(
                self.axes_dims, self.axes_lens, theta=self.theta
            )
            self.cos_cached = [c.to(device) for c in self.cos_cached]
            self.sin_cached = [s.to(device) for s in self.sin_cached]
        else:
            if self.cos_cached[0].device != device:
                self.cos_cached = [c.to(device) for c in self.cos_cached]
                self.sin_cached = [s.to(device) for s in self.sin_cached]

        cos_out = []
        sin_out = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            cos_out.append(self.cos_cached[i][index])
            sin_out.append(self.sin_cached[i][index])

        return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


class ZImageTransformer2DModel(CachableDiT, OffloadableDiTMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]
    param_names_mapping = ZImageDitConfig().arch_config.param_names_mapping

    param_names_mapping = ZImageDitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        ZImageDitConfig().arch_config.reverse_param_names_mapping
    )

    def __init__(
        self,
        config: ZImageDitConfig,
        hf_config: dict[str, Any],
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        self.config_data = config  # Store config
        arch_config = config.arch_config

        self.in_channels = arch_config.in_channels
        self.out_channels = arch_config.out_channels
        self.all_patch_size = arch_config.all_patch_size
        self.all_f_patch_size = arch_config.all_f_patch_size
        self.dim = arch_config.dim
        self.n_heads = arch_config.num_attention_heads
        self.siglip_feat_dim = arch_config.siglip_feat_dim

        self.rope_theta = arch_config.rope_theta
        self.t_scale = arch_config.t_scale
        self.gradient_checkpointing = False

        assert len(self.all_patch_size) == len(self.all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(
            zip(self.all_patch_size, self.all_f_patch_size)
        ):
            x_embedder = ColumnParallelLinear(
                f_patch_size * patch_size * patch_size * self.in_channels,
                self.dim,
                bias=True,
                gather_output=True,
            )
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(
                self.dim, patch_size * patch_size * f_patch_size * self.out_channels
            )
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    self.dim,
                    self.n_heads,
                    arch_config.n_kv_heads,
                    arch_config.norm_eps,
                    arch_config.qk_norm,
                    modulation=True,
                )
                for layer_id in range(arch_config.n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    self.dim,
                    self.n_heads,
                    arch_config.n_kv_heads,
                    arch_config.norm_eps,
                    arch_config.qk_norm,
                    modulation=False,
                )
                for layer_id in range(arch_config.n_refiner_layers)
            ]
        )

        # Optional SigLIP components (for Omni variant)
        if self.siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(self.siglip_feat_dim, eps=arch_config.norm_eps),
                nn.Linear(self.siglip_feat_dim, self.dim, bias=True),
            )
            self.siglip_refiner = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        2000 + layer_id,
                        self.dim,
                        self.n_heads,
                        arch_config.n_kv_heads,
                        arch_config.norm_eps,
                        arch_config.qk_norm,
                        modulation=False,
                    )
                    for layer_id in range(arch_config.n_refiner_layers)
                ]
            )
            self.siglip_pad_token = nn.Parameter(torch.empty((1, self.dim)))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

        self.t_embedder = TimestepEmbedder(
            min(self.dim, ADALN_EMBED_DIM), mid_size=1024
        )

        self.cap_embedder = nn.Sequential(
            RMSNorm(arch_config.cap_feat_dim, eps=arch_config.norm_eps),
            ReplicatedLinear(arch_config.cap_feat_dim, self.dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, self.dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, self.dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    self.dim,
                    self.n_heads,
                    arch_config.n_kv_heads,
                    arch_config.norm_eps,
                    arch_config.qk_norm,
                )
                for layer_id in range(arch_config.num_layers)
            ]
        )
        head_dim = self.dim // self.n_heads
        assert head_dim == sum(arch_config.axes_dims)
        self.axes_dims = arch_config.axes_dims
        self.axes_lens = arch_config.axes_lens

        self.rotary_emb = RopeEmbedder(
            theta=self.rope_theta, axes_dims=self.axes_dims, axes_lens=self.axes_lens
        )
        self.layer_names = ["layers"]

    # Copied from diffusers.models.transformers.transformer_z_image.unpatchify
    def unpatchify(
        self,
        x: List[torch.Tensor],
        size: List[Tuple],
        patch_size,
        f_patch_size,
        x_pos_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz

        if x_pos_offsets is not None:
            # Omni: extract target image from unified sequence (cond_images + target)
            result = []
            for i in range(bsz):
                unified_x = x[i][x_pos_offsets[i][0] : x_pos_offsets[i][1]]
                cu_len = 0
                x_item = None
                for j in range(len(size[i])):
                    if size[i][j] is None:
                        ori_len = 0
                        pad_len = SEQ_MULTI_OF
                        cu_len += pad_len + ori_len
                    else:
                        F, H, W = size[i][j]
                        ori_len = (F // pF) * (H // pH) * (W // pW)
                        pad_len = (-ori_len) % SEQ_MULTI_OF
                        x_item = (
                            unified_x[cu_len : cu_len + ori_len]
                            .view(
                                F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels
                            )
                            .permute(6, 0, 3, 1, 4, 2, 5)
                            .reshape(self.out_channels, F, H, W)
                        )
                        cu_len += ori_len + pad_len
                result.append(x_item)  # Return only the last (target) image
            return result
        else:
            # Original mode: simple unpatchify
            for i in range(bsz):
                F, H, W = size[i]
                ori_len = (F // pF) * (H // pH) * (W // pW)
                # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
                x[i] = (
                    x[i][:ori_len]
                    .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, F, H, W)
                )
            return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [
            torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
            for x0, span in zip(start, size)
        ]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed_omni(
        self,
        all_image: List[List[torch.Tensor]],
        all_cap_feats: List[List[torch.Tensor]],
        patch_size: int,
        f_patch_size: int,
        all_siglip_feats: List[List[torch.Tensor]],
        images_noise_mask: List[List[int]],
    ):
        """
        Patchify and padding for omni mode: multiple images per batch item with noise masks.
        Process siglip.

        Args:
            all_image (List[tensor]): [condition..., latent]
            all_cap_feats (List[tensor]): ["...<|vision_start|>", ..., "<|vision_end|><|im_end|>"]
            patch_size (int): patch_size for h and w
            f_patch_size (int): patch_size for f
            all_siglip_feats (List[tensor]): [num_images...] * batch_size
        Returns:

        """
        assert len(all_image) == len(all_cap_feats) == 1
        bsz = len(all_image)

        device = all_image[0][-1].device
        dtype = all_image[0][-1].dtype

        images = all_image[0]  # List of [C, F, H, W]
        cap_feats = all_cap_feats[0]  # List of [L, D]
        siglips = all_siglip_feats[0]  # List of [H, W, C]

        all_image_out = []
        all_image_size = []
        all_cap_feats_out = []
        all_sig_out = []

        all_image_noise_mask = []
        all_cap_noise_mask = []
        all_sig_noise_mask = []

        all_image_len, all_cap_len, all_sig_len = [], [], []

        # ------------ Process Caption ------------
        # support bsz==1 only
        i = 0

        all_cap_feat_list = []
        all_cap_noise_mask_list = []
        cap_lens = []
        for j, cap_feat in enumerate(cap_feats):
            # cap_ori_len = cap_feat.size(0)
            # cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF

            # # padded feature
            # cap_padded_feat = torch.cat(
            #     [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
            #     dim=0,
            # )

            # noise_val == 1 fill noise
            # noise_val == 0 fill empty
            noise_val = images_noise_mask[i][j] if j < len(images_noise_mask[i]) else 1
            cap_padded_feat, cap_len, cap_nm = self._pad_and_prepare_noise_mask(
                cap_feat,
                noise_val,
            )

            all_cap_feat_list.append(cap_padded_feat)
            all_cap_noise_mask_list.extend(cap_nm)
            cap_lens.append(cap_len)

        all_cap_feats_out.append(torch.cat(all_cap_feat_list, dim=0))
        all_cap_noise_mask.append(all_cap_noise_mask_list)
        all_cap_len.append(cap_lens)

        # ------------ Process Image ------------
        all_image_padded_feat_list = []
        all_image_noise_mask_list = []
        image_lens = []
        image_sizes = []
        for j, image in enumerate(images):
            image, (F, H, W), _ = self._patchify_image(
                image, patch_size=patch_size, f_patch_size=f_patch_size
            )
            image_sizes.append((F, H, W))

            noise_val = images_noise_mask[i][j]
            image_padded_feat, image_len, image_nm = self._pad_and_prepare_noise_mask(
                image,
                noise_val,
            )

            all_image_padded_feat_list.append(image_padded_feat)
            all_image_noise_mask_list.extend(image_nm)
            image_lens.append(image_len)
        all_image_size.append(image_sizes)
        all_image_out.append(torch.cat(all_image_padded_feat_list, dim=0))
        all_image_noise_mask.append(all_image_noise_mask_list)
        all_image_len.append(image_lens)

        # ------------ Process Siglip ------------
        all_sig_feats_list = []
        all_sig_noise_mask_list = []
        sig_lens = []
        for j, sig_item in enumerate(siglips):
            noise_val = images_noise_mask[i][j]
            if sig_item is None:
                sig_len = SEQ_MULTI_OF
                sig_padded_feat = torch.zeros(
                    (sig_len, self.config.siglip_feat_dim), dtype=dtype, device=device
                )
                sig_nm = [noise_val] * sig_len

            else:
                sig_H, sig_W, sig_C = sig_item.size()
                sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)

                sig_padded_feat, sig_len, sig_nm = self._pad_and_prepare_noise_mask(
                    sig_flat,
                    noise_val,
                )

            all_sig_feats_list.append(sig_padded_feat)
            all_sig_noise_mask_list.extend(sig_nm)
            sig_lens.append(sig_len)
        all_sig_out.append(torch.cat(all_sig_feats_list, dim=0))
        all_sig_noise_mask.append(all_sig_noise_mask_list)
        all_sig_len.append(sig_lens)

        # Compute x position offsets
        all_image_pos_offsets = [
            (sum(all_cap_len[i]), sum(all_cap_len[i]) + sum(all_image_len[i]))
            for i in range(bsz)
        ]

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_sig_out,
            all_image_pos_offsets,
            all_image_noise_mask,
            all_cap_noise_mask,
            all_sig_noise_mask,
        )

    def _pad_and_prepare_noise_mask(
        self,
        feat: torch.Tensor,
        noise_mask_val: Optional[int] = None,
    ):
        """
        noise_mask_val == 1 fill noise
        noise_mask_val == 0 fill empty

        Args:
        Returns:
        """
        ori_len = feat.size(0)
        padding_len = (-ori_len) % SEQ_MULTI_OF
        total_len = ori_len + padding_len
        # padded feature
        padded_feat = torch.cat(
            [feat, feat[-1:].repeat(padding_len, 1)],
            dim=0,
        )
        noise_mask = (
            [noise_mask_val] * total_len if noise_mask_val is not None else None
        )  # token level

        return padded_feat, total_len, noise_mask

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        """
        Patchify and padding for basic mode: single image per batch item.

        Args:

        Returns:
            all_image_out (list): List of padded image_feat(patchified image).
            all_cap_feats_out (list): List of padded caption_feat
            all_image_size (list): List of image size (F, H, W)
        """
        assert len(all_image) == len(all_cap_feats) == 1

        image = all_image[0]  # C, F, H, W
        cap_feat = all_cap_feats[0]  # L, D
        # pH = pW = patch_size
        # pF = f_patch_size
        # device = image.device

        all_image_out = []
        all_image_size = []
        all_cap_feats_out = []

        # ------------ Process Caption ------------
        cap_ori_len = cap_feat.size(0)
        cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF

        # padded feature
        cap_padded_feat = torch.cat(
            [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
            dim=0,
        )
        all_cap_feats_out.append(cap_padded_feat)

        # ------------ Process Image ------------
        # C, F, H, W = image.size()

        # all_image_size.append((F, H, W))
        # F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        # image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
        # image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
        #     F_tokens * H_tokens * W_tokens, pF * pH * pW * C
        # )

        image, (F, H, W), _ = self._patchify_image(
            image, patch_size=patch_size, f_patch_size=f_patch_size
        )
        all_image_size.append((F, H, W))

        image_ori_len = image.size(0)
        image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

        # padded feature
        image_padded_feat = torch.cat(
            [image, image[-1:].repeat(image_padding_len, 1)],
            dim=0,
        )
        all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
        )

    def _patchify_image(self, image: torch.Tensor, patch_size: int, f_patch_size: int):
        """
        Patchify a single image tensor: (C, F, H, W) -> (num_patches, patch_dim).

        Args:

        Returns:
            image_feat (tensor): patchify image
            image_size (tuple[int, int, int]): image size in (F, H, W)
            image_num_tokens (tuple[int, int, int]): image num tokens (F // pF, H // pH, W // pW)
        """

        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F, H, W = image.size()
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            F_tokens * H_tokens * W_tokens, pF * pH * pW * C
        )
        return image, (F, H, W), (F_tokens, H_tokens, W_tokens)

    def _build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        x_seqlens: List[int],
        x_noise_mask: Optional[List[List[int]]],
        cap: torch.Tensor,
        cap_freqs: Tuple[torch.Tensor, torch.Tensor],
        cap_seqlens: List[int],
        cap_noise_mask: Optional[List[List[int]]],
        siglip: Optional[torch.Tensor],
        siglip_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]],
        siglip_seqlens: Optional[List[int]],
        siglip_noise_mask: Optional[List[List[int]]],
        omni_mode: bool,
        device: torch.device,
    ):
        """
        Concat feats into a single.

        Build unified sequence: x, cap, and optionally siglip.
        - Basic mode order: [x, cap]
        - Omni mode order: [cap, x, siglip]
        """

        # single batch only for now.
        bsz = len(x_seqlens)
        unified = []
        unified_freqs_cos = []
        unified_freqs_sin = []
        unified_noise_mask = []

        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]

            if omni_mode:
                # Omni: [cap, x, siglip]
                if siglip is not None and siglip_seqlens is not None:
                    sig_len = siglip_seqlens[i]
                    unified.append(
                        torch.cat([cap[i][:cap_len], x[i][:x_len], siglip[i][:sig_len]])
                    )
                    unified_freqs_cos.append(
                        torch.cat(
                            [
                                cap_freqs[0][:cap_len],
                                x_freqs[0][:x_len],
                                siglip_freqs[0][:sig_len],
                            ]
                        )
                    )
                    unified_freqs_sin.append(
                        torch.cat(
                            [
                                cap_freqs[1][:cap_len],
                                x_freqs[1][:x_len],
                                siglip_freqs[1][:sig_len],
                            ]
                        )
                    )
                    unified_noise_mask.append(
                        torch.tensor(
                            cap_noise_mask[i] + x_noise_mask[i] + siglip_noise_mask[i],
                            dtype=torch.long,
                            device=device,
                        )
                    )
                else:
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len]]))
                    unified_freqs_cos.append(
                        torch.cat([cap_freqs[0][:cap_len], x_freqs[0][:x_len]])
                    )
                    unified_freqs_sin.append(
                        torch.cat([cap_freqs[1][:cap_len], x_freqs[1][:x_len]])
                    )
                    unified_noise_mask.append(
                        torch.tensor(
                            cap_noise_mask[i] + x_noise_mask[i],
                            dtype=torch.long,
                            device=device,
                        )
                    )
            else:
                # Basic: [x, cap]
                unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
                unified_freqs_cos.append(
                    torch.cat([x_freqs[0][:x_len], cap_freqs[0][:cap_len]])
                )
                unified_freqs_sin.append(
                    torch.cat([x_freqs[1][:x_len], cap_freqs[1][:cap_len]])
                )

        assert len(unified) == 1, "Single batch only for now."
        assert len(unified_freqs_cos) == 1, "Single batch only for now."
        unified = unified[0].unsqueeze(0)
        # unified_freqs = unified_freqs[0].unsqueeze(0)
        unified_freqs = (unified_freqs_cos[0], unified_freqs_sin[0])

        # Noise mask
        noise_mask_tensor = None
        if omni_mode:
            assert len(unified_noise_mask) == 1, "Single batch only for now."
            noise_mask_tensor = unified_noise_mask[0][: unified.shape[1]].unsqueeze(0)

        return unified, unified_freqs, noise_mask_tensor

    # TODO: copy to configs/zimage.py later
    def get_freqs_cis(
        self,
        prompt_embeds: List[torch.Tensor],
        images: List[torch.Tensor],
        siglips: List[torch.Tensor],
        device,
        rotary_emb,
    ):
        def _get_pos_ids(
            ori_len: int,
            pos_grid_size: Tuple,
            pos_start: Tuple,
            device: torch.device,
        ):
            pad_len = (-ori_len) % SEQ_MULTI_OF
            total_len = ori_len + pad_len

            # Pos IDs
            ori_pos_ids = self.create_coordinate_grid(
                size=pos_grid_size, start=pos_start, device=device
            ).flatten(0, 2)
            if pad_len > 0:
                pad_pos_ids = (
                    self.create_coordinate_grid(
                        size=(1, 1, 1), start=(0, 0, 0), device=device
                    )
                    .flatten(0, 2)
                    .repeat(pad_len, 1)
                )
                pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            else:
                pos_ids = ori_pos_ids

            return pos_ids

        # TODO: assert batch size == 1

        # TODO: hard code....
        PATCH_SIZE = 2
        F_PATCH_SIZE = 1
        SEQ_MULTI_OF = 32

        # cap_start_pos for cap pos ids
        cap_cu_len = 1
        # cap_end_pos + 0 for image pos ids
        # cap_end_pos + 1 for image siglip ids
        cap_end_pos = []

        image_size = []

        cap_pos_ids_list = []
        image_pos_ids_list = []
        siglip_pos_ids_list = []

        for cap_item in prompt_embeds:
            cap_padded_pos_ids = _get_pos_ids(
                len(cap_item),
                (len(cap_item) + (-len(cap_item)) % SEQ_MULTI_OF, 1, 1),
                (cap_cu_len, 0, 0),
                device,
            )
            cap_cu_len += len(cap_item)
            cap_end_pos.append(cap_cu_len)
            cap_cu_len += 2  # for image vae and siglip tokens
            cap_pos_ids_list.append(cap_padded_pos_ids)

        for j, image in enumerate(images):
            if image is not None:
                pH, pW, pF = PATCH_SIZE, PATCH_SIZE, F_PATCH_SIZE
                C, F, H, W = image.size()
                F_t, H_t, W_t = F // pF, H // pH, W // pW
                image = image.view(C, F_t, pF, H_t, pH, W_t, pW)
                image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                    F_t * H_t * W_t, pF * pH * pW * C
                )
                image_pos = _get_pos_ids(
                    F_t * H_t * W_t, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), device
                )
                image_size.append((F, H, W))
            else:
                image_len = SEQ_MULTI_OF
                image_pos = (
                    self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device)
                    .flatten(0, 2)
                    .repeat(image_len, 1)
                )
                image_size.append(None)

            image_pos_ids_list.append(image_pos)

        for j, sig_item in enumerate(siglips if siglips is not None else []):
            if sig_item is not None:
                sig_H, sig_W, sig_C = sig_item.size()
                sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                sig_pos = _get_pos_ids(
                    len(sig_flat),
                    (1, sig_H, sig_W),
                    (cap_end_pos[j] + 1, 0, 0),
                    device,
                )
                # Scale position IDs to match x resolution
                if image_size[j] is not None:
                    sig_pos = sig_pos.float()
                    sig_pos[..., 1] = (
                        sig_pos[..., 1] / max(sig_H - 1, 1) * (image_size[j][1] - 1)
                    )
                    sig_pos[..., 2] = (
                        sig_pos[..., 2] / max(sig_W - 1, 1) * (image_size[j][2] - 1)
                    )
                    sig_pos = sig_pos.to(torch.int32)
            else:
                sig_len = SEQ_MULTI_OF
                sig_pos = (
                    self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device)
                    .flatten(0, 2)
                    .repeat(sig_len, 1)
                )
            siglip_pos_ids_list.append(sig_pos)

        cap_freqs_cis = rotary_emb(torch.cat(cap_pos_ids_list, dim=0))
        x_freqs_cis = rotary_emb(torch.cat(image_pos_ids_list, dim=0))
        siglip_freqs_cis = (
            rotary_emb(torch.cat(siglip_pos_ids_list, dim=0))
            if len(siglip_pos_ids_list) != 0
            else None
        )
        return (cap_freqs_cis, x_freqs_cis, siglip_freqs_cis)

    def forward(
        self,
        hidden_states: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        encoder_hidden_states: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        timestep,
        guidance=0,
        patch_size=2,
        f_patch_size=1,
        freqs_cis=None,
        siglip_feats: Optional[List[List[torch.Tensor]]] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
        condition_latents: Optional[List[List[torch.Tensor]]] = None,
        token_lens: List[int] = None,
        **kwargs,
    ):
        # TODO: ignore control net for now.
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        assert len(hidden_states) == 1, f"{len(hidden_states)=}"
        assert len(encoder_hidden_states) == 1, f"{len(encoder_hidden_states)=}"
        assert siglip_feats is None or len(siglip_feats) == 1, f"{len(siglip_feats)=}"

        batch_size = len(hidden_states)
        # construct a 2d list for omni mode
        #   dim0 = batch_size
        #   dim1 = latents: condition images + 1 target image
        # for a single latent[i] = [cond_images, ..., target_image]
        if condition_latents is not None:
            assert batch_size == 1, "Only batch_size=1 is tested for now."
            assert isinstance(condition_latents, list) and isinstance(
                condition_latents[0], list
            ), "condition_latents should be a 2d list."
            assert len(condition_latents) == batch_size, "batch size mismatch"
            hidden_states = [
                condition_latents[i] + [hidden_states[i]] for i in range(batch_size)
            ]

        x = hidden_states

        # omni_mode:
        #  split from a flatten view to a 2D list.
        #  for a single prompt: embed[i] = ["...<|vision_start|>", "<|vision_start|>", ..., "<|vision_end|><|im_end|>"]
        omni_mode = isinstance(x[0], list)
        if omni_mode:
            encoder_hidden_states = [
                list(encoder_hidden_states[0].split_with_sizes(token_lens, dim=0))
            ]
        cap_feats = encoder_hidden_states
        timestep = 1000.0 - timestep
        t = timestep
        device = x[0][-1].device if omni_mode else x[0].device

        if omni_mode:
            # Dual embeddings: noisy (t) and clean (t=1)
            t_noisy = self.t_embedder(t).type_as(x[0][-1])
            t_clean = self.t_embedder(torch.ones_like(t) * self.t_scale).type_as(
                x[0][-1]
            )
            adaln_input = None
        else:
            # Single embedding for all tokens
            adaln_input = self.t_embedder(t).type_as(x[0])
            t_noisy = t_clean = None

        # TODO: overwrite freqs_cis for debug
        # TODO: single batch only
        # compute freqs before patchify
        # TODO: ugly
        if freqs_cis is None:
            # omni-mode online compute
            # TODO: try pre-compute
            freqs_cis = self.get_freqs_cis(
                prompt_embeds=(
                    encoder_hidden_states[0]
                    if isinstance(encoder_hidden_states[0], list)
                    else encoder_hidden_states
                ),
                images=(
                    hidden_states[0]
                    if isinstance(hidden_states[0], list)
                    else hidden_states
                ),
                siglips=siglip_feats[0] if siglip_feats is not None else None,
                device=device,
                rotary_emb=getattr(self, "rotary_emb", None),
            )

        # Patchify
        if omni_mode:
            (
                x,
                cap_feats,
                x_size,
                siglip_feats,
                x_pos_offsets,
                x_noise_mask,
                cap_noise_mask,
                siglip_noise_mask,
            ) = self.patchify_and_embed_omni(
                x,
                cap_feats,
                patch_size,
                f_patch_size,
                siglip_feats,
                image_noise_mask,
            )
        else:
            (
                x,
                cap_feats,
                x_size,
            ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
            x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        x = torch.cat(x, dim=0)
        x, _ = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        x_freqs_cis = freqs_cis[1]

        x = x.unsqueeze(0)
        x_freqs_cis = x_freqs_cis
        x_noise_tensor = None
        if x_noise_mask is not None:
            x_noise_tensor = torch.tensor(x_noise_mask, dtype=torch.long, device=device)

        for layer in self.noise_refiner:
            x = layer(x, x_freqs_cis, adaln_input, x_noise_tensor, t_noisy, t_clean)

        cap_feats = torch.cat(cap_feats, dim=0)

        cap_feats, _ = self.cap_embedder(cap_feats)

        cap_freqs_cis = freqs_cis[0]

        cap_feats = cap_feats.unsqueeze(0)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs_cis)

        siglip_freqs_cis = siglip_seqlens = siglip_freqs = None

        if (
            omni_mode
            and siglip_feats[0] is not None
            and self.siglip_embedder is not None
        ):
            siglip_freqs_cis = freqs_cis[2]
            # process siglip
            # TODO: review shape
            siglip_feats = torch.cat(siglip_feats, dim=0)
            siglip_feats = siglip_feats.unsqueeze(0)

            siglip_seqlens = [len(si) for si in siglip_feats]
            # embed
            siglip_feats = self.siglip_embedder(siglip_feats)
            # TODO:
            # single batch
            # no pad

            for layer in self.siglip_refiner:
                siglip_feats = layer(siglip_feats, siglip_freqs_cis)

        assert len(x) == 1, "single batch only."
        assert len(cap_feats) == 1, "single batch only."
        assert isinstance(
            x_freqs_cis, Tuple
        ), f"should be tuple of (cos, sin), got {type(x_freqs_cis)}"
        assert isinstance(
            cap_freqs_cis, Tuple
        ), f"should be tuple of (cos, sin), got {type(cap_freqs_cis)}"
        assert siglip_freqs_cis is None or isinstance(
            siglip_freqs_cis, Tuple
        ), f"should be tuple of (cos, sin), got {type(siglip_freqs_cis)}"

        x_seqlens = [len(xi) for xi in x]
        cap_seqlens = [len(ci) for ci in cap_feats]
        unified, unified_freqs_cis, unified_noise_tensor = self._build_unified_sequence(
            x=x,
            x_freqs=x_freqs_cis,
            x_seqlens=x_seqlens,
            x_noise_mask=x_noise_mask,
            cap=cap_feats,
            cap_freqs=cap_freqs_cis,
            cap_seqlens=cap_seqlens,
            cap_noise_mask=cap_noise_mask,
            siglip=siglip_feats,
            siglip_freqs=siglip_freqs_cis,
            siglip_seqlens=siglip_seqlens,
            siglip_noise_mask=siglip_noise_mask,
            omni_mode=omni_mode,
            device=device,
        )

        for layer in self.layers:
            unified = layer(
                unified,
                unified_freqs_cis,
                adaln_input,
                unified_noise_tensor,
                t_noisy,
                t_clean,
            )
            # TODO: controlnet is ignore for now.
            # if controlnet ...

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified,
            adaln_input,
            noise_mask=unified_noise_tensor,
            c_noisy=t_noisy,
            c_clean=t_clean,
        )
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size, x_pos_offsets)

        return -x[0]


EntryClass = ZImageTransformer2DModel
