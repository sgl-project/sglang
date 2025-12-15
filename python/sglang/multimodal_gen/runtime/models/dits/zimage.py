import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.rotary_embedding import _apply_rotary_emb
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
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
                ReplicatedLinear(frequency_embedding_size, mid_size, bias=True),
                nn.SiLU(),
                ReplicatedLinear(mid_size, out_size, bias=True),
            ]
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
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


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = ReplicatedLinear(dim, hidden_dim, bias=False)
        self.w2 = ReplicatedLinear(hidden_dim, dim, bias=False)
        self.w3 = ReplicatedLinear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        x1, _ = self.w1(x)
        x3, _ = self.w3(x)
        out, _ = self.w2(self._forward_silu_gating(x1, x3))
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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.to_q = ReplicatedLinear(dim, dim, bias=False)
        self.to_k = ReplicatedLinear(dim, self.head_dim * num_kv_heads, bias=False)
        self.to_v = ReplicatedLinear(dim, self.head_dim * num_kv_heads, bias=False)

        if self.qk_norm:
            self.norm_q = RMSNorm(self.head_dim, eps=eps)
            self.norm_k = RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.ModuleList([ReplicatedLinear(dim, dim, bias=False)])

        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            num_kv_heads=num_kv_heads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)

        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.num_kv_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        # Apply RoPE
        def apply_rotary_emb(
            x_in: torch.Tensor,
            freqs_cis: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            cos, sin = freqs_cis
            x_out = _apply_rotary_emb(x_in, cos, sin, is_neox_style=False)
            return x_out

        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

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
    ):
        if self.modulation:
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
        self.linear = ReplicatedLinear(hidden_size, out_channels, bias=True)

        self.act = nn.SiLU()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale, _ = self.adaLN_modulation(c)
        scale = 1.0 + scale
        x = self.norm_final(x) * scale.unsqueeze(1)
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


class ZImageTransformer2DModel(CachableDiT):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]

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

        self.rope_theta = arch_config.rope_theta
        self.t_scale = arch_config.t_scale
        self.gradient_checkpointing = False

        assert len(self.all_patch_size) == len(self.all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(
            zip(self.all_patch_size, self.all_f_patch_size)
        ):
            x_embedder = ReplicatedLinear(
                f_patch_size * patch_size * patch_size * self.in_channels,
                self.dim,
                bias=True,
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

    def unpatchify(
        self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size
    ) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
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

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        assert len(all_image) == len(all_cap_feats) == 1

        image = all_image[0]  # C, F, H, W
        cap_feat = all_cap_feats[0]  # L, D
        pH = pW = patch_size
        pF = f_patch_size
        device = image.device

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
        C, F, H, W = image.size()
        all_image_size.append((F, H, W))

        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            F_tokens * H_tokens * W_tokens, pF * pH * pW * C
        )
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

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        encoder_hidden_states: List[torch.Tensor],
        timestep,
        guidance=0,
        patch_size=2,
        f_patch_size=1,
        freqs_cis=None,
        **kwargs,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        x = hidden_states
        cap_feats = encoder_hidden_states
        timestep = 1000.0 - timestep
        t = timestep
        bsz = 1
        device = x[0].device
        t = self.t_embedder(t)
        adaln_input = t.type_as(x)
        (
            x,
            cap_feats,
            x_size,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        x = torch.cat(x, dim=0)
        x, _ = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        x_freqs_cis = freqs_cis[1]

        x = x.unsqueeze(0)
        x_freqs_cis = x_freqs_cis
        for layer in self.noise_refiner:
            x = layer(x, x_freqs_cis, adaln_input)

        cap_feats = torch.cat(cap_feats, dim=0)

        cap_feats, _ = self.cap_embedder(cap_feats)

        cap_freqs_cis = freqs_cis[0]

        cap_feats = cap_feats.unsqueeze(0)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs_cis)

        unified = torch.cat([x, cap_feats], dim=1)
        unified_freqs_cis = (
            torch.cat([x_freqs_cis[0], cap_freqs_cis[0]], dim=0),
            torch.cat([x_freqs_cis[1], cap_freqs_cis[1]], dim=0),
        )

        for layer in self.layers:
            unified = layer(unified, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified, adaln_input
        )
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        return -x[0]


EntryClass = ZImageTransformer2DModel
