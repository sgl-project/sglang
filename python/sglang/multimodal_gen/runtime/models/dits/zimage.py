import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import (
    UlyssesAttention,
    USPAttention,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
    apply_rmsnorm_tanh_mul_add,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
    is_nunchaku_available,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.managers.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

try:
    from nunchaku.models.attention import NunchakuFeedForward  # type: ignore[import]
except Exception:
    NunchakuFeedForward = None

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()

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


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Use MergedColumnParallelLinear for gate and up projection (fused)
        self.w13 = MergedColumnParallelLinear(
            dim,
            [hidden_dim, hidden_dim],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
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
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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

        kv_dim = self.head_dim * num_kv_heads
        self.use_fused_qkv = True

        if self.use_fused_qkv:
            self.to_qkv = MergedColumnParallelLinear(
                dim,
                [dim, kv_dim, kv_dim],
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_qkv",
            )
        else:
            self.to_q = ColumnParallelLinear(
                dim,
                dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_q",
            )
            self.to_k = ColumnParallelLinear(
                dim,
                kv_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_k",
            )
            self.to_v = ColumnParallelLinear(
                dim,
                kv_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_v",
            )

        if self.qk_norm:
            self.norm_q = RMSNorm(self.head_dim, eps=eps)
            self.norm_k = RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )

        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.local_num_kv_heads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )
        self.ulysses_attn = UlyssesAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.local_num_kv_heads,
            softmax_scale=None,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_replicated_prefix: int = 0,
        num_replicated_suffix: int = 0,
    ):
        if self.use_fused_qkv:
            qkv, _ = self.to_qkv(hidden_states)
            q, k, v = qkv.split(
                [
                    self.local_num_heads * self.head_dim,
                    self.local_num_kv_heads * self.head_dim,
                    self.local_num_kv_heads * self.head_dim,
                ],
                dim=-1,
            )
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            q, _ = self.to_q(hidden_states)
            k, _ = self.to_k(hidden_states)
            v, _ = self.to_v(hidden_states)
        q = q.view(*q.shape[:-1], self.local_num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.local_num_kv_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.local_num_kv_heads, self.head_dim)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            if _is_cuda and q.shape == k.shape:
                cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
                if self.qk_norm:
                    q, k = apply_qk_norm_with_optional_rope(
                        q=q,
                        k=k,
                        q_norm=self.norm_q,
                        k_norm=self.norm_k,
                        head_dim=self.head_dim,
                        cos_sin_cache=cos_sin_cache,
                        is_neox=False,
                        allow_inplace=True,
                    )
                else:
                    q, k = apply_flashinfer_rope_qk_inplace(
                        q, k, cos_sin_cache, is_neox=False
                    )
            else:
                if self.qk_norm:
                    q, k = apply_qk_norm_with_optional_rope(
                        q=q,
                        k=k,
                        q_norm=self.norm_q,
                        k_norm=self.norm_k,
                        head_dim=self.head_dim,
                        allow_inplace=True,
                    )
                q = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
                k = _apply_rotary_emb(k, cos, sin, is_neox_style=False)
        elif self.qk_norm:
            q, k = apply_qk_norm_with_optional_rope(
                q=q,
                k=k,
                q_norm=self.norm_q,
                k_norm=self.norm_k,
                head_dim=self.head_dim,
                allow_inplace=True,
            )

        if (
            num_replicated_suffix > 0
            and get_sp_world_size() > 1
            and get_ring_parallel_world_size() == 1
        ):
            # the cap (last num_replicated_suffix tokens), as condition, should be replicated
            q_shard, q_rep = (
                q[:, :-num_replicated_suffix],
                q[:, -num_replicated_suffix:],
            )
            k_shard, k_rep = (
                k[:, :-num_replicated_suffix],
                k[:, -num_replicated_suffix:],
            )
            v_shard, v_rep = (
                v[:, :-num_replicated_suffix],
                v[:, -num_replicated_suffix:],
            )
            hidden_states, hidden_states_rep = self.ulysses_attn(
                q_shard,
                k_shard,
                v_shard,
                replicated_q=q_rep,
                replicated_k=k_rep,
                replicated_v=v_rep,
            )
            assert hidden_states_rep is not None
            hidden_states = torch.cat([hidden_states, hidden_states_rep], dim=1)
        else:
            hidden_states = self.attn(
                q,
                k,
                v,
                num_replicated_prefix=num_replicated_prefix,
                num_replicated_suffix=num_replicated_suffix,
            )
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
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        if not modulation:
            # Context refiner runs on fully replicated caption tokens only.
            # Bypass Ulysses here to preserve the single-GPU attention semantics.
            self.attention.attn.skip_sequence_parallel = True

        hidden_dim = int(dim / 3 * 8)
        nunchaku_enabled = (
            isinstance(quant_config, NunchakuConfig) and is_nunchaku_available()
        )
        if nunchaku_enabled:
            import diffusers

            ff = diffusers.models.attention.FeedForward(
                dim=dim,
                dim_out=dim,
                activation_fn="swiglu",
                inner_dim=hidden_dim,
                bias=False,
            )
            nunchaku_kwargs = {
                "precision": quant_config.precision,
                "rank": quant_config.rank,
                "act_unsigned": quant_config.act_unsigned,
            }
            self.feed_forward = NunchakuFeedForward(ff, **nunchaku_kwargs)
            # NunchakuFeedForward overrides net[2].act_unsigned=True for int4 (GELU-specific
            # optimization for non-negative activations). Z-Image uses SwiGLU whose output
            # can be negative, so we must restore the original act_unsigned value.
            if hasattr(self.feed_forward, "net") and len(self.feed_forward.net) > 2:
                self.feed_forward.net[2].act_unsigned = quant_config.act_unsigned
        else:
            self.feed_forward = FeedForward(
                dim=dim,
                hidden_dim=hidden_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )

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
        num_replicated_prefix: int = 0,
        num_replicated_suffix: int = 0,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa_gate, _ = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = scale_msa_gate.unsqueeze(
                1
            ).chunk(4, dim=2)
            scale_msa = 1.0 + scale_msa

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
                num_replicated_prefix=num_replicated_prefix,
                num_replicated_suffix=num_replicated_suffix,
            )
            if (
                _is_cuda
                and attn_out.is_cuda
                and attn_out.shape[-1] % 256 == 0
                and attn_out.shape[-1] <= 8192
                and self.attention_norm2.variance_epsilon
                == self.ffn_norm1.variance_epsilon
            ):
                from sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale import (
                    fused_norm_tanh_mul_add_norm_scale,
                )

                x, ffn_in = fused_norm_tanh_mul_add_norm_scale(
                    attn_out.contiguous(),
                    self.attention_norm2.weight.data.contiguous(),
                    None,
                    gate_msa.contiguous(),
                    x.contiguous(),
                    self.ffn_norm1.weight.data.contiguous(),
                    None,
                    scale_mlp.contiguous(),
                    "rms",
                    self.attention_norm2.variance_epsilon,
                )
            else:
                x = apply_rmsnorm_tanh_mul_add(
                    attn_out, gate_msa, x, self.attention_norm2
                )
                ffn_in = self.ffn_norm1(x) * (1.0 + scale_mlp)

            # FFN block
            ffn_out = self.feed_forward(ffn_in)
            x = apply_rmsnorm_tanh_mul_add(ffn_out, gate_mlp, x, self.ffn_norm2)
        else:
            # Attention block
            attn_input = self.attention_norm1(x)
            attn_out = self.attention(
                attn_input,
                freqs_cis=freqs_cis,
                num_replicated_prefix=num_replicated_prefix,
                num_replicated_suffix=num_replicated_suffix,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            ffn_input = self.ffn_norm1(x)
            ffn_out = self.feed_forward(
                ffn_input,
            )
            x = x + self.ffn_norm2(ffn_out)

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


class ZImageTransformer2DModel(CachableDiT, OffloadableDiTMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]
    _fsdp_shard_conditions = ZImageDitConfig().arch_config._fsdp_shard_conditions
    param_names_mapping = ZImageDitConfig().arch_config.param_names_mapping

    param_names_mapping = ZImageDitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        ZImageDitConfig().arch_config.reverse_param_names_mapping
    )

    # Maps fused runtime layer names to their checkpoint shard names.
    # Used by is_layer_skipped() to correctly handle --quantization-ignored-layers
    # Only list fusions that are unconditional. Conditional fusions (e.g. to_qkv for
    # Nunchaku) are handled by their own quant path.
    packed_modules_mapping = {
        "w13": ["w1", "w3"],
    }

    @classmethod
    def get_nunchaku_quant_rules(cls) -> dict[str, list[str]]:
        return {
            "skip": [
                "norm",
                "embed",
                "rotary",
                "pos_embed",
            ],
            "svdq_w4a4": [
                "attention.to_qkv",
                "attention.to_out",
                "img_mlp",
                "txt_mlp",
            ],
            "awq_w4a16": [
                "img_mod",
                "txt_mod",
            ],
        }

    def __init__(
        self,
        config: ZImageDitConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
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
                    quant_config=quant_config,
                    prefix=f"noise_refiner.{layer_id}",
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
                    quant_config=quant_config,
                    prefix=f"context_refiner.{layer_id}",
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
                    quant_config=quant_config,
                    prefix=f"layers.{layer_id}",
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

    @staticmethod
    def _ceil_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            return value
        return int(math.ceil(value / multiple) * multiple)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        image_seq_len_target: int | None = None,
    ):
        """Patchify images and pad image/caption tokens to batch targets.

        Each image is [C, F, H, W] and has one [L, D] caption. Returned tensors
        are stacked as [B, S, D], while valid lengths keep track of real tokens
        before learned pad tokens are restored. `image_seq_len_target`, when
        set, is the SP-local padded image-token target.
        """
        if len(all_image) != len(all_cap_feats):
            raise ValueError(
                f"Z-Image expects one caption embedding per image, got {len(all_image)} images and {len(all_cap_feats)} captions"
            )
        if not all_image:
            raise ValueError("Z-Image batch must contain at least one image latent")

        pH = pW = patch_size
        pF = f_patch_size
        all_image_out = []
        all_image_size = []
        all_cap_feats_out = []
        all_image_valid_lens = []
        all_cap_valid_lens = []
        image_records = []

        cap_seq_len_target = max(
            self._ceil_to_multiple(cap_feat.size(0), SEQ_MULTI_OF)
            for cap_feat in all_cap_feats
        )

        for cap_feat in all_cap_feats:
            cap_ori_len = cap_feat.size(0)
            cap_padding_len = cap_seq_len_target - cap_ori_len
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)
            all_cap_valid_lens.append(cap_ori_len)

        target_image_seq_len = image_seq_len_target or 0
        for image in all_image:
            # ------------ Process Image ------------
            C, F, H, W = image.size()
            image_size = (F, H, W)

            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                F_tokens * H_tokens * W_tokens, pF * pH * pW * C
            )
            image_ori_len = image.size(0)
            target_image_seq_len = max(
                target_image_seq_len,
                self._ceil_to_multiple(image_ori_len, SEQ_MULTI_OF),
            )
            image_records.append((image, image_size, image_ori_len))

        for image, image_size, image_ori_len in image_records:
            image_padding_len = target_image_seq_len - image_ori_len
            image_padded_feat = torch.cat(
                [image, image[-1:].repeat(image_padding_len, 1)],
                dim=0,
            )
            all_image_out.append(image_padded_feat)
            all_image_size.append(image_size)
            all_image_valid_lens.append(image_ori_len)

        return (
            torch.stack(all_image_out, dim=0),
            torch.stack(all_cap_feats_out, dim=0),
            all_image_size,
            all_image_valid_lens,
            all_cap_valid_lens,
        )

    @staticmethod
    def _as_image_list(hidden_states) -> list[torch.Tensor]:
        """Normalize 4D/5D image latents into per-sample tensors."""
        if torch.is_tensor(hidden_states):
            if hidden_states.dim() == 5:
                return list(hidden_states.unbind(dim=0))
            if hidden_states.dim() == 4:
                return [hidden_states]
        return list(hidden_states)

    @staticmethod
    def _as_caption_list(encoder_hidden_states) -> list[torch.Tensor]:
        """Normalize caption tensors into per-sample tensors."""
        if torch.is_tensor(encoder_hidden_states):
            if encoder_hidden_states.dim() == 3:
                return list(encoder_hidden_states.unbind(dim=0))
            if encoder_hidden_states.dim() == 2:
                return [encoder_hidden_states]

        cap_feats = list(encoder_hidden_states)
        if len(cap_feats) == 1 and torch.is_tensor(cap_feats[0]):
            if cap_feats[0].dim() == 3:
                return list(cap_feats[0].unbind(dim=0))
            if cap_feats[0].dim() == 2:
                return cap_feats
        return cap_feats

    @staticmethod
    def _replace_padding_with_token(
        tensor: torch.Tensor,
        valid_lens: list[int],
        pad_token: torch.Tensor,
    ) -> torch.Tensor:
        """Replace padded token rows after each valid sequence length."""
        positions = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0)
        lengths = torch.tensor(valid_lens, device=tensor.device).unsqueeze(1)
        pad_mask = positions >= lengths
        if pad_mask.any():
            tensor = tensor.clone()
            tensor[pad_mask] = pad_token.to(device=tensor.device, dtype=tensor.dtype)
        return tensor

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        encoder_hidden_states: List[torch.Tensor],
        timestep,
        guidance=0,
        patch_size=2,
        f_patch_size=1,
        freqs_cis=None,
        image_seq_len_target: int | None = None,
        **kwargs,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        x = self._as_image_list(hidden_states)
        cap_feats = self._as_caption_list(encoder_hidden_states)
        timestep = 1000.0 - timestep
        t = timestep
        device = x[0].device
        t = self.t_embedder(t)
        adaln_input = t.to(dtype=x[0].dtype)
        (
            x,
            cap_feats,
            x_size,
            x_valid_lens,
            cap_valid_lens,
        ) = self.patchify_and_embed(
            x,
            cap_feats,
            patch_size,
            f_patch_size,
            image_seq_len_target=image_seq_len_target,
        )

        x, _ = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        x = self._replace_padding_with_token(x, x_valid_lens, self.x_pad_token)
        x_freqs_cis = freqs_cis[1]

        for layer_id, layer in enumerate(self.noise_refiner):
            x = layer(x, x_freqs_cis, adaln_input)

        cap_feats, _ = self.cap_embedder(cap_feats)
        cap_feats = self._replace_padding_with_token(
            cap_feats, cap_valid_lens, self.cap_pad_token
        )

        cap_freqs_cis = freqs_cis[0]

        for layer_id, layer in enumerate(self.context_refiner):
            cap_feats = layer(
                cap_feats,
                cap_freqs_cis,
            )

        cap_seq_len = cap_feats.shape[1]
        use_full_unified_sequence = (
            get_sp_world_size() > 1 and get_ring_parallel_world_size() > 1
        )
        x_local_seq_len = x.shape[1]
        if use_full_unified_sequence:
            x = sequence_model_parallel_all_gather(x.contiguous(), dim=1)
            x_freqs_cis = (
                sequence_model_parallel_all_gather(x_freqs_cis[0].contiguous(), dim=0),
                sequence_model_parallel_all_gather(x_freqs_cis[1].contiguous(), dim=0),
            )
        unified = torch.cat([x, cap_feats], dim=1)
        unified_freqs_cis = (
            torch.cat([x_freqs_cis[0], cap_freqs_cis[0]], dim=0),
            torch.cat([x_freqs_cis[1], cap_freqs_cis[1]], dim=0),
        )
        num_replicated_suffix = cap_seq_len if not use_full_unified_sequence else 0

        for layer_id, layer in enumerate(self.layers):
            layer.attention.attn.skip_sequence_parallel = use_full_unified_sequence
            unified = layer(
                unified,
                unified_freqs_cis,
                adaln_input,
                num_replicated_suffix=num_replicated_suffix,
            )

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified, adaln_input
        )
        if use_full_unified_sequence:
            sp_rank = get_sp_parallel_rank()
            start = sp_rank * x_local_seq_len
            end = start + x_local_seq_len
            unified = unified[:, start:end]
        x = list(unified.unbind(dim=0))
        x = self.unpatchify(x, x_size, patch_size, f_patch_size)

        # Keep batch dim so output shape matches input (e.g. rollout/scheduler expect same ndim).
        return -torch.stack(x)


EntryClass = ZImageTransformer2DModel
