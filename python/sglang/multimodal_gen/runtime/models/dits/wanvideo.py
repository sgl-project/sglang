# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import (
    FUSED_LINEAR_GELU,
    NVFP4_BIAS_GELU,
    BitExactFusionGate,
    can_use_fused_rmsnorm_interleaved_rope,
    can_use_linear_gelu,
    fused_linear_gelu_tanh,
    fused_rmsnorm_interleaved_rope_,
    fused_rmsnorm_interleaved_rope_into_blocks,
    fused_temb_table_slices,
    rmsnorm_interleaved_rope_block_cols,
    tensors_equal,
)
from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.models.fsdp import is_block
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import (
    MinimalA2AAttnOp,
    UlyssesAttention_VSA,
    USPAttention,
)
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidualLayerNormScaleShift,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
    apply_unquantized_linear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.platforms.aiter import USE_AITER
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()

if USE_AITER:
    from aiter.ops.rope import rope_cached_2c_fwd_inplace


class _WanGELUMLP(MLP):
    """Wan FFN with request-scoped GELU fast paths."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        prefix: str,
        quant_config: QuantizationConfig | None,
    ):
        super().__init__(
            dim,
            ffn_dim,
            act_type="gelu_pytorch_tanh",
            prefix=prefix,
            quant_config=quant_config,
            fuse_bias_gelu_tanh=False,
        )
        FUSED_LINEAR_GELU.mark(self, "fc_in")
        self.fuse_bias_gelu_tanh = isinstance(
            self.fc_in.quant_method, ModelOptFp4LinearMethod
        )
        if self.fuse_bias_gelu_tanh:
            NVFP4_BIAS_GELU.mark(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FUSED_LINEAR_GELU.is_enabled(self) and can_use_linear_gelu(self.fc_in, x):
            x = fused_linear_gelu_tanh(x, self.fc_in.weight, self.fc_in.bias)
        else:
            x, bias = self.fc_in(x)
            x = self._apply_activation(
                x,
                bias,
                use_fused_bias_gelu=NVFP4_BIAS_GELU.is_enabled(self),
            )
        x, _ = self.fc_out(x)
        return x


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        dtype = encoder_hidden_states_image.dtype
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states).to(dtype)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu"
        )
        self.time_modulation = ModulateProjection(dim, factor=6, act_layer="silu")
        self.text_embedder = MLP(
            text_embed_dim, dim, dim, bias=True, act_type="gelu_pytorch_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
        prefix: str = "",
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        is_cross_attention: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        tp_size = get_tp_world_size()

        # layers
        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_v", prefix),
        )
        self.to_out = RowParallelLinear(
            dim,
            dim,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("to_out", prefix),
        )
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.tp_rmsnorm = tp_size > 1 and qk_norm
        self.local_num_heads = divide(num_heads, tp_size)

        # Scaled dot product attention
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            skip_sequence_parallel=is_cross_attention,
            is_cross_attention=is_cross_attention,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: int):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is_cross_attention=True)

    def forward(self, x, context, context_lens, context_kv=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            context_kv: ``project_context_kv(context)``, when already projected
        """
        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))

        k, v = self.project_context_kv(context) if context_kv is None else context_kv

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x

    def project_context_kv(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The normed K and the V of ``context``, as [B, L2, heads, head_dim]"""
        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))
        v, _ = self.to_v(context)
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))
        return k, v


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        prefix: str = "",
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
            is_cross_attention=True,
            quant_config=quant_config,
        )

        self.add_k_proj = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("add_k_proj", prefix),
        )
        self.add_v_proj = ColumnParallelLinear(
            dim,
            dim,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("add_v_proj", prefix),
        )
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]

        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))

        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))

        v, _ = self.to_v(context)
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        k_img, _ = self.add_k_proj(context_img)
        if self.tp_rmsnorm:
            k_img = tensor_parallel_rms_norm(k_img, self.norm_added_k)
        else:
            k_img = self.norm_added_k(k_img)
        k_img = k_img.unflatten(2, (self.local_num_heads, self.head_dim))

        v_img, _ = self.add_v_proj(context_img)
        v_img = v_img.unflatten(2, (self.local_num_heads, self.head_dim))

        img_x = self.attn(q, k_img, v_img)
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


_WAN_TEMB_SLICES = BitExactFusionGate("Wan fused temb-table slices")


def _eager_temb_table_slices(
    table: torch.Tensor, temb: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    parts = (table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
    return tuple(part.squeeze(2) for part in parts)


def _wan_temb_table_slices(
    table: torch.Tensor, temb: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Per-token adaLN slices ``(table + temb.float()).chunk(6)`` in one pass.

    The fused kernel writes each ``(B, S, D)`` slice contiguously, so the
    downstream fused-norm wrappers' ``.contiguous()`` calls stop copying the
    full activation.  A float32 add of widened values involves no rounding,
    so the result is bit-identical to the eager chain; the first call still
    verifies ``torch.equal`` and falls back permanently on mismatch.
    """

    def fused():
        buf = fused_temb_table_slices(table, temb)
        return tuple(buf[j] for j in range(6))

    if _is_cuda and temb.is_cuda:
        return _WAN_TEMB_SLICES.run(
            fused,
            lambda: _eager_temb_table_slices(table, temb),
            equal=tensors_equal,
            logger=logger,
        )
    return _eager_temb_table_slices(table, temb)


def _rows_of_one_buffer(parts: list[torch.Tensor]) -> torch.Tensor | None:
    """``torch.cat(parts)`` without a copy, when the parts are consecutive equal-shape
    rows of one buffer"""
    first = parts[0]
    start, nbytes = first.data_ptr(), first.nbytes
    storage = first.untyped_storage()
    if start + len(parts) * nbytes > storage.data_ptr() + storage.nbytes():
        return None
    for i, part in enumerate(parts):
        if (
            part.data_ptr() != start + i * nbytes
            or part.shape != first.shape
            or not part.is_contiguous()
        ):
            return None
    return first.as_strided(
        (len(parts) * first.shape[0], *first.shape[1:]), first.stride()
    )


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        attention_type: str = "original",
        sla_topk: float = 0.1,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_v", prefix),
        )

        self.to_out = RowParallelLinear(
            dim,
            dim,
            bias=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("to_out", prefix),
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.dim_head = dim // num_heads
        self.use_offline_qk_rotation = False

        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)
        self_attn_backends = supported_attention_backends

        if attention_type in ("sla", "sagesla"):
            self.attn1 = MinimalA2AAttnOp(
                num_heads=self.local_num_heads,
                head_size=dim // num_heads,
                attention_type=attention_type,
                topk=sla_topk,
                supported_attention_backends={
                    AttentionBackendEnum.SLA_ATTN,
                    AttentionBackendEnum.SAGE_SLA_ATTN,
                },
                prefix=add_prefix("attn1", prefix),
            )
        else:
            # TODO Need to create mxfp8 attention scheme and port the code below
            from sglang.multimodal_gen import envs

            quant_description = getattr(quant_config, "quant_description", {})
            self.use_offline_qk_rotation = (
                quant_description.get(f"{prefix}.attn1.q_rot") == "FLOAT"
                and quant_description.get(f"{prefix}.attn1.k_rot") == "FLOAT"
                and envs.SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION
            )
            if self.use_offline_qk_rotation:
                self.register_buffer(
                    "q_rot",
                    torch.empty(
                        self.dim_head,
                        self.dim_head,
                        dtype=torch.bfloat16,
                    ),
                    persistent=True,
                )
                self.register_buffer(
                    "k_rot",
                    torch.empty(
                        self.dim_head,
                        self.dim_head,
                        dtype=torch.bfloat16,
                    ),
                    persistent=True,
                )
                quant_config.use_offline_qk_rotation = True

            self.attn1 = USPAttention(
                num_heads=self.local_num_heads,
                head_size=dim // num_heads,
                causal=False,
                supported_attention_backends=self_attn_backends,
                prefix=add_prefix("attn1", prefix),
                quant_config=quant_config,
                is_cross_attention=False,
            )

        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(self.dim_head, eps=eps)
            self.norm_k = RMSNorm(self.dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.qk_norm = qk_norm
        self.tp_rmsnorm = qk_norm == "rms_norm_across_heads" and tp_size > 1
        # the across-heads q/k RMSNorm runs FlashInfer's kernel, which the RoPE can join
        self.fuse_qk_norm_rope = (
            qk_norm == "rms_norm_across_heads"
            and not self.tp_rmsnorm
            and not self.norm_q.cast_x_before_out_mul
            and self.norm_q._forward_method == self.norm_q.forward_cuda
        )
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention
        cross_attn_backends = {
            b for b in supported_attention_backends if not b.is_sparse
        }
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = _WanGELUMLP(
            dim,
            ffn_dim,
            prefix=add_prefix("ffn", prefix),
            quant_config=quant_config,
        )
        self.mlp_residual = MulAdd()
        # norm1 that first adds the previous block's gated FFN output (its MulAdd)
        self.ffn_residual_norm1 = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # set by fuse_qkv_weights: to_q / to_k / to_v are plain unquantized linears
        self.qkv_fusable = False
        # exchange-block addresses -> (q, k) destination tables of _attend_qk_filled
        self._qk_tables: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def fuse_qkv_weights(self) -> None:
        """Move to_q / to_k / to_v into consecutive rows of one weight and one bias, so a
        single GEMM projects all three (``_fused_qkv``)"""
        layers = (self.to_q, self.to_k, self.to_v)
        self.qkv_fusable = all(
            type(layer) is ColumnParallelLinear
            and isinstance(layer.quant_method, UnquantizedLinearMethod)
            and layer.bias is not None
            and not layer.skip_bias_add
            and not layer.gather_output
            for layer in layers
        )
        if not self.qkv_fusable:
            return
        with torch.no_grad():
            weight = torch.cat([layer.weight for layer in layers])
            bias = torch.cat([layer.bias for layer in layers])
        rows = self.to_q.weight.shape[0]
        for i, layer in enumerate(layers):
            layer.weight.data = weight[i * rows : (i + 1) * rows]
            layer.bias.data = bias[i * rows : (i + 1) * rows]

    def _fused_qkv(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The [3 * dim, dim] weight and [3 * dim] bias of ``fuse_qkv_weights``, read back
        from the layers while they still share them on ``device``. Only the layers hold
        the shared buffers, so moving the module or offloading or swapping a layer's
        weight frees them."""
        layers = (self.to_q, self.to_k, self.to_v)
        if (
            not self.qkv_fusable
            or torch.compiler.is_compiling()
            or not all(type(layer) is ColumnParallelLinear for layer in layers)
            or self.to_q.weight.device != device
        ):
            return None
        weight = _rows_of_one_buffer([layer.weight for layer in layers])
        bias = _rows_of_one_buffer([layer.bias for layer in layers])
        if weight is None or bias is None:
            return None
        return weight, bias

    def _qkv(
        self, norm_hidden_states: torch.Tensor, local_heads: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._qkv_rope(norm_hidden_states, local_heads, None)[:3]

    def _qkv_rope(
        self,
        norm_hidden_states: torch.Tensor,
        local_heads: int,
        rope_cos_sin_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """q, k, v as [B, S, heads, head_dim], and whether q and k already carry the
        RoPE of rope_cos_sin_cache (applied in the q/k norm pass)"""
        return self._norm_qk(
            *self._project_qkv(norm_hidden_states), local_heads, rope_cos_sin_cache
        )

    def _project_qkv(
        self, norm_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self._fused_qkv(norm_hidden_states.device)
        if fused is not None:
            qkv = apply_unquantized_linear(norm_hidden_states, *fused)
            return qkv.split(self.to_q.weight.shape[0], dim=-1)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        return query, key, value

    def _fuses_qk_norm_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cos_sin_cache: torch.Tensor | None,
    ) -> bool:
        return (
            rope_cos_sin_cache is not None
            and self.fuse_qk_norm_rope
            and not torch.compiler.is_compiling()
            and all(
                can_use_fused_rmsnorm_interleaved_rope(
                    x.view(-1, x.shape[-1]),
                    norm.weight,
                    rope_cos_sin_cache,
                    self.dim_head,
                )
                for x, norm in ((query, self.norm_q), (key, self.norm_k))
            )
        )

    def _attend_qk_filled(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_cos_sin_cache: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Self-attention whose q/k norm-RoPE pass writes straight into the Ulysses
        exchange blocks (this rank's own slot rows and the peers' send blocks), so no
        own-row copy of q and k runs; None, with q/k untouched, when it cannot."""
        if (
            type(self.attn1) is not USPAttention
            or self.use_offline_qk_rotation
            or not self._fuses_qk_norm_rope(query, key, rope_cos_sin_cache)
        ):
            return None
        q2d, k2d = (x.view(-1, x.shape[-1]) for x in (query, key))
        block_cols = rmsnorm_interleaved_rope_block_cols(q2d.shape[-1])

        def fill_all(blocks) -> None:
            tables = self._qk_block_tables(blocks, block_cols, q2d.shape[-1])
            if tables is None:
                # blocks the kernel cannot address: norm in place, then copy them
                for x, norm in ((q2d, self.norm_q), (k2d, self.norm_k)):
                    fused_rmsnorm_interleaved_rope_(
                        x,
                        norm.weight,
                        rope_cos_sin_cache,
                        norm.variance_epsilon,
                        self.dim_head,
                    )
                for head_start, head_count, q_dst, k_dst in blocks:
                    cols = slice(
                        head_start * self.dim_head,
                        (head_start + head_count) * self.dim_head,
                    )
                    q_dst.copy_(q2d[:, cols].view_as(q_dst))
                    k_dst.copy_(k2d[:, cols].view_as(k_dst))
                return
            for x, norm, table in zip((q2d, k2d), (self.norm_q, self.norm_k), tables):
                fused_rmsnorm_interleaved_rope_into_blocks(
                    x,
                    norm.weight,
                    rope_cos_sin_cache,
                    norm.variance_epsilon,
                    self.dim_head,
                    table,
                )

        return self.attn1.forward_qk_filled(
            *(
                x.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
                for x in (query, key, value)
            ),
            fill_all,
        )

    def _qk_block_tables(
        self, blocks, block_cols: int, hidden: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        # destination address of every block_cols-wide column run, for q and for k;
        # the exchange blocks keep their addresses, so the tables are built once
        if (
            any(
                head_count * self.dim_head != block_cols
                or (head_start * self.dim_head) % block_cols
                or not (q_dst.is_contiguous() and k_dst.is_contiguous())
                for head_start, head_count, q_dst, k_dst in blocks
            )
            or len(blocks) * block_cols != hidden
        ):
            return None
        ordered = sorted(blocks, key=lambda block: block[0])
        key = tuple((q.data_ptr(), k.data_ptr()) for _, _, q, k in ordered)
        tables = self._qk_tables.get(key)
        if tables is None:
            tables = self._qk_tables[key] = tuple(
                torch.tensor(
                    [block[i].data_ptr() for block in ordered],
                    dtype=torch.int64,
                    device=ordered[0][2].device,
                )
                for i in (2, 3)
            )
        return tables

    def _norm_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        local_heads: int,
        rope_cos_sin_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        qk_norms = ((query, self.norm_q), (key, self.norm_k))
        roped = self._fuses_qk_norm_rope(query, key, rope_cos_sin_cache)
        if roped:
            for x, norm in qk_norms:
                fused_rmsnorm_interleaved_rope_(
                    x.view(-1, x.shape[-1]),
                    norm.weight,
                    rope_cos_sin_cache,
                    norm.variance_epsilon,
                    self.dim_head,
                )
        else:
            if self.norm_q is not None:
                query = (
                    tensor_parallel_rms_norm(query, self.norm_q)
                    if self.tp_rmsnorm
                    else self.norm_q(query)
                )
            if self.norm_k is not None:
                key = (
                    tensor_parallel_rms_norm(key, self.norm_k)
                    if self.tp_rmsnorm
                    else self.norm_k(key)
                )
        query, key, value = (
            tensor.squeeze(1).unflatten(2, (local_heads, self.dim_head))
            for tensor in (query, key, value)
        )
        return query, key, value, roped

    def _cross_and_ffn(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        gate_msa: torch.Tensor,
        c_shift_msa: torch.Tensor,
        c_scale_msa: torch.Tensor,
        c_gate_msa: torch.Tensor,
        orig_dtype: torch.dtype,
        defer_ffn_residual: bool = False,
        text_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)
        null_shift = null_scale = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)
        if text_kv is None:
            attn_output = self.attn2(
                norm_hidden_states, context=encoder_hidden_states, context_lens=None
            )
        else:
            attn_output = self.attn2(
                norm_hidden_states,
                context=encoder_hidden_states,
                context_lens=None,
                context_kv=text_kv,
            )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)
        ff_output = self.ffn(norm_hidden_states)
        if defer_ffn_residual:
            return hidden_states, (ff_output, c_gate_msa)
        return self.mlp_residual(ff_output, c_gate_msa, hidden_states).to(orig_dtype)

    @property
    def chains_ffn_residual(self) -> bool:
        """Whether the FFN residual (MulAdd) and the next norm1 run as one kernel with
        the same bits: both sides on their CUDA kernels"""
        return all(
            op._forward_method == op.forward_cuda
            for op in (self.mlp_residual, self.norm1, self.ffn_residual_norm1)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        rope_cos_sin_cache: torch.Tensor | None = None,
        ffn_residual: tuple[torch.Tensor, torch.Tensor] | None = None,
        defer_ffn_residual: bool = False,
        text_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """``ffn_residual`` is the previous block's (ff_output, gate), still to be added
        to ``hidden_states``; ``defer_ffn_residual`` hands this block's back the same way.
        ``text_kv`` is the cross-attention's projected (K, V) of ``encoder_hidden_states``."""
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        if temb.dim() == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            (
                shift_msa,
                scale_msa,
                gate_msa,
                c_shift_msa,
                c_scale_msa,
                c_gate_msa,
            ) = _wan_temb_table_slices(self.scale_shift_table, temb)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            e = self.scale_shift_table + temb.float()
            (
                shift_msa,
                scale_msa,
                gate_msa,
                c_shift_msa,
                c_scale_msa,
                c_gate_msa,
            ) = e.chunk(6, dim=1)

        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        if ffn_residual is None:
            norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        else:
            ff_output, ff_gate = ffn_residual
            norm_hidden_states, hidden_states = self.ffn_residual_norm1(
                hidden_states, ff_output, ff_gate, shift_msa, scale_msa
            )
        query, key, value = self._project_qkv(norm_hidden_states)
        attn_output = self._attend_qk_filled(query, key, value, rope_cos_sin_cache)
        if attn_output is None:
            attn_output = self._self_attention(
                query, key, value, freqs_cis, rope_cos_sin_cache
            )
        attn_output = attn_output.flatten(2)
        return self._cross_and_ffn(
            hidden_states,
            attn_output,
            encoder_hidden_states,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            orig_dtype,
            defer_ffn_residual,
            text_kv,
        )

    def _self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        rope_cos_sin_cache: torch.Tensor | None,
    ) -> torch.Tensor:
        query, key, value, roped = self._norm_qk(
            query, key, value, self.local_num_heads, rope_cos_sin_cache
        )

        # Apply rotary embeddings
        cos, sin = freqs_cis
        if roped:
            pass
        elif _is_cuda and query.shape == key.shape:
            # The concatenated cache only depends on freqs_cis, which is fixed
            # for the whole forward; the transformer builds it once per call.
            cos_sin_cache = rope_cos_sin_cache
            if cos_sin_cache is None:
                cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        elif USE_AITER:
            query_shape = query.shape
            key_shape = key.shape
            num_tokens = query.shape[:-2].numel()
            q_sbhd = query.view(num_tokens, 1, query_shape[-2], query_shape[-1])
            k_sbhd = key.view(num_tokens, 1, key_shape[-2], key_shape[-1])
            cos_sbhd = cos.contiguous().view(num_tokens, 1, 1, -1)
            sin_sbhd = sin.contiguous().view(num_tokens, 1, 1, -1)
            rope_cached_2c_fwd_inplace(
                q_sbhd,
                k_sbhd,
                cos_sbhd,
                sin_sbhd,
                1,  # GPTJ rotate style
                True,  # reuse_freqs_front_part
                False,  # nope_first
            )
            query = q_sbhd.view(query_shape)
            key = k_sbhd.view(key_shape)
        else:
            query, key = (
                _apply_rotary_emb(query, cos, sin, is_neox_style=False),
                _apply_rotary_emb(key, cos, sin, is_neox_style=False),
            )

        if (
            self.use_offline_qk_rotation
            and self.attn1.backend is AttentionBackendEnum.FA
            and query.shape[1:3] == key.shape[1:3]
            and key.shape == value.shape
            and (query.shape[0] * query.shape[1]) % 64 == 0
        ):
            self.q_rot = self.q_rot.to(device=query.device, dtype=query.dtype)
            self.k_rot = self.k_rot.to(device=key.device, dtype=key.dtype)
            query = torch.matmul(query, self.q_rot)
            key = torch.matmul(key, self.k_rot)

        return self.attn1(query, key, value)


class WanTransformerBlock_VSA(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        attention_type: str = "original",
        sla_topk: float = 0.0,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("to_v", prefix),
        )
        self.to_gate_compress = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_gate_compress", prefix),
        )

        self.to_out = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("to_out", prefix),
        )
        self.attn1 = UlyssesAttention_VSA(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("attn1", prefix),
            quant_config=quant_config,
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention
        cross_attn_backends = {
            b for b in supported_attention_backends if not b.is_sparse
        }
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = _WanGELUMLP(
            dim,
            ffn_dim,
            prefix=add_prefix("ffn", prefix),
            quant_config=quant_config,
        )
        self.mlp_residual = MulAdd()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        rope_cos_sin_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1
        )
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        gate_compress, _ = self.to_gate_compress(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        gate_compress = gate_compress.squeeze(1).unflatten(
            2, (self.num_attention_heads, -1)
        )

        # Apply rotary embeddings
        cos, sin = freqs_cis
        if _is_cuda and query.shape == key.shape:
            # The concatenated cache only depends on freqs_cis, which is fixed
            # for the whole forward; the transformer builds it once per call.
            cos_sin_cache = rope_cos_sin_cache
            if cos_sin_cache is None:
                cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        elif USE_AITER:
            query_shape = query.shape
            key_shape = key.shape
            num_tokens = query.shape[:-2].numel()
            q_sbhd = query.view(num_tokens, 1, query_shape[-2], query_shape[-1])
            k_sbhd = key.view(num_tokens, 1, key_shape[-2], key_shape[-1])
            cos_sbhd = cos.contiguous().view(num_tokens, 1, 1, -1)
            sin_sbhd = sin.contiguous().view(num_tokens, 1, 1, -1)
            rope_cached_2c_fwd_inplace(
                q_sbhd,
                k_sbhd,
                cos_sbhd,
                sin_sbhd,
                1,  # GPTJ rotate style
                True,  # reuse_freqs_front_part
                False,  # nope_first
            )
            query = q_sbhd.view(query_shape)
            key = k_sbhd.view(key_shape)
        else:
            query, key = (
                _apply_rotary_emb(query, cos, sin, is_neox_style=False),
                _apply_rotary_emb(key, cos, sin, is_neox_style=False),
            )

        attn_output = self.attn1(query, key, value, gate_compress=gate_compress)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros((1,), device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = (
            norm_hidden_states.to(orig_dtype),
            hidden_states.to(orig_dtype),
        )

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = (
            norm_hidden_states.to(orig_dtype),
            hidden_states.to(orig_dtype),
        )

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


# DENOISING_LOOP_STATE entry of the text K/V (WanTransformer3DModel._text_kvs)
_TEXT_KV_STATE = "wan_text_kv"


class WanTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = [is_block]
    _compile_conditions = [is_block]
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: WanVideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                self._make_block(index=i, config=config, quant_config=quant_config)
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = ColumnParallelLinear(
            inner_dim,
            config.out_channels * math.prod(config.patch_size),
            bias=True,
            gather_output=True,
            prefix="proj_out",
            quant_config=quant_config,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        # For type checking

        self.cnt = 0
        self.__post_init__()

        # misc
        self.sp_size = get_sp_world_size()

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]

        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
        )

        self.layer_names = ["blocks"]

    def _text_kvs(
        self, source: torch.Tensor, context: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Every block's cross-attention (K, V) of ``context``, the embedding of ``source``.

        The denoising stage hands the same prompt tensor to every step of a loop, so the
        K/V are projected once per (model, prompt tensor) and kept in the request's
        DENOISING_LOOP_STATE, which the stage drops when the loop ends; the module holds
        none of them. Holding the tensor keeps its id from being reused while cached.
        Off when the projections could differ step to step (LoRA wrappers, offloaded
        weights) or would be captured.
        """
        req = get_forward_context().forward_batch
        if (
            req is None
            or not isinstance(source, torch.Tensor)
            or self.layerwise_offload_managers
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
            or not self._plain_blocks()
            or not all(
                type(block.attn2) is WanT2VCrossAttention
                and type(block.attn2.to_k) is ColumnParallelLinear
                and type(block.attn2.to_v) is ColumnParallelLinear
                for block in self.blocks
            )
        ):
            return None
        # (id(model), id(prompt tensor)) -> (that tensor, every block's K/V)
        cached = req.extra.setdefault(DENOISING_LOOP_STATE, {}).setdefault(
            _TEXT_KV_STATE, {}
        )
        key = (id(self), id(source))
        entry = cached.pop(key, None)
        if (
            entry is not None
            and entry[0] is source
            and entry[1][0][0].device == context.device
        ):
            cached[key] = entry
            return entry[1]
        # the other expert is done with its K/V; keep this model's other CFG branch
        mine = [k for k in cached if k[0] == id(self)]
        for stale in [k for k in cached if k[0] != id(self)] + mine[:-1]:
            del cached[stale]
        del entry
        kvs = [block.attn2.project_context_kv(context) for block in self.blocks]
        cached[key] = (source, kvs)
        return kvs

    def _plain_blocks(self) -> bool:
        # plain Wan blocks in a plain list: no cache adapter or block hook replaced them
        return type(self.blocks) is nn.ModuleList and all(
            type(block) is WanTransformerBlock and "forward" not in vars(block)
            for block in self.blocks
        )

    def _chains_ffn_residuals(self) -> bool:
        return (
            len(self.blocks) > 1
            and self._plain_blocks()
            and all(block.chains_ffn_residual for block in self.blocks)
        )

    def post_load_weights(self) -> None:
        super().post_load_weights()
        for block in self.blocks:
            if isinstance(block, WanTransformerBlock):
                block.fuse_qkv_weights()
        fused = sum(
            isinstance(block, WanTransformerBlock)
            and block._fused_qkv(block.to_q.weight.device) is not None
            for block in self.blocks
        )
        logger.info(
            "Fused q/k/v projection weights in %d/%d blocks", fused, len(self.blocks)
        )

    def _make_block(
        self,
        *,
        index: int,
        config: WanVideoConfig,
        quant_config: QuantizationConfig | None,
    ) -> nn.Module:
        """Block factory; subclasses override it to substitute their block class."""
        attn_backend = get_global_server_args().attention_backend
        transformer_block = (
            WanTransformerBlock_VSA
            if (attn_backend and attn_backend.lower() == "video_sparse_attn")
            else WanTransformerBlock
        )
        return transformer_block(
            config.num_attention_heads * config.attention_head_dim,
            config.ffn_dim,
            config.num_attention_heads,
            config.qk_norm,
            config.cross_attn_norm,
            config.eps,
            config.added_kv_proj_dim,
            self._supported_attention_backends
            | {AttentionBackendEnum.VIDEO_SPARSE_ATTN},
            prefix=f"blocks.{index}",
            attention_type=config.attention_type,
            sla_topk=config.sla_topk,
            quant_config=quant_config,
        )

    @lru_cache(maxsize=1)
    def _compute_rope_for_sequence_shard(
        self,
        local_len: int,
        rank: int,
        frame_stride_local: int,
        width_local: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb.forward_3d_sequence_shard(
            local_len, rank, frame_stride_local, width_local, device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        text_source = encoder_hidden_states
        forward_batch = get_forward_context().forward_batch
        if forward_batch is not None:
            sequence_shard_enabled = (
                forward_batch.enable_sequence_shard and self.sp_size > 1
            )
        else:
            sequence_shard_enabled = False
        self.enable_teacache = (
            forward_batch is not None and forward_batch.enable_teacache
        )
        enable_spectrum = forward_batch is not None and forward_batch.enable_spectrum

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if not sequence_shard_enabled:
            # The rotary embedding layer correctly handles SP offsets internally.
            freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
                (
                    post_patch_num_frames * self.sp_size,
                    post_patch_height,
                    post_patch_width,
                ),
                shard_dim=0,
                start_frame=0,
                device=hidden_states.device,
            )
            assert freqs_cos.dtype == torch.float32
            assert freqs_cos.device == hidden_states.device
            freqs_cis = (
                (freqs_cos.float(), freqs_sin.float())
                if freqs_cos is not None
                else None
            )

        # token-major straight from the patchify GEMM, no [B, C, T, H, W] round trip
        hidden_states = self.patch_embedding(hidden_states, tokens=True).contiguous()

        # shape is [B, T' * H' * W', C]
        seq_len_orig = hidden_states.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (batch_size, seq_shard_pad, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                hidden_states = torch.cat([hidden_states, pad], dim=1)
            sp_rank = get_sp_group().rank_in_group
            local_seq_len = hidden_states.shape[1] // self.sp_size
            hidden_states = hidden_states.view(
                batch_size, self.sp_size, local_seq_len, hidden_states.shape[2]
            )
            hidden_states = hidden_states[:, sp_rank, :, :]

            frame_stride = post_patch_height * post_patch_width
            freqs_cos, freqs_sin = self._compute_rope_for_sequence_shard(
                local_seq_len,
                sp_rank,
                frame_stride,
                post_patch_width,
                hidden_states.device,
            )
            freqs_cis = (freqs_cos.float(), freqs_sin.float())

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.dim() == 2:
            # ti2v
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if sequence_shard_enabled and ts_seq_len is not None:
            if seq_shard_pad > 0:
                pad = torch.zeros(
                    (
                        batch_size,
                        seq_shard_pad,
                        timestep_proj.shape[2],
                        timestep_proj.shape[3],
                    ),
                    dtype=timestep_proj.dtype,
                    device=timestep_proj.device,
                )
                timestep_proj = torch.cat([timestep_proj, pad], dim=1)
            timestep_proj = timestep_proj.view(
                batch_size,
                self.sp_size,
                local_seq_len,
                timestep_proj.shape[2],
                timestep_proj.shape[3],
            )
            timestep_proj = timestep_proj[:, sp_rank, :, :, :]

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if not current_platform.is_amp_supported()
            else encoder_hidden_states
        )  # cast to orig_dtype if amp is not supported

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        run_transformer_blocks = self.begin_spectrum_step()
        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb
        )

        if enable_spectrum and not run_transformer_blocks:
            hidden_states = self.spectrum_predict_features(hidden_states)
        elif should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            # if teacache is enabled, we need to cache the original hidden states
            if self.enable_teacache:
                original_hidden_states = hidden_states.clone()

            # cast once for every block's fp32 add (the TI2V table stays bf16)
            if timestep_proj.dim() == 3:
                timestep_proj = timestep_proj.float()
            rope_cos_sin_cache = None
            if _is_cuda and freqs_cis is not None:
                cos, sin = freqs_cis
                rope_cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
            text_kvs = self._text_kvs(text_source, encoder_hidden_states)
            text_kv_kwargs = [
                {} if text_kvs is None else {"text_kv": kv}
                for kv in (text_kvs or [None] * len(self.blocks))
            ]
            if self._chains_ffn_residuals():
                # each block's FFN residual is added by the next block's norm1
                ffn_residual = None
                for block, kv_kwargs in zip(self.blocks[:-1], text_kv_kwargs):
                    hidden_states, ffn_residual = block(
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        freqs_cis,
                        rope_cos_sin_cache=rope_cos_sin_cache,
                        ffn_residual=ffn_residual,
                        defer_ffn_residual=True,
                        **kv_kwargs,
                    )
                hidden_states = self.blocks[-1](
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    rope_cos_sin_cache=rope_cos_sin_cache,
                    ffn_residual=ffn_residual,
                    **text_kv_kwargs[-1],
                )
            else:
                for block, kv_kwargs in zip(self.blocks, text_kv_kwargs):
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        freqs_cis,
                        rope_cos_sin_cache=rope_cos_sin_cache,
                        **kv_kwargs,
                    )
            # if teacache is enabled, we need to cache the original hidden states
            if self.enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)
            if enable_spectrum:
                self.spectrum_record_features(hidden_states)
        self.cnt += 1

        if sequence_shard_enabled:
            hidden_states = hidden_states.contiguous()
            hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)
            if seq_shard_pad > 0:
                hidden_states = hidden_states[:, :seq_len_orig, :]

        return self._unpatchify(
            hidden_states,
            temb,
            (post_patch_num_frames, post_patch_height, post_patch_width),
        )

    def _unpatchify(
        self,
        hidden_states: torch.Tensor,
        timestep_embeddings: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """Output norm, projection and token-grid reconstruction for Wan variants."""
        if timestep_embeddings.dim() == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + timestep_embeddings.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (
                self.scale_shift_table + timestep_embeddings.unsqueeze(1)
            ).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states, _ = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], *grid_size, *self.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache residual with CFG positive/negative separation."""
        residual = hidden_states.squeeze(0) - original_hidden_states
        if not self.is_cfg_negative:
            self.previous_residual = residual
        else:
            self.previous_residual_negative = residual

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        if not self.enable_teacache:
            return False
        ctx = self._get_teacache_context()
        if ctx is None:
            return False

        # Initialize Wan-specific parameters
        teacache_params = ctx.teacache_params
        use_ret_steps = teacache_params.use_ret_steps
        start_skipping, end_skipping = teacache_params.get_skip_step_range(
            ctx.num_inference_steps
        )

        # Count the window in denoising steps: the number of local forwards per
        # step varies with CFG parallel and CFG gating.
        is_boundary_step = not start_skipping <= ctx.current_timestep < end_skipping

        timestep_proj = kwargs["timestep_proj"]
        temb = kwargs["temb"]
        modulated_inp = timestep_proj if use_ret_steps else temb

        self.is_cfg_negative = ctx.is_cfg_negative

        # Use shared helper to compute cache decision
        should_calc = self._compute_teacache_decision(
            modulated_inp=modulated_inp,
            is_boundary_step=is_boundary_step,
            coefficients=ctx.coefficients,
            teacache_thresh=ctx.teacache_thresh,
        )

        return not should_calc

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retrieve cached residual with CFG positive/negative separation."""
        if not self.is_cfg_negative:
            return hidden_states + self.previous_residual
        else:
            return hidden_states + self.previous_residual_negative


EntryClass = WanTransformer3DModel
