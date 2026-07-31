# SPDX-License-Identifier: Apache-2.0
"""DreamZero causal Wan DiT pieces.

Adapted from the official DreamZero causal Wan/action implementation:
https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.dreamzero_causal import (
    DreamZeroCausalWanConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ulysses_parallel_rank,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_varlen,
    _usp_output_all_to_all_varlen,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.utils import (
    sp_gather_tensor,
    sp_shard_sequence,
    sp_shard_tensor,
)

_DREAMZERO_SUPPORTED_ATTENTION_BACKENDS = {
    AttentionBackendEnum.FA,
    AttentionBackendEnum.TORCH_SDPA,
}


def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    return rope_params_polar(max_seq_len, dim, theta)


def rope_params_polar(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0
        / torch.pow(
            theta,
            torch.arange(0, dim, 2).to(torch.float64).div(dim),
        ),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = layer(x)
    if not isinstance(out, tuple):
        return out
    output, output_bias = out
    if output_bias is not None:
        output = output + output_bias
    return output


def _maybe_qk_norm(
    x: torch.Tensor, norm: nn.Module, *, tensor_parallel: bool
) -> torch.Tensor:
    if isinstance(norm, nn.Identity):
        return x
    if tensor_parallel:
        return tensor_parallel_rms_norm(x, norm)
    return norm(x)


def align_modulation(
    parts: tuple[torch.Tensor, ...], target_len: int
) -> tuple[torch.Tensor, ...]:
    aligned = []
    for part in parts:
        part_len = part.shape[1]
        if part_len == target_len:
            aligned.append(part)
        elif part_len >= target_len:
            aligned.append(part[:, :target_len])
        else:
            repeat = (target_len + part_len - 1) // part_len
            aligned.append(part.repeat_interleave(repeat, dim=1)[:, :target_len])
    return tuple(aligned)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=timesteps.device
        ) * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        action_emb = self.W1(actions, cat_ids)
        timestep_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        x = torch.cat([action_emb, timestep_emb], dim=-1)
        x = self.W2(x, cat_ids)
        x = x * torch.sigmoid(x)
        return self.W3(x, cat_ids)


def rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    batch, seq_len, num_heads, _ = x.shape
    out_dtype = x.dtype
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(batch, seq_len, num_heads, -1, 2)
    )

    if action_register_length is not None:
        assert num_action_per_block is not None
        assert num_state_per_block is not None
        chunk_size = action_register_length // (
            num_action_per_block + num_state_per_block
        )
        freqs_action = freqs_action[: chunk_size * num_action_per_block].view(
            chunk_size * num_action_per_block, 1, -1
        )
        freqs_state = freqs_state[: chunk_size * num_state_per_block].view(
            chunk_size * num_state_per_block, 1, -1
        )
        freqs = torch.cat([freqs, freqs_action, freqs_state], dim=0)

    freqs = freqs.unsqueeze(0)
    return torch.view_as_real(x * freqs).flatten(3).to(out_dtype)


class DreamZeroT2VCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm=True,
        eps: float = 1e-6,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.local_num_heads = (
            divide(num_heads, get_tp_world_size()) if use_tensor_parallel else num_heads
        )
        self.head_dim = dim // num_heads
        self.use_tensor_parallel = use_tensor_parallel
        def linear_out():
            return (
                RowParallelLinear(
                    dim,
                    dim,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    reduce_results=True,
                )
                if use_tensor_parallel
                else nn.Linear(dim, dim)
            )

        def linear_in():
            return (
                ColumnParallelLinear(dim, dim, gather_output=False)
                if use_tensor_parallel
                else nn.Linear(dim, dim)
            )

        self.q = linear_in()
        self.k = linear_in()
        self.v = linear_in()
        self.o = linear_out()
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=_DREAMZERO_SUPPORTED_ATTENTION_BACKENDS,
            skip_sequence_parallel=True,
        )

    def _project_query(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        return _maybe_qk_norm(
            _linear(self.q, x), self.norm_q, tensor_parallel=self.use_tensor_parallel
        ).view(batch, -1, self.local_num_heads, self.head_dim)

    def _project_text_kv(
        self,
        context: torch.Tensor,
        batch: int,
        crossattn_cache: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if crossattn_cache is not None and crossattn_cache["is_init"]:
            return crossattn_cache["k"], crossattn_cache["v"]

        k = _maybe_qk_norm(
            _linear(self.k, context),
            self.norm_k,
            tensor_parallel=self.use_tensor_parallel,
        ).view(batch, -1, self.local_num_heads, self.head_dim)
        v = _linear(self.v, context).view(
            batch, -1, self.local_num_heads, self.head_dim
        )
        if crossattn_cache is not None:
            crossattn_cache["is_init"] = True
            crossattn_cache["k"] = k
            crossattn_cache["v"] = v
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        q = self._project_query(x, batch)
        k, v = self._project_text_kv(context, batch, crossattn_cache)
        return _linear(self.o, self.attn(q, k, v).flatten(2))


class DreamZeroI2VCrossAttention(DreamZeroT2VCrossAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm=True,
        eps: float = 1e-6,
        use_tensor_parallel: bool = False,
    ):
        super().__init__(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            use_tensor_parallel=use_tensor_parallel,
        )
        self.k_img = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.v_img = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.norm_k_img = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        context_img = context[:, :257]
        context = context[:, 257:]
        batch = x.shape[0]
        q = self._project_query(x, batch)
        k, v = self._project_text_kv(context, batch, crossattn_cache)
        text_x = self.attn(q, k, v).flatten(2)
        k_img = _maybe_qk_norm(
            _linear(self.k_img, context_img),
            self.norm_k_img,
            tensor_parallel=self.use_tensor_parallel,
        ).view(batch, -1, self.local_num_heads, self.head_dim)
        v_img = _linear(self.v_img, context_img).view(
            batch, -1, self.local_num_heads, self.head_dim
        )
        img_x = self.attn(q, k_img, v_img).flatten(2)
        merged_x = text_x + img_x
        out = _linear(self.o, merged_x)
        return out


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": DreamZeroT2VCrossAttention,
    "i2v_cross_attn": DreamZeroI2VCrossAttention,
}


class DreamZeroCausalWanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        num_frame_per_block: int = 1,
        qk_norm=True,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.local_num_heads = (
            divide(num_heads, get_tp_world_size()) if use_tensor_parallel else num_heads
        )
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.num_frame_per_block = num_frame_per_block
        self.use_tensor_parallel = use_tensor_parallel
        self.max_attention_size = (
            21 * frame_seqlen
            if local_attn_size == -1
            else local_attn_size * frame_seqlen
        )
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.q = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.k = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.v = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.o = (
            RowParallelLinear(
                dim,
                dim,
                input_is_parallel=True,
                skip_bias_add=False,
                reduce_results=True,
            )
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=_DREAMZERO_SUPPORTED_ATTENTION_BACKENDS,
            skip_sequence_parallel=True,
        )
        self.sequence_parallel_attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=_DREAMZERO_SUPPORTED_ATTENTION_BACKENDS,
        )

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(q, k, v).contiguous()

    def _sequence_parallel_attention_with_replicated_prefix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        seq_lens: list[int] | None,
    ) -> torch.Tensor:
        prefix_k = kv_cache[0]
        prefix_v = kv_cache[1]
        if seq_lens is None:
            return self.sequence_parallel_attn.forward_with_replicated_kv_prefix(
                q.contiguous(),
                prefix_k.contiguous(),
                prefix_v.contiguous(),
                k.contiguous(),
                v.contiguous(),
            ).contiguous()

        q = _usp_input_all_to_all_varlen(q.contiguous(), seq_lens, head_dim=2)
        k = _usp_input_all_to_all_varlen(k.contiguous(), seq_lens, head_dim=2)
        v = _usp_input_all_to_all_varlen(v.contiguous(), seq_lens, head_dim=2)

        h_local = k.shape[2]
        h_start = get_ulysses_parallel_rank() * h_local
        prefix_k = prefix_k[:, :, h_start : h_start + h_local].contiguous()
        prefix_v = prefix_v[:, :, h_start : h_start + h_local].contiguous()
        out = self.sequence_parallel_attn.attn_impl.forward(
            q,
            torch.cat([prefix_k, k], dim=1),
            torch.cat([prefix_v, v], dim=1),
            get_forward_context().attn_metadata,
        )
        return _usp_output_all_to_all_varlen(out, seq_lens, head_dim=2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        seq_lens: list[int] | None = None,
        video_sequence_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kv_cache is None:
            raise RuntimeError("DreamZero SGLang inference requires a KV cache")
        if video_sequence_length is None:
            video_sequence_length = x.shape[1] - (action_register_length or 0)

        batch, seq_len = x.shape[:2]
        q = _maybe_qk_norm(
            _linear(self.q, x), self.norm_q, tensor_parallel=self.use_tensor_parallel
        ).view(batch, seq_len, self.local_num_heads, self.head_dim)
        k = _maybe_qk_norm(
            _linear(self.k, x), self.norm_k, tensor_parallel=self.use_tensor_parallel
        ).view(batch, seq_len, self.local_num_heads, self.head_dim)
        v = _linear(self.v, x).view(batch, seq_len, self.local_num_heads, self.head_dim)

        roped_query = rope_action_apply_polar(
            q,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length=None,
        ).type_as(v)
        roped_key = rope_action_apply_polar(
            k,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length=None,
        ).type_as(v)

        if seq_lens is None:
            current_video_k = roped_key[:, :video_sequence_length]
            current_video_v = v[:, :video_sequence_length]
            out = self._attention(
                roped_query,
                torch.cat([kv_cache[0], roped_key], dim=1),
                torch.cat([kv_cache[1], v], dim=1),
            )
        else:
            current_video_k = sp_gather_tensor(roped_key, seq_lens)[
                :, :video_sequence_length
            ]
            current_video_v = sp_gather_tensor(v, seq_lens)[
                :, :video_sequence_length
            ]
            out = self._sequence_parallel_attention_with_replicated_prefix(
                roped_query,
                roped_key,
                v,
                kv_cache,
                seq_lens,
            )

        updated_k = torch.cat([kv_cache[0], current_video_k], dim=1)
        updated_v = torch.cat([kv_cache[1], current_video_v], dim=1)
        updated_k = updated_k[:, -self.max_attention_size :]
        updated_v = updated_v[:, -self.max_attention_size :]
        updated_kv_cache = torch.stack([updated_k, updated_v], dim=0)

        out = _linear(self.o, out.flatten(2))
        return out, updated_kv_cache


class DreamZeroCausalWanTransformerBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        num_frame_per_block: int = 1,
        qk_norm=True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_tensor_parallel = use_tensor_parallel
        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.self_attn = DreamZeroCausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            use_tensor_parallel=use_tensor_parallel,
        )
        self.norm3 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, qk_norm, eps, use_tensor_parallel=use_tensor_parallel
        )
        self.norm2 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        if use_tensor_parallel:
            self.ffn = nn.ModuleList(
                [
                    ColumnParallelLinear(dim, ffn_dim, gather_output=False),
                    nn.GELU(approximate="tanh"),
                    RowParallelLinear(
                        ffn_dim,
                        dim,
                        input_is_parallel=True,
                        skip_bias_add=False,
                        reduce_results=True,
                    ),
                ]
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(ffn_dim, dim),
            )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _run_ffn(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_tensor_parallel:
            return self.ffn(x)
        col = _linear(self.ffn[0], x)
        act = self.ffn[1](col)
        return _linear(self.ffn[2], act)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
        seq_lens: list[int] | None = None,
        video_sequence_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        e_parts = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        e_parts = align_modulation(e_parts, x.shape[1])

        self_attn_input = self.norm1(
            x,
            e_parts[0].squeeze(2),
            e_parts[1].squeeze(2),
        )
        y, updated_kv_cache = self.self_attn(
            x=self_attn_input,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            seq_lens=seq_lens,
            video_sequence_length=video_sequence_length,
        )
        x = x + y * e_parts[2].squeeze(2)
        cross_attn_input = self.norm3(x)
        cross = self.cross_attn(
            cross_attn_input,
            context,
            crossattn_cache=crossattn_cache,
        )
        x = x + cross
        norm2_input = self.norm2(
            x,
            e_parts[3].squeeze(2),
            e_parts[4].squeeze(2),
        )
        y = self._run_ffn(norm2_input)
        x = x + y * e_parts[5].squeeze(2)
        return x, updated_kv_cache


class DreamZeroCausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, ...], eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.norm = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        e_parts = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        shift, scale = align_modulation(e_parts, x.shape[1])
        return self.head(self.norm(x, shift.squeeze(2), scale.squeeze(2)))


class MLPProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(image_embeds)


class DreamZeroCausalWanModel(CachableDiT):
    _fsdp_shard_conditions = (
        DreamZeroCausalWanConfig().arch_config._fsdp_shard_conditions
    )
    _compile_conditions = DreamZeroCausalWanConfig().arch_config._compile_conditions
    _supported_attention_backends = (
        DreamZeroCausalWanConfig().arch_config._supported_attention_backends
    )
    param_names_mapping = DreamZeroCausalWanConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        DreamZeroCausalWanConfig().arch_config.reverse_param_names_mapping
    )
    lora_param_names_mapping = (
        DreamZeroCausalWanConfig().arch_config.lora_param_names_mapping
    )

    def __init__(
        self,
        config: DreamZeroCausalWanConfig | None = None,
        hf_config: dict | None = None,
        quant_config=None,
    ):
        config = config or DreamZeroCausalWanConfig()
        arch = config.arch_config
        super().__init__(config=config, hf_config=hf_config or {})

        model_type = arch.model_type
        patch_size = arch.patch_size
        frame_seqlen = arch.frame_seqlen
        text_len = arch.text_len
        in_dim = arch.in_dim
        dim = arch.dim
        ffn_dim = arch.ffn_dim
        freq_dim = arch.freq_dim
        text_dim = arch.text_dim
        out_dim = arch.out_dim
        num_heads = arch.num_heads
        num_layers = arch.num_layers
        max_chunk_size = arch.max_chunk_size
        qk_norm = arch.qk_norm
        cross_attn_norm = arch.cross_attn_norm
        eps = arch.eps
        num_frame_per_block = arch.num_frame_per_block
        action_dim = arch.action_dim
        max_state_dim = arch.max_state_dim
        hidden_size = arch.hidden_size
        num_action_per_block = arch.num_action_per_block
        num_state_per_block = arch.num_state_per_block
        concat_first_frame_latent = arch.concat_first_frame_latent
        rope_video_max_positions = arch.rope_video_max_positions
        rope_action_max_positions = arch.rope_action_max_positions
        rope_state_max_positions = arch.rope_state_max_positions
        use_tensor_parallel = arch.use_tensor_parallel

        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.local_num_heads = (
            divide(num_heads, get_tp_world_size()) if use_tensor_parallel else num_heads
        )
        self.num_layers = num_layers
        self.local_attn_size = (
            max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        )
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.action_dim = action_dim
        self.max_state_dim = max_state_dim
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.concat_first_frame_latent = concat_first_frame_latent
        self.rope_video_max_positions = rope_video_max_positions
        self.rope_action_max_positions = rope_action_max_positions
        self.rope_state_max_positions = rope_state_max_positions
        self.use_tensor_parallel = use_tensor_parallel

        self.state_encoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=1,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                DreamZeroCausalWanTransformerBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    frame_seqlen,
                    self.local_attn_size,
                    num_frame_per_block,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    num_action_per_block,
                    num_state_per_block,
                    use_tensor_parallel,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = DreamZeroCausalHead(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self._init_rope_freqs(torch.device("cpu"))
        if model_type in ("i2v", "ti2v"):
            self.img_emb = MLPProj(1280, dim)

    def _init_rope_freqs(self, device: torch.device) -> None:
        head_dim = self.dim // self.num_heads
        max_frames, max_height, max_width = self.rope_video_max_positions
        with torch.device(device):
            self.freqs_action = rope_params(self.rope_action_max_positions, head_dim)
            self.freqs_state = rope_params(self.rope_state_max_positions, head_dim)
            self.freqs = [
                rope_params(max_frames, head_dim - 4 * (head_dim // 6)),
                rope_params(max_height, 2 * (head_dim // 6)),
                rope_params(max_width, 2 * (head_dim // 6)),
            ]

    def post_load_weights(self) -> None:
        self._init_rope_freqs(self.patch_embedding.weight.device)

    def _create_freqs(self, grid_size: torch.Tensor, start_frame: int) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        frames, height, width = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame : start_frame + frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                self.freqs[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                self.freqs[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1)
        return freqs

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[dict] | None,
        current_start_frame: int,
        enable_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        x = x.flatten(start_dim=2).transpose(1, 2)
        batch = x.shape[0]
        num_timestep_frames = timestep.shape[1]

        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(
                action, timestep_action, embodiment_id
            )
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_length = 0
            action_register_length = None
            state_features = None

        if action_register_length is not None:
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block
            action_freqs = self.freqs_action[
                action_state_index
                * self.num_action_per_block : (action_state_index + 1)
                * self.num_action_per_block
            ].view(self.num_action_per_block, 1, -1)
            state_freqs = self.freqs_state[
                action_state_index
                * self.num_state_per_block : (action_state_index + 1)
                * self.num_state_per_block
            ].view(self.num_state_per_block, 1, -1)
            freqs = torch.cat([freqs, action_freqs, state_freqs], dim=0)

        seq_lens = None
        if enable_sequence_parallel:
            x, freqs, seq_lens = sp_shard_sequence(x, freqs)

        if num_timestep_frames <= seq_len:
            repeat = (seq_len + num_timestep_frames - 1) // num_timestep_frames
            timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(
                0,
                num_timestep_frames - 1,
                seq_len,
                device=timestep.device,
                dtype=torch.long,
            )
            timestep = timestep[:, indices]

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        assert self.freq_dim % 2 == 0
        e = self.time_embedding(
            timestep_embedding(
                timestep.flatten(), self.freq_dim, dtype=torch.float64
            ).type_as(x)
        )
        e = e.unflatten(dim=0, sizes=(batch, -1))
        e0 = self.time_projection(e).unflatten(dim=2, sizes=(6, self.dim))
        if enable_sequence_parallel:
            assert seq_lens is not None
            e0 = sp_shard_tensor(e0, seq_lens)

        context = self.text_embedding(context)
        if clip_feature is not None:
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                crossattn_cache=(
                    crossattn_cache[block_index]
                    if crossattn_cache is not None
                    else None
                ),
                seq_lens=seq_lens,
                video_sequence_length=seq_len,
            )
            updated_kv_caches.append(updated_kv_cache)

        if enable_sequence_parallel:
            assert seq_lens is not None
            x = sp_gather_tensor(x, seq_lens)

        if action is not None:
            action_noise_pred = x[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))
        return x_video, action_noise_pred, updated_kv_caches

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[dict] | None,
        current_start_frame: int,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        enable_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if self.model_type == "i2v":
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size=grid_size, start_frame=current_start_frame)

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=current_start_frame,
            enable_sequence_parallel=enable_sequence_parallel,
        )
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        video_noise_pred = self.unpatchify(x_video, grid_size)
        return video_noise_pred, action_noise_pred, updated_kv_caches

    def forward(self, *args, **kwargs):
        return self._forward_inference(*args, **kwargs)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        channels = self.out_dim
        grid = grid_size.tolist()
        assert x.shape[1] == math.prod(grid)
        x = x.view(batch, *grid, *self.patch_size, channels)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(
            batch,
            channels,
            *[axis * patch for axis, patch in zip(grid, self.patch_size)],
        )


EntryClass = DreamZeroCausalWanModel
