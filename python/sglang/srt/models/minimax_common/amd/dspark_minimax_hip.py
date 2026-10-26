# SPDX-License-Identifier: Apache-2.0
"""ROCm glue of dspark_minimax: the MiniMax DSpark draft writes every layer's context K/V
through one block-fp8 GEMM and computes each markov step bias in one kernel on gfx950;
bound as _hip."""

from __future__ import annotations

import weakref
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_add_rmsnorm_fp8
from sglang.kernels.ops.speculative.dspark.dspark_gfx950 import (
    fused_kv_gemma_norm_rope_write,
)
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.quantization.fp8_utils import (
    aiter_w8a8_block_fp8_linear,
    fp8_input_quant_format,
)
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_hip import FusedStepBiasMarkov
from sglang.srt.utils import is_gfx95_supported


def markov_head(head):
    """The draft's markov head, a FusedStepBiasMarkov where gfx950 runs its kernel."""
    if (
        type(head) is not VanillaMarkov
        or not is_gfx95_supported()
        or head.markov_w2.weight.dtype != torch.bfloat16
        # the kernel's rank tiles are 64 wide
        or head.markov_rank % 64 != 0
    ):
        return head
    weight = head.markov_w2.weight
    return FusedStepBiasMarkov(
        vocab_size=head.vocab_size, markov_rank=head.markov_rank
    ).to(device=weight.device, dtype=weight.dtype)


def ctx_kv_writer(model) -> Optional[BlockFp8CtxKvWriter]:
    return BlockFp8CtxKvWriter(model) if is_gfx95_supported() else None


class _BlockFp8CtxKv(msgspec.Struct, frozen=True):
    """Every layer's K/V rows of qkv_proj as one block-fp8 linear (with the rowwise
    copy) and what the K-norm / rope / store kernel reads per layer."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    # [layers, 4] int64: k_buffer ptr, v_buffer ptr, k row stride, v row stride
    meta: torch.Tensor
    k_norm_weights: torch.Tensor
    # the bf16 copy sgl_kernel.rotary_embedding reads on HIP
    cos_sin_cache: torch.Tensor
    num_kv_heads: int
    head_dim: int
    eps: float


class BlockFp8CtxKvWriter:
    """Writes every draft layer's context K/V from the target hidden with one block-fp8
    GEMM over the K/V rows of their qkv_proj and one K-norm / rope / store kernel."""

    def __init__(self, model):
        self.model = model
        # (weakref to the pool, its _BlockFp8CtxKv or None); meta holds the pool's buffer
        # addresses, and the weakref keeps a new pool at a freed one's id from reusing them
        self._cache = None

    def reset(self) -> None:
        self._cache = None

    def write(
        self,
        target_hidden: torch.Tensor,
        pool,
        *,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
    ) -> bool:
        """False, having written nothing, where the input, the layers or the row count rule
        it out."""
        if target_hidden_is_projected:
            return False
        ctx_kv = self._ctx_kv(pool)
        if ctx_kv is None:
            return False
        quant_format = fp8_input_quant_format(ctx_kv.weight, target_hidden.shape[0])
        if not quant_format:
            return False
        model = self.model
        if target_hidden.shape[-1] != model.fc.input_size:
            raise ValueError(
                "DSpark target hidden features do not match checkpoint projection"
            )
        projected = model._fc_gatherer(model.fc(target_hidden)[0])
        # hidden_norm and the quant each layer's qkv_proj would apply, done once;
        # "fp8" is the bpreshuffle block GEMM, which reads its scale column-major
        (q, scale), _ = gemma_add_rmsnorm_fp8(
            projected,
            None,
            model.hidden_norm.weight.data,
            model.hidden_norm.variance_epsilon,
            group_size=128 if quant_format == "fp8" else None,
            transpose_scale=quant_format == "fp8",
        )
        kv = aiter_w8a8_block_fp8_linear(
            q, ctx_kv.weight, [128, 128], ctx_kv.weight_scale, input_scale=scale
        )
        if cache_loc_2d is not None and commit_lens is not None:
            locs, locs_row_width = cache_loc_2d.reshape(-1), cache_loc_2d.shape[1]
        else:
            locs, locs_row_width, commit_lens = cache_loc, None, None
        fused_kv_gemma_norm_rope_write(
            kv,
            ctx_kv.meta,
            ctx_kv.k_norm_weights,
            ctx_kv.cos_sin_cache,
            positions,
            locs,
            ctx_kv.num_kv_heads,
            ctx_kv.head_dim,
            ctx_kv.eps,
            commit_lens=commit_lens,
            locs_row_width=locs_row_width,
        )
        return True

    def _ctx_kv(self, pool) -> Optional[_BlockFp8CtxKv]:
        cached = self._cache
        if cached is None or cached[0]() is not pool:
            cached = (weakref.ref(pool), self._build(pool))
            self._cache = cached
        return cached[1]

    def _build(self, pool) -> Optional[_BlockFp8CtxKv]:
        """None unless every layer projects through an MXFP8-converted block-fp8
        qkv_proj (aiter_w8a8_block_fp8_linear) and keeps a bf16 K/V cache."""
        model = self.model
        attns = [layer.self_attn for layer in model.layers]
        attn0 = attns[0]
        rotary = attn0.rotary_emb
        if not (
            envs.SGLANG_DSPARK_STACKED_CTX_KV.get()
            and isinstance(model.hidden_norm, GemmaRMSNorm)
            and type(rotary) is RotaryEmbedding
            and rotary.is_neox_style
            and rotary.use_fallback_kernel
        ):
            return None
        q_size, kv_size, head_dim = attn0.q_size, attn0.kv_size, attn0.head_dim
        block_n = 128
        meta = []
        for attn in attns:
            proj = attn.qkv_proj
            k_buf = pool.get_key_buffer(attn.attn.layer_id)
            v_buf = pool.get_value_buffer(attn.attn.layer_id)
            if (
                getattr(proj.weight, "_ptpc_weight", None) is None
                or proj.bias is not None
                or proj.quant_method.weight_block_size != [block_n, block_n]
                or (attn.q_size, attn.kv_size, attn.head_dim)
                != (q_size, kv_size, head_dim)
                or q_size % block_n
                or (2 * kv_size) % block_n
                or attn.rotary_emb is not rotary
                or not isinstance(attn.k_norm, GemmaRMSNorm)
                or attn.k_norm.variance_epsilon != attn0.k_norm.variance_epsilon
                # a bf16 pool stores bf16 K/V as is, whatever the layer's kv scales
                or any(
                    buf.dtype != torch.bfloat16
                    or buf.shape[1:] != (attn.num_kv_heads, head_dim)
                    or buf.stride(1) != head_dim
                    or buf.stride(2) != 1
                    for buf in (k_buf, v_buf)
                )
            ):
                return None
            meta.append(
                [k_buf.data_ptr(), v_buf.data_ptr(), k_buf.stride(0), v_buf.stride(0)]
            )
        rows = slice(q_size, q_size + 2 * kv_size)
        blocks = slice(q_size // block_n, (q_size + 2 * kv_size) // block_n)
        weight = torch.cat([a.qkv_proj.weight[rows] for a in attns])
        # bpreshuffle tiles are 16 rows, so the K/V row slices keep their layout
        weight._ptpc_weight = torch.cat(
            [a.qkv_proj.weight._ptpc_weight[rows] for a in attns]
        )
        weight._ptpc_scale = torch.cat(
            [a.qkv_proj.weight._ptpc_scale[rows] for a in attns]
        )
        return _BlockFp8CtxKv(
            weight=weight,
            weight_scale=torch.cat(
                [a.qkv_proj.weight_scale_inv[blocks] for a in attns]
            ),
            meta=torch.tensor(meta, dtype=torch.int64, device=weight.device),
            k_norm_weights=torch.stack([a.k_norm.weight.data for a in attns]),
            cos_sin_cache=rotary.cos_sin_cache.to(torch.bfloat16),
            num_kv_heads=attn0.num_kv_heads,
            head_dim=head_dim,
            eps=attn0.k_norm.variance_epsilon,
        )
