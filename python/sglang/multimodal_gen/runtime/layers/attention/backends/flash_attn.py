# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0
import inspect
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.utils import register_custom_op
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
)


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _is_fp4_dtype(dtype: torch.dtype) -> bool:
    return hasattr(torch, "float4_e2m1fn_x2") and dtype == torch.float4_e2m1fn_x2


def _is_nvfp4_fa4_enabled() -> bool:
    return (
        bool(envs.SGLANG_DIFFUSION_NVFP4_FA4)
        or os.environ.get("FASTVIDEO_NVFP4_FA4", "0") == "1"
    )


def _has_fa4_fp4_support() -> bool:
    try:
        from flash_attn.cute import flash_attn_func

        return "mSFQ" in inspect.signature(flash_attn_func).parameters
    except Exception:
        return False


def _nvfp4_quantize_for_fa4(
    tensor_4d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [batch, seqlen, heads, head_dim] Q/K into FA4's FP4/SF layout."""
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    batch, seqlen, nheads, headdim = tensor_4d.shape
    if headdim % 64 != 0:
        raise ValueError(f"NVFP4 FA4 requires head_dim divisible by 64, got {headdim}.")

    sf_vec_size = 16
    tile_m = 128
    seqlen_padded = (seqlen + tile_m - 1) // tile_m * tile_m
    if seqlen_padded != seqlen:
        tensor_4d = F.pad(tensor_4d, (0, 0, 0, 0, 0, seqlen_padded - seqlen))

    t2d = tensor_4d.reshape(batch * seqlen_padded, nheads * headdim)
    global_sf = torch.ones(1, device=t2d.device, dtype=torch.float32)
    fp4_data, sf_data = nvfp4_quantize(
        t2d, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )

    fp4_tensor = (
        fp4_data.reshape(batch, seqlen_padded, nheads, headdim // 2)
        .view(torch.int8)
        .view(torch.float4_e2m1fn_x2)
    )

    atom_m0, atom_m1, atom_k = 32, 4, 4
    rest_m = seqlen_padded // tile_m
    sf_k_per_head = headdim // sf_vec_size
    rest_k = sf_k_per_head // atom_k

    total_m_tiles = batch * rest_m
    total_k_tiles = (nheads * sf_k_per_head) // atom_k
    sf_swizzled = sf_data.reshape(
        total_m_tiles, total_k_tiles, atom_m0, atom_m1, atom_k
    )
    sf_decomposed = sf_swizzled.reshape(
        batch, rest_m, nheads, rest_k, atom_m0, atom_m1, atom_k
    )
    sf_canonical = sf_decomposed.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    sf_mma = sf_canonical.permute(4, 5, 2, 6, 3, 1, 0)

    return fp4_tensor, sf_mma


def _validate_fake_input_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mSFQ: Optional[torch.Tensor],
    mSFK: Optional[torch.Tensor],
) -> None:
    if mSFQ is not None or mSFK is not None or _is_fp4_dtype(q.dtype):
        assert _is_fp4_dtype(q.dtype), "FP4 path expects FP4-packed Q"
        assert q.dtype == k.dtype, "Q and K must have the same dtype"
        assert v.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "V must be float16 or bfloat16 for FP4 Q/K attention"
    else:
        assert q.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "inputs must be float16 or bfloat16"
        assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"


# -----------------------------
# Fake implementations for schema / tracing
# custom op schema requires FIXED return structure.
# We provide TWO ops:
# 1) out-only op: always returns Tensor
# 2) out+lse op: always returns Tuple[Tensor, Tensor]
# -----------------------------
def flash_attn_varlen_func_fake_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
    mSFV: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> torch.Tensor:
    assert ver == 4, "only support flash attention v4"
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
    head_dim_v = v.shape[-1]

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
        assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
        assert cu_seqlens_q.stride(0) == 1, "cu_seqlens_q must be contiguous"

    _validate_fake_input_dtypes(q, k, v, mSFQ, mSFK)
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // v.element_size()
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"

    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (q.shape[0],)
    )
    out = v.new_empty(*q_batch_seqlen_shape, num_head, head_dim_v)
    return out


def flash_attn_varlen_func_fake_out_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = True,
    sinks: Optional[torch.Tensor] = None,
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
    mSFV: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert ver == 4, "only support flash attention v4"
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    head_dim_v = v.shape[-1]

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
        assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
        assert cu_seqlens_q.stride(0) == 1, "cu_seqlens_q must be contiguous"

    _validate_fake_input_dtypes(q, k, v, mSFQ, mSFK)
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // v.element_size()
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"

    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )

    out = v.new_empty(*q_batch_seqlen_shape, num_head, head_dim_v)
    lse = v.new_empty(lse_shape, dtype=torch.float32)
    return out, lse


# -----------------------------
# Registered custom ops
# NOTE: fixed return schemas to avoid:
# "Object of type 'Tensor' is not an instance of 'sequence'"
# -----------------------------
@register_custom_op(fake_impl=flash_attn_varlen_func_fake_out)
def flash_attn_varlen_func_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
    mSFV: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> torch.Tensor:
    if window_size is None:
        window_size = [-1, -1]
    if return_softmax_lse:
        raise ValueError(
            "flash_attn_varlen_func_op is out-only op; return_softmax_lse must be False. "
            "Use flash_attn_varlen_func_op_lse for (out, lse)."
        )
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=tuple(window_size),
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=False,
        sinks=sinks,
        mSFQ=mSFQ,
        mSFK=mSFK,
        mSFV=mSFV,
        ver=ver,
    )


@register_custom_op(fake_impl=flash_attn_varlen_func_fake_out_lse)
def flash_attn_varlen_func_op_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Optional[List[int]] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = True,
    sinks: Optional[torch.Tensor] = None,
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
    mSFV: Optional[torch.Tensor] = None,
    ver: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if window_size is None:
        window_size = [-1, -1]
    if not return_softmax_lse:
        raise ValueError(
            "flash_attn_varlen_func_op_lse is out+lse op; return_softmax_lse must be True. "
            "Use flash_attn_varlen_func_op for out-only."
        )
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=tuple(window_size),
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=True,
        sinks=sinks,
        mSFQ=mSFQ,
        mSFK=mSFK,
        mSFV=mSFV,
        ver=ver,
    )


fa_ver = 3


def set_fa_ver(ver: int) -> None:
    global fa_ver
    fa_ver = ver


@dataclass
class FlashAttentionMetadata:
    # Sequence lengths for the forward batch
    # Maximum sequence length for query
    max_seqlen_q: int = 1
    # Maximum sequence length for key
    max_seqlen_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore
        self,
        raw_latent_shape=list,
        **kwargs: dict[str, Any],
    ) -> FlashAttentionMetadata:
        # TODO: put empty values here to be set at first-run, since the q_len calculation can be complicated
        return FlashAttentionMetadata(max_seqlen_q=None, max_seqlen_k=None)


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder


class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.attention_metadata = FlashAttentionMetadata()
        self.nvfp4_fa4 = bool(extra_impl_args.get("nvfp4_fa4", False)) or (
            _is_nvfp4_fa4_enabled()
        )
        if self.nvfp4_fa4:
            cap = torch.cuda.get_device_capability()
            if cap not in [(10, 0), (10, 3)]:
                raise RuntimeError(
                    "NVFP4 FA4 attention requires Blackwell "
                    f"(sm100a/sm103a), got sm{cap[0]}{cap[1]}."
                )
            if fa_ver != 4:
                raise RuntimeError("NVFP4 FA4 attention requires FA4 backend.")
            if head_size < 128:
                raise RuntimeError(
                    f"NVFP4 FA4 attention requires head_size >= 128, got {head_size}."
                )
            if not _has_fa4_fp4_support():
                raise ImportError(
                    "NVFP4 FA4 attention requires flash_attn.cute with "
                    "mSFQ/mSFK support. Install hao-ai-lab/flash-attention-fp4@fp4."
                )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        return_softmax_lse: bool = False,
    ):
        if attn_metadata is not None:
            if attn_metadata.max_seqlen_q is None:
                attn_metadata.max_seqlen_q = query.shape[1]
            if attn_metadata.max_seqlen_k is None:
                attn_metadata.max_seqlen_k = key.shape[1]
            max_seqlen_q = attn_metadata.max_seqlen_q
            max_seqlen_k = attn_metadata.max_seqlen_k
        else:
            max_seqlen_q = query.shape[1]
            max_seqlen_k = key.shape[1]

        if self.nvfp4_fa4:
            if return_softmax_lse:
                raise NotImplementedError(
                    "NVFP4 FA4 attention does not support return_softmax_lse yet."
                )
            return self._forward_nvfp4(
                query,
                key,
                value,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )

        # FA version selection:
        # - fa_ver == 3: call python function (can return Tensor or (Tensor, Tensor) depending on flag)
        # - fa_ver == 4: call custom ops with FIXED return schema
        if fa_ver == 3:
            flash_attn_op = flash_attn_varlen_func
            output = flash_attn_op(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_softmax_lse=return_softmax_lse,
                ver=fa_ver,
            )
            return output

        if fa_ver == 4:
            if return_softmax_lse:
                out_tensor, softmax_lse = flash_attn_varlen_func_op_lse(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                    return_softmax_lse=True,
                    ver=fa_ver,
                )
                return out_tensor, softmax_lse
            out_tensor = flash_attn_varlen_func_op(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_softmax_lse=False,
                ver=fa_ver,
            )
            return out_tensor

        raise ValueError(f"flash attention version {fa_ver} is not supported.")

    def _forward_nvfp4(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        orig_seqlen_q = query.shape[1]
        orig_seqlen_k = key.shape[1]

        q_fp4, q_sf = _nvfp4_quantize_for_fa4(query)
        k_fp4, k_sf = _nvfp4_quantize_for_fa4(key)

        q_fp4 = q_fp4[:, :orig_seqlen_q]
        k_fp4 = k_fp4[:, :orig_seqlen_k]

        output = flash_attn_varlen_func_op(
            q=q_fp4,
            k=k_fp4,
            v=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            return_softmax_lse=False,
            mSFQ=q_sf,
            mSFK=k_sf,
            ver=fa_ver,
        )
        if isinstance(output, tuple):
            return output[0]
        return output
