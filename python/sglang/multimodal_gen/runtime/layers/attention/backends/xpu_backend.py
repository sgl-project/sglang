# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import math
import re
from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    flash_attn_func = flash_attn_varlen_func
except ImportError as e:
    raise e

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
)

try:
    from sgl_kernel import sageattn_v3, sageattn_v3_hybrid
except ImportError:
    sageattn_v3 = sageattn_v3_hybrid = None

logger = init_logger(__name__)

_SAGE_V3_KINDS = ("pure", "hybrid")
_DEFAULT_SAGE_V3_KIND = "hybrid"
# --attention-backend-config keys of --attention-backend sage_attn_3 on XPU.
SAGE_V3_KIND_KEY = "sage_v3_kind"
SAGE_V3_BF16_BLOCKS_KEY = "sage_v3_bf16_blocks"
_BLOCK_INDEX_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


@torch.library.custom_op("sglang::xpu_sage_v3_attn", mutates_args=())
def _xpu_sage_v3_attn(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kind: str
) -> torch.Tensor:
    attn = sageattn_v3 if kind == "pure" else sageattn_v3_hybrid
    return attn(query, key, value, tensor_layout="NHD")


@_xpu_sage_v3_attn.register_fake
def _(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kind: str
) -> torch.Tensor:
    # The kernels take and return BF16 only; XPUSageAttention3Impl calls this op
    # for BF16 queries and keeps flash attention for other dtypes.
    return torch.empty_like(query)


@lru_cache(maxsize=None)
def _check_sage_v3_runs(kind: str) -> None:
    """Run the kernel once so a missing kernel backend or a non-Xe3P device fails at
    layer construction, not at the first denoising step."""
    if sageattn_v3 is None:
        raise ImportError(
            "--attention-backend sage_attn_3 on XPU needs an sgl-kernel-xpu with "
            "sageattn_v3 / sageattn_v3_hybrid (Xe3P)"
        )
    x = torch.zeros((1, 64, 1, 128), device="xpu", dtype=torch.bfloat16)
    try:
        _xpu_sage_v3_attn(x, x, x, kind)
    except Exception as e:
        raise RuntimeError(
            f"--attention-backend sage_attn_3 ({SAGE_V3_KIND_KEY}={kind}) is set, "
            f"but SageAttention V3 does not run on this device: {e}"
        ) from e


def _parse_bf16_blocks(value) -> frozenset[int]:
    if value is None or value == "":
        return frozenset()
    if isinstance(value, int) and not isinstance(value, bool):
        parts = [value]
    elif isinstance(value, str):
        # JSON / YAML give a list; a string may use "," (JSON string), ";" or
        # spaces (the key=value form splits pairs on ",").
        parts = [b for b in re.split(r"[,;\s]+", value.strip()) if b]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = None
    try:
        if parts is None or any(
            isinstance(b, bool) or (isinstance(b, float) and not b.is_integer())
            for b in parts
        ):
            raise ValueError
        return frozenset(int(b) for b in parts)
    except (TypeError, ValueError):
        raise ValueError(
            f"--attention-backend-config {SAGE_V3_BF16_BLOCKS_KEY} must be a list "
            f"of block indices, e.g. [0, 33, 34, 38], got {value!r}"
        ) from None


def _sage_v3_options() -> tuple[str, frozenset[int]]:
    """(kind, BF16 block indices) from --attention-backend-config."""
    config = get_global_server_args().attention_backend_config or {}
    kind = config.get(SAGE_V3_KIND_KEY)
    kind = _DEFAULT_SAGE_V3_KIND if kind is None else str(kind).strip().lower()
    if kind not in _SAGE_V3_KINDS:
        raise ValueError(
            f"--attention-backend-config {SAGE_V3_KIND_KEY} must be one of "
            f"{_SAGE_V3_KINDS}, got {kind!r}"
        )
    return kind, _parse_bf16_blocks(config.get(SAGE_V3_BF16_BLOCKS_KEY))


def _resolve_sage_v3_kind(
    *,
    kind: str,
    bf16_blocks: frozenset[int],
    prefix: str,
    head_size: int,
    num_heads: int,
    num_kv_heads: int | None,
    softmax_scale: float,
    causal: bool,
    is_cross_attention: bool,
) -> str | None:
    """The Sage V3 kind for this layer, or None to keep BF16 flash attention."""
    _check_sage_v3_runs(kind)
    if (
        causal
        or is_cross_attention
        or head_size != 128
        or (num_kv_heads is not None and num_kv_heads != num_heads)
        or not math.isclose(softmax_scale, head_size**-0.5, rel_tol=1e-6)
    ):
        return None
    block = _BLOCK_INDEX_RE.search(prefix)
    if bf16_blocks and block is None:
        raise ValueError(
            f"--attention-backend-config {SAGE_V3_BF16_BLOCKS_KEY} is set, but "
            f"attention layer prefix {prefix!r} has no 'blocks.<N>.' index, so the "
            "list cannot apply to this model; remove it"
        )
    if block is not None and int(block.group(1)) in bf16_blocks:
        return None
    logger.info_once(
        f"Using SageAttention V3 ({kind}) for eligible XPU self-attention; other "
        f"layers, blocks {sorted(bf16_blocks)} and ring steps keep BF16 flash attention."
    )
    return kind


class XPUAttentionBackend(AttentionBackend):
    @classmethod
    def supports_ring_rotation(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

    @staticmethod
    def get_impl_cls() -> type["XPUAttentionImpl"]:
        return XPUAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        """XPU backend does not require special metadata."""
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder


@lru_cache(maxsize=128)
def _get_cu_seqlens(device_index: int, bsz: int, seqlen: int) -> torch.Tensor:
    return torch.arange(
        0,
        (bsz + 1) * seqlen,
        step=seqlen,
        device=torch.device("xpu", device_index),
        dtype=torch.int32,
    )


def _as_fa_input(t: torch.Tensor, rows: int, heads: int, head_dim: int) -> torch.Tensor:
    """``t`` as the ``[rows, heads, head_dim]`` flash-attention input, copying only if needed.

    The kernel reads rows through their stride, so a head-packed row-strided view
    (e.g. q/k/v slices of a fused projection output) is used in place; measured
    bit-identical to the contiguous input.
    """
    try:
        view = t.view(rows, heads, head_dim)
    except RuntimeError:
        return t.contiguous().view(rows, heads, head_dim)
    if (
        view.stride(-1) == 1
        and view.stride(-2) == head_dim
        and view.stride(0) % 8 == 0
        and view.data_ptr() % 16 == 0
    ):
        return view
    return view.contiguous()


class XPUAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        is_cross_attention: bool = False,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        return_softmax_lse: bool = False,
    ):
        bsz, seqlen_q, nheads_q, d = tuple(query.shape)
        _, seqlen_k, nheads_k, _ = tuple(key.shape)

        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k

        q_ = _as_fa_input(query, bsz * seqlen_q, nheads_q, d)
        k_ = _as_fa_input(key, bsz * seqlen_k, nheads_k, d)
        v_ = _as_fa_input(value, bsz * seqlen_k, nheads_k, d)
        cu_q = _get_cu_seqlens(q_.device.index, bsz, seqlen_q)
        cu_k = _get_cu_seqlens(q_.device.index, bsz, seqlen_k)

        out = flash_attn_func(
            q=q_,
            k=k_,
            v=v_,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            return_softmax_lse=return_softmax_lse,
        )

        if return_softmax_lse:
            out_tensor, softmax_lse = out[:2]
            result = out_tensor.reshape(bsz, seqlen_q, nheads_q, d)
            # The kernel returns lse as (H, B * S); ring attention needs [B, H, S].
            softmax_lse = softmax_lse.view(nheads_q, bsz, seqlen_q).permute(1, 0, 2)
            return result, softmax_lse

        result = out.reshape(bsz, seqlen_q, nheads_q, d)
        return result

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        del cu_seqlens_host
        q_ = query.contiguous()
        k_ = key.contiguous()
        v_ = value.contiguous()
        output = flash_attn_func(
            q=q_,
            k=k_,
            v=v_,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output[0] if isinstance(output, tuple) else output


class XPUSageAttention3Backend(XPUAttentionBackend):
    """``--attention-backend sage_attn_3`` on Intel XPU (Xe3P): SageAttention V3
    from sgl-kernel-xpu for eligible self-attention, XPU flash attention for the
    rest. Configured by --attention-backend-config ``sage_v3_kind`` and
    ``sage_v3_bf16_blocks``."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN_3

    @staticmethod
    def get_impl_cls() -> type["XPUSageAttention3Impl"]:
        return XPUSageAttention3Impl


class XPUSageAttention3Impl(XPUAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        is_cross_attention: bool = False,
        **extra_impl_args,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            is_cross_attention=is_cross_attention,
            **extra_impl_args,
        )
        kind, bf16_blocks = _sage_v3_options()
        self.sage_v3_kind = _resolve_sage_v3_kind(
            kind=kind,
            bf16_blocks=bf16_blocks,
            prefix=prefix,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            softmax_scale=softmax_scale,
            causal=causal,
            is_cross_attention=is_cross_attention,
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
        if (
            self.sage_v3_kind is not None
            and not return_softmax_lse
            and query.dtype == torch.bfloat16
            and query.shape[0] == 1
            and query.shape[1] == key.shape[1]
        ):
            return torch.ops.sglang.xpu_sage_v3_attn(
                query, key, value, self.sage_v3_kind
            )
        return super().forward(
            query,
            key,
            value,
            attn_metadata,
            return_softmax_lse=return_softmax_lse,
        )
