"""Fused K normalization, NeoX RoPE and masked cache store for AdaFlash on Ascend."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties
from triton.language.extra.cann import extension as al

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding


@triton.jit(do_not_specialize=["num_rows"])
def _store_packed_kv(
    k_ptr,
    v_ptr,
    weight_ptr,
    rope_ptr,
    positions_ptr,
    slots_ptr,
    k_cache_ptr,
    v_cache_ptr,
    num_rows,
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    cols = tl.arange(0, HEADS * DIM)
    half = tl.arange(0, DIM // 2)
    weight = tl.load(weight_ptr + tl.arange(0, DIM)).to(tl.float32)

    for row in range(tl.program_id(0), num_rows, tl.num_programs(0)):
        slot = tl.load(slots_ptr + row).to(tl.int64)
        valid = slot >= 0
        safe_slot = tl.where(valid, slot, 0)
        store_mask = (cols < HEADS * DIM) & valid

        k = tl.load(k_ptr + row * K_STRIDE + cols).to(tl.float32)
        k = k.reshape(HEADS, DIM)
        inv_rms = tl.rsqrt(tl.sum(k * k, axis=1) / DIM + EPS)
        # The standalone RMSNorm rounds to BF16 before passing K to RoPE.
        norm = (k * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)
        norm = norm.to(tl.float32)
        lo = al.extract_slice(norm, (0, 0), (HEADS, DIM // 2), (1, 1))
        hi = al.extract_slice(norm, (0, DIM // 2), (HEADS, DIM // 2), (1, 1))

        pos = tl.load(positions_ptr + row).to(tl.int64)
        cos = tl.load(rope_ptr + pos * DIM + half).to(tl.float32)
        sin = tl.load(rope_ptr + pos * DIM + DIM // 2 + half).to(tl.float32)
        rotated = tl.zeros((HEADS, DIM), tl.float32)
        rotated = al.insert_slice(
            rotated,
            lo * cos[None, :] - hi * sin[None, :],
            (0, 0),
            (HEADS, DIM // 2),
            (1, 1),
        )
        rotated = al.insert_slice(
            rotated,
            hi * cos[None, :] + lo * sin[None, :],
            (0, DIM // 2),
            (HEADS, DIM // 2),
            (1, 1),
        )

        cache_offsets = safe_slot * (HEADS * DIM) + cols
        tl.store(
            k_cache_ptr + cache_offsets,
            rotated.reshape(HEADS * DIM),
            mask=store_mask,
        )
        v = tl.load(v_ptr + row * V_STRIDE + cols)
        tl.store(v_cache_ptr + cache_offsets, v, mask=store_mask)


def store_packed_adaflash_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """Write BF16 K/V into physical slots (rows with slot < 0 are skipped)."""
    num_rows = positions.numel()
    if num_rows == 0:
        return
    _, vector_cores = get_device_properties()
    _store_packed_kv[(min(num_rows, vector_cores),)](
        k,
        v,
        weight,
        cos_sin,
        positions,
        slots,
        k_cache,
        v_cache,
        num_rows,
        k.stride(0),
        v.stride(0),
        num_heads,
        head_dim,
        eps,
    )


def _layer_supports_packed_adaflash_kv(attn, cache_layer) -> bool:
    pool = cache_layer.view
    norm = getattr(attn, "k_norm", None)
    rope = getattr(attn, "rotary_emb", None)
    if (
        not cache_layer.plain
        or type(pool) is not NPUMHATokenToKVPool
        or pool.store_dtype != torch.bfloat16
        or pool.dtype != torch.bfloat16
        or pool.is_modelslim_mxfp8
        or pool.head_dim != pool.v_head_dim
        or pool.head_dim % 2 != 0
        or (pool.head_num, pool.head_dim) != (attn.num_kv_heads, attn.head_dim)
        or type(norm) is not RMSNorm
        or type(rope) is not RotaryEmbedding
        or not rope.is_neox_style
        or rope.rotary_dim != attn.head_dim
        or not rope.cos_sin_cache.is_contiguous()
        or norm.weight.dtype != torch.bfloat16
    ):
        return False
    k_cache, v_cache = cache_layer.kv()
    return k_cache.is_contiguous() and v_cache.is_contiguous()


def can_store_packed_adaflash_kv(draft_model, token_to_kv_pool) -> bool:
    """Return True when every draft layer can use the masked fused store."""
    layers = getattr(draft_model, "layers", None)
    router = getattr(token_to_kv_pool, "layer_router", None)
    if not layers or router is None:
        return False
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "attn"):
            return False
        cache_layer = router.layer(attn.attn.layer_id)
        if not _layer_supports_packed_adaflash_kv(attn, cache_layer):
            return False
    return True


def try_store_packed_adaflash_kv(attn, cache_layer, k, v, positions, slots) -> bool:
    """Use direct masked stores for standard BF16 Ascend MHA draft caches."""
    if not _layer_supports_packed_adaflash_kv(attn, cache_layer):
        return False
    rope = attn.rotary_emb
    norm = attn.k_norm
    if (
        rope.cos_sin_cache.device != k.device
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or k.stride(-1) != 1
        or v.stride(-1) != 1
    ):
        return False

    k_cache, v_cache = cache_layer.kv()
    store_packed_adaflash_kv(
        k,
        v,
        norm.weight,
        rope.cos_sin_cache,
        positions,
        slots,
        k_cache,
        v_cache,
        num_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
        eps=norm.variance_epsilon,
    )
    return True
