# SPDX-License-Identifier: Apache-2.0
# Torch-native attention implemented with PyTorch SDPA instead of FA4/CUTLASS.
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F

_BLOCK_CAUSAL_MASK_MOD_CACHE = {}


def _auto_sdpa_backend_name() -> str | None:
    """Return the ROCm-only correctness fallback for H3 video-VAE SDPA."""
    if torch.version.hip is None:
        return None

    from sglang.srt.utils import is_gfx95_supported

    # Fused ROCm SDPA corrupts the dense ViT decode on gfx950. Keep every
    # non-gfx950 platform, including CUDA, on PyTorch's unchanged auto path.
    return "math" if is_gfx95_supported() else None


_AUTO_SDPA_BACKEND = _auto_sdpa_backend_name()


def _as_bool_mask(mask, *, device):
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=device)
    return mask.to(device=device, dtype=torch.bool)


def _ensure_nonempty_rows(mask):
    if mask.numel() == 0 or mask.shape[-1] == 0:
        return mask
    empty = ~mask.any(dim=-1)
    mask[..., 0] |= empty
    return mask


def _sdpa_kernel_context():
    backend_name = os.environ.get("MINIMAX_H3_TORCH_SDPA_BACKEND", "auto").lower()
    if backend_name in {"", "auto", "default"}:
        backend_name = _AUTO_SDPA_BACKEND
    if backend_name is None:
        return nullcontext()

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "flash_attention": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "cudnn_attention": SDPBackend.CUDNN_ATTENTION,
    }
    if backend_name not in backends:
        raise ValueError(
            "MINIMAX_H3_TORCH_SDPA_BACKEND must be one of "
            f"{sorted([*backends, 'auto', 'default'])}, got {backend_name!r}"
        )
    return sdpa_kernel(backends=[backends[backend_name]])


def _sdpa_attention(query, key, value, causal=False, attn_mask=None):
    # query/key/value arrive as [B, S, H, D]; PyTorch SDPA expects
    # [B, H, S, D].
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if attn_mask is not None and attn_mask.dim() == 3:
        attn_mask = attn_mask.unsqueeze(0)
    with _sdpa_kernel_context():
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
        )
    return out.transpose(1, 2).nan_to_num(0.0)


def _mask_mod_to_dense(mask_mod, batch, heads, q_len, kv_len, device, aux_tensors=None):
    q_idx = torch.arange(q_len, device=device).view(q_len, 1)
    kv_idx = torch.arange(kv_len, device=device).view(1, kv_len)
    dense = torch.empty((batch, heads, q_len, kv_len), dtype=torch.bool, device=device)
    for b in range(batch):
        b_idx = torch.tensor(b, device=device)
        for h in range(heads):
            h_idx = torch.tensor(h, device=device)
            mask = mask_mod(b_idx, h_idx, q_idx, kv_idx, None, aux_tensors)
            dense[b, h] = _as_bool_mask(mask, device=device)
    return _ensure_nonempty_rows(dense)


#########################################################
# Block causal attention
#########################################################


def make_block_causal_mask_mod(num_tokens, block_size, num_special=0, suffix=False):
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_special < 0:
        raise ValueError(f"num_special must be non-negative, got {num_special}")

    cache_key = (num_tokens, block_size, num_special, suffix)
    if cache_key in _BLOCK_CAUSAL_MASK_MOD_CACHE:
        return _BLOCK_CAUSAL_MASK_MOD_CACHE[cache_key]

    if suffix:

        def mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
            del b, h, seqlen_info, aux_tensors
            q_is_special = q_idx >= num_tokens
            kv_is_special = kv_idx >= num_tokens
            return (
                q_is_special
                | kv_is_special
                | (q_idx // block_size >= kv_idx // block_size)
            )

    else:

        def mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
            del b, h, seqlen_info, aux_tensors
            q_is_special = q_idx < num_special
            kv_is_special = kv_idx < num_special
            q_block_idx = (q_idx - num_special) // block_size
            kv_block_idx = (kv_idx - num_special) // block_size
            return q_is_special | kv_is_special | (q_block_idx >= kv_block_idx)

    mask_mod.block_sparse_cache_key = (
        "block_causal",
        num_tokens,
        block_size,
        num_special,
        suffix,
    )
    _BLOCK_CAUSAL_MASK_MOD_CACHE[cache_key] = mask_mod
    return mask_mod


#########################################################
# Public entry point
#########################################################


@torch.compiler.disable
def flash_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    mask_mod=None,
    block_sparse=None,
    aux_tensors=None,
) -> torch.Tensor:
    use_masked = mask_mod is not None or block_sparse is not None

    if block_sparse is not None and mask_mod is None:
        raise ValueError("block_sparse requires mask_mod")
    if causal and mask_mod is not None:
        raise ValueError(
            "causal must be encoded in mask_mod when using masked attention"
        )
    if aux_tensors is not None and not use_masked:
        raise ValueError("aux_tensors is only supported with masked attention")

    if use_masked:
        batch, q_len, heads, _ = query.shape
        kv_len = key.shape[1]
        dense_mask = _mask_mod_to_dense(
            mask_mod,
            batch,
            heads,
            q_len,
            kv_len,
            query.device,
            aux_tensors=aux_tensors,
        )
        return _sdpa_attention(query, key, value, attn_mask=dense_mask)

    return _sdpa_attention(query, key, value, causal=causal)
