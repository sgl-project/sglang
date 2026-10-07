"""Torch export contract and reference for read-only MLX radix attention."""

from math import isfinite

import torch


def _validate_shapes(q, k, v, kp, vp, table, requests, lengths, scale):
    if q.ndim != 3 or k.ndim != 3 or kp.ndim != 3 or table.ndim != 2:
        raise ValueError("Unsupported MLX radix tensor rank")
    batch, heads, dim = q.shape
    kv_heads = k.shape[1]
    if (
        batch < 1
        or heads < 1
        or kv_heads < 1
        or dim not in (64, 128, 256)
        or heads % kv_heads
        or q.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or k.shape != (batch, kv_heads, dim)
        or v.shape != k.shape
        or kp.shape[1:] != (kv_heads, dim)
        or vp.shape != kp.shape
        or any(t.dtype != q.dtype for t in (k, v, kp, vp))
        or table.dtype != torch.int32
        or requests.dtype not in (torch.int32, torch.int64)
        or lengths.dtype not in (torch.int32, torch.int64)
        or requests.shape != (batch,)
        or lengths.shape != (batch,)
        or not isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Unsupported MLX radix attention shape or dtype")


@torch.library.custom_op("sglang::mlx_radix_decode", mutates_args=())
def radix_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend over cached prefix rows plus current K/V without modifying the pool.

    Q is [batch, query_heads, dim]; K/V are [batch, kv_heads, dim].
    Pools are [slots, kv_heads, dim], and table maps [request, position] to slots.
    Lengths include the current token. Serving must validate table indices before
    executing the exported graph; the Metal kernel emits NaNs for invalid rows.
    """
    _validate_shapes(q, k, v, k_pool, v_pool, table, requests, lengths, scale)
    outputs = []
    for batch, (request, length) in enumerate(
        zip(requests.cpu().tolist(), lengths.cpu().tolist())
    ):
        if not 0 <= request < table.shape[0] or not 1 <= length <= table.shape[1]:
            raise ValueError("Invalid MLX radix request or sequence length")
        slots = table[request, : length - 1].long()
        if bool(((slots < 0) | (slots >= k_pool.shape[0])).any()):
            raise ValueError("Invalid MLX radix cache slot")
        keys = torch.cat((k_pool[slots], k[batch : batch + 1]))
        values = torch.cat((v_pool[slots], v[batch : batch + 1]))
        groups = q.shape[1] // k.shape[1]
        keys = keys.repeat_interleave(groups, dim=1).float()
        values = values.repeat_interleave(groups, dim=1).float()
        scores = torch.einsum("hd,thd->ht", q[batch].float(), keys) * scale
        outputs.append(
            torch.einsum("ht,thd->hd", scores.softmax(-1), values).to(q.dtype)
        )
    return torch.stack(outputs)


@radix_decode.register_fake
def _fake(q, k, v, k_pool, v_pool, table, requests, lengths, scale):
    _validate_shapes(q, k, v, k_pool, v_pool, table, requests, lengths, scale)
    return torch.empty_like(q)
