from __future__ import annotations

"""SM70 indexer score.

Software E4M3FN decode plus the DeepGEMM fp8 MQA formula:
``sum_h relu(q[h] dot k) * weight[h] * k_scale``.
"""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

_PAGE_BYTES = 64 * 132


@cache_once
def _mqa_logits_sm70_module():
    return load_jit(
        "mqa_logits_sm70",
        cuda_files=["attention/mqa_logits_sm70.cuh"],
        cuda_wrappers=[
            ("ragged", "MqaLogitsSm70Kernel::ragged"),
            ("paged", "MqaLogitsSm70Kernel::paged"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def _fp8_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.contiguous()
    if tensor.dtype != torch.float8_e4m3fn:
        raise ValueError(f"indexer q/k must be float8_e4m3fn or uint8, got {tensor.dtype}")
    return tensor.contiguous().view(torch.uint8)


def _require_sm70(tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda" or torch.cuda.get_device_capability(tensor.device) != (7, 0):
        raise ValueError("SM70 MQA logits require an SM70 CUDA device")


def fp8_mqa_logits_sm70(
    q_fp8: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Ragged scores. ``kv`` is ``(k_fp8 [N, 128], k_scale [N])``.

    ``q_fp8`` is ``[M, H, 128]``. Returns ``[M, N]`` fp32. Positions outside
    ``[start, end)`` are ``-inf`` when ``clean_logits`` is set, else 0.
    """
    _require_sm70(q_fp8)
    k_fp8, k_scale = kv
    if q_fp8.ndim != 3 or q_fp8.shape[-1] != 128:
        raise ValueError(f"q must be [M, H, 128], got {tuple(q_fp8.shape)}")
    if k_fp8.ndim == 3 and k_fp8.shape[1] == 1:
        k_fp8 = k_fp8[:, 0]
    if k_fp8.ndim != 2 or k_fp8.shape[-1] != 128:
        raise ValueError(f"k must be [N, 128], got {tuple(k_fp8.shape)}")
    heads = q_fp8.shape[1]
    queries = q_fp8.shape[0]
    keys = k_fp8.shape[0]
    weights = weights.reshape(queries, heads).contiguous().float()
    k_scale = k_scale.reshape(keys).contiguous().float()
    starts = starts.reshape(queries).to(torch.int32).contiguous()
    ends = ends.reshape(queries).to(torch.int32).contiguous()
    out = torch.empty(queries, keys, dtype=torch.float32, device=q_fp8.device)
    masked = float("-inf") if clean_logits else 0.0
    _mqa_logits_sm70_module().ragged(
        _fp8_bytes(q_fp8),
        _fp8_bytes(k_fp8),
        k_scale,
        weights,
        starts,
        ends,
        out,
        masked,
    )
    return out


def fp8_paged_mqa_logits_sm70(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Paged scores. ``q_fp8`` is ``[B, 1, H, 128]`` or ``[B, H, 128]``.

    ``kv_cache_fp8`` is ``[pages, 64, 1, 132]`` or ``[pages, 8448]`` bytes:
    64 packed fp8 tokens, then 64 fp32 scales. Returns ``[B, max_seq_len]``.
    """
    _require_sm70(q_fp8)
    if q_fp8.ndim == 4:
        if q_fp8.shape[1] != 1:
            raise ValueError(f"paged MQA next_n must be 1, got {tuple(q_fp8.shape)}")
        q_fp8 = q_fp8[:, 0]
    if q_fp8.ndim != 3 or q_fp8.shape[-1] != 128:
        raise ValueError(f"q must be [B, H, 128], got {tuple(q_fp8.shape)}")
    batch, heads, _ = q_fp8.shape
    kv = kv_cache_fp8
    if kv.ndim == 4:
        kv = kv.reshape(kv.shape[0], -1)
    if kv.ndim != 2 or kv.shape[1] != _PAGE_BYTES:
        raise ValueError(f"kv pages must be [pages, {_PAGE_BYTES}], got {tuple(kv_cache_fp8.shape)}")
    if seq_lens.ndim > 1:
        seq_lens = seq_lens.reshape(-1)
    weights = weights.reshape(batch, heads).contiguous().float()
    seq_lens = seq_lens.to(torch.int32).contiguous()
    page_table = page_table.to(torch.int32).contiguous()
    fill = float("-inf") if clean_logits else 0.0
    out = torch.full((batch, max_seq_len), fill, dtype=torch.float32, device=q_fp8.device)
    _mqa_logits_sm70_module().paged(
        _fp8_bytes(q_fp8),
        _fp8_bytes(kv),
        weights,
        seq_lens,
        page_table,
        out,
    )
    return out
