"""SM70 sparse MLA for the NoPE 512-d latent.

Fp16 query and KV, fp32 softmax. The bf16 Triton and TileLang DSA kernels
do not run on Volta. The indexer has already chosen the keys; this kernel
only attends to those rows.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _sparse_mla_sm70_module():
    return load_jit(
        "sparse_mla_sm70",
        cuda_files=["attention/sparse_mla_sm70.cuh"],
        cuda_wrappers=[("run", "SparseMlaSm70Kernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def sparse_mla_sm70(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Sparse attention for one NoPE latent.

    ``q`` is ``[S, H, 512]`` fp16. ``kv`` is ``[N, 512]`` or ``[N, 1, 512]``
    fp16, and ``indices`` is ``[S, topk]`` (or ``[S, 1, topk]``). Negative
    indices are ignored. Returns ``[1, S, H, 512]`` fp16.
    """
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("sparse_mla_sm70 requires an SM70 CUDA device")
    if q.dtype != torch.float16 or kv.dtype != torch.float16:
        raise ValueError(
            f"SM70 sparse MLA requires fp16 q and kv, got {q.dtype} and {kv.dtype}"
        )
    if q.ndim != 3 or q.shape[-1] != 512:
        raise ValueError(f"q must be [S, H, 512], got {tuple(q.shape)}")
    if indices.ndim == 3:
        if indices.shape[1] != 1:
            raise ValueError(
                f"indices must be [S, topk] or [S, 1, topk], got {tuple(indices.shape)}"
            )
        indices = indices[:, 0, :]
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"indices must be [S, topk] with S={q.shape[0]}, got {tuple(indices.shape)}"
        )
    if kv.ndim == 3:
        if kv.shape[1] != 1 or kv.shape[-1] != 512:
            raise ValueError(f"kv must be [N, 1, 512], got {tuple(kv.shape)}")
    elif kv.ndim != 2 or kv.shape[-1] != 512:
        raise ValueError(f"kv must be [N, 512] or [N, 1, 512], got {tuple(kv.shape)}")

    # The kernel takes q's token and head strides; it reads each row as fp16 pairs.
    if q.stride(-1) != 1 or q.stride(0) % 2 or q.stride(1) % 2 or q.data_ptr() % 4:
        q = q.contiguous()
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    indices = indices.contiguous()
    if indices.shape[-1] == 0:
        return torch.zeros((1, *q.shape), dtype=torch.float16, device=q.device)
    out = torch.empty(q.shape, dtype=torch.float16, device=q.device)
    splits = _num_splits(q.shape[0], q.shape[1], indices.shape[-1])
    part_shape = (q.shape[0], q.shape[1], splits) if splits > 1 else (0,)
    part_acc = torch.empty((*part_shape, 512) if splits > 1 else (0,), dtype=torch.float32, device=q.device)
    part_ml = torch.empty((*part_shape, 2) if splits > 1 else (0,), dtype=torch.float32, device=q.device)
    _sparse_mla_sm70_module().run(q, kv, indices, out, part_acc, part_ml, float(sm_scale), splits)
    return out.unsqueeze(0)


# Two blocks per V100 SM (80 SMs); fewer (token, head-tile) blocks than that
# split the topk slots, with at least one 32-row chunk per split.
_TARGET_BLOCKS = 160
_ROWS_PER_CHUNK = 32


def _num_splits(tokens: int, heads: int, topk: int) -> int:
    # Decode and MTP verify (up to four tokens) share the one-token split, so a
    # verify row merges its softmax partials in the same order as at decode.
    blocks = (1 if tokens <= 4 else tokens) * ((heads + 7) // 8)
    if blocks >= _TARGET_BLOCKS // 2 or topk <= _ROWS_PER_CHUNK:
        return 1
    max_splits = (topk + _ROWS_PER_CHUNK - 1) // _ROWS_PER_CHUNK
    return max(1, min(max_splits, (_TARGET_BLOCKS + blocks - 1) // blocks))
