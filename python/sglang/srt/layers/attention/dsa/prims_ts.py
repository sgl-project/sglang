"""FlashInfer Prims-TS sparse MLA for the DSA backend (``prims_ts``).

The kernel reads the raw FP8 MLA cache rows that the ``trtllm`` DSA backend writes
(576 B/token: 512 NoPE + 64 RoPE E4M3, one per-tensor scale) with an E4M3 query
and FP8 QK/PV MMA, and returns BF16 output with a natural-log LSE. It shares the
``trtllm`` route's query preparation, KV store, top-k transforms and DCP slot
localization; only the attention kernel differs. With ``--kv-cache-dtype
nvfp4_ds_mla`` it reads 352 B NVFP4 records instead (``kv_cache_format
"nvfp4.fp8rope"``; layout in ``sglang.kernels.ops.kvcache.mla_buffer``).
"""

from __future__ import annotations

from typing import Optional

import torch

# Each row count up to this gets its own kept plan, for graph shapes and eager
# batches alike; larger batches share power-of-two capacity plans.
_EXACT_PLAN_MAX_ROWS = 1024

# B200, B300/GB300 and Rubin; SM107 needs flashinfer-ai/flashinfer#8186.
_SUPPORTED_DEVICE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))


def _plan_capacity(rows: int) -> Optional[int]:
    """None plans exactly ``rows`` fixed rows; otherwise the packed row capacity."""
    if rows <= _EXACT_PLAN_MAX_ROWS:
        return None
    return 1 << (rows - 1).bit_length()


# Bytes per cache row: raw E4M3 D576, and the nvfp4_ds_mla record
# (sglang.kernels.ops.kvcache.mla_buffer.NVFP4_DS_MLA_ROW_BYTES).
_ROW_BYTES = {"native": 576, "nvfp4.fp8rope": 352}


def check_prims_ts_sparse_mla_support(
    *,
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    kv_cache_format: str = "native",
) -> None:
    """Fail at startup, not at the first forward, when the installed FlashInfer
    cannot plan the D576 native-E4M3 sparse MLA this backend needs."""
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        raise ValueError(
            "--dsa-*-backend prims_ts supports D576 MLA "
            f"(kv_lora_rank 512, qk_rope_head_dim 64); got {kv_lora_rank}, "
            f"{qk_rope_head_dim}."
        )
    if torch.cuda.get_device_capability(device) not in _SUPPORTED_DEVICE_CAPABILITIES:
        raise ValueError("Prims-TS sparse MLA requires SM100, SM103 or SM107.")
    try:
        from flashinfer.attention.prims_ts import (
            get_prims_ts_sparse_mla_decode_workspace_size,
        )

        get_prims_ts_sparse_mla_decode_workspace_size(
            device,
            1,
            num_heads,
            max_topk=2048,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_cache_format=kv_cache_format,
            return_lse=True,
        )
    except (ImportError, TypeError, ValueError, NotImplementedError, KeyError) as exc:
        raise RuntimeError(
            f"--dsa-*-backend prims_ts needs FlashInfer Prims-TS with D576 "
            f"{kv_cache_format} sparse MLA (flashinfer-ai/flashinfer#7410 and #7984; "
            f"#8186 on SM107): {exc}"
        ) from exc


class PrimsTSSparseMLA:
    """Planned Prims-TS sparse MLA launches, keyed by row plan, heads and top-k width.

    Plans compile and allocate, so they are made on eager calls; the CUDA graph
    runner's eager warmups plan every captured row count before its capture.
    """

    def __init__(self, device: torch.device, kv_cache_format: str = "native"):
        # Plans compare tensor devices with this one, so it carries the index.
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        # "native": raw E4M3 rows of 576 B. "nvfp4.fp8rope": 352 B records of
        # E2M1 NoPE + unscaled E4M3 RoPE + E4M3 block16 scales (no global scale).
        assert kv_cache_format in ("native", "nvfp4.fp8rope"), kv_cache_format
        self.kv_cache_format = kv_cache_format
        self.nvfp4 = kv_cache_format == "nvfp4.fp8rope"
        self.row_bytes = _ROW_BYTES[kv_cache_format]
        self._plans: dict[tuple[int, int, int, bool], object] = {}
        # packed plans' [0, rows] query offsets, keyed like _plans
        self._qo_indptr: dict[tuple[int, int, int, bool], torch.Tensor] = {}

    def _wrapper(self, *, rows: int, heads: int, width: int):
        from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

        capacity = _plan_capacity(rows)
        packed = capacity is not None
        key = (capacity if packed else rows, heads, width, packed)
        wrapper = self._plans.get(key)
        if wrapper is not None:
            return wrapper, self._qo_indptr.get(key)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Prims-TS sparse MLA has no plan for {rows} rows x {heads} heads; "
                "graph capture must follow an eager run of the same shape."
            )
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        wrapper.plan(
            self.device,
            1 if packed else rows,
            heads,
            max_topk=width,
            max_q_len=capacity if packed else 1,
            packed_query=packed,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            kv_cache_format=self.kv_cache_format,
            return_lse=True,
        )
        self._plans[key] = wrapper
        if packed:
            self._qo_indptr[key] = torch.zeros(2, dtype=torch.int32, device=self.device)
        return wrapper, self._qo_indptr.get(key)

    def forward(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        sm_scale: float,
        kv_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q [rows, H, 576] E4M3; kv_cache raw rows viewable as [N, 576] E4M3
        (native) or [N, 352] uint8 records (nvfp4.fp8rope); indices [rows, K]
        int32 cache rows (-1 holes); lengths [rows] int32 <= K.
        Returns BF16 [rows, H, 512] and natural-log LSE [rows, H] (-inf, zero
        output for a row with no keys)."""
        rows, heads, head_dim = q.shape
        out = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
        if rows == 0:
            return out, lse
        width = indices.shape[1]
        row_bytes = kv_cache.shape[-1] * kv_cache.element_size()
        if row_bytes != self.row_bytes:
            raise ValueError(
                f"Prims-TS {self.kv_cache_format} reads {self.row_bytes} B cache rows; "
                f"got {row_bytes} B (--kv-cache-dtype and the pool layout disagree)."
            )
        wrapper, qo_indptr = self._wrapper(rows=rows, heads=heads, width=width)
        if self.nvfp4:
            kv_rows = kv_cache.view(torch.uint8).view(-1, self.row_bytes)
        else:
            kv_rows = kv_cache.view(torch.float8_e4m3fn).view(-1, head_dim)
        # The per-tensor KV descale multiplies both the QK logits and the
        # weighted V sum; the query is cast unscaled. nvfp4_ds_mla has none
        # (kv_scale 1, checked at backend init).
        kwargs = dict(
            softmax_scale=sm_scale * kv_scale,
            output_scale=kv_scale,
            validate=False,
        )
        if qo_indptr is not None:
            qo_indptr[1:].fill_(rows)
            wrapper.run(
                q,
                kv_rows,
                indices,
                lengths,
                qo_indptr=qo_indptr,
                out=out,
                lse=lse,
                **kwargs,
            )
        else:
            wrapper.run(
                q.view(rows, 1, heads, head_dim),
                kv_rows,
                indices,
                lengths,
                out=out.view(rows, 1, heads, 512),
                lse=lse.view(rows, 1, heads),
                **kwargs,
            )
        return out, lse
