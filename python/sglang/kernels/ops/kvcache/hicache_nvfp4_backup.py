"""Fused HiCache backup of FP8 device KV rows into an NVFP4 host copy
(``kvcacheio/hicache_nvfp4_backup.cuh``; used by ``srt/mem_cache/pool_host/mha_fp8_nvfp4.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(*, row: int, inv_global_scale: int, warps: int, group: int) -> Module:
    args = make_cpp_args(row, inv_global_scale, warps, group)
    return load_jit(
        "hicache_nvfp4_backup",
        *args,
        cuda_files=["kvcacheio/hicache_nvfp4_backup.cuh"],
        cuda_wrappers=[("launch", f"&HiCacheNvfp4BackupKernel<{args}>::launch")],
    )


class Nvfp4HostBackup:
    """One host pool's fused backup into its NVFP4 host buffers."""

    def __init__(
        self,
        *,
        row: int,
        global_scale: float,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tokens: int,
        layers: int,
        warps: int,
        group: int,
        device: torch.device,
        draft_k: torch.Tensor | None = None,
        draft_v: torch.Tensor | None = None,
        draft_layers: int = 0,
        draft_row_bytes: int = 0,
    ):
        inv = 1.0 / global_scale
        if inv != int(inv) or int(inv) & (int(inv) - 1):
            raise ValueError(f"NVFP4 backup: 1 / global scale {inv} is not a power of two")
        for name, buf, cols in (
            ("k_data", k_data, row // 2),
            ("v_data", v_data, row // 2),
            ("k_scale", k_scale, row // 16),
            ("v_scale", v_scale, row // 16),
        ):
            # [tokens, layers, cols] contiguous: a token's rows of every layer are one span.
            if buf.dtype != torch.uint8 or not buf.is_contiguous() or buf.numel() != tokens * layers * cols:
                raise ValueError(f"NVFP4 backup: {name} is not a contiguous [{tokens}, {layers}, {cols}] uint8 buffer")
        if draft_layers:
            for name, buf in (("draft_k", draft_k), ("draft_v", draft_v)):
                # [tokens, draft_layers, row] contiguous: a token's draft rows are one span.
                if not buf.is_contiguous() or buf.nbytes != tokens * draft_layers * draft_row_bytes:
                    raise ValueError(f"NVFP4 backup: {name} is not a contiguous [{tokens}, {draft_layers}, {draft_row_bytes} B] buffer")
            if draft_row_bytes % 16:
                raise ValueError(f"NVFP4 backup: draft rows of {draft_row_bytes} B are not 16 B multiples")
        self.module = _jit_module(row=row, inv_global_scale=int(inv), warps=warps, group=group)
        self.warps = warps
        self.group = group
        self.draft_layers = draft_layers
        self._draft_row_bytes = draft_row_bytes
        self._dst = (k_data.data_ptr(), v_data.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr())
        self._draft_dst = (draft_k.data_ptr(), draft_v.data_ptr()) if draft_layers else (0, 0)
        self._no_draft = torch.empty(0, dtype=torch.int64, device=device)

    def __call__(
        self,
        k_ptrs: torch.Tensor,
        v_ptrs: torch.Tensor,
        host_idx: torch.Tensor,
        dev_idx: torch.Tensor,
        src_stride: int,
        block_quota: int,
        draft_k_ptrs: torch.Tensor | None = None,
        draft_v_ptrs: torch.Tensor | None = None,
        draft_src_stride: int = 0,
    ) -> None:
        """Back up K and V rows of every layer, and the draft rows when built with them, for int64 CUDA
        indices on the current stream; the ``*_ptrs`` are int64 CUDA tables of the layer buffers' addresses."""
        if self.draft_layers:
            if draft_k_ptrs is None or draft_k_ptrs.numel() != self.draft_layers or draft_v_ptrs.numel() != self.draft_layers:
                raise ValueError(f"NVFP4 backup: needs {self.draft_layers}-layer draft tables")
        else:
            draft_k_ptrs = draft_v_ptrs = self._no_draft
        self.module.launch(
            k_ptrs,
            v_ptrs,
            draft_k_ptrs,
            draft_v_ptrs,
            host_idx,
            dev_idx,
            *self._dst,
            *self._draft_dst,
            src_stride,
            draft_src_stride,
            self._draft_row_bytes,
            block_quota,
            torch.cuda.current_stream(host_idx.device).cuda_stream,
        )
