"""Fused HiCache load-back of an NVFP4 host copy into FP8 device KV rows
(``kvcacheio/hicache_nvfp4_load.cuh``; used by ``srt/mem_cache/pool_host/mha_fp8_nvfp4.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Rows the kernel moves per stage at most (kNvfp4LoadMaxRows): one 64-token page.
MAX_ROWS_PER_CHUNK = 64


@cache_once
def _jit_module(*, row: int, stages: int, store_warps: int) -> Module:
    args = make_cpp_args(row, stages, store_warps)
    kernel = f"HiCacheNvfp4LoadKernel<{args}>"
    return load_jit(
        "hicache_nvfp4_load",
        *args,
        cuda_files=["kvcacheio/hicache_nvfp4_load.cuh"],
        cuda_wrappers=[
            ("params_bytes", f"&{kernel}::params_bytes"),
            ("fits_device", f"&{kernel}::fits_device"),
            ("encode", f"&{kernel}::encode"),
            ("launch", f"&{kernel}::launch"),
        ],
    )


def rows_per_chunk(page_size: int) -> int:
    """The largest power of two <= 64 dividing the page size, so a chunk never straddles a page."""
    rows = 1
    while rows * 2 <= MAX_ROWS_PER_CHUNK and page_size % (rows * 2) == 0:
        rows *= 2
    return rows


class Nvfp4HostLoad:
    """One host pool's fused load: four boxes encoded once over its pinned host buffers."""

    def __init__(
        self,
        *,
        row: int,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tokens: int,
        layers: int,
        page_size: int,
        lut: torch.Tensor,
        dst_stride: int,
        device: torch.device,
        stages: int,
        store_warps: int,
    ):
        if lut.shape != (256, 16) or lut.dtype != torch.uint8 or not lut.is_cuda:
            raise ValueError("NVFP4 load table must be a [256, 16] uint8 CUDA tensor")
        for name, buf, cols in (
            ("k_data", k_data, row // 2),
            ("v_data", v_data, row // 2),
            ("k_scale", k_scale, row // 16),
            ("v_scale", v_scale, row // 16),
        ):
            # [tokens, layers, ...] contiguous: token stride layers * cols bytes, layer stride cols bytes.
            if buf.dtype != torch.uint8 or not buf.is_contiguous() or buf.numel() != tokens * layers * cols:
                raise ValueError(f"NVFP4 load: {name} is not a contiguous [{tokens}, {layers}, {cols}] uint8 buffer")
        self.module = _jit_module(row=row, stages=stages, store_warps=store_warps)
        index = device.index if device.index is not None else torch.cuda.current_device()
        if not self.module.fits_device(index):
            raise RuntimeError(f"NVFP4 load: {stages} stages do not fit the device's shared memory")
        self.rows_per_chunk = rows_per_chunk(page_size)
        self.stages = stages
        self.store_warps = store_warps
        self.lut = lut  # read by every launch
        self.blob = torch.empty(self.module.params_bytes(), dtype=torch.uint8)
        self.module.encode(
            self.blob,
            k_data.data_ptr(),
            v_data.data_ptr(),
            k_scale.data_ptr(),
            v_scale.data_ptr(),
            tokens,
            layers,
            self.rows_per_chunk,
            lut.data_ptr(),
            dst_stride,
        )

    def __call__(
        self,
        host_idx: torch.Tensor,
        dev_idx: torch.Tensor,
        k_dst_ptr: int,
        v_dst_ptr: int,
        layer: int,
        block_quota: int,
    ) -> None:
        """Load one layer's K and V rows for int64 CUDA indices on the current stream."""
        self.module.launch(
            self.blob,
            host_idx,
            dev_idx,
            k_dst_ptr,
            v_dst_ptr,
            layer,
            block_quota,
            torch.cuda.current_stream(host_idx.device).cuda_stream,
        )
