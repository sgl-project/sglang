# SPDX-License-Identifier: Apache-2.0
"""Experimental FN TP1/TP4 NVFP4 fused expert phase; no serving binding yet.

The prototype consumes current routing and linear-layout quantized input. All
scratch and maps are allocated before capture. Expert rows keep native BF16
rounding; the single-layer harness compares them before considering finalize.
"""
from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sglang.kernels.ops.moe import q38_router_jit as jit

ROOT = Path(__file__).resolve().parents[2] / "jit/csrc/moe"
H, I, E, TOP_K = 2560, 640, 512, 10


def pipeline_smem_bytes(intermediate):
    tile = 128 if intermediate == 640 else 64
    c = intermediate // tile
    stages = 2 if intermediate == 640 else 5
    fields = [(stages * 2 * tile * 144, 1024), (20480, 1024), (16384, 1024),
              (2560, 4), (16 * (tile // 64) * 4, 4), (16 * (2 * tile + 4) * 4, 4),
              (c * 16 * 260 * 4, 1024), (stages * 8, 8), (stages * 8, 8),
              (stages * 8, 8), (32, 8), (80, 8), (80, 8), (4, 4), (64, 4)]
    pos = 0
    for size, align in fields:
        pos = (pos + align - 1) // align * align + size
    return (pos + 1023) // 1024 * 1024 + 1024


def smem_bytes(intermediate):
    tile = 128 if intermediate == 640 else 64
    fields = [(tile * 2 * 144, 1024), (20480, 1024),
              ((intermediate // tile + 7) * 2048, 1024), (2560, 4),
              (16 * (intermediate // 64) * 4, 4), (16 * (tile * 2 + 4) * 4, 4),
              (8, 8), (8, 8), (4, 4), (64, 4)]
    pos = 0
    for size, align in fields:
        pos = (pos + align - 1) // align * align + size
    return (pos + 1023) // 1024 * 1024 + 1024


class Params(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "x", "xsf", "sf13", "sf2", "g1", "g1c", "g2", "ids",
        "unique_ids", "num_unique", "out",
    )] + [("m", ctypes.c_int)]


@cache
def kernel(device: int, intermediate: int = 640, pipeline: bool = False, instrumented: bool = False):
    major, minor = torch.cuda.get_device_capability(device)
    name = "q38_mk_moe_pipeline" if pipeline else "q38_mk_moe"
    shared = pipeline_smem_bytes(intermediate) if pipeline else smem_bytes(intermediate)
    cubin = jit.compile_source(
        (ROOT / (name + ".cu")).read_text(), name, f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", "--generate-line-info", f"-DQ38_MK_I={intermediate}",
         f"-DQ38_MK_SMEM_BYTES={shared}", f"-DQ38_MK_TRACE={int(instrumented)}"),
        {"q38_mk_moe_primitives.cuh": (ROOT / "q38_mk_moe_primitives.cuh").read_text()},
    )
    return jit.load(cubin, name, (256, 1, 1), shared)


def tensor_map(tensor, rows, row_bytes, box_bytes, box_rows, swizzle):
    cu = jit.driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    result, = jit.check(cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, tensor.data_ptr(),
        [u64(row_bytes), u64(rows)], [u64(row_bytes)],
        [u32(box_bytes), u32(box_rows)], [u32(1), u32(1)],
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    ), "FN weight map")
    return result


def weight_maps(w13, w2):
    intermediate = w2.shape[-1] * 2
    sw = jit.driver().CUtensorMapSwizzle
    return (
        tensor_map(w13, E * 2 * intermediate, H // 2, 128, 128, sw.CU_TENSOR_MAP_SWIZZLE_128B),
        tensor_map(w2, E * H, intermediate // 2, 64 if intermediate == 640 else 32, 256,
                   sw.CU_TENSOR_MAP_SWIZZLE_64B if intermediate == 640 else sw.CU_TENSOR_MAP_SWIZZLE_32B),
    )


def run(maps, x, xsf, sf13, sf2, g1, g1c, g2, ids, unique_ids, num_unique, out, intermediate=640, pipeline=False, trace=None):
    m = x.shape[0]
    if not 1 <= m <= 16:
        raise ValueError("FN prototype supports M1..16")
    p = Params(*(t.data_ptr() for t in (
        x, xsf, sf13, sf2, g1, g1c, g2, ids, unique_ids, num_unique, out,
    )), m)
    cluster_size = 5 if intermediate == 640 else 3
    clusters = torch.cuda.get_device_properties(x.device).multi_processor_count // cluster_size if pipeline else 30
    args = [*maps, p]
    if pipeline and trace is not None:
        args.append(ctypes.c_void_p(trace.data_ptr()))
    jit.launch(kernel(x.device.index, intermediate, pipeline, trace is not None), (clusters * cluster_size, 1, 1), args,
               torch.cuda.current_stream().cuda_stream, cluster=(cluster_size, 1, 1))
    return out
