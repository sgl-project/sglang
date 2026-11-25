"""NVRTC / CUDA driver helpers for the Muse Spark Hopper kernels (needs cuda.bindings)."""

import ctypes
import functools
import math
import os
import re
import struct


def expand(src):
    return re.sub(r'#include "([\w.]+)"\n', lambda m: read_source(m.group(1)), src)


def read_source(name):
    with open(os.path.join(os.path.dirname(__file__), "csrc", "muse_spark", name)) as f:
        return expand(f.read())


def fields(*groups):
    return [(n, t) for t, names in groups for n in names.split()]


def check(res, what):
    if int(res[0]):
        raise RuntimeError(f"{what} failed: {res[0]}")
    return res[-1]


@functools.cache
def driver():
    from cuda.bindings import driver as d

    return d


SM90A_OPTS = [b"--gpu-architecture=sm_90a", b"-std=c++17"]


def build_module(name, src, opts):
    from cuda.bindings import nvrtc

    prog = check(
        nvrtc.nvrtcCreateProgram(expand(src).encode(), name.encode(), 0, [], []),
        "nvrtcCreateProgram",
    )
    try:
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log = b" " * nvrtc.nvrtcGetProgramLogSize(prog)[1]
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"NVRTC failed on {name}:\n" + log.decode(errors="replace")
            )
        cubin = b" " * check(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")
        check(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    return check(driver().cuModuleLoadData(cubin), "cuModuleLoadData")


def device_module(dev, name, src):
    """Build for dev's sm_XYa (context current) -> (module, SM count)."""
    import torch

    with torch.cuda.device(dev):
        torch.zeros(1, device=dev)
        mod = build_module(
            name,
            src,
            [
                b"--gpu-architecture=sm_%d%da" % torch.cuda.get_device_capability(dev),
                b"-std=c++17",
                b"-default-device",
            ],
        )
        return mod, torch.cuda.get_device_properties(dev).multi_processor_count


def scale_log2(sm_scale, qk_scale_factor):
    """FA3's softmax_scale_log2 (same float32 roundings)."""
    sm = struct.unpack("f", struct.pack("f", sm_scale * qk_scale_factor))[0]
    return struct.unpack("f", struct.pack("f", sm * math.log2(math.e)))[0]


def function(module, name, smem=0, nonportable_cluster=0):
    d = driver()
    f = check(d.cuModuleGetFunction(module, name.encode()), name)
    A = d.CUfunction_attribute
    for attr, v in (
        (A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem),
        (A.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, nonportable_cluster),
    ):
        if v:
            check(d.cuFuncSetAttribute(f, attr, v), "cuFuncSetAttribute")
    return f


def tensor_map(ptr, dims, strides, box, words=False):
    d = driver()
    dtype, swizzle, l2 = (
        d.CUtensorMapDataType,
        d.CUtensorMapSwizzle,
        d.CUtensorMapL2promotion,
    )
    tm = check(
        d.cuTensorMapEncodeTiled(
            dtype.CU_TENSOR_MAP_DATA_TYPE_UINT32
            if words
            else dtype.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            d.cuuint32_t(len(dims)),
            ptr,
            [d.cuuint64_t(v) for v in dims],
            [d.cuuint64_t(v) for v in strides],
            [d.cuuint32_t(v) for v in box],
            [d.cuuint32_t(1)] * len(dims),
            d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle.CU_TENSOR_MAP_SWIZZLE_NONE
            if words
            else swizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            l2.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled",
    )
    return struct.pack("16Q", *map(int, tm.opaque))


@functools.cache
def cluster_dim(x):
    d = driver()
    at = d.CUlaunchAttribute()
    at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = x, 1, 1
    return at


@functools.cache
def pdl():
    d = driver()
    at = d.CUlaunchAttribute()
    at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    at.value.programmaticStreamSerializationAllowed = 1
    return at


def launch_config(grid, block, smem, stream, attrs):
    d = driver()
    cfg = d.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = (
        (grid, 1, 1) if isinstance(grid, int) else (grid[0], grid[1], 1)
    )
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
    cfg.sharedMemBytes = smem
    if stream is not None:
        cfg.hStream = d.CUstream(stream)
    cfg.attrs, cfg.numAttrs = attrs, len(attrs)
    return cfg


_CFG = {}


def launch(f, grid, block, smem, stream, args, attrs=None):
    if isinstance(args, bytes):
        raw = ctypes.create_string_buffer(len(args) + 128)
        ptrs = [(ctypes.addressof(raw) + 63) // 64 * 64]
        ctypes.memmove(ptrs[0], args, len(args))
    else:
        ptrs = [ctypes.addressof(a) for a in args]
    argv = (ctypes.c_void_p * len(ptrs))(*ptrs)
    key = (grid, block, smem, stream, *map(id, attrs or ()))
    if key not in _CFG:
        if len(_CFG) >= 4096:
            _CFG.clear()
        _CFG[key] = launch_config(grid, block, smem, stream, attrs or []), attrs
    check(
        driver().cuLaunchKernelEx(_CFG[key][0], f, ctypes.addressof(argv), 0),
        "kernel launch",
    )
