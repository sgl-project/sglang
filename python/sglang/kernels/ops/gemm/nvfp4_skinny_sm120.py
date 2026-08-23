"""Qwen3.8 NVFP4 W4A4 skinny GEMM for high-SM-count SM120 GPUs.

The kernel is specialized for the three dense GEMMs observed in Qwen3.8
decode and DSpark verification. It covers M=1..16 and leaves all other shapes
on the general FlashInfer path.
"""

import ctypes
import logging
from pathlib import Path

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_logged_fast_path = False

_SRC_PATH = (
    Path(__file__).resolve().parents[2]
    / "jit"
    / "csrc"
    / "gemm"
    / "nvfp4"
    / "fp4_skinny_sm120.cu"
)

# (K, N) -> (NWARPS, STAGES, PREFETCH, GRID[, extra defines])
# GRID aligned to (N/NTILE) * split compiles the static schedule (-DSPLIT=s,
# loop-invariant producers, out-of-loop epilogue); any other GRID falls back
# to the generalized stream-K walk (-DSPLIT=0).
# evict_first policy on weight copies dodges part of the harness dirty-L2
# writeback tax; the winning scope is shape-specific (measured): gate_up
# both B copies at fraction 0.5 (+3%), lm_head both at fraction 1.0 (+2.8%),
# down none (policy evicts prefetched lines early and loses).
_CONFIGS = {
    (5120, 34816): (
        4,
        8,
        1,
        544,
        ("-DL2POLICY=1", "-DPOLSFB=0", "-DL2DESC=0x1270000000000000ULL"),
    ),
    # mlp down: 40 tiles x 136, 4-way split; FENCEREL swaps the epilogue's
    # 3x full __threadfence() for release/acquire sem atomics (-0.5us).
    # S7 won the dev protocol by ~0.2us but LOST the CUPTI full run
    # (1.1191 vs 1.1201) — sub-0.3us dev deltas do not transfer; keep S6.
    (17408, 5120): (8, 6, 1, 160, ("-DFENCEREL=1",)),
    (5120, 248320): (4, 12, 1, 3880, ("-DL2POLICY=1", "-DPOLSFB=0", "-DL1SFA=1")),
}

_ROLE_BY_SHAPE = {
    (5120, 34816): "gate_up",
    (17408, 5120): "down",
    (5120, 248320): "lm_head",
}


def _role_enabled(kdim: int, ndim: int, m: int) -> bool:
    configured = envs.SGLANG_QWEN38_FP4_SKINNY_GEMM_ROLES.get()
    roles = {role.strip() for role in configured.split(",") if role.strip()}
    role = _ROLE_BY_SHAPE[(kdim, ndim)]
    if "all" in roles or role in roles:
        return True
    return role == "down" and m == 1 and "down_m1" in roles


def _stage_bytes(nwarps):
    ntile = 16 * nwarps
    return 4 * (ntile * 20 + 320 + 256 + ntile * 2)


def _compile(kdim, ndim, nwarps, stages, prefetch, split, extra=()):
    with open(_SRC_PATH, "rb") as f:
        src = f.read()
    err, prog = nvrtc.nvrtcCreateProgram(src, b"fp4_gemm_sm120.cu", 0, [], [])
    if int(err) != 0:
        raise RuntimeError(f"nvrtcCreateProgram: {err}")
    opts = [
        b"--gpu-architecture=sm_120a",
        b"--std=c++17",
        f"-DKDIM={kdim}".encode(),
        f"-DNDIM={ndim}".encode(),
        f"-DNWARPS={nwarps}".encode(),
        f"-DSTAGES={stages}".encode(),
        f"-DPREFETCH={prefetch}".encode(),
        f"-DSPLIT={split}".encode(),
        *[e.encode() for e in extra],
    ]
    (res,) = (nvrtc.nvrtcCompileProgram(prog, len(opts), opts),)
    _, logsz = nvrtc.nvrtcGetProgramLogSize(prog)
    log = bytearray(logsz)
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(res[0]) != 0:
        raise RuntimeError(f"NVRTC compile failed:\n{bytes(log).decode()}")
    _, sz = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = bytearray(sz)
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    return bytes(cubin)


class _Variant:
    __slots__ = (
        "cubin",
        "func",
        "nwarps",
        "stages",
        "prefetch",
        "split",
        "extra",
        "grid",
        "kdim",
        "ndim",
        "ntile",
        "ktiles",
        "units",
        "maxc",
        "smem",
        "ws",
        "sem",
        "ws_donor",
    )

    def __init__(self, kdim, ndim, cfg):
        self.ws_donor = None
        self.kdim, self.ndim = kdim, ndim
        self.nwarps, self.stages, self.prefetch, grid = cfg[:4]
        extra = cfg[4] if len(cfg) > 4 else ()
        self.ntile = 16 * self.nwarps
        self.ktiles = kdim // 128
        nt = ndim // self.ntile
        self.units = nt * self.ktiles
        self.grid = min(grid, self.units)
        # Aligned grid -> static schedule; otherwise generalized stream-K.
        split = self.grid // nt
        if not (split >= 1 and self.grid == nt * split and self.ktiles % split == 0):
            split = 0
        # The parallel spin-reduce epilogue requires all SPLIT contributors
        # co-resident: safe within one full wave (188 SMs on the target).
        # Larger grids use the serial ticket reduction instead.
        spinred = 1 if (split <= 1 or self.grid <= 188) else 0
        extra = extra + (f"-DSPINRED={spinred}",)
        # Max partial contributors per tile (+1 covers boundary straddle).
        self.maxc = -(-self.ktiles * self.grid // self.units) + 1
        self.smem = self.stages * _stage_bytes(self.nwarps)
        self.split = split
        self.extra = extra
        self.cubin = None
        self.func = None
        self.ws = None
        self.sem = None

    def ensure_loaded(self, device):
        if self.func is not None:
            return
        if self.cubin is None:
            self.cubin = _compile(
                self.kdim,
                self.ndim,
                self.nwarps,
                self.stages,
                self.prefetch,
                self.split,
                self.extra,
            )
        err, mod = cu.cuModuleLoadData(self.cubin)
        if int(err) != 0:
            raise RuntimeError(f"cuModuleLoadData: {err}")
        err, func = cu.cuModuleGetFunction(mod, b"fp4gemm")
        if int(err) != 0:
            raise RuntimeError(f"cuModuleGetFunction: {err}")
        (err,) = cu.cuFuncSetAttribute(
            func,
            cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            self.smem,
        )
        if int(err) != 0:
            raise RuntimeError(f"cuFuncSetAttribute: {err}")
        self.func = func
        if self.ws_donor is not None:
            # Same-shape sibling variants never co-launch (per-row fixed M,
            # serialized stream, sem self-resets each launch), so they can
            # share one ws/sem. Allocating a second copy shifts the caching
            # allocator's layout and reproducibly costs lm_head ~8.7us per
            # 16 rows (~5bp) — see the round-25/26 forensics.
            self.ws_donor.ensure_loaded(device)
            self.ws = self.ws_donor.ws
            self.sem = self.ws_donor.sem
            return
        self.ws = torch.empty(
            (self.maxc, 16, self.ndim), dtype=torch.float32, device=device
        )
        self.sem = torch.zeros(
            (2 * (self.ndim // self.ntile),), dtype=torch.int32, device=device
        )


_variants = {}
for _key, _cfg in _CONFIGS.items():
    _variants[_key] = _Variant(_key[0], _key[1], _cfg)

# mlp_down per-M crossover: KROT=1 rotates each CTA's k-phase start inside its
# split slice. Full-run A/B (20260822T160702Z vs T150312Z, CUPTI): M7-16 all
# faster (-0.03..-0.48us, the M11-13 gains are 8-12x the same-config row CV),
# M1-6 all slower (+0.03..+0.16us). Dispatching at M>=7 keeps the M=1 decode
# anchor on the byte-identical base cubin and takes only the measured wins.
_variants[(17408, 5120, "krot7")] = _Variant(
    17408, 5120, (8, 6, 1, 160, ("-DFENCEREL=1", "-DKROT=1"))
)
_variants[(17408, 5120, "krot7")].ws_donor = _variants[(17408, 5120)]

_ARGTYPES = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
)

_cuda_ready = False


def _ensure_cuda():
    global _cuda_ready
    if not _cuda_ready:
        torch.cuda.init()
        cu.cuInit(0)
        _cuda_ready = True


def _get_variant(kdim, ndim):
    key = (kdim, ndim)
    var = _variants.get(key)
    if var is None:
        raise ValueError(f"unsupported NVFP4 skinny GEMM shape K={kdim}, N={ndim}")
    return var


def use_nvfp4_skinny_sm120(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> bool:
    """Return whether the SM120 fast path supports this exact call."""
    if not envs.SGLANG_ENABLE_QWEN38_FP4_SKINNY_GEMM.get() or not input.is_cuda:
        return False
    if input.ndim != 2 or not 1 <= input.shape[0] <= 16:
        return False

    kbytes = input.shape[1]
    kdim = kbytes * 2
    ndim = int(out_features)
    if (kdim, ndim) not in _CONFIGS:
        return False
    if not _role_enabled(kdim, ndim, input.shape[0]):
        return False
    if input.dtype != torch.uint8 or out_dtype != torch.bfloat16:
        return False
    if weight.dtype != torch.uint8:
        return False
    if input_sf.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return False
    if weight_sf.dtype != torch.float8_e4m3fn:
        return False
    if alpha.dtype != torch.float32 or alpha.numel() != 1:
        return False
    if input.stride() != (kbytes, 1):
        return False
    if weight.shape != (kbytes, ndim) or weight.stride() != (1, kbytes):
        return False
    scale_k = kdim // 16
    if input_sf.shape != (128, scale_k) or input_sf.stride() != (scale_k, 1):
        return False
    if weight_sf.shape != (scale_k, ndim) or weight_sf.stride() != (1, scale_k):
        return False
    tensors = (weight, input_sf, weight_sf, alpha)
    if any(t.device != input.device for t in tensors):
        return False

    props = torch.cuda.get_device_properties(input.device)
    return (props.major, props.minor) == (12, 0) and props.multi_processor_count >= 160


def _gemm(input, weight, input_sf, weight_sf, alpha, out_dtype, out_features):
    m, kbytes = input.shape
    if not 1 <= m <= 16:
        # Decode/verify scope only: accumulators cover 16 output rows and SFA
        # is staged for the 128-row pad. Production must route M>16 (prefill)
        # to the general fp4_gemm path.
        raise ValueError(f"fp4 skinny GEMM supports M=1..16, got M={m}")
    kdim = kbytes * 2
    ndim = out_features
    _ensure_cuda()
    var = _get_variant(kdim, ndim)
    if m >= 7:
        var = _variants.get((kdim, ndim, "krot7"), var)
    var.ensure_loaded(input.device)
    out = torch.empty((m, ndim), dtype=out_dtype, device=input.device)
    stream = torch.cuda.current_stream(input.device).cuda_stream
    args = (
        (
            input.data_ptr(),
            weight.data_ptr(),
            input_sf.data_ptr(),
            weight_sf.data_ptr(),
            alpha.data_ptr(),
            out.data_ptr(),
            var.ws.data_ptr(),
            var.sem.data_ptr(),
            m,
        ),
        _ARGTYPES,
    )
    res = cu.cuLaunchKernel(
        var.func,
        var.grid,
        1,
        1,
        var.nwarps * 32,
        1,
        1,
        var.smem,
        stream,
        args,
        0,
    )
    if int(res[0]) != 0:
        raise RuntimeError(f"cuLaunchKernel: {res[0]}")
    return out


def nvfp4_skinny_sm120(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    global _logged_fast_path
    if not use_nvfp4_skinny_sm120(
        input,
        weight,
        input_sf,
        weight_sf,
        alpha,
        out_dtype,
        out_features,
    ):
        raise ValueError("unsupported call for Qwen3.8 NVFP4 skinny GEMM")
    if not _logged_fast_path:
        logger.info("Using Qwen3.8 NVFP4 skinny GEMM on SM120")
        _logged_fast_path = True
    return _gemm(
        input,
        weight,
        input_sf,
        weight_sf,
        alpha,
        out_dtype,
        out_features,
    )


__all__ = ["nvfp4_skinny_sm120", "use_nvfp4_skinny_sm120"]
