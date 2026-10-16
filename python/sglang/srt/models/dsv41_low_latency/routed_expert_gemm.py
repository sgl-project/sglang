# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Low-batch routed-expert GEMMs for DeepSeek-V4.1 decode.

A TP MoE layer (all experts local, intermediate 576
padded to 640) replaces FlashInfer's TRT-LLM-gen routing + two batched GEMMs by a route
kernel and a GEMM pair of 8-row tcgen05 block-scaled tiles (persistent schedule, PDL),
compiled with NVRTC on first use. An expert chosen by more than 8 rows gets several
consecutive tiles (route_wide.cu). The kernels read the same shuffled MXFP4 weights and
MXFP8 activations as FlashInfer and return its deferred-finalize triple, so the fused
finalize + all-reduce + mHC consumer is unchanged.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
from functools import cache
from pathlib import Path
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)

SOURCE_DIR = Path(__file__).resolve().parent / "routed_expert_gemm_src"

HIDDEN = 5120
SLICE = 576
PADDED = 640
MAX_EXPERTS = 384
MAX_TOP_K = 8
TILE = 8
MAX_ROWS = 8
WIDE_MAX_ROWS = 64  # route_wide.cu: one 64-bit row mask per expert
GEMM1_UNITS = 9  # 128-row tiles of w13 holding the 576-wide slice (10th is padding)
GEMM2_UNITS = HIDDEN // 128
ROUTE_THREADS = 64
ROUTE_WIDE_THREADS = 512
NVRTC_OPTIONS = ("-std=c++17", "-default-device", "--use_fast_math")
HEADERS = ("route.cuh",)
GEMM1_PARAMETERS = ("A", "SFA", "X", "XS", "route", "clamp", "H", "HS", "cta_words")
GEMM2_PARAMETERS = ("A", "SFA", "H", "HS", "route", "Y", "cta_words")
PARAMETERS = {"gemm1": GEMM1_PARAMETERS, "gemm2": GEMM2_PARAMETERS}


def max_ctas(rows: int, top_k: int, experts: int) -> int:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = min(experts, slots)
    return filled + (slots - filled) // TILE


def cta_words(ctas: int) -> int:
    return (ctas + 3) // 4 * 4


def route_words(ctas: int) -> int:
    """int32 length of the route record for ``ctas`` token tiles."""
    return 4 + 2 * cta_words(ctas) + TILE * ctas


def _driver():
    from cuda.bindings import driver

    return driver


def _check(result: tuple, what: str) -> tuple:
    if int(result[0]) != 0:
        raise RuntimeError(f"{what} failed: {result[0]!r}")
    return result[1:]


def _include_dirs() -> list[str]:
    candidates: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "include")
    candidates.append(Path("/usr/local/cuda/include"))
    try:
        import nvidia

        for base in nvidia.__path__:
            candidates.extend(sorted(Path(base).glob("*/include")))
    except ImportError:
        pass
    found: list[str] = []
    for directory in candidates:
        if (directory / "cuda.h").is_file() and str(directory) not in found:
            found.append(str(directory))
            if (directory / "cccl" / "cuda" / "std").is_dir():
                found.append(str(directory / "cccl"))
    return found


def nvrtc_version() -> tuple[int, int]:
    from cuda.bindings import nvrtc

    major, minor = _check(nvrtc.nvrtcVersion(), "nvrtcVersion")
    return int(major), int(minor)


def _cache_dir() -> Path:
    from sglang.srt.environ import envs

    return (
        Path(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()))
        / "dsv41_routed_expert_gemm"
    )


def compile_source(
    source: str,
    name: str,
    arch: str,
    options: tuple[str, ...] = NVRTC_OPTIONS,
    headers: Optional[dict[str, str]] = None,
) -> bytes:
    """Compile ``source`` to a cubin for ``arch`` with NVRTC, cached on disk by the
    hash of source, headers, options and NVRTC version."""
    from cuda.bindings import nvrtc

    headers = dict(headers or {})
    options = [f"--gpu-architecture={arch}", *options]
    options += [f"-I{d}" for d in _include_dirs()]
    if nvrtc_version() >= (12, 9):
        options.append("--no-cache")
    header_text = [f"{k}\0{v}" for k, v in sorted(headers.items())]
    key = hashlib.sha256(
        "\0".join([source, *header_text, *options, repr(nvrtc_version())]).encode()
    ).hexdigest()
    cached = _cache_dir() / f"{name}-{key[:24]}.cubin"
    if cached.is_file():
        return cached.read_bytes()
    (program,) = _check(
        nvrtc.nvrtcCreateProgram(
            source.encode(),
            f"{name}.cu".encode(),
            len(headers),
            [text.encode() for text in headers.values()],
            [header.encode() for header in headers],
        ),
        "nvrtcCreateProgram",
    )
    try:
        encoded = [option.encode() for option in options]
        (status,) = nvrtc.nvrtcCompileProgram(program, len(encoded), encoded)
        if int(status) != 0:
            (size,) = _check(nvrtc.nvrtcGetProgramLogSize(program), "log size")
            log = b" " * size
            _check(nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
            raise RuntimeError(
                f"NVRTC failed to compile {name}: {log.decode(errors='replace')}"
            )
        (size,) = _check(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
        cubin = b" " * size
        _check(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(cubin)
    temporary.replace(cached)
    return cubin


class _Kernel(NamedTuple):
    function: object
    block: tuple[int, int, int]
    shared_memory_bytes: int
    cubin: bytes


def _load(cubin: bytes, symbol: str, block, shared: int) -> _Kernel:
    cu = _driver()
    (module,) = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
    (function,) = _check(cu.cuModuleGetFunction(module, symbol.encode()), symbol)
    if shared > 48 * 1024:
        _check(
            cu.cuFuncSetAttribute(
                function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            ),
            "cuFuncSetAttribute",
        )
    return _Kernel(function, tuple(block), shared, cubin)


@cache
def load_kernels(device_index: int) -> dict[str, _Kernel]:
    """Compile (or read from the cache) and load the route kernel and GEMMs."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        arch = f"sm_{major}{minor}a"
        facts = json.loads((SOURCE_DIR / "launch.json").read_text())
        headers = {name: (SOURCE_DIR / name).read_text() for name in HEADERS}
        kernels = {
            "route": _load(
                compile_source(
                    (SOURCE_DIR / "route.cu").read_text(),
                    "route",
                    arch,
                    headers=headers,
                ),
                "dsv41_routed_expert_route",
                (ROUTE_THREADS, 1, 1),
                0,
            )
        }
        for name, fact in facts.items():
            if tuple(fact["parameters"]) != PARAMETERS.get(name):
                raise RuntimeError(
                    f"{name}: unexpected launch.json parameters {fact['parameters']}"
                )
            source = (SOURCE_DIR / f"{name}.cu").read_text()
            kernels[name] = _load(
                compile_source(source, name, arch),
                fact["symbol"],
                fact["block"],
                int(fact["shared_memory_bytes"]),
            )
        return kernels


@cache
def load_route_wide(device_index: int) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        source = (SOURCE_DIR / "route_wide.cu").read_text()
        cubin = compile_source(source, "route_wide", f"sm_{major}{minor}a")
        return _load(
            cubin, "dsv41_routed_expert_route_wide", (ROUTE_WIDE_THREADS, 1, 1), 0
        )


def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _launch(
    kernel: _Kernel, grid: tuple[int, int, int], arguments: list, stream: int
) -> None:
    """Launch with programmatic stream serialization (PDL); graph-capturable."""
    cu = _driver()
    addresses = [
        argument.getPtr()
        if isinstance(argument, cu.CUtensorMap)
        else ctypes.addressof(argument)
        for argument in arguments
    ]
    pointer_array = (ctypes.c_void_p * len(addresses))(*addresses)
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = grid
    config.blockDimX, config.blockDimY, config.blockDimZ = kernel.block
    config.sharedMemBytes = kernel.shared_memory_bytes
    config.hStream = cu.CUstream(stream)
    attribute = cu.CUlaunchAttribute()
    attribute.id = (
        cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    )
    attribute.value.programmaticStreamSerializationAllowed = 1
    config.attrs = [attribute]
    config.numAttrs = 1
    _check(
        cu.cuLaunchKernelEx(
            config, kernel.function, ctypes.addressof(pointer_array), 0
        ),
        "cuLaunchKernelEx",
    )


def _weight_map(weights: torch.Tensor, m: int, k: int, chunks: int):
    """TMA map of MXFP4 weights ``[E, m, k / 2]`` (16U4, 128B swizzle)."""
    cu = _driver()
    if weights.data_ptr() % 32:
        raise ValueError("packed expert weights must be 32-byte aligned")
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
            4,
            weights.data_ptr(),
            [u64(128), u64(m), u64(k // 128), u64(weights.size(0))],
            [u64(k // 2), u64(64), u64(m * k // 2)],
            [u32(128), u32(128), u32(chunks), u32(1)],
            [u32(1)] * 4,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(weights)",
    )
    return tensor_map


def _scale_map(scales: torch.Tensor, groups: int):
    """TMA map of 128x4-layout scale bytes viewed as ``[bytes/256, 256]``."""
    cu = _driver()
    total = scales.numel() * scales.element_size()
    if total % 256 or scales.data_ptr() % 16:
        raise ValueError("weight scales must be whole 256-byte rows, 16-byte aligned")
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            scales.data_ptr(),
            [u64(256), u64(total // 256)],
            [u64(256)],
            [u32(256), u32(2 * groups)],
            [u32(1)] * 2,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(scales)",
    )
    return tensor_map


def check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k: int) -> None:
    """Raise ValueError unless the weights have the layout the kernels read."""
    experts = w13.size(0)
    expected = (
        (w13, (experts, 2 * PADDED, HIDDEN // 2)),
        (w13_scale, (experts, 2 * PADDED, HIDDEN // 32)),
        (w2, (experts, HIDDEN, PADDED // 2)),
        (w2_scale, (experts, HIDDEN, PADDED // 32)),
    )
    for tensor, shape in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected a contiguous {shape} 1-byte tensor, got {tuple(tensor.shape)} {tensor.dtype}"
            )
    if experts % 32 or not 0 < experts <= MAX_EXPERTS:
        raise ValueError(
            f"expected a multiple of 32 local experts <= {MAX_EXPERTS}, got {experts}"
        )
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"expected a top-k of 1-{MAX_TOP_K}, got {top_k}")
    if (
        clamp is None
        or clamp.dtype != torch.float32
        or tuple(clamp.shape) != (experts,)
    ):
        raise ValueError("expected an FP32 [E] SwiGLU clamp limit")
    # GEMM1 never reads w13's padding tile, so the padding must be zero, as must the
    # matching K columns of w2.
    if (
        torch.count_nonzero(w13.view(torch.uint8)[:, 2 * SLICE :]).item()
        or torch.count_nonzero(w2.view(torch.uint8)[:, :, SLICE // 2 :]).item()
    ):
        raise ValueError("the intermediate padding of w13/w2 is not zero")


class RoutedExpertGemm:
    """The route kernel and GEMM pair bound to one MoE layer's weights."""

    def __init__(self, w13, w13_scale, w2, w2_scale, clamp, top_k: int, max_rows: int):
        w13, w13_scale = w13.view(torch.uint8), w13_scale.view(torch.uint8)
        w2, w2_scale = w2.view(torch.uint8), w2_scale.view(torch.uint8)
        check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k)
        if not MAX_ROWS <= max_rows <= WIDE_MAX_ROWS:
            raise ValueError(
                f"max_rows {max_rows} is outside {MAX_ROWS}-{WIDE_MAX_ROWS}"
            )
        self.experts = int(w13.size(0))
        self.top_k = top_k
        self.max_rows = max_rows
        self.kernels = load_kernels(w13.device.index)
        self.route_wide = (
            load_route_wide(w13.device.index) if max_rows > MAX_ROWS else None
        )
        # route.cu flips bit 31 of this word on every launch.
        self.barrier = torch.zeros(1, dtype=torch.int32, device=w13.device)
        self._tensors = (
            w13,
            w13_scale,
            w2,
            w2_scale,
            clamp,
        )  # the maps hold their addresses
        self.clamp = clamp
        self.gemm1_maps = (
            _weight_map(w13, 2 * PADDED, HIDDEN, 2),
            _scale_map(w13_scale, 4),
        )
        self.gemm2_maps = (_weight_map(w2, HIDDEN, PADDED, 2), _scale_map(w2_scale, 1))

    def supports(self, x_quant, x_scale, topk_output) -> bool:
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        if not TopKOutputChecker.format_is_standard(topk_output):
            return False
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
        rows = x_quant.size(0)
        return (
            1 <= rows <= self.max_rows
            and topk_ids.dim() == 2
            and topk_ids.size(1) == self.top_k
            and tuple(x_quant.shape) == (rows, HIDDEN)
            and x_quant.element_size() == 1
            and x_quant.is_contiguous()
            and x_scale is not None
            and x_scale.element_size() == 1
            and x_scale.is_contiguous()
            and x_scale.numel() >= rows * HIDDEN // 32
            and topk_weights.dtype == torch.float32
        )

    def apply(self, x_quant, x_scale, topk_output):
        """The MoE output as FlashInfer's deferred finalize returns it."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        gemm2_out, expert_weights, permuted = self(
            x_quant, x_scale, topk_output.topk_ids, topk_output.topk_weights
        )
        return StandardCombineInput(
            hidden_states=FlashInferTrtllmDeferredFinalizeOutput(
                gemm2_out=gemm2_out,
                expert_weights=expert_weights,
                # Flat [rows * top_k] slot -> row map, as FlashInfer's routed MoE returns it.
                expanded_idx_to_permuted_idx=permuted.view(-1),
                top_k=self.top_k,
            )
        )

    def __call__(self, x_quant, x_scale, topk_ids, topk_weights):
        """Route + GEMM1 + GEMM2 on the current stream; returns FlashInfer's
        deferred-finalize triple (gemm2 rows, weights, slot -> permuted row)."""
        rows, top_k = topk_ids.shape
        device = x_quant.device
        stream = torch.cuda.current_stream(device).cuda_stream
        ids = topk_ids.to(torch.int32).contiguous()
        ctas = max_ctas(rows, top_k, self.experts)
        permuted = torch.empty((rows, top_k), dtype=torch.int32, device=device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=device)
        if rows > MAX_ROWS:
            _launch(
                self.route_wide,
                (1, 1, 1),
                [
                    _pointer(ids),
                    _pointer(permuted),
                    _pointer(record),
                    ctypes.c_int32(rows),
                    ctypes.c_int32(top_k),
                    ctypes.c_int32(self.experts),
                ],
                stream,
            )
        else:
            _launch(
                self.kernels["route"],
                (1, 1, 1),
                [
                    _pointer(ids),
                    _pointer(permuted),
                    _pointer(record),
                    _pointer(self.barrier),
                    ctypes.c_int32(rows),
                    ctypes.c_int32(top_k),
                    ctypes.c_int32(self.experts),
                ],
                stream,
            )
        padded_rows = TILE * ctas
        h_values = torch.empty((padded_rows, PADDED), dtype=torch.uint8, device=device)
        h_scales = torch.empty(
            (padded_rows + 127) // 128 * 128 * PADDED // 32,
            dtype=torch.uint8,
            device=device,
        )
        gemm2_rows = torch.empty(
            (rows * top_k * TILE, HIDDEN), dtype=torch.bfloat16, device=device
        )
        words = ctypes.c_int32(cta_words(ctas))
        _launch(
            self.kernels["gemm1"],
            (GEMM1_UNITS, ctas, 1),
            [
                *self.gemm1_maps,
                _pointer(x_quant),
                _pointer(x_scale),
                _pointer(record),
                _pointer(self.clamp),
                _pointer(h_values),
                _pointer(h_scales),
                words,
            ],
            stream,
        )
        _launch(
            self.kernels["gemm2"],
            (GEMM2_UNITS, ctas, 1),
            [
                *self.gemm2_maps,
                _pointer(h_values),
                _pointer(h_scales),
                _pointer(record),
                _pointer(gemm2_rows),
                words,
            ],
            stream,
        )
        return gemm2_rows, topk_weights.contiguous().view(rows, top_k), permuted


def build(layer, clamp) -> Optional[RoutedExpertGemm]:
    """The pair for a TP MoE layer's processed weights; None (logged) when the
    kernels cannot take them, and FlashInfer serves every batch."""
    from sglang.srt.models.dsv41_low_latency.switch import ROUTED_EXPERT_MAX_ROWS

    try:
        if layer.num_local_experts != layer.num_experts:
            raise ValueError("expert parallelism: the pair needs every expert local")
        return RoutedExpertGemm(
            layer.w13_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight,
            layer.w2_weight_scale_inv,
            clamp,
            top_k=layer.top_k,
            max_rows=ROUTED_EXPERT_MAX_ROWS,
        )
    except (ValueError, ImportError, OSError, RuntimeError) as error:
        logger.warning(
            f"DSV4.1 routed-expert GEMMs not installed, FlashInfer kept: "
            f"{type(error).__name__}: {error}"
        )
        return None
