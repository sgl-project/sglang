"""Nemotron 3 Super Mamba-2 decode mega kernel for RTX PRO 6000.

`run` issues exactly one CUDA device activity: a cooperative launch of
`mamba2_mega_kernel`, specialized on the (in_proj, out_proj) precision pair.
Everything else -- weight-pointer discovery, scale folding, scratch allocation,
nvcc compilation -- happens on the host, once per layer object.

The SGLang integration intentionally enables this only for TP=1, batch-1
decode. Full CUDA Graph capture is supported because SGLang runs two eager
warmups on the capture stream first, which completes JIT compilation, layer
binding, and scratch allocation. Prefill, speculative verify, tracked Mamba
cache, and other shapes keep using the production Mamba2 path.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "mamba2_mega.cu"
_ARCH = os.environ.get("KDA_MEGA_ARCH", "sm_120a")
# Launch geometry.  Defaults are the measured best on RTX PRO 6000 / sm_120a;
# the env overrides exist so tools/tune_geometry.py can sweep them.
_NWARPS = int(os.environ.get("KDA_MEGA_NWARPS", "8"))
_P1_ROWS = int(os.environ.get("KDA_MEGA_P1_ROWS", "1"))
_P3_ROWS = int(os.environ.get("KDA_MEGA_P3_ROWS", "1"))
_GRID_OVERRIDE = int(os.environ.get("KDA_MEGA_GRID", "0"))
_DEFINES = (
    f"-DNWARPS={_NWARPS}",
    f"-DP1_ROWS={_P1_ROWS}",
    f"-DP3_ROWS={_P3_ROWS}",
    f"-DWSTREAM={os.environ.get('KDA_MEGA_WSTREAM', '1')}",
    f"-DPREFETCH={os.environ.get('KDA_MEGA_PREFETCH', '0')}",
    f"-DMAX_PHASE={os.environ.get('KDA_MEGA_MAX_PHASE', '6')}",
    f"-DPF_OUT={os.environ.get('KDA_MEGA_PF_OUT', '2')}",
    f"-DPF_IN={os.environ.get('KDA_MEGA_PF_IN', '0')}",
    f"-DP2_STAGE={os.environ.get('KDA_MEGA_P2_STAGE', '0')}",
    f"-DP3_TRIM={os.environ.get('KDA_MEGA_P3_TRIM', '1')}",
    f"-DPF_OUT_FP8={os.environ.get('KDA_MEGA_PF_OUT_FP8', '1')}",
    f"-DPF_OUT_PAGES={os.environ.get('KDA_MEGA_PF_OUT_PAGES', '1')}",
    f"-DPF_OUT_COV={os.environ.get('KDA_MEGA_PF_OUT_COV', '12')}",
    f"-DPF_OUT_COV_BF16={os.environ.get('KDA_MEGA_PF_OUT_COV_BF16', '12')}",
    f"-DPF_OUT_EARLY={os.environ.get('KDA_MEGA_PF_OUT_EARLY', '0')}",
    f"-DPF_OUT_SKIP={os.environ.get('KDA_MEGA_PF_OUT_SKIP', '0')}",
    f"-DPF_P3={os.environ.get('KDA_MEGA_PF_P3', '0')}",
    f"-DPF_P3_FP8={os.environ.get('KDA_MEGA_PF_P3_FP8', '0')}",
    f"-DGNORM_P2={os.environ.get('KDA_MEGA_GNORM_P2', '1')}",
    f"-DNO_EPI={os.environ.get('KDA_MEGA_NO_EPI', '0')}",
)

HSZ = 4096
ISZ = 8192
NHEAD = 128
HDIM = 64
NGROUP = 8
DSTATE = 128
CONVDIM = 10240
CONVW = 4
PROJW = 18560


class _Params(ctypes.Structure):
    _fields_ = [
        ("hidden", ctypes.c_void_p),
        ("residual", ctypes.c_void_p),
        ("out_hidden", ctypes.c_void_p),
        ("out_residual", ctypes.c_void_p),
        ("norm_w", ctypes.c_void_p),
        ("in_w", ctypes.c_void_p),
        ("conv_w", ctypes.c_void_p),
        ("conv_b", ctypes.c_void_p),
        ("A", ctypes.c_void_p),
        ("D", ctypes.c_void_p),
        ("dt_bias", ctypes.c_void_p),
        ("gnorm_w", ctypes.c_void_p),
        ("out_w", ctypes.c_void_p),
        ("conv_state", ctypes.c_void_p),
        ("ssm_state", ctypes.c_void_p),
        ("cache_idx", ctypes.c_void_p),
        ("scratch", ctypes.c_void_p),
        ("in_alpha", ctypes.c_float),
        ("in_inv_scale", ctypes.c_float),
        ("out_alpha", ctypes.c_float),
        ("out_inv_scale", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("idx_offset", ctypes.c_int),
    ]


# ------------------------------------------------------------------ compilation


def _build_dir(digest: str) -> Path:
    for base in (os.environ.get("KDA_MEGA_BUILD_DIR"), tempfile.gettempdir(), "/tmp"):
        if not base:
            continue
        path = Path(base) / f"kda_mamba2_mega_{digest}"
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError("no writable build directory")


def _compile() -> ctypes.CDLL:
    source = _SRC.read_bytes()
    key = source + _ARCH.encode() + " ".join(_DEFINES).encode()
    digest = hashlib.sha256(key).hexdigest()[:16]
    out_dir = _build_dir(digest)
    lib_path = out_dir / "libmamba2_mega.so"
    if not lib_path.is_file():
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        tmp = lib_path.with_suffix(f".{os.getpid()}.so")
        cmd = [
            nvcc,
            "-O3",
            "-lineinfo",
            f"-arch={_ARCH}",
            "-std=c++17",
            *_DEFINES,
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "--cudart=shared",
            "-o",
            str(tmp),
            str(_SRC),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{proc.stdout}\n{proc.stderr}")
        os.replace(tmp, lib_path)
        print(f"[mamba2_mega] built {lib_path}", file=sys.stderr)
    lib = ctypes.CDLL(str(lib_path))
    lib.mamba2_mega_launch.restype = ctypes.c_int
    lib.mamba2_mega_launch.argtypes = [
        ctypes.POINTER(_Params),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.mamba2_mega_config.restype = ctypes.c_int
    lib.mamba2_mega_config.argtypes = [ctypes.c_int, ctypes.c_int] + [
        ctypes.POINTER(ctypes.c_int)
    ] * 5
    return lib


_LIB = None
_CONFIG: dict[tuple[int, int], tuple[int, int]] = {}
_SCRATCH: dict[tuple, torch.Tensor] = {}
_LAYERS: dict[int, dict] = {}
_ENABLED_LAYERS: set[int] = set()


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _compile()
    return _LIB


def _config(in_fp8: bool, out_fp8: bool) -> tuple[int, int]:
    """(grid, scratch_floats) for this specialization, from the compiled cubin."""
    key = (int(in_fp8), int(out_fp8))
    cached = _CONFIG.get(key)
    if cached is not None:
        return cached
    occ = ctypes.c_int()
    regs = ctypes.c_int()
    threads = ctypes.c_int()
    smem = ctypes.c_int()
    scratch = ctypes.c_int()
    rc = _lib().mamba2_mega_config(
        key[0],
        key[1],
        ctypes.byref(occ),
        ctypes.byref(regs),
        ctypes.byref(threads),
        ctypes.byref(smem),
        ctypes.byref(scratch),
    )
    if rc != 0:
        raise RuntimeError(f"mamba2_mega_config failed: cuda error {rc}")
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    resident = max(1, occ.value) * sms
    # Filling the residency limit is never the best grid.  Measured with
    # tools/grid_ab.py (5 interleaved repeats per point, so clock drift cancels):
    # 3.5 CTAs/SM is best or tied-best on all four rows at
    # 116.86/139.62/189.89/167.70 us, against 118.03/141.38/191.10/168.50 at the
    # residency-filling 940/940/752/940 -- -0.5% to -1.2% per row, +0.85% geomean.
    # A CTA that adds no bandwidth to a DRAM-bound kernel still pays for its
    # redundant P0 (24 KB of L2 traffic each) and widens both grid barriers.
    grid = _GRID_OVERRIDE or min(round(3.5 * sms), resident)
    if grid > resident:
        raise RuntimeError(
            f"grid {grid} exceeds the cooperative residency limit "
            f"{occ.value} blocks/sm * {sms} sm"
        )
    print(
        f"[mamba2_mega] in_fp8={key[0]} out_fp8={key[1]} regs={regs.value} "
        f"threads={threads.value} smem={smem.value} blocks/sm={occ.value} grid={grid}",
        file=sys.stderr,
    )
    value = (grid, scratch.value)
    _CONFIG[key] = value
    return value


def _scratch(device: torch.device, floats: int) -> torch.Tensor:
    """Phase-handoff workspace, one per (device, stream).

    The caller-owned conv/SSM state is never staged here -- the kernel updates the
    committed slot in place.  This holds only the projection/recurrence handoff,
    but keying it by stream keeps two concurrent layer instances from sharing it.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, torch.cuda.current_stream(device).cuda_stream)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < floats:
        buf = torch.empty(floats, dtype=torch.float32, device=device)
        _SCRATCH[key] = buf
    return buf


# --------------------------------------------------------------- layer binding


def _row_major(
    weight: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    """Return an [out_features, in_features] contiguous view of a linear weight."""
    if weight.shape == (out_features, in_features):
        if weight.is_contiguous():
            return weight
        return weight.contiguous()
    if weight.shape == (in_features, out_features):
        transposed = weight.t()
        if transposed.is_contiguous():
            return transposed
        return transposed.contiguous()
    raise RuntimeError(f"unexpected linear weight shape {tuple(weight.shape)}")


def _scale(module, name: str) -> float:
    value = getattr(module, name, None)
    if value is None:
        raise RuntimeError(f"quantized linear is missing {name}")
    flat = value.reshape(-1).float()
    lo = float(flat.min().item())
    hi = float(flat.max().item())
    if hi - lo > 1e-6 * max(hi, 1.0):
        raise RuntimeError(f"{name} is not per-tensor (min={lo}, max={hi})")
    return hi


def _bind(layer) -> dict:
    key = id(layer)
    cached = _LAYERS.get(key)
    if cached is not None:
        return cached

    mixer = layer.mixer
    in_proj = mixer.in_proj
    out_proj = mixer.out_proj
    in_w = _row_major(in_proj.weight.data, PROJW, HSZ)
    out_w = _row_major(out_proj.weight.data, HSZ, ISZ)
    in_fp8 = in_w.dtype == torch.float8_e4m3fn
    out_fp8 = out_w.dtype == torch.float8_e4m3fn
    if not in_fp8 and in_w.dtype != torch.bfloat16:
        raise RuntimeError(f"unsupported in_proj dtype {in_w.dtype}")
    if not out_fp8 and out_w.dtype != torch.bfloat16:
        raise RuntimeError(f"unsupported out_proj dtype {out_w.dtype}")

    in_alpha = in_inv = 1.0
    if in_fp8:
        s_in = _scale(in_proj, "input_scale")
        s_w = _scale(in_proj, "weight_scale")
        in_alpha = s_in * s_w
        in_inv = 1.0 / s_in
    out_alpha = out_inv = 1.0
    if out_fp8:
        s_in = _scale(out_proj, "input_scale")
        s_w = _scale(out_proj, "weight_scale")
        out_alpha = s_in * s_w
        out_inv = 1.0 / s_in

    # conv1d.weight is [conv_dim, 1, conv_kernel] after MambaMixer2's unsqueeze.
    conv_w = mixer.conv1d.weight.data.reshape(CONVDIM, -1)
    if conv_w.shape[1] != CONVW or not conv_w.is_contiguous():
        raise RuntimeError(
            f"unexpected conv1d weight {tuple(mixer.conv1d.weight.shape)}"
        )
    # The pre-norm and the gated norm share config.layer_norm_epsilon; the kernel
    # carries one value, so refuse to guess if the checkpoint ever splits them.
    eps = float(layer.norm.variance_epsilon)
    if float(mixer.norm.variance_epsilon) != eps:
        raise RuntimeError("pre-norm and gated-norm epsilon differ")
    # A is stored as -exp(A_log) in fp32 by the production weight loader.
    A = mixer.A.data
    if A.dtype != torch.float32:
        raise RuntimeError(f"expected fp32 A, got {A.dtype}")
    grid, scratch_floats = _config(in_fp8, out_fp8)

    info = {
        "in_w": in_w,
        "out_w": out_w,
        "in_fp8": int(in_fp8),
        "out_fp8": int(out_fp8),
        "in_alpha": in_alpha,
        "in_inv": in_inv,
        "out_alpha": out_alpha,
        "out_inv": out_inv,
        "norm_w": layer.norm.weight.data,
        "conv_w": conv_w,
        "conv_b": mixer.conv1d.bias.data,
        "A": A,
        "D": mixer.D.data,
        "dt_bias": mixer.dt_bias.data,
        "gnorm_w": mixer.norm.weight.data,
        "eps": eps,
        "grid": grid,
        "scratch_floats": scratch_floats,
    }
    _LAYERS[key] = info
    return info


# ------------------------------------------------------------------ entry point


def run(layer, hidden, residual, cache, metadata, out_hidden, out_residual):
    info = _bind(layer)
    conv_state = cache.conv[0]
    ssm_state = cache.temporal
    scratch = _scratch(hidden.device, info["scratch_floats"])

    p = _Params()
    p.hidden = hidden.data_ptr()
    p.residual = 0 if residual is None else residual.data_ptr()
    p.out_hidden = out_hidden.data_ptr()
    p.out_residual = out_residual.data_ptr()
    p.norm_w = info["norm_w"].data_ptr()
    p.in_w = info["in_w"].data_ptr()
    p.conv_w = info["conv_w"].data_ptr()
    p.conv_b = info["conv_b"].data_ptr()
    p.A = info["A"].data_ptr()
    p.D = info["D"].data_ptr()
    p.dt_bias = info["dt_bias"].data_ptr()
    p.gnorm_w = info["gnorm_w"].data_ptr()
    p.out_w = info["out_w"].data_ptr()
    p.conv_state = conv_state.data_ptr()
    p.ssm_state = ssm_state.data_ptr()
    p.cache_idx = metadata.mamba_cache_indices.data_ptr()
    p.scratch = scratch.data_ptr()
    p.in_alpha = info["in_alpha"]
    p.in_inv_scale = info["in_inv"]
    p.out_alpha = info["out_alpha"]
    p.out_inv_scale = info["out_inv"]
    p.eps = info["eps"]
    p.idx_offset = int(metadata.num_prefills)

    rc = _lib().mamba2_mega_launch(
        ctypes.byref(p),
        info["in_fp8"],
        info["out_fp8"],
        info["grid"],
        ctypes.c_void_p(torch.cuda.current_stream(hidden.device).cuda_stream),
    )
    if rc != 0:
        raise RuntimeError(f"cooperative launch failed: cuda error {rc}")
    return out_hidden, out_residual, conv_state, ssm_state


def maybe_run_sglang(layer, hidden, residual, forward_batch, backend):
    """Run the production fast path when the current request matches its contract."""
    if not forward_batch.forward_mode.is_decode():
        return None
    if tuple(hidden.shape) != (1, HSZ):
        return None
    if hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
        return None
    if residual is not None and (
        residual.dtype != torch.bfloat16
        or tuple(residual.shape) != (1, HSZ)
        or not residual.is_contiguous()
    ):
        return None
    if torch.cuda.get_device_capability(hidden.device) != (12, 0):
        return None

    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_size != 1:
        return None
    if forward_batch.mamba_track_mask is not None:
        return None

    metadata = backend.forward_metadata
    if (
        metadata is None
        or metadata.num_prefills != 0
        or metadata.num_prefill_tokens != 0
        or metadata.num_decodes != 1
        or metadata.is_target_verify
        or metadata.draft_token_num != 1
        or metadata.mamba_cache_indices.dtype != torch.int32
        or not metadata.mamba_cache_indices.is_contiguous()
    ):
        return None

    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_state = cache.conv[0]
    ssm_state = cache.temporal
    if (
        not conv_state.is_contiguous()
        or not ssm_state.is_contiguous()
        or tuple(conv_state.shape[1:]) != (CONVDIM, CONVW - 1)
        or tuple(ssm_state.shape[1:]) != (NHEAD, HDIM, DSTATE)
    ):
        return None

    out_hidden = torch.empty_like(hidden)
    out_residual = torch.empty_like(hidden)
    run(layer, hidden, residual, cache, metadata, out_hidden, out_residual)
    if layer.layer_id not in _ENABLED_LAYERS:
        print(
            f"[nemotron_mamba2_mega] active on layer {layer.layer_id}",
            file=sys.stderr,
        )
        _ENABLED_LAYERS.add(layer.layer_id)
    return out_hidden, out_residual


def prepare_benchmark(*inputs):
    _bind(inputs[0])
    _scratch(inputs[1].device, _LAYERS[id(inputs[0])]["scratch_floats"])
    return run, inputs, {}
