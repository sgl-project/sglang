"""gfx950 small-M MXFP4 fused-MoE kernel for the ROCm aiter MoE path (Qwen3.5-397B-A17B TP4 shape: hidden 4096, per-rank intermediate 256).

Two HIP kernels (source `smallm_moe.hip` next to this file, compiled with hipcc at first use) replace
aiter.fused_moe for small token counts:
  p1  : gate/up GEMM + silu*up per (expert, column slice) with expert dedup, bf16 activations, fp4->bf16 dequant
        in registers, MFMA 16x16x32 bf16; writes h[tok][slot][inter] (bf16).
  p2  : down GEMM per (token, 64-row slice): the token's h rows staged in LDS, all 11 experts' w2 tiles in flight
        up front, fp32 dot products (v_dot2_f32_bf16), one bf16 store per output element (no atomics).
Weights/scales are consumed in aiter's production layouts (shuffle_weight(layout=(16,16)) and e8m0_shuffle), so
the checkpoint-loading path is untouched. Activations are NOT quantized (production quantizes them to MXFP4).

On by default; SGLANG_ROCM_SMALLM_MOE=0 turns it off. Any build/load failure disables it for the process
(aiter path is used).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile

import torch

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    find_loaded_library,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "smallm_moe.hip")
_ENV = "SGLANG_ROCM_SMALLM_MOE"
DIM = 4096
MAX_TOK = 64  # kernel list capacity (tokens per expert) and workspace size
# Dispatch cap: above this token count aiter's flydsl path is faster (crossover ~44
# tokens at per-rank intermediate 256 / TP4). Only that shape is dispatched here; the
# TP2 shape (intermediate 512) has a lower crossover and is left to a follow-up.
MAX_TOK_DISPATCH = {256: 40}

_hip = None
_kernels: dict = {}
_available = None
_disabled_reason = None
_build_dir = None
_ws: dict = {}


class SmallMMoeUnavailable(RuntimeError):
    pass


class Args(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        (n, ctypes.c_void_p)
        for n in (
            "x",
            "w1",
            "s1",
            "w2",
            "s2",
            "ids",
            "wts",
            "hbuf",
            "ctrl",
            "out",
            "acc",
        )
    ] + [
        ("tok", ctypes.c_int32),
        ("slots", ctypes.c_int32),
        ("E", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


assert ctypes.sizeof(Args) == 104, ctypes.sizeof(
    Args
)  # must match static_assert(sizeof(Args) == 104) in smallm_moe.hip


def _hip_lib():
    global _hip
    if _hip is None:
        # Bind to the HIP runtime torch already mapped. ROCm 10 ships a second
        # libamdhip64 in _rocm_sdk_devel; an unversioned CDLL("libamdhip64.so")
        # picks that one, and every launch on a torch stream then fails with
        # hipErrorContextIsDestroyed (709).
        torch.cuda.current_device()
        _hip = ctypes.CDLL(find_loaded_library("libamdhip64") or "libamdhip64.so")
        _hip.hipModuleLoad.restype = ctypes.c_int
        _hip.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _hip.hipModuleGetFunction.restype = ctypes.c_int
        _hip.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _hip.hipModuleLaunchKernel.restype = ctypes.c_int
        _hip.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        _hip.hipGetErrorString.restype = ctypes.c_char_p
    return _hip


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(
            f"smallm_moe: {what} failed: {rc} ({_hip_lib().hipGetErrorString(rc).decode()})"
        )


class _Kernel:
    def __init__(self, mod, name):
        self.fn = ctypes.c_void_p()
        _check(
            _hip_lib().hipModuleGetFunction(ctypes.byref(self.fn), mod, name.encode()),
            f"hipModuleGetFunction {name}",
        )

    def launch(self, grid, args, stream):
        size = ctypes.c_size_t(ctypes.sizeof(args))
        extra = (ctypes.c_void_p * 5)(
            1,
            ctypes.cast(ctypes.byref(args), ctypes.c_void_p).value,
            2,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value,
            3,
        )
        _check(
            _hip_lib().hipModuleLaunchKernel(
                self.fn,
                grid,
                1,
                1,
                256,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                None,
                ctypes.cast(extra, ctypes.c_void_p),
            ),
            "hipModuleLaunchKernel",
        )


def _hipcc():
    return os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")


def _hip_version():
    if torch.version.hip:
        return tuple(int(x) for x in torch.version.hip.split("-")[0].split(".")[:3])
    return (0, 0, 0)


def smallm_moe_enabled() -> bool:
    """True unless SGLANG_ROCM_SMALLM_MOE=0, when the device is exactly gfx950 on ROCm >= 7.2 (the kernel uses
    v_cvt_scalef32_pk_bf16_fp4 / v_dot2_f32_bf16 and needs that toolchain), hipcc is present, and no build/load
    failure has disabled the kernel in this process. Numerics differ slightly from the aiter path (bf16 activations
    instead of MXFP4 a4w4); GSM8K matches within run-to-run noise.

    Model coverage: the kernel is compiled for one shape only, Qwen3.5-397B-A17B MXFP4 at TP4 (hidden 4096,
    per-rank intermediate 256, top-10 plus the optional fused shared expert). There is no model-name check;
    smallm_moe_supported() enforces the shape, and every other model or parallel layout keeps aiter fused_moe."""
    global _available
    if os.environ.get(_ENV, "1") == "0":
        return False
    if _available is None:
        arch = ""
        if torch.version.hip and torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        _available = (
            arch == "gfx950"
            and _hip_version() >= (7, 2, 0)
            and os.path.exists(_hipcc())
        )
        if not _available:
            print(
                f"[smallm_moe] not enabled (arch={arch!r}, hip={_hip_version()}, hipcc={os.path.exists(_hipcc())}); aiter fused_moe stays in use",
                flush=True,
            )
    return _available and _disabled_reason is None


def _disable(reason):
    global _disabled_reason
    _disabled_reason = reason
    print(
        f"[smallm_moe] disabling the gfx950 small-M MoE kernel, falling back to aiter fused_moe: {reason}",
        flush=True,
    )
    raise SmallMMoeUnavailable(reason)


def _get_kernels(inter: int):
    kk = _kernels.get(inter)
    if kk is None:
        global _build_dir
        if _build_dir is None:
            _build_dir = tempfile.mkdtemp(prefix="sglang_smallm_moe_")
            co = os.path.join(_build_dir, "smallm_moe.co")
            try:
                subprocess.run(
                    [
                        _hipcc(),
                        "--genco",
                        "--offload-arch=gfx950",
                        "-O3",
                        "-o",
                        co,
                        _SRC,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as e:
                detail = getattr(e, "stderr", None) or str(e)
                _disable(f"hipcc failed: {detail.strip()[-400:]}")
            mod = ctypes.c_void_p()
            try:
                _check(
                    _hip_lib().hipModuleLoad(ctypes.byref(mod), co.encode()),
                    "hipModuleLoad",
                )
            except RuntimeError as e:
                _disable(str(e))
            _kernels["_mod"] = mod
            print(
                f"[smallm_moe] gfx950 small-M MoE kernels active (built {co})",
                flush=True,
            )
        mod = _kernels["_mod"]
        try:
            kk = {
                "p1": _Kernel(mod, f"smallm_p1_i{inter}_c16"),
                "p2": {
                    sl: _Kernel(mod, f"smallm_p2_i{inter}_s{sl}") for sl in (10, 11)
                },
            }
        except RuntimeError as e:
            _disable(str(e))
        _kernels[inter] = kk
    return kk


def _workspace(device, slots: int, inter: int):
    key = (device, slots, inter)
    ws = _ws.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # never allocate under graph capture (the captured tensors would be garbage)
        ws = (
            torch.zeros(MAX_TOK * slots * inter, dtype=torch.bfloat16, device=device),
            torch.zeros(MAX_TOK * DIM, dtype=torch.float32, device=device),
            torch.zeros(1024, dtype=torch.int32, device=device),
        )
        _ws[key] = ws
    return ws


def smallm_moe_supported(
    hidden_states,
    w13,
    w2,
    topk_ids,
    expert_mask,
    doweight_stage1,
    activation_is_silu,
    has_bias,
    a1_scale,
) -> bool:
    if not smallm_moe_enabled():
        return False
    tok = hidden_states.shape[0]
    if (
        tok < 1
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] != DIM
    ):
        return False
    if (
        w13.dim() != 3
        or w2.dim() != 3
        or w13.element_size() != 1
        or w2.element_size() != 1
    ):
        return False
    inter = w2.shape[2] * 2
    if (
        inter not in MAX_TOK_DISPATCH
        or w13.shape[1] != 2 * inter
        or w13.shape[2] != DIM // 2
        or w2.shape[1] != DIM
    ):
        return False
    if tok > min(MAX_TOK, MAX_TOK_DISPATCH[inter]):
        return False
    # 704 = MAX_TOK (64) * 11 slots: the kernels stage the whole topk_ids table in a
    # fixed-size LDS array (MAX_NSLOT in smallm_moe.hip); larger tables would overrun it.
    if topk_ids.shape[1] not in (10, 11) or tok * topk_ids.shape[1] > MAX_TOK * 11:
        return False
    if (
        expert_mask is not None
        or doweight_stage1
        or not activation_is_silu
        or has_bias
        or a1_scale is not None
    ):
        return False
    return True


def smallm_moe_fwd(hidden_states, w13, w2, topk_weights, topk_ids, w13_scale, w2_scale):
    """out[tok, 4096] (bf16) = sum_j w_tj * down_e(silu(gate_e(x_t)) * up_e(x_t)); e = topk_ids[t, j].
    w13/w2: fp4x2 [E, 2*inter, 2048] / [E, 4096, inter/2] in aiter shuffle_weight((16,16)) layout;
    w13_scale/w2_scale: e8m0 in e8m0_shuffle layout. Returns None if the workspace cannot be allocated (capture)."""
    tok, slots = topk_ids.shape
    E = w13.shape[0]
    inter = w2.shape[2] * 2
    ws = _workspace(hidden_states.device, slots, inter)
    if ws is None:
        return None
    kk = _get_kernels(inter)
    hbuf, acc, cnt = ws
    ids = (
        topk_ids
        if (topk_ids.dtype == torch.int32 and topk_ids.is_contiguous())
        else topk_ids.to(torch.int32).contiguous()
    )
    wts = (
        topk_weights
        if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous())
        else topk_weights.to(torch.float32).contiguous()
    )
    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    out = torch.empty(tok, DIM, dtype=torch.bfloat16, device=hidden_states.device)
    args = Args(
        x.data_ptr(),
        w13.data_ptr(),
        w13_scale.data_ptr(),
        w2.data_ptr(),
        w2_scale.data_ptr(),
        ids.data_ptr(),
        wts.data_ptr(),
        hbuf.data_ptr(),
        cnt.data_ptr(),
        out.data_ptr(),
        acc.data_ptr(),
        tok,
        slots,
        E,
        0,
    )
    stream = torch.cuda.current_stream().cuda_stream
    try:
        kk["p1"].launch(
            tok * slots * (inter // 16), args, stream
        )  # (expert-slot, 16-column slice) blocks
        kk["p2"][slots].launch(
            tok * (DIM // 64), args, stream
        )  # (token, 64-row slice) blocks
    except RuntimeError as e:
        if torch.cuda.is_current_stream_capturing():
            raise  # a failed launch inside capture cannot be undone; surface it
        _disable(f"kernel launch failed: {e}")
    return out
