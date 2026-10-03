"""gfx950 small-M FP8 block-scale linear: aiter's 1x128 activation quant fused into a GEMM over the (16, 16)-
preshuffled weight. hipcc-built at first use like smallm_moe_gfx950; SGLANG_ROCM_SMALLM_FP8_BS=0 turns it off."""

import ctypes
import os
import subprocess
import tempfile

import torch

from sglang.kernels.ops.moe.smallm_moe_gfx950 import _check, _hip_lib, _hipcc

_SRC = os.path.join(os.path.dirname(__file__), "smallm_fp8_blockscale_gemm.hip")
# (N, K) -> ((max M, n-tiles per block, k steps per wave, waves), ...), first match wins. Qwen3.5-397B-A17B-FP8 TP4:
# packed GDN in_proj_qkvz, attention qkv_proj, GDN out_proj / attention o_proj.
SHAPES = {
    (5120, 4096): ((32, 2, 4, 16),),
    (4608, 4096): ((32, 2, 4, 16),),
    (4096, 2048): ((8, 1, 4, 8), (16, 1, 2, 16), (32, 1, 4, 8)),
}
_mod = None
_fns = {}


class Args(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("x", "w", "ws", "out")] + [
        (n, ctypes.c_int32) for n in ("M", "N", "lda")
    ]


def smallm_fp8_bs_enabled() -> bool:
    global _mod
    if os.environ.get("SGLANG_ROCM_SMALLM_FP8_BS", "1") == "0":
        return False
    if _mod is None:
        co = os.path.join(tempfile.mkdtemp(), "k.co")
        cmd = [_hipcc(), "--genco", "--offload-arch=gfx950", "-O3", "-o", co, _SRC]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            _mod = ctypes.c_void_p()
            _check(_hip_lib().hipModuleLoad(ctypes.byref(_mod), co.encode()), "load")
        except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
            _mod = False
            print(f"[smallm_fp8_bs] disabled: {e}", flush=True)
    return _mod is not False


def _config(M, N, K):
    return next((cfg for max_m, *cfg in SHAPES.get((N, K), ()) if M <= max_m), None)


def smallm_fp8_bs_supported(x, WQ, w_scale) -> bool:
    """bf16 x [M, K] with 16 B aligned rows, WQ aiter-shuffled fp8 [N, K], contiguous w_scale [N / 128, K / 128]."""
    (M, K), N = x.shape, WQ.shape[0]
    return (
        x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and getattr(WQ, "is_shuffled", False)
        and w_scale.shape == (N // 128, K // 128)
        and w_scale.is_contiguous()
        and _config(M, N, K) is not None
        and not torch.compiler.is_compiling()
        and smallm_fp8_bs_enabled()
    )


def smallm_fp8_bs_linear(x, WQ, w_scale):
    """aiter per-1x128 quant + gemm_a8w8_blockscale_bpreshuffle in one kernel; same fp8 bytes and scales, bf16 out."""
    (M, K), N = x.shape, WQ.shape[0]
    key = ((M + 15) // 16, *_config(M, N, K))
    hip, fn = _hip_lib(), _fns.get(key)
    if fn is None:
        fn = _fns[key] = ctypes.c_void_p()
        name = "smallm_fp8_bs_m{}_n{}_s{}_w{}".format(*key)
        _check(hip.hipModuleGetFunction(ctypes.byref(fn), _mod, name.encode()), name)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    args = Args(*(t.data_ptr() for t in (x, WQ, w_scale, out)), M, N, x.stride(0))
    params = (ctypes.c_void_p * 1)(ctypes.addressof(args))
    stream = torch.cuda.current_stream().cuda_stream
    grid, block = N // (16 * key[1]), 64 * key[3]
    rc = hip.hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, params, None)
    _check(rc, "launch")
    return out
