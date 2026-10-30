"""gfx950 small-M FP8 block-scale GEMM with fused activation quant."""

import ctypes
import os
import subprocess
import tempfile

import torch

from sglang.kernels.ops.moe.smallm_moe_gfx950 import _check, _hip_lib, _hipcc, _Kernel

_SRC = os.path.join(os.path.dirname(__file__), "smallm_fp8_blockscale_gemm.hip")
# (N, K) -> (max M, M tiles/block, N tiles/block, K steps/wave, waves),
# first match wins. Multiple blocks cover M when it exceeds the per-block tile.
SHAPES = {
    (3392, 6144): ((32, 1, 2, 6, 16),),
    (5120, 4096): ((32, 2, 2, 4, 16),),
    (4608, 4096): ((32, 2, 2, 4, 16),),
    (4096, 2048): (
        (8, 1, 1, 4, 8),
        (16, 1, 1, 2, 16),
        (32, 2, 1, 4, 8),
    ),
}
_mod = None
_fns = {}


class Args(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("x", "w", "ws", "out")] + [
        (n, ctypes.c_int32) for n in ("M", "N", "lda")
    ]


def try_smallm_fp8_bs_linear(x, WQ, w_scale):
    """Run the fused kernel, or return None for fallback."""
    global _mod
    (M, K), N = x.shape, WQ.shape[0]
    cfg = next((cfg for max_m, *cfg in SHAPES.get((N, K), ()) if M <= max_m), None)
    if not (
        _mod is not False
        and os.environ.get("SGLANG_ROCM_MIMO_FUSED_QKV", "0") == "1"
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and getattr(WQ, "is_shuffled", False)
        and w_scale.shape == ((N + 127) // 128, K // 128)
        and w_scale.is_contiguous()
        and cfg is not None
        and not torch.compiler.is_compiling()
    ):
        return None
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
            return None
    key = tuple(cfg)
    hip, fn = _hip_lib(), _fns.get(key)
    if fn is None:
        fn = _fns[key] = _Kernel(_mod, "smallm_fp8_bs_m{}_n{}_s{}_w{}".format(*key)).fn
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    args = Args(*(t.data_ptr() for t in (x, WQ, w_scale, out)), M, N, x.stride(0))
    params = (ctypes.c_void_p * 1)(ctypes.addressof(args))
    stream = torch.cuda.current_stream().cuda_stream
    grid_x = N // (16 * key[1])
    grid_y = (M + 16 * key[0] - 1) // (16 * key[0])
    block = 64 * key[3]
    rc = hip.hipModuleLaunchKernel(
        fn, grid_x, grid_y, 1, block, 1, 1, 0, stream, params, None
    )
    _check(rc, "launch")
    return out
