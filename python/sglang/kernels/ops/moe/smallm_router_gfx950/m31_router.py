"""gfx950 one-launch MiniMax-M3.1 decode router for 1..16 tokens, bitwise identical to router_gemv ->
moe_fused_gate (sigmoid, correction bias, top-4, renormalize, routed scaling) -> aiter's MoE sort (block_m 16).
The sorted buffers reach the aiter runner through moe_sorting_small.presorted_moe_sort().
m31_shared_router() runs it in one launch with the shared expert's gate_up GEMM.
SGLANG_ROCM_SMALLM_ROUTER=0 disables it; a build/load failure disables it too."""

import ctypes
import hashlib
import logging
import os
import pathlib
import subprocess
import tempfile

import torch

from sglang.kernels.jit.utils.compile.cache import cache_root
from sglang.kernels.jit.utils.hipcc import HipKernel, check_hip, hip_lib, hipcc_path
from sglang.kernels.ops.gemm import router_gemv, smallm_fp8_gfx950
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported

logger = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m31_router.hip")
# the GEMM body m31_router.hip includes
_GEMM_SRC = os.path.join(
    os.path.dirname(os.path.abspath(smallm_fp8_gfx950.__file__)), "smallm_fp8_gemm.hip"
)
_HIPCC_FLAGS = ("--genco", "--offload-arch=gfx950", "-O3")
MAX_TOK = 16
NUM_EXPERTS, TOPK, HIDDEN, BLOCK_M = 128, 4, 6144, 16
# router_gemv tiles (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, warps) the kernels replicate, with the kernel and grid;
# a different router_gemv tiling would change the logits' rounding, so it disables this path.
_TILES = {
    8: ((16, 32, 256, 12, 8), "m31_router_t8", 48),
    16: ((16, 16, 512, 8, 4), "m31_router_t16", 64),
}
_MAX_SPLIT = 12
SHARED_N = 1536
# the fused kernel per router tile: name, router CTAs, the shared expert gate_up's smallm_fp8_gemm config
# (n-tiles, k steps, waves, LDS-staged x) it runs on the remaining CTAs
_SHARED = {
    8: ("m31_shared_router_t8", 48, (1, 12, 8, 1)),
    16: ("m31_shared_router_t16", 64, (1, 8, 12, 1)),
}
_kernels: dict = {}
_ws: dict = {}
_disabled = False


class Args(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        (n, ctypes.c_void_p)
        for n in (
            "x",
            "w",
            "bias",
            "part",
            "cnt",
            "logits",
            "topk_w",
            "topk_ids",
            "sorted_ids",
            "sorted_w",
            "sorted_eids",
            "num_valid",
        )
    ] + [
        ("tok", ctypes.c_int32),
        ("x_stride", ctypes.c_int32),
        ("scale", ctypes.c_float),
    ]


assert ctypes.sizeof(Args) == 108, (
    "kernarg ABI drift: keep Args in step with m31_router.hip"
)


class SharedArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("r", Args), ("g", smallm_fp8_gfx950.Args)]


assert ctypes.sizeof(SharedArgs) == 164, (
    "kernarg ABI drift: keep SharedArgs in step with m31_router.hip"
)


def _token_bucket(m: int) -> int:
    return 8 if m <= 8 else 16


def _tile(m: int):
    return _TILES[_token_bucket(m)]


def m31_router_supported(
    x: torch.Tensor, gate_weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    m = x.shape[0]
    return (
        not _disabled
        and envs.SGLANG_ROCM_SMALLM_ROUTER.get()
        and is_gfx95_supported()
        and x.dim() == 2
        and 1 <= m <= MAX_TOK
        and x.shape[1] == HIDDEN
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and gate_weight.dtype == torch.bfloat16
        and gate_weight.shape == (NUM_EXPERTS, HIDDEN)
        and gate_weight.is_contiguous()
        and bias is not None
        and bias.dtype == torch.float32
        and bias.shape == (NUM_EXPERTS,)
        and bias.is_contiguous()
        and router_gemv._config(m, gate_weight.dtype) == _tile(m)[0]
    )


def _code_object() -> pathlib.Path:
    """m31_router.hip built once per source, flags and ROCm under the JIT cache, so
    restarts and the other ranks load it instead of running hipcc again."""
    digest = hashlib.sha256(repr((_HIPCC_FLAGS, torch.version.hip)).encode())
    for src in (_SRC, _GEMM_SRC):
        digest.update(pathlib.Path(src).read_bytes())
    cache_dir = cache_root() / "hipcc"
    co = cache_dir / f"m31_router-{digest.hexdigest()[:16]}.co"
    if co.is_file():
        return co
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as build_dir:
        built = os.path.join(build_dir, "r.co")
        cmd = [hipcc_path(), *_HIPCC_FLAGS, "-o", built, _SRC]
        subprocess.run(cmd, check=True, capture_output=True)
        # ranks building at once each publish the same bytes
        os.replace(built, co)
    return co


def _load():
    global _disabled
    try:
        co = str(_code_object())
        mod = ctypes.c_void_p()
        check_hip(hip_lib().hipModuleLoad(ctypes.byref(mod), co.encode()), "load")
        _kernels.update({name: HipKernel(mod, name) for _, name, _ in _TILES.values()})
        _kernels.update({name: HipKernel(mod, name) for name, _, _ in _SHARED.values()})
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        logger.warning("M3.1 fused decode router disabled: %s", e)
        _disabled = True


def _router_lds(m: int) -> int:
    """router_lds() in m31_router.hip: the router's LDS-staged weight and x rows."""
    _, bn, bk, split, _ = _tile(m)[0]
    rowb = -(-HIDDEN // (bk * split)) * bk * 2 + 16
    return (bn + BLOCK_M) * rowb


def _router_launch(x, gate_weight, bias, routed_scaling_factor):
    """(Args, outputs) of one router launch, or None while the build or workspace is
    missing under graph capture."""
    ws = _ws.get(x.device)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        # split partials (always fully overwritten) and the self-resetting ticket counter
        ws = _ws[x.device] = (
            torch.empty(
                _MAX_SPLIT * BLOCK_M * NUM_EXPERTS, dtype=torch.float32, device=x.device
            ),
            torch.zeros(1, dtype=torch.int32, device=x.device),
        )
    if not _kernels:
        if torch.cuda.is_current_stream_capturing():
            return None
        _load()
        if _disabled:
            return None
    m = x.shape[0]
    dev = x.device
    logits = torch.empty(m, NUM_EXPERTS, dtype=torch.float32, device=dev)
    topk_w = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
    topk_ids = torch.empty(m, TOPK, dtype=torch.int32, device=dev)
    # aiter's moe_sorting buffer sizes
    max_padded = m * TOPK + NUM_EXPERTS * BLOCK_M - TOPK
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(max_padded, dtype=torch.float32, device=dev)
    sorted_eids = torch.empty(
        (max_padded + BLOCK_M - 1) // BLOCK_M, dtype=torch.int32, device=dev
    )
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    # the stage2 output; the presorted sort zero-fills it only for a stage2 accumulating into it
    moe_buf = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev)
    ptrs = (
        x,
        gate_weight,
        bias,
        *ws,
        logits,
        topk_w,
        topk_ids,
        sorted_ids,
        sorted_w,
        sorted_eids,
        num_valid,
    )
    args = Args(
        *(t.data_ptr() for t in ptrs),
        m,
        x.stride(0),
        float(routed_scaling_factor),
    )
    return args, (
        topk_w,
        topk_ids,
        logits,
        (sorted_ids, sorted_w, sorted_eids, num_valid, moe_buf),
    )


def m31_router(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
):
    """(topk_weights [M, 4] fp32, topk_ids [M, 4] int32, router_logits [M, 128] fp32, sorted) or None when
    unavailable (build failure, first call under graph capture). sorted = (sorted_ids, sorted_weights,
    sorted_expert_ids, num_valid_ids, moe_buf) in aiter's moe_sorting layout, moe_buf [M, 6144] bf16 unfilled."""
    launch = _router_launch(x, gate_weight, bias, routed_scaling_factor)
    if launch is None:
        return None
    args, out = launch
    _, name, grid = _tile(x.shape[0])
    _kernels[name].launch(grid, args, torch.cuda.current_stream().cuda_stream)
    return out


def m31_shared_router_supported(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor | None,
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    wq: torch.Tensor,
) -> bool:
    """m31_router_supported(x, ...) and smallm_fp8_gemm(xq, wq, x_scale) on the shared expert's
    [1536, 6144] gate_up with the instance the fused kernel runs."""
    m = x.shape[0]
    return (
        m31_router_supported(x, gate_weight, bias)
        and xq.shape[0] == m
        and wq.shape == (SHARED_N, HIDDEN)
        and smallm_fp8_gfx950.smallm_fp8_gemm_supported(xq, wq, x_scale, torch.bfloat16)
        and tuple(smallm_fp8_gfx950.smallm_fp8_gemm_config(m, *wq.shape))
        == _SHARED[_token_bucket(m)][2]
    )


def m31_shared_router(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
):
    """(smallm_fp8_gemm(xq, wq, x_scale, w_scale), m31_router(x, gate_weight, bias, routed_scaling_factor))
    in one launch, bitwise equal to the two calls, where m31_shared_router_supported(); None as m31_router."""
    launch = _router_launch(x, gate_weight, bias, routed_scaling_factor)
    if launch is None:
        return None
    r_args, out = launch
    m, n = x.shape[0], wq.shape[0]
    name, r_grid, (nt, _, waves, _) = _SHARED[_token_bucket(m)]
    gate_up = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    xs = x_scale.view(m, -1)[:, 0]
    g_args = smallm_fp8_gfx950.Args(
        *(t.data_ptr() for t in (xq, wq, xs, w_scale.contiguous(), gate_up)),
        m,
        n,
        xq.stride(0),
        xs.stride(0),
    )
    _kernels[name].launch(
        r_grid + n // (16 * nt),
        SharedArgs(r_args, g_args),
        torch.cuda.current_stream().cuda_stream,
        64 * waves,
        max(m * (HIDDEN + 16), _router_lds(m)),
    )
    return gate_up, out
