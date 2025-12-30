import itertools
import os
from functools import lru_cache
from typing import Tuple

import torch
import triton
import triton.testing

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

IS_QUICK = os.getenv("SGL_BENCH_QUICK", "0").lower() in ("1", "true", "yes", "y")
ONLY_PROVIDER = os.getenv("SGL_BENCH_PROVIDER", "").strip() or None
SHOW_SPEEDUP = os.getenv("SGL_BENCH_SPEEDUP", "1").lower() in ("1", "true", "yes", "y")

DEVICE = "cuda"
MAX_SEQ_LEN = 8192
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
DTYPE = torch.bfloat16

try:
    from vllm.model_executor.layers.rotary_embedding import (
        RotaryEmbedding as vLLMRotaryEmbedding,
    )

    HAS_VLLM = True
except Exception:
    vLLMRotaryEmbedding = None
    HAS_VLLM = False

if IS_CI or IS_QUICK:
    BS_RANGE = [16]
    SEQ_RANGE = [1, 128]
    HEAD_SIZE_RANGE = [128]
    INTERLEAVED_RANGE = [True]
else:
    BS_RANGE = [1, 8, 64]
    SEQ_RANGE = [1, 4, 128, 2048]
    HEAD_SIZE_RANGE = [64, 96, 128, 256]
    INTERLEAVED_RANGE = [True, False]


def _compute_cos_sin_cache_half(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


@lru_cache(maxsize=None)
def _get_cos_sin_cache_half_cuda(rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _compute_cos_sin_cache_half(MAX_SEQ_LEN, rotary_dim, dtype=DTYPE)
    return cos.to(device=DEVICE, dtype=DTYPE), sin.to(device=DEVICE, dtype=DTYPE)


@torch.no_grad()
def torch_impl_rotary_fp32(
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
) -> None:
    orig_dtype = q.dtype
    if interleaved and cos.shape[1] == head_size:
        half = head_size // 2
        cos = cos.view(cos.shape[0], half, 2).select(2, 0).contiguous()
        sin = sin.view(sin.shape[0], half, 2).select(2, 1).contiguous()

    cos_f, sin_f = cos.float(), sin.float()
    q_f, k_f = q.float(), k.float()

    if interleaved:
        embed_dim = int(cos_f.shape[1])
        rot_dim = embed_dim * 2
        cos_b = cos_f[:, None, :embed_dim]
        sin_b = sin_f[:, None, :embed_dim]

        def _apply(x: torch.Tensor) -> None:
            xr = x[..., :rot_dim]
            xr2 = xr.view(xr.shape[0], xr.shape[1], embed_dim, 2)
            x0 = xr2[..., 0].clone()
            x1 = xr2[..., 1].clone()
            xr2[..., 0].copy_(x0 * cos_b - x1 * sin_b)
            xr2[..., 1].copy_(x1 * cos_b + x0 * sin_b)

    else:
        if cos_f.shape[1] == head_size // 2:
            embed_dim = int(cos_f.shape[1])
            rot_dim = embed_dim * 2
            cos_x, sin_x = cos_f[:, None, :], sin_f[:, None, :]
            cos_y, sin_y = cos_x, sin_x
        else:
            embed_dim = int(cos_f.shape[1]) // 2
            rot_dim = embed_dim * 2
            cos_x, sin_x = cos_f[:, None, :embed_dim], sin_f[:, None, :embed_dim]
            cos_y, sin_y = (
                cos_f[:, None, embed_dim:rot_dim],
                sin_f[:, None, embed_dim:rot_dim],
            )

        def _apply(x: torch.Tensor) -> None:
            xr = x[..., :rot_dim]
            x0 = xr[..., :embed_dim].clone()
            x1 = xr[..., embed_dim:rot_dim].clone()
            xr[..., :embed_dim].copy_(x0 * cos_x - x1 * sin_x)
            xr[..., embed_dim:rot_dim].copy_(x1 * cos_y + x0 * sin_y)

    _apply(q_f)
    _apply(k_f)
    q.copy_(q_f.to(orig_dtype))
    k.copy_(k_f.to(orig_dtype))


def sglang_aot_rotary_positions(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
    cos_sin_cache: torch.Tensor,
) -> None:
    from sgl_kernel.rotary_embedding import rotary_embedding

    rotary_embedding(
        positions=positions,
        query=q,
        key=k,
        head_size=head_size,
        is_neox=not interleaved,
        cos_sin_cache=cos_sin_cache,
    )


def sglang_jit_rotary_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
) -> None:
    from sglang.jit_kernel.rotary_embedding import rotary_embedding_cos_sin

    rotary_embedding_cos_sin(cos, sin, q, k, head_size, interleaved)


BASE_LINE_VALS = ["jit_cos_sin", "aot_pos"]
BASE_LINE_NAMES = ["SGL JIT (cos/sin)", "SGL AOT (positions)"]
BASE_STYLES = [("blue", "-"), ("orange", "--")]

LINE_VALS = list(BASE_LINE_VALS)
LINE_NAMES = list(BASE_LINE_NAMES)
STYLES = list(BASE_STYLES)

if HAS_VLLM:
    LINE_VALS.append("vllm_pos")
    LINE_NAMES.append("vLLM (positions)")
    STYLES.append(("green", "-."))

if ONLY_PROVIDER is not None:
    if ONLY_PROVIDER not in LINE_VALS:
        raise ValueError(
            f"Unknown SGL_BENCH_PROVIDER={ONLY_PROVIDER}. Allowed: {LINE_VALS}"
        )
    idx = LINE_VALS.index(ONLY_PROVIDER)
    LINE_VALS = [ONLY_PROVIDER]
    LINE_NAMES = [LINE_NAMES[idx]]
    STYLES = [STYLES[idx]]

configs = list(
    itertools.product(BS_RANGE, SEQ_RANGE, HEAD_SIZE_RANGE, INTERLEAVED_RANGE)
)
_SANITY_DONE = False


def _assert_close_gpu(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    name: str,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: dtype mismatch {actual.dtype} vs {expected.dtype}"
        )
    if actual.device != expected.device:
        raise AssertionError(
            f"{name}: device mismatch {actual.device} vs {expected.device}"
        )

    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    denom = atol + rtol * expected.abs()
    max_rel = float((diff / denom).max().item())
    if max_abs > atol and max_rel > 1.0:
        raise AssertionError(
            f"{name}: not close (atol={atol} rtol={rtol}) max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )


@lru_cache(maxsize=None)
def _get_vllm_rope(
    head_size: int, rotary_dim: int, interleaved: bool, dtype: torch.dtype
):
    if not HAS_VLLM or vLLMRotaryEmbedding is None:
        raise RuntimeError("vLLM not available")
    return vLLMRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=MAX_SEQ_LEN,
        base=10000,
        is_neox_style=interleaved,
        dtype=dtype,
    ).cuda()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "head_size", "interleaved"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="rotary-embedding-performance",
        args={},
    )
)
def benchmark(
    batch_size: int,
    seq_len: int,
    head_size: int,
    interleaved: bool,
    provider: str,
) -> Tuple[float, float, float]:
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    num_tokens = batch_size * seq_len
    rotary_dim = head_size

    cos_cache_half, sin_cache_half = _get_cos_sin_cache_half_cuda(rotary_dim)
    cos_sin_cache_aot = torch.cat([cos_cache_half, sin_cache_half], dim=-1).contiguous()

    positions = torch.arange(seq_len, device=DEVICE, dtype=torch.int64).repeat(
        batch_size
    )
    cos_half = cos_cache_half[positions].contiguous()
    sin_half = sin_cache_half[positions].contiguous()

    if interleaved:
        cos, sin = cos_half, sin_half
    else:
        cos = torch.cat([cos_half, cos_half], dim=-1).contiguous()
        sin = torch.cat([sin_half, sin_half], dim=-1).contiguous()

    q = torch.randn(num_tokens, NUM_Q_HEADS, head_size, device=DEVICE, dtype=DTYPE)
    k = torch.randn(num_tokens, NUM_KV_HEADS, head_size, device=DEVICE, dtype=DTYPE)

    global _SANITY_DONE
    if (
        (not _SANITY_DONE)
        and provider == "aot_pos"
        and batch_size == BS_RANGE[0]
        and seq_len == SEQ_RANGE[0]
        and head_size == HEAD_SIZE_RANGE[0]
        and interleaved == INTERLEAVED_RANGE[0]
    ):
        q_ref = q.clone()
        k_ref = k.clone()
        torch_impl_rotary_fp32(cos, sin, q_ref, k_ref, head_size, interleaved)
        q_out = q.clone()
        k_out = k.clone()
        sglang_aot_rotary_positions(
            positions, q_out, k_out, head_size, interleaved, cos_sin_cache_aot
        )
        ref_atol = 2e-2 if DTYPE == torch.bfloat16 else 2e-3
        ref_rtol = 2e-2 if DTYPE == torch.bfloat16 else 2e-3
        _assert_close_gpu(q_out, q_ref, atol=ref_atol, rtol=ref_rtol, name="AOT(q)")
        _assert_close_gpu(k_out, k_ref, atol=ref_atol, rtol=ref_rtol, name="AOT(k)")
        _SANITY_DONE = True

    FN_MAP = {
        "aot_pos": lambda: sglang_aot_rotary_positions(
            positions, q, k, head_size, interleaved, cos_sin_cache_aot
        ),
        "jit_cos_sin": lambda: sglang_jit_rotary_cos_sin(
            cos, sin, q, k, head_size, interleaved
        ),
        "torch_fp32": lambda: torch_impl_rotary_fp32(
            cos, sin, q, k, head_size, interleaved
        ),
    }
    if HAS_VLLM:
        vllm_rope = _get_vllm_rope(head_size, rotary_dim, interleaved, DTYPE)
        FN_MAP["vllm_pos"] = lambda: vllm_rope.forward_cuda(positions, q, k)

    fn = FN_MAP[provider]
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


SPEEDUP_LINE_VALS = [p for p in LINE_VALS if p != "jit_cos_sin"]
SPEEDUP_LINE_NAMES = [LINE_NAMES[LINE_VALS.index(p)] for p in SPEEDUP_LINE_VALS]
SPEEDUP_STYLES = [STYLES[LINE_VALS.index(p)] for p in SPEEDUP_LINE_VALS]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "head_size", "interleaved"],
        x_vals=configs,
        line_arg="provider",
        line_vals=SPEEDUP_LINE_VALS,
        line_names=SPEEDUP_LINE_NAMES,
        styles=SPEEDUP_STYLES,
        ylabel="speedup (x) vs SGL JIT (cos/sin)",
        plot_name="rotary-embedding-speedup-vs-jit",
        args={},
    )
)
def benchmark_speedup_vs_jit(
    batch_size: int,
    seq_len: int,
    head_size: int,
    interleaved: bool,
    provider: str,
) -> Tuple[float, float, float]:
    if provider == "jit_cos_sin":
        return 1.0, 1.0, 1.0
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    num_tokens = batch_size * seq_len
    rotary_dim = head_size

    cos_cache_half, sin_cache_half = _get_cos_sin_cache_half_cuda(rotary_dim)
    cos_sin_cache_aot = torch.cat([cos_cache_half, sin_cache_half], dim=-1).contiguous()

    positions = torch.arange(seq_len, device=DEVICE, dtype=torch.int64).repeat(
        batch_size
    )
    cos_half = cos_cache_half[positions].contiguous()
    sin_half = sin_cache_half[positions].contiguous()

    if interleaved:
        cos, sin = cos_half, sin_half
    else:
        cos = torch.cat([cos_half, cos_half], dim=-1).contiguous()
        sin = torch.cat([sin_half, sin_half], dim=-1).contiguous()

    q_base = torch.randn(num_tokens, NUM_Q_HEADS, head_size, device=DEVICE, dtype=DTYPE)
    k_base = torch.randn(
        num_tokens, NUM_KV_HEADS, head_size, device=DEVICE, dtype=DTYPE
    )

    q_jit = q_base.clone()
    k_jit = k_base.clone()
    q_p = q_base.clone()
    k_p = k_base.clone()

    FN_MAP = {
        "jit_cos_sin": lambda: sglang_jit_rotary_cos_sin(
            cos, sin, q_jit, k_jit, head_size, interleaved
        ),
        "aot_pos": lambda: sglang_aot_rotary_positions(
            positions, q_p, k_p, head_size, interleaved, cos_sin_cache_aot
        ),
        "torch_fp32": lambda: torch_impl_rotary_fp32(
            cos, sin, q_p, k_p, head_size, interleaved
        ),
    }
    if HAS_VLLM:
        vllm_rope = _get_vllm_rope(head_size, rotary_dim, interleaved, DTYPE)
        FN_MAP["vllm_pos"] = lambda: vllm_rope.forward_cuda(positions, q_p, k_p)

    quantiles = [0.5, 0.2, 0.8]
    jit_ms, jit_min_ms, jit_max_ms = triton.testing.do_bench_cudagraph(
        FN_MAP["jit_cos_sin"], quantiles=quantiles
    )  # type: ignore
    p_ms, p_min_ms, p_max_ms = triton.testing.do_bench_cudagraph(
        FN_MAP[provider], quantiles=quantiles
    )  # type: ignore

    speed_med = float(jit_ms / p_ms)
    speed_max = float(jit_max_ms / p_min_ms)
    speed_min = float(jit_min_ms / p_max_ms)
    return speed_med, speed_max, speed_min


if __name__ == "__main__":
    benchmark.run(print_data=True)
    if (
        SHOW_SPEEDUP
        and ("jit_cos_sin" in LINE_VALS)
        and (len(LINE_VALS) > 1)
        and (ONLY_PROVIDER is None)
    ):
        benchmark_speedup_vs_jit.run(print_data=True)
