"""
Benchmark for Rotary Embedding implementations.

Key test scenarios:
- jit_fused: JIT kernel with fused gathering (positions + full cache) - OPTIMIZED
  * Avoids explicit index_select in Python
  * Gathering happens inside the kernel
  
- jit_unfused: JIT kernel with explicit gathering (positions + full cache) - BASELINE
  * Includes index_select overhead in Python
  * Shows the cost we're trying to avoid
  
- jit_cos_sin: JIT kernel with pre-gathered cos/sin (backward compatibility)
  * Traditional approach: cos/sin already gathered before kernel call
  * Does not include gathering overhead in timing
  
- aot_pos: AOT kernel with positions
- vllm_pos: vLLM's implementation
- flashinfer_rope: FlashInfer's RoPE implementation
"""
import itertools
import os
from functools import lru_cache
from typing import Tuple

import torch
import triton
import triton.testing

try:
    import sgl_kernel  # noqa: F401
except Exception:
    sgl_kernel = None  # type: ignore

HAS_SGL_POS = hasattr(torch.ops.sgl_kernel, "rotary_embedding")

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

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

try:
    from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
        apply_flashinfer_rope_qk_inplace,
    )

    try:
        from flashinfer.rope import (  # noqa: F401
            apply_rope_with_cos_sin_cache_inplace as _,
        )

        HAS_FLASHINFER = True
    except Exception:
        HAS_FLASHINFER = False
except Exception:
    apply_flashinfer_rope_qk_inplace = None  # type: ignore
    HAS_FLASHINFER = False

if IS_CI:
    BS_RANGE = [16]
    SEQ_RANGE = [1, 128]
    HEAD_SIZE_RANGE = [128]
    INTERLEAVED_RANGE = [True]
else:
    BS_RANGE = [1, 8, 64]
    SEQ_RANGE = [1, 4, 128, 2048]
    HEAD_SIZE_RANGE = [64, 96, 128, 256]
    INTERLEAVED_RANGE = [True, False]


def _is_flashinfer_unsupported_shape_error(e: BaseException) -> bool:
    msg = str(e)
    return ("Unsupported head_dim" in msg) or ("cos_sin_cache should be float32" in msg)


def _bench(
    fn,
    *,
    quantiles: list[float],
    use_cudagraph: bool,
) -> tuple[float, float, float]:
    if use_cudagraph:
        return triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return triton.testing.do_bench(fn, quantiles=quantiles)  # type: ignore


def _bench_provider(
    provider: str,
    fn,
    *,
    quantiles: list[float],
    use_cudagraph: bool,
) -> tuple[float, float, float]:
    """Bench a provider and return (median, min, max) in ms."""
    try:
        return _bench(fn, quantiles=quantiles, use_cudagraph=use_cudagraph)
    except Exception as e:  # noqa: BLE001
        if provider == "flashinfer_rope" and _is_flashinfer_unsupported_shape_error(e):
            nan = float("nan")
            return nan, nan, nan
        raise


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
    if not HAS_SGL_POS:
        raise RuntimeError("torch.ops.sgl_kernel.rotary_embedding is not available")
    torch.ops.sgl_kernel.rotary_embedding(
        positions,
        q,
        k,
        head_size,
        cos_sin_cache,
        not interleaved,  # sgl-kernel positions op uses is_neox (split-halves) flag
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


def sglang_jit_rotary_with_positions(
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
) -> None:
    """
    JIT kernel with fused gathering: positions + full cache.
    This is the optimized version that avoids explicit index_select in Python.
    """
    from sglang.jit_kernel.rotary_embedding import rotary_embedding_cos_sin

    rotary_embedding_cos_sin(
        cos_cache, sin_cache, q, k, head_size, interleaved, positions=positions
    )


def sglang_jit_rotary_with_gather(
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
) -> None:
    """
    JIT kernel WITHOUT fused gathering: explicit index_select in Python.
    This is the baseline version that includes the gathering overhead.
    """
    from sglang.jit_kernel.rotary_embedding import rotary_embedding_cos_sin

    # Explicit gathering in Python (the overhead we want to avoid)
    cos_gathered = cos_cache[positions].contiguous()
    sin_gathered = sin_cache[positions].contiguous()
    rotary_embedding_cos_sin(
        cos_gathered, sin_gathered, q, k, head_size, interleaved
    )


BASE_PROVIDER = "jit_fused"
BASE_NAME = "SGL JIT (fused)"
BASE_STYLE = ("blue", "-")

BASE_LINE_VALS = [BASE_PROVIDER]
BASE_LINE_NAMES = [BASE_NAME]
BASE_STYLES = [BASE_STYLE]

LINE_VALS = list(BASE_LINE_VALS)
LINE_NAMES = list(BASE_LINE_NAMES)
STYLES = list(BASE_STYLES)

# Add unfused version for comparison
LINE_VALS.append("jit_unfused")
LINE_NAMES.append("SGL JIT (unfused)")
STYLES.append(("purple", ":"))

if HAS_SGL_POS:
    LINE_VALS.append("aot_pos")
    LINE_NAMES.append("SGL AOT (positions)")
    STYLES.append(("orange", "--"))

if HAS_VLLM:
    LINE_VALS.append("vllm_pos")
    LINE_NAMES.append("vLLM (positions)")
    STYLES.append(("green", "-."))

if HAS_FLASHINFER:
    LINE_VALS.append("flashinfer_rope")
    LINE_NAMES.append("FlashInfer RoPE")
    STYLES.append(("red", "--"))

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

    # Prepare full cache (for fused gathering tests)
    cos_cache_half, sin_cache_half = _get_cos_sin_cache_half_cuda(rotary_dim)
    cos_sin_cache_aot = torch.cat([cos_cache_half, sin_cache_half], dim=-1).contiguous()
    
    # FlashInfer requires cos_sin_cache to be float32.
    if HAS_FLASHINFER:
        cos_f32, sin_f32 = _compute_cos_sin_cache_half(
            MAX_SEQ_LEN, rotary_dim, dtype=torch.float32
        )
        cos_sin_cache_flashinfer = (
            torch.cat([cos_f32, sin_f32], dim=-1)
            .to(device=DEVICE, dtype=torch.float32)
            .contiguous()
        )

    # Prepare positions (simulate real decoding scenario)
    # Use arange for simplicity, but in real scenario this could be non-contiguous
    positions = torch.arange(seq_len, device=DEVICE, dtype=torch.int64).repeat(
        batch_size
    )
    
    # Pre-gathered cos/sin for unfused tests (traditional approach)
    cos_half = cos_cache_half[positions].contiguous()
    sin_half = sin_cache_half[positions].contiguous()

    if interleaved:
        cos_gathered, sin_gathered = cos_half, sin_half
        # For fused tests, prepare cache in interleaved format
        cos_cache_for_fused = cos_cache_half
        sin_cache_for_fused = sin_cache_half
    else:
        cos_gathered = torch.cat([cos_half, cos_half], dim=-1).contiguous()
        sin_gathered = torch.cat([sin_half, sin_half], dim=-1).contiguous()
        # For fused tests, prepare cache in non-interleaved format
        cos_cache_for_fused = torch.cat([cos_cache_half, cos_cache_half], dim=-1).contiguous()
        sin_cache_for_fused = torch.cat([sin_cache_half, sin_cache_half], dim=-1).contiguous()

    # Keep a canonical 4D layout for providers that require it (e.g. FlashInfer).
    # View as 3D for providers that accept [num_tokens, nheads, head_size].
    q4 = torch.randn(
        batch_size, seq_len, NUM_Q_HEADS, head_size, device=DEVICE, dtype=DTYPE
    ).contiguous()
    k4 = torch.randn(
        batch_size, seq_len, NUM_Q_HEADS, head_size, device=DEVICE, dtype=DTYPE
    ).contiguous()
    q = q4.view(num_tokens, NUM_Q_HEADS, head_size)
    k = k4.view(num_tokens, NUM_Q_HEADS, head_size)

    global _SANITY_DONE
    if (
        (not _SANITY_DONE)
        and provider == "jit_fused"
        and batch_size == BS_RANGE[0]
        and seq_len == SEQ_RANGE[0]
        and head_size == HEAD_SIZE_RANGE[0]
        and interleaved == INTERLEAVED_RANGE[0]
    ):
        # Reference implementation
        q_ref = q.clone()
        k_ref = k.clone()
        torch_impl_rotary_fp32(cos_gathered, sin_gathered, q_ref, k_ref, head_size, interleaved)
        
        # Test fused version
        q_fused_test = q.clone()
        k_fused_test = k.clone()
        sglang_jit_rotary_with_positions(
            positions, cos_cache_for_fused, sin_cache_for_fused, 
            q_fused_test, k_fused_test, head_size, interleaved
        )
        
        # Test unfused version
        q_unfused_test = q.clone()
        k_unfused_test = k.clone()
        sglang_jit_rotary_with_gather(
            positions, cos_cache_for_fused, sin_cache_for_fused,
            q_unfused_test, k_unfused_test, head_size, interleaved
        )
        
        ref_atol = 2e-2 if DTYPE == torch.bfloat16 else 2e-3
        ref_rtol = 2e-2 if DTYPE == torch.bfloat16 else 2e-3
        _assert_close_gpu(q_fused_test, q_ref, atol=ref_atol, rtol=ref_rtol, name="JIT-fused(q)")
        _assert_close_gpu(k_fused_test, k_ref, atol=ref_atol, rtol=ref_rtol, name="JIT-fused(k)")
        _assert_close_gpu(q_unfused_test, q_ref, atol=ref_atol, rtol=ref_rtol, name="JIT-unfused(q)")
        _assert_close_gpu(k_unfused_test, k_ref, atol=ref_atol, rtol=ref_rtol, name="JIT-unfused(k)")
        print("✅ Sanity check passed: fused and unfused versions match reference!")
        _SANITY_DONE = True

    FN_MAP = {
        "aot_pos": lambda: sglang_aot_rotary_positions(
            positions, q, k, head_size, interleaved, cos_sin_cache_aot
        ),
        # NEW: Fused gathering - positions + full cache (OPTIMIZED)
        "jit_fused": lambda: sglang_jit_rotary_with_positions(
            positions, cos_cache_for_fused, sin_cache_for_fused, q, k, head_size, interleaved
        ),
        # NEW: Unfused - explicit gathering in Python (BASELINE)
        "jit_unfused": lambda: sglang_jit_rotary_with_gather(
            positions, cos_cache_for_fused, sin_cache_for_fused, q, k, head_size, interleaved
        ),
        # OLD: Pre-gathered cos/sin (for backward compatibility)
        "jit_cos_sin": lambda: sglang_jit_rotary_cos_sin(
            cos_gathered, sin_gathered, q, k, head_size, interleaved
        ),
        "torch_fp32": lambda: torch_impl_rotary_fp32(
            cos_gathered, sin_gathered, q, k, head_size, interleaved
        ),
    }
    if HAS_FLASHINFER and apply_flashinfer_rope_qk_inplace is not None:
        FN_MAP["flashinfer_rope"] = lambda: apply_flashinfer_rope_qk_inplace(
            q4,
            k4,
            cos_sin_cache_flashinfer,
            is_neox=not interleaved,
        )
    if HAS_VLLM:
        vllm_rope = _get_vllm_rope(head_size, rotary_dim, interleaved, DTYPE)
        FN_MAP["vllm_pos"] = lambda: vllm_rope.forward_cuda(positions, q, k)

    fn = FN_MAP[provider]
    quantiles = [0.5, 0.2, 0.8]
    # If FlashInfer is available, unify timing across providers by using do_bench for all.
    use_cudagraph = not HAS_FLASHINFER
    ms, min_ms, max_ms = _bench_provider(
        provider, fn, quantiles=quantiles, use_cudagraph=use_cudagraph
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


SPEEDUP_LINE_VALS = [p for p in LINE_VALS if p != BASE_PROVIDER]
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
        ylabel=f"speedup (x) vs {BASE_NAME}",
        plot_name=f"rotary-embedding-speedup-vs-{BASE_PROVIDER}",
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
    if provider == BASE_PROVIDER:
        return 1.0, 1.0, 1.0
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    num_tokens = batch_size * seq_len
    rotary_dim = head_size

    # Prepare full cache
    cos_cache_half, sin_cache_half = _get_cos_sin_cache_half_cuda(rotary_dim)
    cos_sin_cache_aot = torch.cat([cos_cache_half, sin_cache_half], dim=-1).contiguous()
    if HAS_FLASHINFER:
        cos_f32, sin_f32 = _compute_cos_sin_cache_half(
            MAX_SEQ_LEN, rotary_dim, dtype=torch.float32
        )
        cos_sin_cache_flashinfer = (
            torch.cat([cos_f32, sin_f32], dim=-1)
            .to(device=DEVICE, dtype=torch.float32)
            .contiguous()
        )

    positions = torch.arange(seq_len, device=DEVICE, dtype=torch.int64).repeat(
        batch_size
    )
    cos_half = cos_cache_half[positions].contiguous()
    sin_half = sin_cache_half[positions].contiguous()

    if interleaved:
        cos_gathered, sin_gathered = cos_half, sin_half
        cos_cache_for_fused = cos_cache_half
        sin_cache_for_fused = sin_cache_half
    else:
        cos_gathered = torch.cat([cos_half, cos_half], dim=-1).contiguous()
        sin_gathered = torch.cat([sin_half, sin_half], dim=-1).contiguous()
        cos_cache_for_fused = torch.cat([cos_cache_half, cos_cache_half], dim=-1).contiguous()
        sin_cache_for_fused = torch.cat([sin_cache_half, sin_cache_half], dim=-1).contiguous()

    q_base4 = torch.randn(
        batch_size, seq_len, NUM_Q_HEADS, head_size, device=DEVICE, dtype=DTYPE
    ).contiguous()
    k_base4 = torch.randn(
        batch_size, seq_len, NUM_Q_HEADS, head_size, device=DEVICE, dtype=DTYPE
    ).contiguous()
    q_base = q_base4.view(num_tokens, NUM_Q_HEADS, head_size)
    k_base = k_base4.view(num_tokens, NUM_Q_HEADS, head_size)

    q_fused = q_base.clone()
    k_fused = k_base.clone()
    q_unfused = q_base.clone()
    k_unfused = k_base.clone()
    q_jit = q_base.clone()
    k_jit = k_base.clone()
    q_p = q_base.clone()
    k_p = k_base.clone()

    FN_MAP = {
        "jit_fused": lambda: sglang_jit_rotary_with_positions(
            positions, cos_cache_for_fused, sin_cache_for_fused, q_fused, k_fused, head_size, interleaved
        ),
        "jit_unfused": lambda: sglang_jit_rotary_with_gather(
            positions, cos_cache_for_fused, sin_cache_for_fused, q_unfused, k_unfused, head_size, interleaved
        ),
        "jit_cos_sin": lambda: sglang_jit_rotary_cos_sin(
            cos_gathered, sin_gathered, q_jit, k_jit, head_size, interleaved
        ),
        "aot_pos": lambda: sglang_aot_rotary_positions(
            positions, q_p, k_p, head_size, interleaved, cos_sin_cache_aot
        ),
        "torch_fp32": lambda: torch_impl_rotary_fp32(
            cos_gathered, sin_gathered, q_p, k_p, head_size, interleaved
        ),
    }
    if HAS_FLASHINFER and apply_flashinfer_rope_qk_inplace is not None:
        q_f = q_base4.clone()
        k_f = k_base4.clone()
        FN_MAP["flashinfer_rope"] = lambda: apply_flashinfer_rope_qk_inplace(
            q_f,
            k_f,
            cos_sin_cache_flashinfer,
            is_neox=not interleaved,
        )
    if HAS_VLLM:
        vllm_rope = _get_vllm_rope(head_size, rotary_dim, interleaved, DTYPE)
        FN_MAP["vllm_pos"] = lambda: vllm_rope.forward_cuda(positions, q_p, k_p)

    quantiles = [0.5, 0.2, 0.8]
    # If FlashInfer is available, use do_bench for all providers to keep methodology
    # consistent and avoid CUDA-graph sensitivity.
    use_cudagraph = not HAS_FLASHINFER

    base_ms, base_min_ms, base_max_ms = _bench_provider(
        BASE_PROVIDER,
        FN_MAP[BASE_PROVIDER],
        quantiles=quantiles,
        use_cudagraph=use_cudagraph,
    )
    if base_ms != base_ms:  # NaN
        nan = float("nan")
        return nan, nan, nan

    p_ms, p_min_ms, p_max_ms = _bench_provider(
        provider, FN_MAP[provider], quantiles=quantiles, use_cudagraph=use_cudagraph
    )

    speed_med = float(base_ms / p_ms)
    speed_max = float(base_max_ms / p_min_ms)
    speed_min = float(base_min_ms / p_max_ms)
    return speed_med, speed_max, speed_min


if __name__ == "__main__":
    benchmark.run(print_data=True)
    if (
        SHOW_SPEEDUP
        and (BASE_PROVIDER in LINE_VALS)
        and (len(LINE_VALS) > 1)
        and (ONLY_PROVIDER is None)
    ):
        benchmark_speedup_vs_jit.run(print_data=True)
