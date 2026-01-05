"""
Lean Rotary Embedding benchmark.

Providers:
- jit_fused   : SGL JIT with fused gather (positions + full cache)
- jit_unfused : SGL JIT with Python gather (indexing cos/sin in Python)
- aot_pos     : SGL AOT with fused gather
- vllm_pos    : vLLM
- flashinfer  : FlashInfer
- torch_fp32  : Torch
"""

import itertools
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Tuple

import torch
import triton.testing

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 8192
NUM_HEADS = 32
BS_RANGE = [1, 64]
if not os.getenv("CI"):
    BS_RANGE = [1, 2, 4, 8, 16, 32, 64, 128]
SEQ_RANGE = [1, 2048]
HEAD_SIZE_RANGE = [128]
INTERLEAVED_RANGE = [True, False]

ONLY_PROVIDER = os.getenv("SGL_BENCH_PROVIDER", "").strip() or None
INCLUDE_TORCH_REF = os.getenv("SGL_BENCH_INCLUDE_TORCH", "0").lower() in (
    "1",
    "true",
    "yes",
    "y",
)
SHOW_SPEEDUP = os.getenv("SGL_BENCH_SPEEDUP", "1").lower() in ("1", "true", "yes", "y")

try:
    import sgl_kernel  # noqa: F401
except Exception:
    sgl_kernel = None  # type: ignore

HAS_SGL_POS = hasattr(torch.ops, "sgl_kernel") and hasattr(
    torch.ops.sgl_kernel, "rotary_embedding"
)

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

    HAS_FLASHINFER = True
except Exception:
    apply_flashinfer_rope_qk_inplace = None  # type: ignore
    HAS_FLASHINFER = False


def _compute_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


@lru_cache(maxsize=None)
def _cos_sin_cache_half_cuda(rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _compute_cos_sin_cache(MAX_SEQ_LEN, rotary_dim, dtype=DTYPE)
    return cos.to(device=DEVICE, dtype=DTYPE), sin.to(device=DEVICE, dtype=DTYPE)


@lru_cache(maxsize=None)
def _vllm_rope(head_size: int, rotary_dim: int, interleaved: bool, dtype: torch.dtype):
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


@torch.no_grad()
def torch_impl_rotary_fp32(cos, sin, q, k, head_size: int, interleaved: bool) -> None:
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

        def _apply(x):
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

        def _apply(x):
            xr = x[..., :rot_dim]
            x0 = xr[..., :embed_dim].clone()
            x1 = xr[..., embed_dim:rot_dim].clone()
            xr[..., :embed_dim].copy_(x0 * cos_x - x1 * sin_x)
            xr[..., embed_dim:rot_dim].copy_(x1 * cos_y + x0 * sin_y)

    _apply(q_f)
    _apply(k_f)
    q.copy_(q_f.to(orig_dtype))
    k.copy_(k_f.to(orig_dtype))


def _assert_close_gpu(actual, expected, atol: float, rtol: float, name: str):
    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    denom = atol + rtol * expected.abs()
    max_rel = float((diff / denom).max().item())
    if max_abs > atol and max_rel > 1.0:
        raise AssertionError(
            f"{name}: not close max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )


def sgl_jit_fused(
    positions, cos_cache, sin_cache, q, k, head_size: int, interleaved: bool
):
    from sglang.jit_kernel.rotary_embedding import rotary_embedding_cos_sin

    rotary_embedding_cos_sin(
        cos_cache, sin_cache, q, k, head_size, interleaved, positions=positions
    )


def sgl_jit_unfused(
    positions, cos_cache, sin_cache, q, k, head_size: int, interleaved: bool
):
    from sglang.jit_kernel.rotary_embedding import rotary_embedding_cos_sin

    cos_g = cos_cache[positions].contiguous()
    sin_g = sin_cache[positions].contiguous()
    rotary_embedding_cos_sin(cos_g, sin_g, q, k, head_size, interleaved)


def sgl_aot_pos(positions, q, k, head_size: int, interleaved: bool, cos_sin_cache):
    if not HAS_SGL_POS:
        raise RuntimeError("torch.ops.sgl_kernel.rotary_embedding not available")
    torch.ops.sgl_kernel.rotary_embedding(
        positions, q, k, head_size, cos_sin_cache, not interleaved  # is_neox flag
    )


def _is_flashinfer_unsupported(e: BaseException) -> bool:
    msg = str(e)
    return ("Unsupported head_dim" in msg) or ("cos_sin_cache should be float32" in msg)


@dataclass
class Case:
    positions: torch.Tensor
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    cos_gathered: torch.Tensor
    sin_gathered: torch.Tensor
    cos_sin_cache_aot: torch.Tensor
    q_base: torch.Tensor  # [T, H, D]
    k_base: torch.Tensor  # [T, H, D]
    q4_base: torch.Tensor  # [B,S,H,D] for flashinfer
    k4_base: torch.Tensor
    cos_sin_cache_flashinfer: torch.Tensor | None


def prepare_case(
    batch_size: int, seq_len: int, head_size: int, interleaved: bool
) -> Case:
    num_tokens = batch_size * seq_len
    rotary_dim = head_size

    cos_cache_half, sin_cache_half = _cos_sin_cache_half_cuda(rotary_dim)
    positions = torch.arange(seq_len, device=DEVICE, dtype=torch.int64).repeat(
        batch_size
    )

    cos_half = cos_cache_half[positions].contiguous()
    sin_half = sin_cache_half[positions].contiguous()

    if interleaved:
        cos_cache, sin_cache = cos_cache_half, sin_cache_half
        cos_g, sin_g = cos_half, sin_half
    else:
        # non-interleaved: kernel expects R=2*embed_dim (x and y halves)
        cos_cache = torch.cat([cos_cache_half, cos_cache_half], dim=-1).contiguous()
        sin_cache = torch.cat([sin_cache_half, sin_cache_half], dim=-1).contiguous()
        cos_g = torch.cat([cos_half, cos_half], dim=-1).contiguous()
        sin_g = torch.cat([sin_half, sin_half], dim=-1).contiguous()

    cos_sin_cache_aot = torch.cat([cos_cache_half, sin_cache_half], dim=-1).contiguous()

    q4 = torch.randn(
        batch_size, seq_len, NUM_HEADS, head_size, device=DEVICE, dtype=DTYPE
    ).contiguous()
    k4 = torch.randn(
        batch_size, seq_len, NUM_HEADS, head_size, device=DEVICE, dtype=DTYPE
    ).contiguous()
    q = q4.view(num_tokens, NUM_HEADS, head_size)
    k = k4.view(num_tokens, NUM_HEADS, head_size)

    cos_sin_cache_flashinfer = None
    if HAS_FLASHINFER:
        cos_f32, sin_f32 = _compute_cos_sin_cache(
            MAX_SEQ_LEN, rotary_dim, dtype=torch.float32
        )
        cos_sin_cache_flashinfer = (
            torch.cat([cos_f32, sin_f32], dim=-1)
            .to(device=DEVICE, dtype=torch.float32)
            .contiguous()
        )

    return Case(
        positions=positions,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        cos_gathered=cos_g,
        sin_gathered=sin_g,
        cos_sin_cache_aot=cos_sin_cache_aot,
        q_base=q,
        k_base=k,
        q4_base=q4,
        k4_base=k4,
        cos_sin_cache_flashinfer=cos_sin_cache_flashinfer,
    )


def make_fn(
    provider: str, case: Case, head_size: int, interleaved: bool
) -> Callable[[], None]:
    if provider in ("jit_fused", "jit_unfused", "aot_pos", "torch_fp32", "vllm_pos"):
        q = case.q_base.clone()
        k = case.k_base.clone()

        if provider == "jit_fused":
            return lambda: sgl_jit_fused(
                case.positions,
                case.cos_cache,
                case.sin_cache,
                q,
                k,
                head_size,
                interleaved,
            )
        if provider == "jit_unfused":
            return lambda: sgl_jit_unfused(
                case.positions,
                case.cos_cache,
                case.sin_cache,
                q,
                k,
                head_size,
                interleaved,
            )
        if provider == "aot_pos":
            return lambda: sgl_aot_pos(
                case.positions, q, k, head_size, interleaved, case.cos_sin_cache_aot
            )
        if provider == "torch_fp32":
            return lambda: torch_impl_rotary_fp32(
                case.cos_gathered, case.sin_gathered, q, k, head_size, interleaved
            )
        if provider == "vllm_pos":
            rope = _vllm_rope(head_size, head_size, interleaved, DTYPE)
            return lambda: rope.forward_cuda(case.positions, q, k)

    if provider == "flashinfer":
        if not (HAS_FLASHINFER and apply_flashinfer_rope_qk_inplace is not None):
            raise RuntimeError("FlashInfer not available")
        q4 = case.q4_base.clone()
        k4 = case.k4_base.clone()
        return lambda: apply_flashinfer_rope_qk_inplace(
            q4,
            k4,
            case.cos_sin_cache_flashinfer,
            is_neox=not interleaved,
            positions=case.positions,
        )

    raise ValueError(f"Unknown provider={provider}")


def bench(fn: Callable[[], None], use_cudagraph: bool) -> Tuple[float, float, float]:
    qs = [0.5, 0.2, 0.8]
    if use_cudagraph:
        return triton.testing.do_bench_cudagraph(fn, quantiles=qs)  # type: ignore
    return triton.testing.do_bench(fn, quantiles=qs)  # type: ignore


def available_providers() -> Dict[str, str]:
    ps: Dict[str, str] = {
        "jit_fused": "SGL JIT (fused gather)",
        "jit_unfused": "SGL JIT (Python gather)",
    }
    if HAS_SGL_POS:
        ps["aot_pos"] = "SGL AOT (fused gather)"
    if HAS_VLLM:
        ps["vllm_pos"] = "vLLM"
    if HAS_FLASHINFER:
        ps["flashinfer"] = "FlashInfer"
    if INCLUDE_TORCH_REF:
        ps["torch_fp32"] = "Torch"
    return ps


PROVIDERS = available_providers()
if ONLY_PROVIDER is not None:
    if ONLY_PROVIDER not in PROVIDERS:
        raise ValueError(
            f"Unknown provider={ONLY_PROVIDER}. Allowed: {list(PROVIDERS)}"
        )
    PROVIDERS = {ONLY_PROVIDER: PROVIDERS[ONLY_PROVIDER]}

configs = list(
    itertools.product(BS_RANGE, SEQ_RANGE, HEAD_SIZE_RANGE, INTERLEAVED_RANGE)
)

_SANITY_DONE = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "head_size", "interleaved"],
        x_vals=configs,
        line_arg="provider",
        line_vals=list(PROVIDERS.keys()),
        line_names=list(PROVIDERS.values()),
        styles=None,
        ylabel="us",
        plot_name="rotary-embedding-performance",
        args={},
    )
)
def benchmark(
    batch_size: int, seq_len: int, head_size: int, interleaved: bool, provider: str
):
    case = prepare_case(batch_size, seq_len, head_size, interleaved)
    fn = make_fn(provider, case, head_size, interleaved)

    use_cudagraph = provider != "flashinfer"
    try:
        ms, min_ms, max_ms = bench(fn, use_cudagraph=use_cudagraph)
    except Exception as e:
        if provider == "flashinfer" and _is_flashinfer_unsupported(e):
            nan = float("nan")
            return nan, nan, nan
        raise

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
