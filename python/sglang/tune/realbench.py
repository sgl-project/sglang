"""Real-silicon attention microbench (needs a CUDA GPU + torch).

AUTHORED-BUT-UNVERIFIED-ON-GPU: written against torch's stable public SDPA API; must be
run on real hardware to confirm. It is deliberately dependency-light: the "backends" are
torch's own scaled-dot-product-attention backends —

    flash          -> SDPBackend.FLASH_ATTENTION
    cudnn          -> SDPBackend.CUDNN_ATTENTION
    mem_efficient  -> SDPBackend.EFFICIENT_ATTENTION
    math           -> SDPBackend.MATH        (universal fallback, always runs)

— so the harness/isolation/crossover/fingerprint machinery can be validated on ANY CUDA
GPU without a full SGLang install. Once this machinery is proven green on silicon, the same
isolation + do_bench scaffolding wraps the real SGLang AttentionBackend kernels (fa3,
flashinfer, flashmla, ...); only the per-backend adapter below changes.

Decode is modeled as q_len=1 attending to a KV of length ctx_len (memory-bound); prefill as
q_len=seq_len self-attention, causal (compute-bound). GQA is expanded to full heads so plain
SDPA handles any qo/kv ratio.
"""

from __future__ import annotations

from typing import List

from .harness import do_bench
from .shapes import AttnProfile, DecodeShape, PrefillShape

REAL_BACKENDS: List[str] = ["flash", "cudnn", "mem_efficient", "math"]


def _sdpa_backend(name: str):  # pragma: no cover - real GPU only
    from torch.nn.attention import SDPBackend

    return {
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }[name]


def _dtype(profile: AttnProfile):  # pragma: no cover - real GPU only
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
    }.get(profile.dtype, torch.bfloat16)


def _make_qkv(B, Hq, Hkv, Sq, Skv, D, dtype):  # pragma: no cover - real GPU only
    import torch

    dev = "cuda"
    q = torch.randn(B, Hq, Sq, D, device=dev, dtype=dtype)
    k = torch.randn(B, Hkv, Skv, D, device=dev, dtype=dtype)
    v = torch.randn(B, Hkv, Skv, D, device=dev, dtype=dtype)
    if Hkv != Hq:  # expand GQA to full heads for plain SDPA
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return q, k, v


def real_decode_latency(
    backend: str, shape: DecodeShape, profile: AttnProfile
) -> float:  # pragma: no cover
    import torch
    from torch.nn.attention import sdpa_kernel

    dt = _dtype(profile)
    q, k, v = _make_qkv(
        shape.batch,
        profile.num_qo_heads,
        profile.num_kv_heads,
        1,
        shape.ctx_len,
        profile.head_dim,
        dt,
    )

    def run():
        with sdpa_kernel(_sdpa_backend(backend)):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    run()  # provoke an incompatible-backend error early (caught upstream)
    return do_bench(run, use_cuda_graph=(shape.batch <= 8)).median_us


def real_prefill_latency(
    backend: str, shape: PrefillShape, profile: AttnProfile
) -> float:  # pragma: no cover
    import torch
    from torch.nn.attention import sdpa_kernel

    dt = _dtype(profile)
    q, k, v = _make_qkv(
        shape.batch,
        profile.num_qo_heads,
        profile.num_kv_heads,
        shape.seq_len,
        shape.seq_len,
        profile.head_dim,
        dt,
    )

    def run():
        with sdpa_kernel(_sdpa_backend(backend)):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    run()
    return do_bench(run).median_us


# --- deliberate-failure helpers for the crash-recovery validation --------------
def force_oom():  # pragma: no cover - real GPU only
    """Allocate until CUDA OOM (run inside an isolated subprocess)."""
    import torch

    blocks = []
    while True:
        blocks.append(
            torch.empty(1 << 30, dtype=torch.float16, device="cuda")
        )  # 2 GiB each


def force_incompatible():  # pragma: no cover - real GPU only
    """Call the flash backend on an input it rejects (fp32) -> a real backend error."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 8, 16, 128, device="cuda", dtype=torch.float32)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
