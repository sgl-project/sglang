import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)

MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0
ROPE_DIM = 128
CACHE_SIZE = 1024 * 1024


def create_cos_sin_cache(
    rotary_dim: int = ROPE_DIM,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEFAULT_DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEFAULT_DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)


# Pre-build the cache once
COS_SIN_CACHE = create_cos_sin_cache()


# ---------------------------------------------------------------------------
# RoPE-only provider implementations
# ---------------------------------------------------------------------------


def flashinfer_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    head_size = q.shape[-1]
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=head_size,
        cos_sin_cache=COS_SIN_CACHE,
        is_neox=is_neox,
    )


def sglang_rope_v0(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    from sglang.jit_kernel.pos_enc import rotary_embedding_with_key

    head_size = q.shape[-1]
    rotary_embedding_with_key(
        positions=positions,
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=head_size,
        cos_sin_cache=COS_SIN_CACHE,
        is_neox=is_neox,
    )


def sglang_rope_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace

    head_size = q.shape[-1]
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=head_size,
        cos_sin_cache=COS_SIN_CACHE,
        is_neox=is_neox,
    )


def sglang_rope_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    from sglang.jit_kernel.rope import apply_rope_inplace

    apply_rope_inplace(q, k, COS_SIN_CACHE, positions, is_neox=is_neox)


# ---------------------------------------------------------------------------
# RoPE + KV cache store provider implementations
# ---------------------------------------------------------------------------


def rope_v0_store(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    is_neox: bool,
) -> None:
    from sglang.jit_kernel.kvcache import store_cache
    from sglang.jit_kernel.rope import apply_rope_inplace

    head_size = q.shape[-1]
    row_dim = k.shape[-2] * head_size
    apply_rope_inplace(
        positions=positions,
        q=q,
        k=k,
        rope_dim=head_size,
        cos_sin_cache=COS_SIN_CACHE,
        is_neox=is_neox,
    )
    store_cache(
        k.view(-1, row_dim),
        v.view(-1, row_dim),
        k_cache,
        v_cache,
        out_loc,
    )


def rope_v1_store(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    is_neox: bool,
) -> None:
    from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace

    head_size = q.shape[-1]
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=head_size,
        cos_sin_cache=COS_SIN_CACHE,
        is_neox=is_neox,
        fused_set_kv_buffer_arg=FusedSetKVBufferArg(
            value=v.view(v.shape[0], -1),
            k_buffer=k_cache,
            v_buffer=v_cache,
            k_scale=None,
            v_scale=None,
            cache_loc=out_loc,
        ),
    )


def rope_v2_store(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    is_neox: bool,
) -> None:
    from sglang.jit_kernel.rope import apply_rope_inplace_with_kvcache

    apply_rope_inplace_with_kvcache(
        q, k, v, k_cache, v_cache, COS_SIN_CACHE, positions, out_loc, is_neox=is_neox
    )


# ---------------------------------------------------------------------------
# Benchmark configuration (shared)
# ---------------------------------------------------------------------------

BS_RANGE = get_benchmark_range(
    full_range=[2**n for n in range(0, 16)],
    ci_range=[16],
)
QK_HEAD_RANGE = get_benchmark_range(
    full_range=[(8, 1), (16, 2), (32, 8)],
    ci_range=[(16, 2)],
)
QK_HEAD_RANGE = [f"{q},{k}" for q, k in QK_HEAD_RANGE]
IS_NEOX_RANGE = get_benchmark_range(
    full_range=[True, False],
    ci_range=[True],
)


# ---------------------------------------------------------------------------
# Benchmark 1: RoPE only
# ---------------------------------------------------------------------------

ROPE_LINE_VALS = ["fi", "rope_v0", "rope_v1", "rope_v2"]
ROPE_LINE_NAMES = ["FlashInfer", "SGL RoPE v0", "SGL RoPE v1", "SGL RoPE v2"]
ROPE_STYLES = [("green", "-."), ("red", "-"), ("orange", "-"), ("blue", "--")]

rope_configs = list(itertools.product(QK_HEAD_RANGE, IS_NEOX_RANGE, BS_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_q_k_heads", "is_neox", "batch_size"],
        x_vals=rope_configs,
        line_arg="provider",
        line_vals=ROPE_LINE_VALS,
        line_names=ROPE_LINE_NAMES,
        styles=ROPE_STYLES,
        ylabel="us",
        plot_name="rope-performance",
        args={},
    )
)
def benchmark(batch_size: int, num_q_k_heads: str, is_neox: bool, provider: str):
    qo, kv = num_q_k_heads.split(",")
    num_qo_heads = int(qo)
    num_kv_heads = int(kv)
    q = torch.randn(
        (batch_size, num_qo_heads, ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    k = torch.randn(
        (batch_size, num_kv_heads, ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    seed = batch_size << 16 | num_qo_heads << 8 | num_kv_heads << 4 | is_neox
    torch.random.manual_seed(seed)
    positions = torch.randint(
        MAX_SEQ_LEN, (batch_size,), device=DEFAULT_DEVICE, dtype=torch.int64
    )
    torch.cuda.synchronize()

    FN_MAP = {
        "fi": flashinfer_rope,
        "rope_v0": sglang_rope_v0,
        "rope_v1": sglang_rope_v1,
        "rope_v2": sglang_rope_v2,
    }
    fn = lambda: FN_MAP[provider](q, k, positions, is_neox)
    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Benchmark 2: RoPE + KV cache store
# ---------------------------------------------------------------------------

STORE_LINE_VALS = ["rope_v0_store", "rope_v1_store", "rope_v2_store"]
STORE_LINE_NAMES = ["SGL RoPE v0 + Store", "SGL RoPE v1 + Store", "SGL RoPE v2 + Store"]
STORE_STYLES = [("red", "-"), ("orange", "-"), ("blue", "--")]

store_configs = list(itertools.product(QK_HEAD_RANGE, IS_NEOX_RANGE, BS_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_q_k_heads", "is_neox", "batch_size"],
        x_vals=store_configs,
        line_arg="provider",
        line_vals=STORE_LINE_VALS,
        line_names=STORE_LINE_NAMES,
        styles=STORE_STYLES,
        ylabel="us",
        plot_name="rope-store-performance",
        args={},
    )
)
def benchmark_store(batch_size: int, num_q_k_heads: str, is_neox: bool, provider: str):
    qo, kv = num_q_k_heads.split(",")
    num_qo_heads = int(qo)
    num_kv_heads = int(kv)
    q = torch.randn(
        (batch_size, num_qo_heads, ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    k = torch.randn(
        (batch_size, num_kv_heads, ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    v = torch.randn(
        (batch_size, num_kv_heads, ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    row_size = num_kv_heads * ROPE_DIM
    k_cache = torch.zeros(
        CACHE_SIZE, row_size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    v_cache = torch.zeros(
        CACHE_SIZE, row_size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    out_loc = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE, dtype=torch.int64)[
        :batch_size
    ]
    seed = batch_size << 16 | num_qo_heads << 8 | num_kv_heads << 4 | is_neox
    torch.random.manual_seed(seed)
    positions = torch.randint(
        MAX_SEQ_LEN, (batch_size,), device=DEFAULT_DEVICE, dtype=torch.int64
    )
    torch.cuda.synchronize()

    FN_MAP = {
        "rope_v0_store": rope_v0_store,
        "rope_v1_store": rope_v1_store,
        "rope_v2_store": rope_v2_store,
    }
    fn = lambda: FN_MAP[provider](
        q, k, v, k_cache, v_cache, positions, out_loc, is_neox
    )
    return run_benchmark(fn)


if __name__ == "__main__":
    print("Running RoPE performance benchmark...")
    benchmark.run(print_data=True)
    print("\nRunning RoPE + KV cache store performance benchmark...")
    benchmark_store.run(print_data=True)
