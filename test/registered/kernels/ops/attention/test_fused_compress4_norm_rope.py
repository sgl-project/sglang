from __future__ import annotations

import sys
from typing import Tuple, Union

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4 import (
    compress_forward,
    compress_forward_norm_rope_store,
    compress_norm_rope_store,
)
from sglang.srt.utils import get_device, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.deepseek_v4.common import (
    LegacyContext,
    PagedContext,
    make_legacy_context,
    make_paged_context,
    make_state_pool,
    to_seq_extend,
)

# HIP/gfx95 only: compress_forward_norm_rope_store is the fused c4 decode path.
register_amd_ci(est_time=90, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(
    not is_hip(), reason="fused compress+norm+rope is HIP/gfx95 only"
)

Context = Union[LegacyContext, PagedContext]

# c4 input row layout: | kv_overlap | kv | score_overlap | score |, each head_dim.
RATIO = 4
WINDOW = 8  # 2 * RATIO
ROPE_DIM = 64
NORM_EPS = 1.0e-6
N_DECODE_STEPS = 8  # spans >=2 compress boundaries plus non-boundary steps

# Per-head-dim store epilogue the fused kernel dispatches to. head_dim 512 is the
# flashmla epilogue (bf16 store into a [slots, head_dim] bf16 buffer, page_size
# 1); head_dim 128 is the indexer epilogue (fp8 store into the page-major
# index-k-with-scale buffer, head_dim + 4 bytes per token, page_size 64). The
# indexer store is always fp8; bf16_store is a flashmla-only option.
CONFIGS = {
    512: dict(page_size=1, bf16_store=True, store="bf16"),
    128: dict(page_size=64, bf16_store=False, store="fp8"),
}

SRC_DTYPES = [
    pytest.param(torch.float32, id="src_fp32"),
    # bf16 source also exercises the kv_score bf16 transport the compressor
    # kernels widen on load.
    pytest.param(torch.bfloat16, id="src_bf16"),
]


def _make_ctx(mode: str, head_dim: int) -> Context:
    if mode == "legacy":
        return make_legacy_context(bs=1, compress_ratio=RATIO, head_dim=head_dim)
    return make_paged_context(bs=1, compress_ratio=RATIO, head_dim=head_dim)


def _make_inputs(
    num_q: int, head_dim: int, seed: int, src_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv_score_input_cpu = torch.randn(
        num_q, head_dim * 4, generator=g, dtype=torch.float32
    )
    # Round to the transport dtype up front so both paths see identical inputs;
    # what is under test is fused-vs-chain, not how far bf16 rounds the inputs.
    kv_score_input_cpu = kv_score_input_cpu.to(src_dtype).float()
    ape_cpu = torch.randn(WINDOW, head_dim, generator=g, dtype=torch.float32)
    return kv_score_input_cpu, ape_cpu


def _to_src(t: torch.Tensor, src_dtype: torch.dtype) -> torch.Tensor:
    return t.to(get_device()).to(src_dtype)


def _make_cache(
    head_dim: int, num_slots: int, page_size: int, store: str
) -> torch.Tensor:
    if store == "bf16":
        return torch.zeros(
            num_slots, head_dim, dtype=torch.bfloat16, device=get_device()
        )
    bytes_per_token = head_dim + 4  # fp8 codes + 4 scale bytes
    num_pages = (num_slots + page_size) // page_size + 1
    return torch.zeros(
        num_pages, page_size * bytes_per_token, dtype=torch.uint8, device=get_device()
    )


def _assert_cache_close(store: str, a: torch.Tensor, b: torch.Tensor) -> None:
    """The two paths must produce interchangeable caches.

    The fused kernel keeps the compressed row in fp32 registers while the
    two-kernel chain rounds it to bf16 between launches, so a handful of fp8
    codes can land one code apart; scales are computed from the same amax and
    stay identical. bf16 store carries the same intermediate difference forward.
    """
    if store == "bf16":
        torch.testing.assert_close(a.float(), b.float(), atol=2e-2, rtol=2e-2)
        return
    diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
    assert diff.max().item() <= 1, f"fp8 store diverged by >1 code: max={diff.max()}"
    assert diff.ne(0).float().mean().item() < 5e-3, "too many differing fp8 codes"


@pytest.mark.parametrize("head_dim", list(CONFIGS))
@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("src_dtype", SRC_DTYPES)
@pytest.mark.parametrize("prefix_len", [0, 6, 256])
def test_fused_matches_chain_decode(
    head_dim: int, mode: str, src_dtype: torch.dtype, prefix_len: int
) -> None:
    """Fused compress+norm+rope+store must match compress_forward + store.

    A prefix that is not a multiple of the ratio (6) forces a partial first
    block whose overlap is read from the state buffer; prefix 256 exercises a
    multi-page paged layout. Stepping N_DECODE_STEPS past the prefix crosses
    several compress boundaries and includes non-compress accumulate steps.
    """
    cfg = CONFIGS[head_dim]
    device = get_device()
    torch.manual_seed(head_dim + prefix_len + int(src_dtype == torch.bfloat16))

    ctx = _make_ctx(mode, head_dim)
    pool_chain = make_state_pool(ctx.num_pages, RATIO, head_dim)
    pool_fused = make_state_pool(ctx.num_pages, RATIO, head_dim)

    seq_len_total = prefix_len + N_DECODE_STEPS
    kv_full_cpu, ape_cpu = _make_inputs(
        seq_len_total, head_dim, seed=seq_len_total, src_dtype=src_dtype
    )
    ape = ape_cpu.to(device)
    norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    freqs_cis = precompute_freqs_cis(
        ROPE_DIM, seq_len_total + 4, 0, 10000, 1, 32, 1
    ).to(device)

    cache_chain = _make_cache(head_dim, N_DECODE_STEPS, cfg["page_size"], cfg["store"])
    cache_fused = _make_cache(head_dim, N_DECODE_STEPS, cfg["page_size"], cfg["store"])

    # Bring both pools to the same state with an identical prefix prefill
    # (compress_forward only -- the fused path is decode-only).
    if prefix_len > 0:
        seq_lens_cpu, extend_lens_cpu, _ = to_seq_extend([(prefix_len, prefix_len)])
        kv_pref = _to_src(kv_full_cpu[:prefix_len], src_dtype)
        for pool in (pool_chain, pool_fused):
            plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, prefix_len)
            compress_forward(
                pool, kv_pref, ape, plan, head_dim=head_dim, compress_ratio=RATIO
            )

    next_slot = 0
    for k in range(N_DECODE_STEPS):
        pos = prefix_len + k
        cur_seq_len = pos + 1
        is_boundary = cur_seq_len % RATIO == 0
        seq_lens_gpu = torch.tensor([cur_seq_len], dtype=torch.int64, device=device)
        kv_step = _to_src(kv_full_cpu[pos : pos + 1], src_dtype)
        out_loc = torch.tensor(
            [next_slot if is_boundary else 0], dtype=torch.int64, device=device
        )

        before_chain = cache_chain.clone()
        before_fused = cache_fused.clone()

        plan = ctx.make_decode_plan(seq_lens_gpu)
        row = compress_forward(
            pool_chain, kv_step, ape, plan, head_dim=head_dim, compress_ratio=RATIO
        )
        compress_norm_rope_store(
            row,
            plan,
            norm_weight=norm_weight,
            norm_eps=NORM_EPS,
            freq_cis=freqs_cis,
            out_loc=out_loc,
            kvcache=cache_chain,
            page_size=cfg["page_size"],
            bf16_store=cfg["bf16_store"],
        )

        plan_fused = ctx.make_decode_plan(seq_lens_gpu)
        compress_forward_norm_rope_store(
            kv_score_buffer=pool_fused,
            kv_score_input=kv_step,
            ape=ape,
            plan=plan_fused,
            head_dim=head_dim,
            norm_weight=norm_weight,
            norm_eps=NORM_EPS,
            freq_cis=freqs_cis,
            out_loc=out_loc,
            kvcache=cache_fused,
            page_size=cfg["page_size"],
            bf16_store=cfg["bf16_store"],
        )

        if is_boundary:
            next_slot += 1
        else:
            # A non-compress decode step must not touch either cache.
            assert torch.equal(
                cache_chain, before_chain
            ), f"chain wrote cache on non-compress step (pos={pos})"
            assert torch.equal(
                cache_fused, before_fused
            ), f"fused wrote cache on non-compress step (pos={pos})"

        _assert_cache_close(cfg["store"], cache_chain, cache_fused)

    assert next_slot >= 2, "test did not exercise at least two compress boundaries"
    # The state buffers accumulate the same verbatim token rows in both paths.
    torch.testing.assert_close(pool_fused, pool_chain, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
