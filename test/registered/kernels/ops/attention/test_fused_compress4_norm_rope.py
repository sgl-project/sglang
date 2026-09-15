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
register_amd_ci(est_time=150, suite="nightly-amd-kernel-1-gpu", nightly=True)

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

# Store epilogue the fused kernel dispatches to, one scenario per reachable
# (head_dim, store) pairing. All caches are page-major uint8 buffers.
#
#   512-bf16 : flashmla epilogue, plain bf16 store (head_dim * 2 bytes/token).
#              This is the unified-KV-triton path (compressor_v2 bf16_store=True).
#   512-fp8  : flashmla epilogue, UE8M0 pack -- 448 nope codes in 7 fp8 groups
#              of 64 + 64 rope elements kept in bf16, with 8 scale bytes/token in
#              a trailing region. This is the production non-indexer extra-key
#              cache, where compressor_v2 leaves bf16_store=False; page_size > 1
#              so the per-token offset math in the store is exercised.
#   128-fp8  : indexer epilogue, always fp8 (head_dim codes + one fp32 scale per
#              token). bf16_store is a flashmla-only option and does not apply.
#
# buffer_dtype is the state-pool (BufferFloat) dtype; the last scenario runs it
# in bf16 while ape/input stay fp32 so the mixed BufferFloat != InputFloat path
# is covered on the reachable 512 FP8 store.
CONFIGS = {
    "512-bf16": dict(
        head_dim=512,
        page_size=1,
        bf16_store=True,
        store="bf16",
        buffer_dtype=torch.float32,
    ),
    "512-fp8": dict(
        head_dim=512,
        page_size=16,
        bf16_store=False,
        store="flashmla_fp8",
        buffer_dtype=torch.float32,
    ),
    "128-fp8": dict(
        head_dim=128,
        page_size=64,
        bf16_store=False,
        store="indexer_fp8",
        buffer_dtype=torch.float32,
    ),
    "512-fp8-bf16buf": dict(
        head_dim=512,
        page_size=16,
        bf16_store=False,
        store="flashmla_fp8",
        buffer_dtype=torch.bfloat16,
    ),
}

# flashmla FP8 pack constants (head_dim 512): the nope half is 448 elements in 7
# UE8M0 groups of 64, the rope half is 64 bf16 elements, and each token's value
# region is 576 bytes with an 8-byte scale slot at 576*page_size.
_FLASHMLA_NOPE = 448
_FLASHMLA_GROUP = 64
_FLASHMLA_VALUE_BYTES = 576
_FLASHMLA_SCALE_BYTES = 8


def _make_ctx(mode: str, head_dim: int) -> Context:
    if mode == "legacy":
        return make_legacy_context(bs=1, compress_ratio=RATIO, head_dim=head_dim)
    return make_paged_context(bs=1, compress_ratio=RATIO, head_dim=head_dim)


def _make_inputs(
    num_q: int, head_dim: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    # The compressor computes in (and consumes kv_score in) the ape/fp32 dtype.
    kv_score_input_cpu = torch.randn(
        num_q, head_dim * 4, generator=g, dtype=torch.float32
    )
    ape_cpu = torch.randn(WINDOW, head_dim, generator=g, dtype=torch.float32)
    return kv_score_input_cpu, ape_cpu


def _to_dev(t: torch.Tensor) -> torch.Tensor:
    return t.to(get_device())


def _flashmla_page_bytes(page_size: int) -> int:
    # Mirrors the kernel's kPageBytes: 576 value bytes + 8 scale bytes per token,
    # rounded up to a multiple of the 576-byte value stride.
    used = _FLASHMLA_VALUE_BYTES * page_size + _FLASHMLA_SCALE_BYTES * page_size
    return -(-used // _FLASHMLA_VALUE_BYTES) * _FLASHMLA_VALUE_BYTES


def _make_cache(
    head_dim: int, num_slots: int, page_size: int, store: str
) -> torch.Tensor:
    # All caches are page-major uint8 buffers.
    #   bf16          : head_dim * 2 bytes/token.
    #   indexer_fp8   : head_dim fp8 codes + 4 scale bytes/token.
    #   flashmla_fp8  : 576 value bytes/token + 8 scale bytes/token, page padded.
    num_pages = (num_slots + page_size) // page_size + 1
    if store == "flashmla_fp8":
        page_bytes = _flashmla_page_bytes(page_size)
    else:
        bytes_per_token = head_dim * 2 if store == "bf16" else head_dim + 4
        page_bytes = page_size * bytes_per_token
    return torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device=get_device())


def _dequant_indexer_fp8(
    cache: torch.Tensor, head_dim: int, page_size: int
) -> torch.Tensor:
    """Decode the index-k store: head_dim e4m3 codes + one fp32 scale per token."""
    n_code = page_size * head_dim
    codes = (
        cache[:, :n_code]
        .view(torch.float8_e4m3fn)
        .float()
        .view(cache.shape[0], page_size, head_dim)
    )
    scale = (
        cache[:, n_code:]
        .reshape(cache.shape[0], page_size, 4)
        .view(torch.float32)  # [pages, page_size, 1]
    )
    return codes * scale


def _dequant_flashmla_fp8(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    """Decode the flashmla nope-fp8 + rope-bf16 pack into [pages, page_size, 512].

    Each token's value region is 576 bytes: 448 e4m3 codes in 7 UE8M0 groups of
    64, then 64 rope elements as bf16 (128 bytes). Scales live in a trailing
    region at 576*page_size, 8 bytes/token (7 used, one UE8M0 exponent byte per
    group); the fp32 scale is 2^(exp - 127).
    """
    pages = cache.shape[0]
    value = cache[:, : _FLASHMLA_VALUE_BYTES * page_size].reshape(
        pages, page_size, _FLASHMLA_VALUE_BYTES
    )
    codes = value[:, :, :_FLASHMLA_NOPE].contiguous().view(torch.float8_e4m3fn).float()
    rope = value[:, :, _FLASHMLA_NOPE:].contiguous().view(torch.bfloat16).float()
    scale_base = _FLASHMLA_VALUE_BYTES * page_size
    scale_bytes = cache[
        :, scale_base : scale_base + _FLASHMLA_SCALE_BYTES * page_size
    ].reshape(pages, page_size, _FLASHMLA_SCALE_BYTES)
    n_groups = _FLASHMLA_NOPE // _FLASHMLA_GROUP
    exp = scale_bytes[:, :, :n_groups].int()
    scale = torch.exp2((exp - 127).float())  # [pages, page_size, n_groups]
    nope = codes.reshape(pages, page_size, n_groups, _FLASHMLA_GROUP) * scale[..., None]
    return torch.cat([nope.reshape(pages, page_size, _FLASHMLA_NOPE), rope], dim=-1)


def _assert_cache_close(
    store: str, a: torch.Tensor, b: torch.Tensor, head_dim: int, page_size: int
) -> None:
    """The two paths must produce interchangeable caches.

    The fused kernel keeps the compressed row in fp32 registers while the
    two-kernel chain rounds it to bf16 between launches. bf16 store carries that
    intermediate difference straight into the cache; the fp8 stores keep codes
    essentially identical while a rare amax tie shifts one token's scale in its
    lowest mantissa bits, so they are compared in dequantized value space where
    at most a handful of elements may land one e4m3 code apart.
    """
    if store == "bf16":
        a_f = a.view(torch.bfloat16).float()
        b_f = b.view(torch.bfloat16).float()
        torch.testing.assert_close(a_f, b_f, atol=2e-2, rtol=2e-2)
        return
    if store == "flashmla_fp8":
        va = _dequant_flashmla_fp8(a, page_size)
        vb = _dequant_flashmla_fp8(b, page_size)
    else:
        va = _dequant_indexer_fp8(a, head_dim, page_size)
        vb = _dequant_indexer_fp8(b, head_dim, page_size)
    diff = (va - vb).abs()
    # No element may diverge by more than a single e4m3 code step...
    step_tol = 2e-1 + 2e-1 * vb.abs()
    assert (diff > step_tol).sum().item() == 0, "value diverged beyond one fp8 code"
    # ...and only a handful may shift at all.
    small_tol = 2e-2 + 2e-2 * vb.abs()
    assert (diff > small_tol).float().mean().item() < 5e-3, "too many fp8 codes shifted"


@pytest.mark.parametrize("scenario", list(CONFIGS))
@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("prefix_len", [0, 6, 256])
def test_fused_matches_chain_decode(scenario: str, mode: str, prefix_len: int) -> None:
    """Fused compress+norm+rope+store must match compress_forward + store.

    Runs across every reachable store epilogue (see CONFIGS): flashmla bf16,
    flashmla FP8/UE8M0, and the indexer FP8 pack, including a mixed
    BufferFloat != InputFloat state pool. A prefix that is not a multiple of the
    ratio (6) forces a partial first block whose overlap is read from the state
    buffer; prefix 256 exercises a multi-page paged layout. Stepping
    N_DECODE_STEPS past the prefix crosses several compress boundaries and
    includes non-compress accumulate steps.
    """
    cfg = CONFIGS[scenario]
    head_dim = cfg["head_dim"]
    buffer_dtype = cfg["buffer_dtype"]
    device = get_device()
    torch.manual_seed(head_dim + prefix_len)

    ctx = _make_ctx(mode, head_dim)
    pool_chain = make_state_pool(ctx.num_pages, RATIO, head_dim, dtype=buffer_dtype)
    pool_fused = make_state_pool(ctx.num_pages, RATIO, head_dim, dtype=buffer_dtype)

    seq_len_total = prefix_len + N_DECODE_STEPS
    kv_full_cpu, ape_cpu = _make_inputs(seq_len_total, head_dim, seed=seq_len_total)
    ape = ape_cpu.to(device)
    norm_weight = torch.randn(head_dim, dtype=torch.float32, device=device)
    freqs_cis = precompute_freqs_cis(
        ROPE_DIM, seq_len_total + 4, 0, 10000, 1, 32, 1
    ).to(device)

    cache_chain = _make_cache(head_dim, N_DECODE_STEPS, cfg["page_size"], cfg["store"])
    cache_fused = _make_cache(head_dim, N_DECODE_STEPS, cfg["page_size"], cfg["store"])

    # Bring both pools to the same state with an identical prefix prefill
    # (compress_forward only -- the fused path is decode-only).
    if prefix_len > 0:
        seq_lens_cpu, extend_lens_cpu, _ = to_seq_extend([(prefix_len, prefix_len)])
        kv_pref = _to_dev(kv_full_cpu[:prefix_len])
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
        kv_step = _to_dev(kv_full_cpu[pos : pos + 1])
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
            assert torch.equal(cache_chain, before_chain), (
                f"chain wrote cache on non-compress step (pos={pos})"
            )
            assert torch.equal(cache_fused, before_fused), (
                f"fused wrote cache on non-compress step (pos={pos})"
            )

        _assert_cache_close(
            cfg["store"], cache_chain, cache_fused, head_dim, cfg["page_size"]
        )

    assert next_slot >= 2, "test did not exercise at least two compress boundaries"
    # The state buffers accumulate the same verbatim token rows in both paths.
    torch.testing.assert_close(pool_fused, pool_chain, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
