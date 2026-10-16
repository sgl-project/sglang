"""The verify attention megakernel (T = 8 and 16) against the stock spec-decode
verify sequence it replaces: QK norm with the FP8 Q/KV store, TRT-LLM-gen paged
decode with q_len_per_req=T, then norm + gate. Single GPU."""

from __future__ import annotations

import functools
import inspect

import flashinfer
import pytest
import torch

from sglang.kernels.ops.muse_spark.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark.norm_gate import rms_norm_gate
from sglang.test.ci.ci_register import register_cuda_ci

try:
    import helion
except ModuleNotFoundError as error:
    if error.name != "helion":
        raise
    HELION_MEGAKERNELS = False
else:
    HELION_MEGAKERNELS = (
        "cross_loop_pipeline" in inspect.signature(helion.Config).parameters
    )
    from sglang.kernels.ops.muse_spark.helion_megakernels.attention_verify_megakernel import (
        BMM1_SCALE,
        EPS,
        HEAD_DIM,
        KV_WIDTH,
        NUM_KV_HEADS,
        NUM_Q_HEADS,
        PAGE_SIZE,
        Q_WIDTH,
        QKV_WIDTH,
        MuseAttentionVerifyMegakernel,
    )

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.skipif(
        not HELION_MEGAKERNELS, reason="requires Helion with cross-loop pipelining"
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the megakernels are specialized for SM10x",
    ),
]

MAX_CONTEXT = 1 << 20
ROPE_CAPACITY = 32768


@functools.cache
def _megakernel(
    tokens: int, sliding_window: bool, bmm1_scale: float, eps: float
) -> MuseAttentionVerifyMegakernel:
    return MuseAttentionVerifyMegakernel(
        torch.device("cuda"),
        tokens=tokens,
        sliding_window=sliding_window,
        bmm1_scale=bmm1_scale,
        eps=eps,
    )


@functools.cache
def _qk_norm(eps: float) -> FusedQKNorm:
    return FusedQKNorm(
        head_dim=HEAD_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        eps=eps,
    )


@functools.cache
def _flashinfer_buffers(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    nbytes = flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(
        tokens, NUM_Q_HEADS, sms
    )
    counters = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    return counters, workspace


def _rope_cache() -> torch.Tensor:
    exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM
    freqs = torch.outer(
        torch.arange(ROPE_CAPACITY, dtype=torch.float64), 1.0 / (10000.0**exponent)
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).float().cuda()


def _make_case(
    tokens: int,
    sliding_window: bool,
    seq_len: int,
    seed: int,
    padded: tuple[int, ...] = (),
) -> dict:
    """One verify step at KV length seq_len; the T new tokens are the last.
    Tokens in padded get cache_loc 0 (the CUDA-graph padding slot)."""

    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    pool_pages = pages + 16
    k_raw = torch.randn(pool_pages * PAGE_SIZE, KV_WIDTH, generator=g, device="cuda")
    rows = k_raw.view(-1, HEAD_DIM)
    k_norm = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + 1e-5)
    v_raw = torch.randn(pool_pages * PAGE_SIZE, KV_WIDTH, generator=g, device="cuda")
    block_tables = torch.zeros(
        1, MAX_CONTEXT // PAGE_SIZE, dtype=torch.int32, device="cuda"
    )
    # Page 0 holds slot 0, the padding sentinel: requests never own it.
    perm = torch.randperm(pool_pages - 1, generator=g, device="cuda")[:pages] + 1
    block_tables[0, :pages] = perm.to(torch.int32)
    positions = torch.arange(seq_len - tokens, seq_len, device="cuda")
    cache_loc = block_tables[0, positions // PAGE_SIZE].long() * PAGE_SIZE
    cache_loc = cache_loc + positions % PAGE_SIZE
    cache_loc[list(padded)] = 0
    return {
        "sliding_window": sliding_window,
        "seq_len": seq_len,
        # Unit-scale QKV projections: the gate spans both sigmoid tails.
        "qkv": torch.randn(tokens, QKV_WIDTH, generator=g, device="cuda").bfloat16(),
        "positions": positions,
        "cos_sin_cache": _rope_cache(),
        "k_cache": k_norm.bfloat16()
        .to(torch.float8_e4m3fn)
        .view(pool_pages, PAGE_SIZE * KV_WIDTH),
        "v_cache": (v_raw * 0.25)
        .bfloat16()
        .to(torch.float8_e4m3fn)
        .view(pool_pages, PAGE_SIZE * KV_WIDTH),
        "cache_loc": cache_loc,
        "block_tables": block_tables,
        "seq_lens": torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
    }


def _clone(case: dict) -> dict:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in case.items()}


def _with_page_metadata_tail(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy page rows into storage with an untouched qparam-sized tail."""

    pages, page_data = cache.shape
    qparam_tail = PAGE_SIZE * NUM_KV_HEADS * torch.int32.itemsize
    page_stride = page_data + qparam_tail
    storage = torch.empty(pages * page_stride, dtype=cache.dtype, device=cache.device)
    storage.view(torch.uint8).fill_(0xA5)
    view = torch.as_strided(storage, cache.shape, (page_stride, 1))
    view.copy_(cache)
    tails = storage.view(torch.uint8).view(pages, page_stride)[:, page_data:]
    return view, tails


def _paged_hnd(cache: torch.Tensor) -> torch.Tensor:
    return cache.view(-1, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)


def _run_megakernel(case: dict, bmm1_scale: float, eps: float) -> torch.Tensor:
    tokens = case["qkv"].shape[0]
    return _megakernel(tokens, case["sliding_window"], bmm1_scale, eps)(
        case["qkv"],
        case["positions"],
        case["cos_sin_cache"],
        case["k_cache"],
        case["v_cache"],
        case["cache_loc"],
        case["block_tables"],
        case["seq_lens"],
    )


def _stock(
    case: dict, bmm1_scale: float, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(gated output, FP8 Q) of the served verify path."""

    qkv, rope = case["qkv"], case["sliding_window"]
    tokens = qkv.shape[0]
    q_fp8, _ = _qk_norm(eps)._forward_triton(
        qkv,
        case["positions"] if rope else None,
        case["cos_sin_cache"] if rope else None,
        (
            case["k_cache"].view(-1, KV_WIDTH),
            case["v_cache"].view(-1, KV_WIDTH),
            case["cache_loc"],
        ),
    )
    counters, workspace = _flashinfer_buffers(tokens)
    attn = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q_fp8.view(tokens, NUM_Q_HEADS, HEAD_DIM),
        kv_cache=(_paged_hnd(case["k_cache"]), _paged_hnd(case["v_cache"])),
        workspace_buffer=workspace,
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        max_seq_len=MAX_CONTEXT,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        window_left=2047 if rope else -1,
        out_dtype=torch.bfloat16,
        backend="trtllm-gen",
        q_len_per_req=tokens,
        multi_ctas_kv_counter_buffer=counters,
    )
    out = rms_norm_gate(
        attn.view(tokens, Q_WIDTH), qkv[:, -Q_WIDTH:], None, head_dim=HEAD_DIM, eps=eps
    )
    return out, q_fp8


def _fp64_reference(
    case: dict, q_fp8: torch.Tensor, bmm1_scale: float, eps: float
) -> torch.Tensor:
    """FP64 causal attention over the post-store cache, then FP64 norm + gate."""

    seq_len, tokens = case["seq_len"], case["qkv"].shape[0]
    pages = case["block_tables"][0, : (seq_len + PAGE_SIZE - 1) // PAGE_SIZE].long()

    def logical(cache: torch.Tensor) -> torch.Tensor:
        c = cache.view(-1, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).index_select(0, pages)
        c = c.permute(2, 0, 1, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)
        return c[:, :seq_len].double()

    key, value = logical(case["k_cache"]), logical(case["v_cache"])
    q = q_fp8.view(tokens, NUM_Q_HEADS, HEAD_DIM).double()
    kv_head = torch.arange(NUM_Q_HEADS, device=q.device) // (
        NUM_Q_HEADS // NUM_KV_HEADS
    )
    scores = torch.einsum("thd,hkd->thk", q, key[kv_head]) * bmm1_scale
    q_pos = torch.arange(seq_len - tokens, seq_len, device=q.device)[:, None]
    back = q_pos - torch.arange(seq_len, device=q.device)[None, :]
    visible = back >= 0
    if case["sliding_window"]:
        visible &= back <= 2047
    scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
    attn = torch.einsum("thk,hkd->thd", torch.softmax(scores, dim=-1), value[kv_head])
    attn = attn * torch.rsqrt(attn.pow(2).mean(-1, keepdim=True) + eps)
    gate = case["qkv"][:, -Q_WIDTH:].double().reshape(tokens, NUM_Q_HEADS, HEAD_DIM)
    return (attn * torch.sigmoid(gate)).reshape(tokens, Q_WIDTH)


def _rrmse_cos(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    a, e = actual.double().reshape(-1), expected.double().reshape(-1)
    rrmse = float((a - e).norm() / e.norm())
    return rrmse, float(torch.dot(a, e) / (a.norm() * e.norm()))


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize(
    "sliding_window,seq_len,padded",
    [
        (True, 300, ()),
        (True, 2500, ()),
        (True, 2100, (0, -1)),
        # Long before short: no state may leak between launches.
        (False, 2500, ()),
        (False, 300, ()),
        (False, 16, ()),
        (False, 17920, (-1,)),
    ],
    ids=lambda v: (
        ("swa" if v else "full")
        if isinstance(v, bool)
        else (f"pad{len(v)}" if isinstance(v, tuple) else f"len{v}")
    ),
)
# The served scalars, then values far enough off that a baked-in constant fails.
@pytest.mark.parametrize(
    "bmm1_scale,eps", [(BMM1_SCALE, EPS), (0.5, 0.25)], ids=["served", "off"]
)
@pytest.mark.parametrize("tokens", [8, 16], ids=lambda t: f"t{t}")
@torch.inference_mode()
def test_attention_verify_matches_stock(
    tokens: int,
    sliding_window: bool,
    seq_len: int,
    padded: tuple[int, ...],
    bmm1_scale: float,
    eps: float,
) -> None:
    base = _make_case(tokens, sliding_window, seq_len, seed=seq_len, padded=padded)
    stock_case, mk_case = _clone(base), _clone(base)
    expected, q_fp8 = _stock(stock_case, bmm1_scale, eps)
    actual = _run_megakernel(mk_case, bmm1_scale, eps)
    torch.cuda.synchronize()

    assert _bytes_equal(stock_case["k_cache"], mk_case["k_cache"])
    assert _bytes_equal(stock_case["v_cache"], mk_case["v_cache"])
    assert torch.isfinite(actual.float()).all()
    rrmse, cos = _rrmse_cos(actual, expected)
    # An MK at least as close to FP64 as stock also passes.
    exact = _fp64_reference(stock_case, q_fp8, bmm1_scale, eps)
    rrmse_mk, cos_mk = _rrmse_cos(actual, exact)
    rrmse_stock, cos_stock = _rrmse_cos(expected, exact)
    max_abs = float((actual.double() - exact).abs().max())
    assert (rrmse <= 1e-2 and cos >= 0.99995) or (
        rrmse_mk <= rrmse_stock and cos_mk >= cos_stock
    ), (
        f"vs stock {rrmse:.3e}/{cos:.7f}, vs fp64 mk {rrmse_mk:.3e} "
        f"stock {rrmse_stock:.3e}, max abs {max_abs:.3e}"
    )


@pytest.mark.parametrize("sliding_window", [False, True])
@pytest.mark.parametrize("tokens", [8, 16], ids=lambda t: f"t{t}")
@torch.inference_mode()
def test_attention_verify_supports_per_page_metadata_gaps(
    tokens: int, sliding_window: bool
) -> None:
    base = _make_case(tokens, sliding_window, seq_len=300, seed=71)
    contiguous = _clone(base)
    strided = _clone(base)
    strided["k_cache"], k_tails = _with_page_metadata_tail(base["k_cache"])
    strided["v_cache"], v_tails = _with_page_metadata_tail(base["v_cache"])
    k_tails_before = k_tails.clone()
    v_tails_before = v_tails.clone()

    expected = _run_megakernel(contiguous, BMM1_SCALE, EPS)
    actual = _run_megakernel(strided, BMM1_SCALE, EPS)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert _bytes_equal(strided["k_cache"], contiguous["k_cache"])
    assert _bytes_equal(strided["v_cache"], contiguous["v_cache"])
    assert torch.equal(k_tails, k_tails_before)
    assert torch.equal(v_tails, v_tails_before)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
