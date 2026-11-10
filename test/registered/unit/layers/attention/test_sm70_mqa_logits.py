"""SM70 DSA indexer scores at 32 heads, the tensor-core (WMMA) kernels.

The fp8 operands are decoded to fp16 with a bit trick that must be exact for
every E4M3FN code. A row's score must not depend on the other rows of the
launch: prefill, decode and the 4-row MTP verify then pick the same top-k keys.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The SM70 indexer kernels require an NVIDIA V100",
)

HEADS = 32
PAGE = 64


def _fp8(*shape):
    return torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)


def _i32(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _reference(q, k, weights, k_scale):
    """fp64 sum_h relu(q[m, h] . k[n]) * weights[m, h] * k_scale[n] -> [M, N]."""
    dots = torch.einsum("mhd,nd->mhn", q.double(), k.double())
    return (dots.clamp(min=0) * weights.double()[:, :, None]).sum(1) * k_scale.double()


def _ragged(q, k, k_scale, weights, starts, ends, clean_logits=True):
    from sglang.kernels.ops.attention.mqa_logits_sm70 import fp8_mqa_logits_sm70

    return fp8_mqa_logits_sm70(
        q, (k, k_scale), weights, _i32(starts), _i32(ends), clean_logits
    )


# With q = -1 the relu keeps the negative codes, so both signs are checked.
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_e4m3_decode_is_exact_for_every_code(sign):
    codes = torch.arange(256, device="cuda", dtype=torch.uint8)
    q = torch.zeros(1, HEADS, 128, device="cuda", dtype=torch.uint8)
    q[0, 0, 0] = torch.tensor(sign).to(torch.float8_e4m3fn).view(torch.uint8).item()
    k = torch.zeros(256, 128, device="cuda", dtype=torch.uint8)
    k[:, 0] = codes
    weights = torch.zeros(1, HEADS, device="cuda")
    weights[0, 0] = 1.0
    got = _ragged(
        q.view(torch.float8_e4m3fn),
        k.view(torch.float8_e4m3fn),
        torch.ones(256, device="cuda"),
        weights,
        [0],
        [256],
    )[0]
    nan = (codes & 0x7F) == 0x7F
    expected = (sign * codes.view(torch.float8_e4m3fn).float()).clamp(min=0)
    assert torch.equal(got[~nan], expected[~nan])
    # A NaN code scores 0, so it cannot take a top-k slot.
    assert not got[nan].any()


# Two query blocks of 4 (the second one partial) over two 1024-key spans (the
# second one partial). Queries 4-6 leave most key tiles unscored.
_STARTS = [0, 5, -3, 1000, 600, 300, 9]
_ENDS = [1100, 5, 40, 5000, 610, 301, 2]


@pytest.mark.parametrize("clean_logits", [True, False])
def test_ragged_matches_reference(clean_logits):
    torch.manual_seed(0)
    queries, keys = len(_STARTS), 1100
    q, k = _fp8(queries, HEADS, 128), _fp8(keys, 128)
    weights = torch.randn(queries, HEADS, device="cuda")
    k_scale = torch.rand(keys, device="cuda") + 0.05
    got = _ragged(q, k, k_scale, weights, _STARTS, _ENDS, clean_logits)

    ref = _reference(q, k, weights, k_scale)
    key_ids = torch.arange(keys, device="cuda")
    window = (key_ids >= _i32(_STARTS)[:, None]) & (key_ids < _i32(_ENDS)[:, None])
    torch.testing.assert_close(got[window].double(), ref[window], atol=1e-4, rtol=1e-5)
    masked = float("-inf") if clean_logits else 0.0
    assert torch.equal(got[~window], torch.full_like(got[~window], masked))


def _paged_case(seed):
    """Three rows over shuffled 64-token pages; row 1 has an unmapped page."""
    torch.manual_seed(seed)
    seq_lens = [70, 200, 10]
    pool_pages = 8
    tokens = _fp8(pool_pages * PAGE, 128)
    scales = torch.rand(pool_pages * PAGE, device="cuda") + 0.05
    kv = torch.zeros(pool_pages, PAGE * 132, device="cuda", dtype=torch.uint8)
    kv[:, : PAGE * 128] = tokens.view(torch.uint8).reshape(pool_pages, -1)
    kv[:, PAGE * 128 :] = scales.view(torch.uint8).reshape(pool_pages, -1)
    table = _i32([[5, 2, -1, -1], [0, -1, 7, 3], [4, -1, -1, -1]])
    return tokens, scales, kv, table, seq_lens


def _paged(q, kv, weights, seq_lens, table, max_seq_len, clean_logits=False):
    from sglang.kernels.ops.attention.mqa_logits_sm70 import (
        fp8_paged_mqa_logits_sm70,
    )

    return fp8_paged_mqa_logits_sm70(
        q, kv, weights, _i32(seq_lens), table, max_seq_len, clean_logits
    )


@pytest.mark.parametrize("clean_logits", [True, False])
def test_paged_matches_reference(clean_logits):
    tokens, scales, kv, table, seq_lens = _paged_case(seed=1)
    q = _fp8(3, HEADS, 128)
    weights = torch.randn(3, HEADS, device="cuda")
    max_seq_len = 192  # cuts row 1 short of its 200 tokens
    got = _paged(q, kv, weights, seq_lens, table, max_seq_len, clean_logits)

    fill = float("-inf") if clean_logits else 0.0
    for row, seq_len in enumerate(seq_lens):
        limit = min(seq_len, max_seq_len)
        for page in range(math.ceil(limit / PAGE)):
            lo, hi = page * PAGE, min((page + 1) * PAGE, limit)
            pool_page = int(table[row, page])
            if pool_page < 0:
                assert not got[row, lo:hi].any(), f"unmapped page {row}/{page}"
                continue
            ids = torch.arange(pool_page * PAGE, pool_page * PAGE + hi - lo)
            ref = _reference(
                q[row : row + 1], tokens[ids], weights[row : row + 1], scales[ids]
            )
            torch.testing.assert_close(
                got[row, lo:hi].double(), ref[0], atol=1e-4, rtol=1e-5
            )
        tail = got[row, limit:]
        assert torch.equal(tail, torch.full_like(tail, fill)), f"row {row} tail"


def test_rows_do_not_depend_on_neighbours():
    torch.manual_seed(2)
    queries, keys = len(_STARTS), 1100
    q, k = _fp8(queries, HEADS, 128), _fp8(keys, 128)
    weights = torch.randn(queries, HEADS, device="cuda")
    k_scale = torch.rand(keys, device="cuda") + 0.05
    batched = _ragged(q, k, k_scale, weights, _STARTS, _ENDS)
    for m in range(queries):
        alone = _ragged(
            q[m : m + 1],
            k,
            k_scale,
            weights[m : m + 1],
            _STARTS[m : m + 1],
            _ENDS[m : m + 1],
        )
        assert torch.equal(alone[0], batched[m]), f"ragged row {m}"

    tokens, scales, kv, table, seq_lens = _paged_case(seed=3)
    q_paged = _fp8(3, HEADS, 128)
    w_paged = torch.randn(3, HEADS, device="cuda")
    batched = _paged(q_paged, kv, w_paged, seq_lens, table, 256)
    for b in range(3):
        alone = _paged(
            q_paged[b : b + 1],
            kv,
            w_paged[b : b + 1],
            seq_lens[b : b + 1],
            table[b : b + 1],
            256,
        )
        assert torch.equal(alone[0], batched[b]), f"paged row {b}"

    # Decode (paged) scores a key exactly as prefill (ragged) does.
    row0_ids = torch.cat(
        [torch.arange(5 * PAGE, 6 * PAGE), torch.arange(2 * PAGE, 3 * PAGE)]
    )[:70]
    prefill = _ragged(
        q_paged[:1], tokens[row0_ids], scales[row0_ids], w_paged[:1], [0], [70]
    )
    assert torch.equal(prefill[0], batched[0, :70])
