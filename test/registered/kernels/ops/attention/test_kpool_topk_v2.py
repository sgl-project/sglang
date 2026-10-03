"""K-pool top-k v2 parity across selection paths, output transforms and routing.

Pool order is unspecified, so expanded token sets are compared sorted against
``torch.topk``. Concentrated scores cover threshold-bin overflow.
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
    fast_kpool_topk_transform_fused,
)
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_kpool_v2
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    build_kpool_topk_v2_plan,
    can_use_kpool_topk_v2,
    kpool_topk_v2_enabled,
    topk_from_pooled_history_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

POOL = 4


def _plan(pool_lens):
    plan = plan_topk_v2(pool_lens)
    # V2 reads the plan before its PDL wait; finish planning before launching top-k.
    torch.cuda.synchronize()
    return plan


def _reference(scores, pool_lens, token_seq_lens, topk, map_token):
    rows = []
    tail_cols = POOL - 1 if token_seq_lens is not None else 0
    for b in range(scores.shape[0]):
        n = int(pool_lens[b])
        k = min(n, topk)
        pools = (
            torch.topk(scores[b, :n].cpu(), k).indices if n > topk else torch.arange(n)
        )
        tokens = (pools[:, None] * POOL + torch.arange(POOL)).flatten().tolist()
        if token_seq_lens is not None:
            tokens += [n * POOL + i for i in range(int(token_seq_lens[b]) % POOL)]
        row = [map_token(b, t) for t in tokens]
        row += [-1] * (topk * POOL + tail_cols - len(row))
        rows.append(sorted(row))
    return torch.tensor(rows, dtype=torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,max_pools",
    list(itertools.product([1, 6, 48], [300, 9000, 20000, 70000])),
)
@pytest.mark.parametrize("mode", ["raw", "page1", "page64", "offset"])
@pytest.mark.parametrize("with_tail", [True, False])
def test_kpool_topk_v2(rows, max_pools, mode, with_tail):
    torch.manual_seed(rows * 7919 + max_pools)
    topk = 512  # GLM-5.3-Flash: index_topk=2048 over index_kpool=4
    pool_lens = torch.randint(0, max_pools + 1, (rows,), dtype=torch.int32)
    pool_lens[0] = max_pools
    token_seq_lens = pool_lens * POOL + torch.randint(
        0, POOL, (rows,), dtype=torch.int32
    )
    # Graph-style wide buffer: columns past each row's length hold garbage.
    width = (max_pools + 3) // 4 * 4 + 64
    scores = torch.randn(rows, width) + 10
    max_tokens = int(token_seq_lens.max()) + POOL

    kwargs = {}
    if mode == "page1":
        table = torch.randperm(rows * max_tokens, dtype=torch.int32).view(
            rows, max_tokens
        )
        kwargs = dict(page_table=table.cuda())
        map_token = lambda b, t: int(table[b, t])
    elif mode == "page64":
        pages = (max_tokens + 63) // 64
        table = torch.randperm(rows * pages, dtype=torch.int32).view(rows, pages)
        kwargs = dict(page_table=table.cuda(), page_size=64)
        map_token = lambda b, t: int(table[b, t // 64]) * 64 + t % 64
    elif mode == "offset":
        offsets = torch.randint(0, 1 << 20, (rows,), dtype=torch.int32)
        kwargs = dict(out_offsets=offsets.cuda())
        map_token = lambda b, t: t + int(offsets[b])
    else:
        map_token = lambda b, t: t

    pool_lens_gpu = pool_lens.cuda()
    out = torch.full(
        (rows, topk * POOL + (POOL - 1 if with_tail else 0)),
        -7,
        dtype=torch.int32,
        device="cuda",
    )
    topk_transform_kpool_v2(
        scores=scores.cuda(),
        pool_lens=pool_lens_gpu,
        out=out,
        pool_size=POOL,
        metadata=_plan(pool_lens_gpu),
        token_seq_lens=token_seq_lens.cuda() if with_tail else None,
        **kwargs,
    )
    expected = _reference(
        scores=scores,
        pool_lens=pool_lens,
        token_seq_lens=token_seq_lens if with_tail else None,
        topk=topk,
        map_token=map_token,
    )
    torch.testing.assert_close(out.cpu().sort(dim=1).values, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not kpool_topk_v2_enabled(),
    reason="k-pool top-k v2 route disabled",
)
@pytest.mark.parametrize("rows", [1, 7, 48, 600])
@pytest.mark.parametrize("mode", ["raw", "page1", "offset"])
def test_routed_kpool_topk_v2(rows, mode):
    torch.manual_seed(rows)
    topk, max_pools, pad_rows = 2048, 20000, 5
    pool_lens = torch.randint(0, max_pools + 1, (rows,), dtype=torch.int32)
    pool_lens[0] = max_pools
    token_seq_lens = pool_lens * POOL + torch.randint(
        0, POOL, (rows,), dtype=torch.int32
    )
    scores = torch.randn(rows, max_pools + 64) + 10
    max_tokens = int(token_seq_lens.max()) + POOL
    kwargs, map_token = {}, lambda b, t: t
    if mode == "page1":
        table = torch.randperm(rows * max_tokens, dtype=torch.int32).view(
            rows, max_tokens
        )
        kwargs, map_token = dict(page_table=table.cuda()), lambda b, t: int(table[b, t])
    elif mode == "offset":
        offsets = torch.randint(0, 1 << 20, (rows,), dtype=torch.int32)
        kwargs = dict(topk_offsets=offsets.cuda())
        map_token = lambda b, t: t + int(offsets[b])

    pool_lens_gpu = pool_lens.cuda()
    call = dict(
        logits=scores.cuda(),
        group_lengths=pool_lens_gpu,
        pool_size=POOL,
        topk=topk,
        seq_lens=token_seq_lens.cuda(),
        out_rows=rows + pad_rows,
        **kwargs,
    )
    # The plan belongs to whoever builds the pooled lengths; routing never recomputes it.
    with pytest.raises(AssertionError, match="plan"):
        topk_from_pooled_history_logits(**call)
    plan = build_kpool_topk_v2_plan(pool_lens_gpu)
    torch.cuda.synchronize()  # see _plan
    out = topk_from_pooled_history_logits(**call, topk_v2_plan=plan).cpu()

    assert out.shape == (rows + pad_rows, topk + POOL - 1)
    assert (out[rows:] == -1).all()
    expected = _reference(
        scores=scores,
        pool_lens=pool_lens,
        token_seq_lens=token_seq_lens,
        topk=topk // POOL,
        map_token=map_token,
    )
    torch.testing.assert_close(out[:rows].sort(dim=1).values, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,num_pools",
    # Register2/4, streaming, small-batch cluster and persistent cluster on GB300;
    # Hopper uses a higher cluster floor and can dispatch the last case to streaming.
    [(1, 3072), (4, 12000), (20, 20000), (1, 70000), (48, 40000)],
)
def test_kpool_topk_v2_concentrated_scores(rows, num_pools):
    """Concentrated distinct scores must select the exact top pools even on tie-buffer overflow."""
    topk = 2048
    # Distinct scores inside one or two coarse fp16 bins around 1.0.
    scores = torch.linspace(1.0, 1.01, num_pools).repeat(rows, 1).cuda()
    pool_lens = torch.full((rows,), num_pools, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    topk_transform_kpool_v2(
        scores=scores,
        pool_lens=pool_lens,
        out=out,
        pool_size=POOL,
        metadata=_plan(pool_lens),
    )
    selected = out[:, ::POOL].cpu() // POOL
    expected = torch.topk(scores.cpu(), topk // POOL, dim=1).indices.to(torch.int32)
    torch.testing.assert_close(
        selected.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0
    )
    if num_pools <= 4096:
        # Within the legacy kernel's candidate buffer: both must agree.
        legacy = fast_kpool_topk_transform_fused(
            score=scores, lengths=pool_lens, pool_size=POOL, topk=topk
        ).cpu()
        torch.testing.assert_close(
            out.cpu().sort(dim=1).values, legacy.sort(dim=1).values, rtol=0, atol=0
        )


def _check_legacy_route_and_v2_rejection(scores, error):
    assert not can_use_kpool_topk_v2(
        logits=scores, pool_size=POOL, row_starts=None, page_table_row_index=None
    )
    rows, num_pools = scores.shape
    pool_lens = torch.full((rows,), num_pools, dtype=torch.int32, device=scores.device)
    out = topk_from_pooled_history_logits(
        logits=scores, group_lengths=pool_lens, pool_size=POOL, topk=2048
    )
    selected = out[:, ::POOL].long() // POOL
    expected = torch.topk(scores, 512, dim=1).indices
    torch.testing.assert_close(
        selected.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0
    )
    with pytest.raises(RuntimeError, match=error):
        topk_transform_kpool_v2(
            scores=scores,
            pool_lens=pool_lens,
            out=torch.empty((rows, 2048), dtype=torch.int32, device=scores.device),
            pool_size=POOL,
            metadata=_plan(pool_lens),
        )
    torch.cuda.synchronize()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not kpool_topk_v2_enabled(),
    reason="k-pool top-k v2 route disabled",
)
def test_kpool_topk_shifted_view():
    """An unaligned column view must route safely and be rejected by the v2 API."""
    scores = torch.empty((1, 3076), device="cuda")[:, 1:3073]
    scores.copy_(torch.randperm(3072, device="cuda").float())
    assert scores.stride() == (3076, 1) and scores.data_ptr() % 16 == 4
    _check_legacy_route_and_v2_rejection(
        scores=scores, error="Tensor data pointer is not aligned to 16 bytes"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not kpool_topk_v2_enabled(),
    reason="k-pool top-k v2 route disabled",
)
@pytest.mark.parametrize("rows", [1, 2])
def test_kpool_topk_unpadded_allocation(rows):
    """An unpadded final vector must route safely and be rejected by the v2 API."""
    scores = torch.empty_strided((rows, 3073), (3584, 1), device="cuda")
    scores.copy_(torch.randperm(3073, device="cuda").float())
    assert scores.untyped_storage().nbytes() == ((rows - 1) * 3584 + 3073) * 4
    assert scores.data_ptr() % 16 == 0 and scores.stride() == (3584, 1)
    _check_legacy_route_and_v2_rejection(
        scores=scores, error="score width must be a multiple of 4"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
