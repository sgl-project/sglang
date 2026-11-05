"""The V4.1 HIP dense prefill indexer on AITER's row-group kernel scores what the schedule kernel
scores."""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    LOW_RATIO_PAGE_TABLE_BUCKET,
    aiter_fp4_paged_mqa_logits,
    aiter_has_fp4_rowgroup,
    pack_fp4_query_flydsl,
    prepare_fp4_rowgroup_prefill_workspace,
    rowgroup_table_columns,
)
from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache_hip import (
    store_fp4_index_k_cache_split,
)
from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
    _score_requests_rowgroup,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported() and aiter_has_fp4_rowgroup()),
    reason="AITER's row-group FP4 MQA logits kernel (gfx95x, ROCm/aiter#6145).",
)

DEVICE = "cuda"
PAGE = 64


def build_prefill_case(extend_lens, contexts, heads=64, seed=0):
    """A ragged prefill step: request b owns extend_lens[b] consecutive rows, row j of it
    sees contexts[b] - (extend_lens[b] - 1 - j) compressed positions (causal)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pages_per_req = [-(-c // PAGE) for c in contexts]
    num_pages = sum(pages_per_req) + 1
    payload = torch.zeros((num_pages, 1, 4, PAGE, 16), dtype=torch.uint8, device=DEVICE)
    scale = torch.zeros((num_pages, 1, 4, PAGE), dtype=torch.uint8, device=DEVICE)
    k = torch.randn(num_pages * PAGE, 128, generator=gen, device=DEVICE)
    slots = torch.arange(num_pages * PAGE, device=DEVICE)
    store_fp4_index_k_cache_split(
        k.to(torch.bfloat16), payload, scale, slots, page_size=PAGE, rne=True
    )
    perm = torch.randperm(num_pages, generator=gen, device=DEVICE).to(torch.int32)
    tables = torch.zeros((len(contexts), max(pages_per_req)), dtype=torch.int32)
    start = 0
    for b, n in enumerate(pages_per_req):
        tables[b, :n] = perm[start : start + n].cpu()
        start += n
    counts = torch.tensor(extend_lens)
    page_table = tables.repeat_interleave(counts, dim=0).to(DEVICE)
    row_ends = torch.cat(
        [
            torch.arange(c - n + 1, c + 1).clamp_min(0)
            for n, c in zip(extend_lens, contexts)
        ]
    ).to(device=DEVICE, dtype=torch.int32)
    rows = int(counts.sum())
    q = torch.randn(rows, heads, 128, generator=gen, device=DEVICE)
    q_fp4, q_scale = pack_fp4_query_flydsl(q.to(torch.bfloat16))
    weights = torch.randn(rows, heads, generator=gen, device=DEVICE) * heads**-0.5
    workspace = prepare_fp4_rowgroup_prefill_workspace(
        page_table,
        list(extend_lens),
        page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
        table_columns=rowgroup_table_columns(1 << 17, 2, LOW_RATIO_PAGE_TABLE_BUCKET),
    )
    return dict(
        q_fp4=q_fp4,
        q_scale=q_scale,
        weights=weights.to(torch.bfloat16),
        k_payload=payload,
        k_scale=scale,
        page_table=page_table,
        row_ends=row_ends,
        extend_lens=list(extend_lens),
        workspace=workspace,
    )


def schedule_logits(case) -> torch.Tensor:
    return aiter_fp4_paged_mqa_logits(
        q_fp4=case["q_fp4"],
        q_scale=case["q_scale"],
        k_payload=case["k_payload"],
        k_scale=case["k_scale"],
        weights=case["weights"],
        page_table=case["page_table"],
        c4_seq_lens=case["row_ends"],
        weight_scale=1.0,
        page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
        is_decode=False,
    ).clone()


def rowgroup_logits(case, **kw) -> torch.Tensor:
    return _score_requests_rowgroup(
        page8=False,
        q_fp4=case["q_fp4"],
        q_scale=case["q_scale"],
        weights=case["weights"],
        k_payload=case["k_payload"],
        k_scale=case["k_scale"],
        workspace=case["workspace"],
        rows=slice(0, case["page_table"].shape[0]),
        req_lo=0,
        rows_per_request=case["extend_lens"],
        row_ends=case["row_ends"],
        **kw,
    ).clone()


def assert_rows_close(got, ref, ends, rtol):
    for r, end in enumerate(ends.tolist()):
        if end:
            e = ref[r, :end].double()
            err = float((got[r, :end].double() - e).abs().max())
            assert err <= rtol * max(1.0, float(e.abs().max())), (r, err)


def test_rowgroup_prefill_matches_schedule_kernel():
    """8-row tiles with short tails, tiny tiles and an empty request: only the fp32
    summation order differs."""
    case = build_prefill_case([16, 3, 9, 1, 0, 5], [700, 64, 1000, 33, 50, 6])
    assert_rows_close(
        rowgroup_logits(case), schedule_logits(case), case["row_ends"], 2e-6
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
