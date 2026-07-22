"""Test for the kpool top-k transform JIT kernel.

Ported from the former AOT sgl-kernel test (sgl-kernel/tests/test_topk.py).
The kernel selects pool groups at pool granularity, expands each selected group
to ``pool_size`` token indices, and optionally transforms those token indices
through a page table or a ragged offset.
"""

from __future__ import annotations

import sys
from typing import Optional

import pytest
import torch

from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _ref_torch_kpool_transform_impl(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows = score.shape[0]
    group_topk = topk // pool_size
    offsets = torch.arange(pool_size, dtype=torch.int32, device=score.device)
    out_cols = topk + (pool_size - 1 if seq_lens is not None else 0)
    out = torch.full((rows, out_cols), -1, dtype=torch.int32, device=score.device)
    for i in range(rows):
        length = int(lengths[i].item())
        valid_count = min(length, group_topk)
        write_pos = 0
        if valid_count == 0:
            token_ids = torch.empty((0,), dtype=torch.int32, device=score.device)
        elif length <= group_topk:
            selected = torch.arange(length, dtype=torch.int32, device=score.device)
            token_ids = (selected.unsqueeze(1) * pool_size + offsets).reshape(-1)
        else:
            selected = torch.topk(
                score[i, :length], group_topk, dim=-1, sorted=False
            ).indices.to(torch.int32)
            token_ids = (selected.unsqueeze(1) * pool_size + offsets).reshape(-1)
        if token_ids.numel() > 0:
            if page_table is not None:
                token_ids = page_table[i, token_ids.long()].to(torch.int32)
            elif topk_indices_offset is not None:
                token_ids = token_ids + topk_indices_offset[i].to(torch.int32)
            write_pos = valid_count * pool_size
            out[i, :write_pos] = token_ids[:write_pos]
        if seq_lens is not None:
            tail_count = int(seq_lens[i].item()) % pool_size
            if tail_count > 0:
                raw_tail = length * pool_size + torch.arange(
                    tail_count, dtype=torch.int32, device=score.device
                )
                if page_table is not None:
                    tail = page_table[i, raw_tail.long()].to(torch.int32)
                elif topk_indices_offset is not None:
                    tail = raw_tail + topk_indices_offset[i].to(torch.int32)
                else:
                    tail = raw_tail
                out[i, write_pos : write_pos + tail_count] = tail
    return out


@pytest.mark.parametrize(
    "pool_size,group_topk",
    [(16, 128), (16, 160), (16, 192), (16, 224), (8, 256), (4, 512)],
)
@pytest.mark.parametrize("mode", ["raw", "paged", "ragged"])
@pytest.mark.parametrize("append_tail", [False, True])
@torch.inference_mode()
def test_kpool_topk_transform_kernel(
    pool_size: int, group_topk: int, mode: str, append_tail: bool
) -> None:
    torch.manual_seed(42)
    bs = 17
    topk = pool_size * group_topk
    num_groups = 4096
    score = torch.randn(bs, num_groups, dtype=torch.float32, device="cuda")
    lengths = torch.randint(
        group_topk + 1, num_groups + 1, (bs,), dtype=torch.int32, device="cuda"
    )

    page_table = None
    topk_indices_offset = None
    seq_lens = None
    tail_counts = torch.randint(0, pool_size, (bs,), dtype=torch.int32, device="cuda")
    if append_tail:
        seq_lens = lengths * pool_size + tail_counts
    if mode == "paged":
        page_table = torch.arange(
            bs * (num_groups * pool_size + pool_size),
            dtype=torch.int32,
            device="cuda",
        ).view(bs, num_groups * pool_size + pool_size)
    elif mode == "ragged":
        topk_indices_offset = torch.randint(
            0, 2048, (bs,), dtype=torch.int32, device="cuda"
        )

    out_ref = _ref_torch_kpool_transform_impl(
        score,
        lengths,
        pool_size,
        topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        seq_lens=seq_lens,
    )
    out_our = fast_kpool_topk_transform_fused(
        score,
        lengths,
        pool_size,
        topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        seq_lens=seq_lens,
    )
    torch.cuda.synchronize()

    out_ref = torch.sort(out_ref, dim=-1).values
    out_our = torch.sort(out_our, dim=-1).values
    torch.testing.assert_close(out_our, out_ref, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
