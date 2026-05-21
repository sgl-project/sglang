from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_topk_v1_module(topk: int):
    args = make_cpp_args(is_arch_support_pdl())
    assert topk in (512, 1024), "Only support topk=512 or 1024"
    return load_jit(
        make_name(f"topk_v1_{topk}"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
        extra_cuda_cflags=[f"-DSGL_TOPK={topk}"],
    )


@cache_once
def _jit_topk_v2_module(topk: int):
    return load_jit(
        make_name(f"topk_v2_{topk}"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform", "CombinedTopKKernel::transform"),
            ("topk_plan", "CombinedTopKKernel::plan"),
        ],
        extra_cuda_cflags=[f"-DSGL_TOPK={topk}"],
    )


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    module = _jit_topk_v1_module(out_page_indices.shape[1])
    module.topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )


_WORKSPACE_INTS_PER_BATCH = 2 + 1024 * 2
_PLAN_METADATA_INTS_PER_BATCH = 4


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    module = _jit_topk_v2_module(512)  # does not matter
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
) -> None:
    module = _jit_topk_v2_module(out_page_indices.shape[1])
    bs = scores.shape[0]
    workspace = seq_lens.new_empty(bs, _WORKSPACE_INTS_PER_BATCH)
    module.topk_transform(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        workspace,
        metadata,
    )
