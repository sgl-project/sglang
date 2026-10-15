"""Shared drivers for the draft sampling tests (CPU unit and CUDA kernel suites)."""

from __future__ import annotations

import torch


def run_prepare_fill(
    impl,
    *,
    device,
    src_top_ks: torch.Tensor,
    src_temperatures: torch.Tensor,
    sampling_top_k: int,
    max_bs: int = 4,
    raw_bs: int = 2,
    bs: int = 3,
    stage_temperatures: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive a fill_draft_extend_prepare_buffers implementation with a graph
    runner's draft sampling buffers; returns (temperatures, top_ks) after it."""
    i64 = dict(dtype=torch.int64, device=device)
    i32 = dict(dtype=torch.int32, device=device)
    num_tokens_per_bs = 2
    max_num_token, num_tokens = max_bs * num_tokens_per_bs, raw_bs * num_tokens_per_bs
    temperatures = torch.full((max_bs,), 9.0, dtype=torch.float32, device=device)
    top_ks = torch.full((max_bs,), 7, **i32)
    impl(
        torch.zeros(max_num_token, **i64),
        torch.zeros(max_num_token, **i64),
        torch.zeros(max_num_token, **i64),
        torch.arange(num_tokens, **i64),
        torch.arange(num_tokens, **i64),
        torch.arange(num_tokens, **i64),
        torch.zeros(max_bs, **i32),
        torch.zeros(max_bs, **i64),
        torch.zeros(max_bs, **i32),
        torch.zeros(max_bs, **i32),
        torch.zeros(max_bs, **i64),
        temperatures if stage_temperatures else None,
        torch.ones(raw_bs, **i32),
        torch.arange(raw_bs, **i64),
        torch.zeros(raw_bs, **i32),
        torch.zeros(raw_bs, **i32),
        src_temperatures.to(device) if stage_temperatures else None,
        None,
        None,
        None,
        None,
        raw_bs,
        bs,
        num_tokens_per_bs,
        0,
        0,
        top_ks=top_ks,
        src_top_ks=src_top_ks.to(device),
        sampling_top_k=sampling_top_k,
    )
    return temperatures, top_ks
