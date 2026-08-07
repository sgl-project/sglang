"""Exactness test for producer-direct DSV4 CP KV-cache publication."""

from __future__ import annotations

import atexit
import logging
import os
from dataclasses import dataclass
from typing import Any

import pytest
import sglang.srt.distributed.parallel_state as ps
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from sglang.kernels.jit.utils import cache_once, get_ci_test_range
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.direct_cp_kv_store import direct_cp_kv_store
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE_SIZE = 256
PAGE_BYTES = ((584 * PAGE_SIZE + 575) // 576) * 576
MAX_TOKENS = 4096
TEST_TOKENS = get_ci_test_range([1024, 4096], [1024])


@dataclass
class _State:
    group: dist.ProcessGroup
    rank: int
    world_size: int
    cache: torch.Tensor
    handle: Any


@cache_once
def _init_state() -> _State:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())

    group = ps._WORLD.device_group
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), world_size * 4))
    with torch.inference_mode(False), torch.no_grad():
        cache = symm_mem.empty(
            (MAX_TOKENS // PAGE_SIZE, PAGE_BYTES),
            dtype=torch.uint8,
            device=f"cuda:{local_rank}",
        )
    cache.zero_()
    handle = symm_mem.rendezvous(cache, group=group)
    if handle.multicast_ptr == 0:
        pytest.skip("NVLink multicast mapping is unavailable")
    return _State(group, local_rank, world_size, cache, handle)


@pytest.mark.parametrize("global_tokens", TEST_TOKENS)
@torch.inference_mode()
def test_dsv4_direct_cp_kv_store_exact(global_tokens: int) -> None:
    state = _init_state()
    assert global_tokens % state.world_size == 0
    device = state.cache.device

    torch.manual_seed(20260801 + global_tokens)
    global_kv = torch.randn((global_tokens, 512), dtype=torch.bfloat16, device=device)
    local_kv = global_kv[state.rank :: state.world_size].contiguous()
    global_indices = torch.randperm(MAX_TOKENS, dtype=torch.int32, device=device)[
        :global_tokens
    ]
    local_indices = global_indices[state.rank :: state.world_size].contiguous()

    gathered = torch.empty_like(global_kv)
    dist.all_gather_into_tensor(gathered, local_kv, group=state.group)
    gathered = (
        gathered.view(state.world_size, -1, 512)
        .transpose(0, 1)
        .reshape(global_tokens, 512)
    )
    reference = torch.zeros_like(state.cache)
    fused_store_cache(
        gathered,
        reference,
        global_indices,
        page_size=PAGE_SIZE,
        type="flashmla",
    )

    state.cache.zero_()
    dist.barrier(group=state.group)
    direct_cp_kv_store(
        cache=state.cache,
        handle=state.handle,
        cache_multicast=state.handle.multicast_ptr,
        local_kv=local_kv,
        local_indices=local_indices,
        rank=state.rank,
        world_size=state.world_size,
        page_size=PAGE_SIZE,
    )
    torch.testing.assert_close(state.cache, reference, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
