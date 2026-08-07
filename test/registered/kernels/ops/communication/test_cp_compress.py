"""Four-GPU correctness for consumer-direct DSV4 CP compression.

Usage::

    python test/registered/kernels/ops/communication/test_cp_compress.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import os
import time

import pytest
import sglang.srt.distributed.parallel_state as ps
import torch
import torch.distributed as dist
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention.dsv4.cp_compress import (
    cp_compress_aligned,
    create_cp_compressor_state,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="extra-b", runner_config="4-gpu-b200")


@cache_once
def _init_group() -> tuple[dist.ProcessGroup, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    assert ps._WORLD.device_group is not None
    return ps._WORLD.device_group, ps._WORLD.rank_in_group


def _gather_interleaved(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    rank_major = torch.empty(
        (local.shape[0] * group.size(), local.shape[1]),
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(rank_major, local, group=group)
    return (
        rank_major.view(group.size(), local.shape[0], local.shape[1])
        .transpose(0, 1)
        .reshape(-1, local.shape[1])
    )


def _softmax_weighted(kv: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(score, dim=1)
    return (kv * weights).sum(dim=1)


def _reference_c128(global_input: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    windows = global_input.shape[0] // 128
    head_dim = ape.shape[1]
    x = global_input.view(windows, 128, 2, head_dim)
    return _softmax_weighted(x[:, :, 0], x[:, :, 1] + ape)


def _reference_c4(
    global_input: torch.Tensor,
    ape: torch.Tensor,
    previous: torch.Tensor | None,
) -> torch.Tensor:
    windows = global_input.shape[0] // 4
    head_dim = ape.shape[1]
    x = global_input.view(windows, 4, 4, head_dim)
    outputs = []
    for window in range(windows):
        current = x[window]
        kv = current[:, 1]
        score = current[:, 3] + ape[4:]
        prior = previous if window == 0 else x[window - 1]
        if prior is not None:
            kv = torch.cat((prior[:, 0], kv), dim=0)
            score = torch.cat((prior[:, 2] + ape[:4], score), dim=0)
        outputs.append(_softmax_weighted(kv[None], score[None])[0])
    return torch.stack(outputs)


@pytest.mark.parametrize("ratio,head_dim", [(4, 128), (4, 512), (128, 512)])
@torch.inference_mode()
def test_changing_input_and_aligned_prefix(ratio: int, head_dim: int) -> None:
    group, rank = _init_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    global_tokens = 512
    local_tokens = global_tokens // group.size()
    width = (4 if ratio == 4 else 2) * head_dim
    state = create_cp_compressor_state(
        group, rank, ratio, head_dim, global_tokens, device
    )
    carry = (
        torch.empty((2, head_dim), dtype=torch.float32, device=device)
        if ratio == 4
        else None
    )
    ape_rows = 8 if ratio == 4 else 128
    torch.manual_seed(1000 + ratio + head_dim)
    ape = torch.randn((ape_rows, head_dim), dtype=torch.float32, device=device)
    previous = None
    for iteration in range(3):
        # Deliberately skew one producer before the intervening publication
        # barrier; iteration 2 then reuses parity zero without stale reads.
        if iteration == 1 and rank == 0:
            time.sleep(0.02)
        torch.manual_seed(28639 + iteration * 10 + rank)
        local = torch.randn((local_tokens, width), dtype=torch.float32, device=device)
        global_input = _gather_interleaved(local, group)
        actual = cp_compress_aligned(
            state,
            local,
            ape,
            prefix_tokens=iteration * global_tokens,
            c4_carry=carry,
        )
        expected = (
            _reference_c4(global_input, ape, previous)
            if ratio == 4
            else _reference_c128(global_input, ape)
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        if ratio == 4:
            previous = global_input[-4:].view(4, 4, head_dim)


@pytest.mark.parametrize("head_dim", [128, 512])
@torch.inference_mode()
def test_c4_sparse_plan_w_state_buffer(head_dim: int) -> None:
    group, rank = _init_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    global_tokens = 512
    local_tokens = global_tokens // group.size()
    width = 4 * head_dim
    state = create_cp_compressor_state(group, rank, 4, head_dim, global_tokens, device)
    carry = torch.empty((2, head_dim), dtype=torch.float32, device=device)
    ape = torch.zeros((8, head_dim), dtype=torch.float32, device=device)

    # These are exactly the four C4 tail rows from each 128-token SWA page.
    # Every owner rank appears once per page under round-robin CP4.
    ragged_ids = torch.tensor(
        [
            124,
            125,
            126,
            127,
            252,
            253,
            254,
            255,
            380,
            381,
            382,
            383,
            508,
            509,
            510,
            511,
        ],
        dtype=torch.int32,
        device=device,
    )
    num_slots = 41
    sentinel = -12345.0
    state_buffer = torch.empty(
        (num_slots, 4, head_dim), dtype=torch.float32, device=device
    )

    # Three generations force reuse of double-buffer slot zero. Inputs and
    # destinations change each time, catching stale peer-stage publication.
    for iteration in range(3):
        dist.barrier(group)
        torch.manual_seed(39000 + iteration * 10 + rank)
        local = torch.randn((local_tokens, width), dtype=torch.float32, device=device)
        global_input = _gather_interleaved(local, group)
        write_locs = (
            torch.arange(ragged_ids.numel(), dtype=torch.int32, device=device) * 7
            + iteration * 3
        ) % num_slots
        plan_i32 = torch.stack((ragged_ids, write_locs), dim=1).contiguous()
        plan_w = plan_i32.view(torch.uint8)
        assert plan_w.shape == (ragged_ids.numel(), 8)

        state_buffer.fill_(sentinel)
        cp_compress_aligned(
            state,
            local,
            ape,
            prefix_tokens=iteration * global_tokens,
            c4_carry=carry,
            c4_plan_w=plan_w,
            c4_state_buffer=state_buffer,
        )

        expected = torch.full_like(state_buffer, sentinel)
        expected[write_locs.long()] = global_input.index_select(
            0, ragged_ids.long()
        ).view(-1, 4, head_dim)
        torch.testing.assert_close(state_buffer, expected, rtol=0, atol=0)

    assert state.generation == 3


@pytest.mark.parametrize("head_dim", [128, 512])
@torch.inference_mode()
def test_c4_paged_prefix_reads_plan_c_history(head_dim: int) -> None:
    group, rank = _init_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    global_tokens = 512
    local_tokens = global_tokens // group.size()
    state = create_cp_compressor_state(group, rank, 4, head_dim, global_tokens, device)
    torch.manual_seed(48000 + head_dim)
    ape = torch.randn((8, head_dim), dtype=torch.float32, device=device)
    state_buffer = torch.zeros((32, 4 * head_dim), dtype=torch.float32, device=device)
    page = 5

    previous = None
    for iteration in range(2):
        torch.manual_seed(49000 + iteration * 10 + rank)
        local = torch.randn(
            (local_tokens, 4 * head_dim), dtype=torch.float32, device=device
        )
        global_input = _gather_interleaved(local, group)
        plan_c_i32 = torch.zeros(
            (global_tokens // 4, 4), dtype=torch.int32, device=device
        )
        plan_c_i32[0, 2] = page
        plan_c = plan_c_i32.view(torch.uint8)
        if iteration == 0:
            ragged = torch.arange(
                global_tokens - 4,
                global_tokens,
                dtype=torch.int32,
                device=device,
            )
            write_locs = torch.arange(
                page * 4, page * 4 + 4, dtype=torch.int32, device=device
            )
            plan_w = torch.stack((ragged, write_locs), dim=1).view(torch.uint8)
        else:
            plan_w = torch.empty((0, 8), dtype=torch.uint8, device=device)

        actual = cp_compress_aligned(
            state,
            local,
            ape,
            prefix_tokens=iteration * global_tokens,
            c4_plan_c=plan_c,
            c4_plan_w=plan_w,
            c4_state_buffer=state_buffer,
        )
        torch.testing.assert_close(
            actual,
            _reference_c4(global_input, ape, previous),
            rtol=2e-5,
            atol=2e-5,
        )
        previous = global_input[-4:].view(4, 4, head_dim)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
