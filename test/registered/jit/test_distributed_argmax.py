"""Correctness test for the sharded-vocab distributed argmax kernel.

Each rank owns a contiguous vocab shard and contributes only its local
``(max_value, global_token_id)`` pair; the ``multimem.st`` NVLink scatter must
reproduce a full-vocab ``torch.argmax`` (lowest global id on ties), across a
sweep of token counts and vocab sizes. Skips when the world size / arch has no
multicast (production falls back to the full-logits path in that case).

Usage::

    python test/registered/jit/test_distributed_argmax.py
    python test/registered/jit/test_distributed_argmax.py --num-gpu 4
    python test/registered/jit/test_distributed_argmax.py -k 32768
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.jit_kernel.utils import cache_once, get_ci_test_range
from sglang.srt.distributed.device_communicators.triton_symm_mem_argmax import (
    _create_multimem_state,
    multimem_argmax,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="8-gpu-h200")
register_cuda_ci(est_time=120, suite="nightly-kernel-8-gpu-h200", nightly=True)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Full (pre-shard) vocab width V; per-rank shard is V / world_size. Each value
# is a multiple of 8 so it splits evenly for world sizes 4 / 8.
TEST_VOCAB = [4096, 32768, 131072]
TEST_NUM_TOKENS = [1, 8, 16, 128]
TEST_LOOP = 8

TEST_VOCAB = get_ci_test_range(TEST_VOCAB, [32768])
TEST_NUM_TOKENS = get_ci_test_range(TEST_NUM_TOKENS, [16])

MAX_TOKENS = max(TEST_NUM_TOKENS)

# ---------------------------------------------------------------------------
# Per-rank distributed setup (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_world_once():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    assert ps._WORLD is not None
    return ps._WORLD


@cache_once
def _init_state_once():
    coord = _init_world_once()
    return _create_multimem_state(
        group=coord.device_group,
        rank_in_group=coord.rank_in_group,
        world_size=coord.world_size,
        max_tokens=MAX_TOKENS,
        device=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
    )


def _make_full_logits(num_tokens: int, vocab: int, seed: int, device) -> torch.Tensor:
    """Full ``[T, V]`` logits, identical on every rank (seeded CPU RNG), so the
    ``torch.argmax`` reference is well-defined across ranks."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, vocab, generator=gen).to(device)


def _local_shard(full: torch.Tensor, rank: int, world_size: int):
    """This rank's ``(local_max_value, global_token_id)`` over its vocab shard."""
    shard = full.shape[-1] // world_size
    sub = full[:, rank * shard : (rank + 1) * shard]
    local_val, local_idx = torch.max(sub, dim=-1)
    return (
        local_val.contiguous(),
        (local_idx.to(torch.int64) + rank * shard).contiguous(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vocab", TEST_VOCAB)
@pytest.mark.parametrize("num_tokens", TEST_NUM_TOKENS)
@torch.inference_mode()
def test_distributed_argmax(num_tokens: int, vocab: int) -> None:
    coord = _init_world_once()
    state = _init_state_once()
    world_size = coord.world_size
    rank = coord.rank_in_group
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    if state is None:
        pytest.skip(f"multimem multicast unavailable for world_size={world_size}")
    if vocab % world_size != 0:
        pytest.skip(f"vocab={vocab} incompatible with world_size={world_size}")

    for it in range(TEST_LOOP):
        # Entry barrier may be skipped on the kernel side; make sure every rank's
        # input is ready before the reduction.
        dist.barrier(coord.device_group)
        full = _make_full_logits(
            num_tokens, vocab, seed=it * 1_000_003 + vocab + num_tokens, device=device
        )
        ref = torch.argmax(full, dim=-1).to(torch.int64)
        local_val, global_id = _local_shard(full, rank, world_size)

        mm_out = multimem_argmax(state, local_val, global_id).clone()
        torch.testing.assert_close(mm_out, ref, atol=0, rtol=0)


@torch.inference_mode()
def test_distributed_argmax_tie_break() -> None:
    # Equal maximum in every shard: torch.argmax picks the lowest global id
    # (rank 0's peak), and the kernel must reproduce that tie-break.
    coord = _init_world_once()
    state = _init_state_once()
    world_size = coord.world_size
    rank = coord.rank_in_group
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    if state is None:
        pytest.skip(f"multimem multicast unavailable for world_size={world_size}")

    shard = 8
    vocab = world_size * shard
    full = torch.full((4, vocab), -1.0, device=device)
    for r in range(world_size):
        full[:, r * shard + 2] = 5.0  # identical peak in each shard
    ref = torch.argmax(full, dim=-1).to(torch.int64)  # == global id 2 (rank 0)
    local_val, global_id = _local_shard(full, rank, world_size)

    dist.barrier(coord.device_group)
    mm_out = multimem_argmax(state, local_val, global_id).clone()
    torch.testing.assert_close(mm_out, ref, atol=0, rtol=0)


if __name__ == "__main__":
    # multimem multicast needs world_size in {4, 6, 8} (cc9) or {6, 8} (cc10);
    # unsupported sizes self-skip via the state=None guard.
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
