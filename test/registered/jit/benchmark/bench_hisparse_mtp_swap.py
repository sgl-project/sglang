from __future__ import annotations

from typing import NamedTuple

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla
from sglang.kernels.ops.kvcache.hisparse_mtp import (
    HiSparseMTPCacheState,
    HiSparseMTPMissWorkspace,
    load_cache_to_device_buffer_mtp_mla,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=45, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
NUM_STEPS = 4
TOP_K = 2048
HOT_BUFFER_SIZE = 4096
PAGE_SIZE = 64
ITEM_WORDS = 72
ITEM_SIZE_BYTES = ITEM_WORDS * 8
MISS_COUNT_PER_STEP = 196
UNIQUE_MISS_COUNT = 782
# CUDA Graph benchmarking replays 100 swap calls per graph. Keep every replay
# on a fresh miss range while staying inside the native GLM-5.2 context length.
MISS_ADVANCE = 800
SEQ_LEN = 1_048_576
TOKEN_SCALE = 1_000_003


class _BenchmarkState(NamedTuple):
    top_k_tokens: torch.Tensor
    device_buffer_tokens: torch.Tensor
    host_cache_locs: torch.Tensor
    device_buffer_locs: torch.Tensor
    host_cache: torch.Tensor
    device_buffer: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    num_real_reqs: torch.Tensor
    lru_slots: torch.Tensor
    mtp_out: torch.Tensor
    lru_out: torch.Tensor
    cache_state: HiSparseMTPCacheState
    miss_workspace: HiSparseMTPMissWorkspace


def _make_top_k_tokens(batch_size: int) -> torch.Tensor:
    hit_count = TOP_K - MISS_COUNT_PER_STEP
    steps = []
    next_miss = HOT_BUFFER_SIZE
    duplicate_tokens = []
    for step in range(NUM_STEPS):
        hits = torch.roll(
            torch.arange(HOT_BUFFER_SIZE, dtype=torch.int32, device=DEVICE),
            step * 137,
        )[:hit_count]
        if step < 2:
            misses = torch.arange(
                next_miss,
                next_miss + MISS_COUNT_PER_STEP,
                dtype=torch.int32,
                device=DEVICE,
            )
            duplicate_tokens.append(misses[step])
            next_miss += MISS_COUNT_PER_STEP
        else:
            unique_misses = torch.arange(
                next_miss,
                next_miss + MISS_COUNT_PER_STEP - 1,
                dtype=torch.int32,
                device=DEVICE,
            )
            misses = torch.cat((duplicate_tokens[step - 2].view(1), unique_misses))
            next_miss += MISS_COUNT_PER_STEP - 1
        steps.append(torch.cat((hits, misses)))
    top_k_tokens = torch.stack(steps).unsqueeze(0)
    top_k_tokens = top_k_tokens.repeat(batch_size, 1, 1).contiguous()
    assert torch.unique(top_k_tokens[0, :, -MISS_COUNT_PER_STEP:]).numel() == (
        UNIQUE_MISS_COUNT
    )
    return top_k_tokens


def _make_ring_hash(
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hash_size = 1 << (2 * HOT_BUFFER_SIZE - 1).bit_length()
    primary = torch.full((batch_size, hash_size), -1, dtype=torch.int64, device=DEVICE)
    secondary = torch.full_like(primary, -1)
    tokens = torch.arange(HOT_BUFFER_SIZE, dtype=torch.int64, device=DEVICE)
    hash_slots = ((tokens * 2654435761) & (hash_size - 1)).to(torch.long)
    slot_bits = (HOT_BUFFER_SIZE - 1).bit_length()
    primary[:, hash_slots] = (tokens << slot_bits) | tokens
    return primary, secondary


def _build_state(batch_size: int) -> _BenchmarkState:
    buffer_size = HOT_BUFFER_SIZE + PAGE_SIZE
    scratch_size = HOT_BUFFER_SIZE
    physical_tokens_per_req = buffer_size + scratch_size

    top_k_tokens = _make_top_k_tokens(batch_size)
    device_buffer_tokens = torch.full(
        (batch_size, buffer_size), -1, dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens[:, :HOT_BUFFER_SIZE] = torch.arange(
        HOT_BUFFER_SIZE, dtype=torch.int32, device=DEVICE
    )
    request_bases = (
        torch.arange(batch_size, dtype=torch.int32, device=DEVICE).view(-1, 1)
        * physical_tokens_per_req
    )
    device_buffer_locs = (
        request_bases
        + torch.arange(buffer_size, dtype=torch.int32, device=DEVICE).view(1, -1)
    ).contiguous()
    scratch_locs = (
        request_bases
        + buffer_size
        + torch.arange(scratch_size, dtype=torch.int32, device=DEVICE).view(1, -1)
    ).contiguous()

    host_cache_locs = torch.arange(SEQ_LEN, dtype=torch.int64, device=DEVICE)
    host_cache_locs = host_cache_locs.view(1, -1).repeat(batch_size, 1).contiguous()
    host_cache = torch.empty((SEQ_LEN, ITEM_WORDS), dtype=torch.int64, pin_memory=True)
    host_cache.copy_(
        torch.arange(SEQ_LEN, dtype=torch.int64).view(-1, 1) * TOKEN_SCALE
        + torch.arange(ITEM_WORDS, dtype=torch.int64).view(1, -1)
    )
    device_buffer = torch.empty(
        (batch_size * physical_tokens_per_req, ITEM_WORDS),
        dtype=torch.int64,
        device=DEVICE,
    )
    hot_locs = device_buffer_locs[:, :HOT_BUFFER_SIZE].to(torch.long)
    device_buffer[hot_locs] = host_cache[:HOT_BUFFER_SIZE].to(DEVICE)

    hash_primary, hash_secondary = _make_ring_hash(batch_size)
    total_occurrences = NUM_STEPS * TOP_K
    return _BenchmarkState(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        req_pool_indices=torch.arange(batch_size, dtype=torch.int32, device=DEVICE),
        seq_lens=torch.full(
            (batch_size * NUM_STEPS,), SEQ_LEN, dtype=torch.int32, device=DEVICE
        ),
        num_real_reqs=torch.tensor([batch_size], dtype=torch.int32, device=DEVICE),
        lru_slots=torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE)
        .view(1, -1)
        .repeat(batch_size, 1),
        mtp_out=torch.full_like(top_k_tokens, -1),
        lru_out=torch.full_like(top_k_tokens, -1),
        cache_state=HiSparseMTPCacheState(
            hash_primary=hash_primary,
            hash_secondary=hash_secondary,
            ring_state=torch.zeros(batch_size, dtype=torch.int32, device=DEVICE),
            ref_epochs=torch.zeros(
                (batch_size, HOT_BUFFER_SIZE),
                dtype=torch.int32,
                device=DEVICE,
            ),
        ),
        miss_workspace=HiSparseMTPMissWorkspace(
            locs=scratch_locs,
            metadata=torch.full(
                (batch_size, 5 * total_occurrences),
                -1,
                dtype=torch.int32,
                device=DEVICE,
            ),
            counters=torch.zeros(4 * batch_size, dtype=torch.int32, device=DEVICE),
        ),
    )


def _run_mtp(state: _BenchmarkState) -> None:
    load_cache_to_device_buffer_mtp_mla(
        top_k_tokens=state.top_k_tokens,
        device_buffer_tokens=state.device_buffer_tokens,
        host_cache_locs=state.host_cache_locs,
        device_buffer_locs=state.device_buffer_locs,
        host_cache=state.host_cache,
        device_buffer=state.device_buffer,
        top_k_device_locs=state.mtp_out,
        req_pool_indices=state.req_pool_indices,
        seq_lens=state.seq_lens,
        cache_state=state.cache_state,
        miss_workspace=state.miss_workspace,
        num_real_reqs=state.num_real_reqs,
    )


def _run_lru_step(state: _BenchmarkState, step: int) -> None:
    batch_size = state.top_k_tokens.size(0)
    load_cache_to_device_buffer_mla(
        top_k_tokens=state.top_k_tokens[:, step],
        device_buffer_tokens=state.device_buffer_tokens,
        host_cache_locs=state.host_cache_locs,
        device_buffer_locs=state.device_buffer_locs,
        host_cache=state.host_cache,
        device_buffer=state.device_buffer,
        top_k_device_locs=state.lru_out[:, step],
        req_pool_indices=state.req_pool_indices,
        # All benchmark steps use the same logical length. Reuse a contiguous
        # slice so the measured path does not allocate a temporary tensor.
        seq_lens=state.seq_lens[:batch_size],
        lru_slots=state.lru_slots,
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=TOP_K,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=PAGE_SIZE,
        block_size=1024,
        num_real_reqs=state.num_real_reqs,
    )


def _assert_current_result(state: _BenchmarkState, impl: str) -> None:
    if impl == "mtp_4step":
        out = state.mtp_out
        tokens = state.top_k_tokens
    else:
        out = state.lru_out
        tokens = state.top_k_tokens
    torch.cuda.synchronize()
    actual = state.device_buffer[:, 0][out.to(torch.long)]
    torch.testing.assert_close(actual, tokens.to(torch.int64) * TOKEN_SCALE)


def _check_initial_result(state: _BenchmarkState, impl: str) -> None:
    if impl == "mtp_4step":
        _run_mtp(state)
    else:
        for step in range(NUM_STEPS):
            _run_lru_step(state, step)
    _assert_current_result(state, impl)


@marker.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64], [1])
@marker.benchmark("impl", ["lru_4step", "mtp_4step"])
def benchmark(batch_size: int, impl: str):
    state = _build_state(batch_size)
    _check_initial_result(state, impl)

    if impl == "mtp_4step":

        def run() -> None:
            state.top_k_tokens[..., -MISS_COUNT_PER_STEP:].add_(MISS_ADVANCE)
            _run_mtp(state)

    else:

        def run() -> None:
            state.top_k_tokens[..., -MISS_COUNT_PER_STEP:].add_(MISS_ADVANCE)
            for step in range(NUM_STEPS):
                _run_lru_step(state, step)

    result = marker.do_bench(
        run,
        use_cuda_graph=True,
        warmup_iters=5,
        replay_iters=40,
        disable_log_bandwidth=True,
        memory_args=None,
        memory_output=None,
    )
    _assert_current_result(state, impl)
    return result


if __name__ == "__main__":
    benchmark.run()
