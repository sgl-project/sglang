"""The sharded DSpark Markov chain kernel (--enable-dsv41-low-latency-decode) draws
MarkovSampleStep's tokens and keeps its step logits bit for bit when the bias products
sum exactly: one rank and four ranks (spawned, symmetric memory), one to four rows,
greedy and sampled temperatures, BF16 and FP32 base logits, local-shard and full-vocab
base logits, three launches (epoch tags and both staging parities) and CUDA graph
replay with the noise forked onto its side stream.
"""

import os
import socket

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

VOCAB = 129_280  # DeepSeek-V4.1
RANK = 256
STEPS = 5
MAX_BS = 8
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = ((0.0,) * 8, (1.0,) * 8, (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1))


def _gpu_ready() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _ternary(g: torch.Generator, device, *shape: int) -> torch.Tensor:
    """Sparse {-1, 0, 1} BF16: |bias| <= 256, exact in BF16 in any summation order."""
    values = torch.randint(-1, 2, shape, generator=g)
    values *= torch.rand(shape, generator=g) < 0.25
    return values.to(device, torch.bfloat16)


def _reference(w1, w2, base, anchor, temperatures, offset, cache):
    """MarkovSampleStep per step over the full vocab, as the unsharded draft step runs it."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    prev, tokens = anchor.long(), []
    for step in range(base.shape[1]):
        bias = F.linear(w1[prev], w2)
        prev = MarkovSampleStep.execute(
            base_logits=base[:, step],
            bias=bias,
            temperatures=temperatures,
            key=KEY,
            offset=offset,
            step=step,
            step_dtype=torch.float32,
            logits_out=None if cache is None else cache[: base.shape[0], step],
            has_sampled_rows=None if cache is not None else False,
        )
        tokens.append(prev)
    return torch.stack(tokens, 1)


def _check_equal(device, group, rank: int, world: int, base_dtype, seed: int):
    from sglang.srt.models.dsv41_low_latency.dspark.sharded_markov import (
        ARRIVAL_WORDS,
        TIMEOUT_MS,
        DSparkShardedMarkovSampler,
    )

    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    sampler = DSparkShardedMarkovSampler(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=group,
        rank=rank,
        world=world,
        max_bs=MAX_BS,
        timeout_ms=TIMEOUT_MS,
    )
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    for rows in (1, 2, 4):
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temps in TEMPERATURE_SETS:
            temperatures = torch.tensor(temps, device=device)
            for keep in (True, False):  # sampled draft vs greedy draft (no logits)
                want_cache = (
                    torch.zeros(MAX_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for local in (True, False):
                    inputs = base[:, :, shard].contiguous() if local else base
                    for _ in range(3):
                        cache = (
                            torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
                            if keep
                            else None
                        )
                        tokens = torch.full(
                            (MAX_BS, STEPS), -7, dtype=torch.int64, device=device
                        )
                        if keep:
                            sampler.launch_noise(rows=rows, key=KEY, offset=offset)
                        sampler.sample(
                            base_logits=inputs,
                            anchor_tokens=anchor,
                            temperatures=temperatures,
                            logits_cache=cache,
                            draft_tokens=tokens,
                        )
                        torch.cuda.synchronize()
                        assert torch.equal(tokens[:rows], want), (
                            rows,
                            temps,
                            keep,
                            local,
                        )
                        if keep:
                            assert torch.equal(cache[:rows], want_cache[:rows]), (
                                rows,
                                temps,
                                local,
                            )

    # CUDA graph: noise forked onto the side stream, replayed with new offsets.
    rows = 2
    base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
    anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)

    def forked_sample():
        sampler.fork_noise(rows=rows, key=KEY, offset=offset)
        sampler.join_noise()
        sampler.sample(
            base_logits=base[:, :, shard].contiguous(),
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache,
            draft_tokens=tokens,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        forked_sample()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forked_sample()
    for step_offset in (5, 6, 1 << 33):
        offset.fill_(step_offset)
        graph.replay()
        want_cache = torch.zeros_like(cache)
        want = _reference(w1, w2, base, anchor, temperatures[:rows], offset, want_cache)
        torch.cuda.synchronize()
        assert torch.equal(tokens[:rows], want)
        assert torch.equal(cache[:rows], want_cache[:rows])
    assert int(sampler.control[ARRIVAL_WORDS].abs().sum()) == 0  # arrivals reset


@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_one_rank_matches_markov_sample_step(base_dtype):
    _check_equal(torch.device("cuda"), None, 0, 1, base_dtype, seed=1)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _four_rank_worker(rank: int, world: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    try:
        for base_dtype in (torch.bfloat16, torch.float32):
            _check_equal(device, group, rank, world, base_dtype, seed=7)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_four_ranks_match_markov_sample_step():
    torch.multiprocessing.spawn(
        _four_rank_worker, args=(4, _free_port()), nprocs=4, join=True
    )
