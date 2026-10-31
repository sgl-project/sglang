from __future__ import annotations

import random

import pytest
import torch

from sglang.kernels.ops.kv_canary.canary_info import launch_canary_info_kernel
from sglang.kernels.ops.kv_canary.canary_info_ref import (
    launch_canary_info_kernel_torch_reference,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.kv_canary._canary_helpers import (
    make_canary_buf,
    run_canary_info_kernel_pair,
    stamp_clean_chain,
)
from sglang.test.kernels.kv_canary._fuzz_driver import FUZZ_SEEDS_PR

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small-amd")


_DEVICE = torch.device("cuda")

_FUZZ_ITER_PER_SEED = 20
_NUM_SLOTS = 2048


def _random_sampler(rng: random.Random, rows: int) -> dict:
    def f32(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=_DEVICE)

    return dict(
        temperatures=f32(
            [rng.choice([0.0, 1.0, rng.uniform(0.01, 2.0)]) for _ in range(rows)]
        ),
        top_ps=f32([rng.choice([1.0, rng.uniform(0.01, 1.0)]) for _ in range(rows)]),
        top_ks=torch.tensor(
            [
                rng.choice([-1, 1, 1 << 30, rng.randint(-(1 << 31), (1 << 31) - 1)])
                for _ in range(rows)
            ],
            dtype=torch.int32,
            device=_DEVICE,
        ),
        min_ps=f32([rng.choice([0.0, rng.uniform(0.0, 1.0)]) for _ in range(rows)]),
    )


def _random_chain_buf(
    rng: random.Random, num_chains: int
) -> tuple[torch.Tensor, list[int]]:
    buf = make_canary_buf(num_slots=_NUM_SLOTS, slot_stride_bytes=32, device=_DEVICE)
    free_slots = list(range(1, _NUM_SLOTS))
    rng.shuffle(free_slots)
    last_slots: list[int] = []
    cursor = 0
    for _ in range(num_chains):
        chain_len = rng.randint(1, 3)
        if rng.random() < 0.1 or cursor + chain_len > len(free_slots):
            last_slots.append(-1)
            continue
        slots = free_slots[cursor : cursor + chain_len]
        cursor += chain_len
        start = rng.randint(0, 4096)
        stamp_clean_chain(
            cuda_buf=buf,
            ref_buf=buf,
            slot_indices=slots,
            tokens=[rng.randint(0, 200000) for _ in slots],
            positions=list(range(start, start + chain_len)),
        )
        last_slots.append(slots[-1])
    return buf, last_slots


def _run_one(rng: random.Random, *, wide: bool) -> None:
    batch_size = rng.randint(1, 128 if wide else 512)
    width = rng.randint(1, 8) if wide else 1
    num_tokens = batch_size * width
    buf, last_slots = _random_chain_buf(rng, num_tokens)
    accept_lens = (
        torch.tensor(
            [rng.randint(0, width) for _ in range(batch_size)],
            dtype=torch.int32,
            device=_DEVICE,
        )
        if wide
        else None
    )
    slots = torch.tensor(
        last_slots, dtype=torch.int64 if wide else torch.int32, device=_DEVICE
    ).view(batch_size, width)
    out_kernel, out_ref = run_canary_info_kernel_pair(
        launch=launch_canary_info_kernel,
        launch_ref=launch_canary_info_kernel_torch_reference,
        out_like=torch.empty((batch_size, width), dtype=torch.int64, device=_DEVICE),
        canary_buf=buf,
        out_cache_loc=slots,
        accept_lens=accept_lens,
        next_token_ids=torch.tensor(
            [rng.randint(0, 200000) for _ in range(num_tokens)],
            dtype=torch.int32 if wide else torch.int64,
            device=_DEVICE,
        ).view(batch_size, width),
        **_random_sampler(rng, batch_size),
    )

    assert torch.equal(out_kernel, out_ref)
    invalid = slots[:, :width] < 0
    if accept_lens is not None:
        invalid |= torch.arange(width, device=_DEVICE)[None, :] >= accept_lens[:, None]
    assert bool((out_kernel[invalid] == 0).all())


@pytest.mark.parametrize("seed", FUZZ_SEEDS_PR)
def test_canary_info_kernel_per_row_matches_reference(seed: int) -> None:
    """Match ordinary sampling against the host reference."""
    rng = random.Random(seed)
    for _ in range(_FUZZ_ITER_PER_SEED):
        _run_one(rng, wide=False)


@pytest.mark.parametrize("seed", FUZZ_SEEDS_PR)
def test_canary_info_kernel_linear_acceptance_matches_reference(seed: int) -> None:
    """Match variable accepted prefixes against the host reference."""
    rng = random.Random(seed)
    for _ in range(_FUZZ_ITER_PER_SEED):
        _run_one(rng, wide=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
