"""Tiled GDN ReplaySSM commit: 2 warps (the late-joined side commit) give the bits of the served 1 warp.

The commit updates the checkpoint in place, so outputs are compared across warp counts and against the untouched
slots' initial values instead of being NaN-poisoned.
"""

from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="the tiled commit is a CUDA Triton kernel",
    ),
]

DEVICE = "cuda"
LAYERS, SLOTS, H, HV, D, RING = 2, 12, 2, 8, 128, 16


def _case(seed: int, *, residual: bool, track: bool):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape, scale):
        return (torch.randn(*shape, device=DEVICE, generator=gen) * scale).to(
            torch.bfloat16
        )

    state = rand(LAYERS, SLOTS, HV, D, D, scale=0.05)
    caches = dict(
        d_cache=rand(LAYERS, SLOTS, HV, RING, D, scale=0.05),
        k_cache=rand(LAYERS, SLOTS, H, RING, D, scale=0.09),
        g_cache=-torch.rand(LAYERS, SLOTS, HV, RING, device=DEVICE, generator=gen)
        * 0.5,
        d_residual_cache=rand(LAYERS, SLOTS, HV, RING, D, scale=1e-4)
        if residual
        else None,
        k_residual_cache=rand(LAYERS, SLOTS, H, RING, D, scale=1e-4)
        if residual
        else None,
    )
    rows = dict(
        state_batch_indices=torch.tensor(
            [3, 7, -1, 1], dtype=torch.int32, device=DEVICE
        ),
        replay_indices=torch.tensor([0, 5, 9, 2], dtype=torch.int64, device=DEVICE),
        accept_lens=torch.tensor([3, 7, 2, 1], dtype=torch.int32, device=DEVICE),
    )
    if track:
        rows.update(
            mamba_track_indices=torch.tensor(
                [10, 11, -1, 4], dtype=torch.int64, device=DEVICE
            ),
            mamba_steps_to_track=torch.tensor(
                [1, 6, -1, 0], dtype=torch.int32, device=DEVICE
            ),
        )
    write_pos = torch.zeros(SLOTS, dtype=torch.int32, device=DEVICE)
    write_pos[rows["replay_indices"]] = torch.tensor([3, 12, 2, 1], device=DEVICE).int()
    cursors = dict(
        write_pos=write_pos,
        cache_base=torch.randint(0, RING, (SLOTS,), device=DEVICE, generator=gen).int(),
        is_flush=torch.ones(SLOTS, dtype=torch.int8, device=DEVICE),
    )
    return state, caches, rows, cursors


def _commit(gdn, state, caches, rows, cursors, *, warps=None):
    out = state.clone()
    cur = {k: v.clone() for k, v in cursors.items()}
    with (
        gdn.tiled_commit_warps(warps)
        if warps
        else mock.patch.object(gdn, "_TILED_COMMIT_WARPS_OVERRIDE", None)
    ):
        gdn.commit_gdn_replayssm_circular(
            checkpoint_state=out,
            **caches,
            **rows,
            **cur,
            null_block_id=-1,
            use_cuda=False,
        )
    torch.cuda.synchronize()
    return out, cur


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("track", [True, False])
def test_two_warps_equal_one(seed, residual, track):
    from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as gdn

    state, caches, rows, cursors = _case(seed, residual=residual, track=track)
    with mock.patch.object(gdn, "_GDN_TILED_COMMIT", True):
        assert gdn._use_tiled_commit(
            caches["d_cache"], caches["k_cache"], H, HV, D, D
        )  # the tiled kernel runs
        served, served_cur = _commit(gdn, state, caches, rows, cursors)
        side, side_cur = _commit(gdn, state, caches, rows, cursors, warps=2)
        assert torch.equal(served.view(torch.int16), side.view(torch.int16))
        for name in served_cur:
            assert torch.equal(served_cur[name], side_cur[name]), name
        written = [3, 7, 1] + ([10, 11, 4] if track else [])
        untouched = [s for s in range(SLOTS) if s not in written]
        assert not torch.equal(
            served[:, written], state[:, written]
        )  # the commit did fold
        assert torch.equal(side[:, untouched], state[:, untouched])

        # Control: the same commit over a doubled value ring must change the folded checkpoints.
        doubled = dict(caches, d_cache=caches["d_cache"] * 2)
        other, _ = _commit(gdn, state, doubled, rows, cursors, warps=2)
        assert not torch.equal(other.view(torch.int16), side.view(torch.int16))
