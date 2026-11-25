from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

import pytest
import torch

from sglang.kernels.ops.mamba.triton_ops import mamba_chunk_scan_combined
from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import (
    _chunk_state_varlen_kernel,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    _ssm_slots_contiguous,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
from sglang.srt.utils import get_device

N_HEADS, N_GROUPS = 8, 2


def _inputs(seqlens, d_head, d_state, gen, device):
    T = sum(seqlens)
    x = torch.randn(1, T, N_HEADS, d_head, generator=gen, device=device)
    dt = torch.randn(1, T, N_HEADS, generator=gen, device=device)
    B = torch.randn(1, T, N_GROUPS, d_state, generator=gen, device=device)
    C = torch.randn(1, T, N_GROUPS, d_state, generator=gen, device=device)
    A = -torch.exp(torch.rand(N_HEADS, generator=gen, device=device))
    D = torch.ones(N_HEADS, device=device, dtype=torch.bfloat16)
    dt_bias = (torch.rand(N_HEADS, generator=gen, device=device) - 2).bfloat16()
    bf = lambda t: t.to(torch.bfloat16)
    return bf(x), bf(dt), A, bf(B), bf(C), D, dt_bias


def _case(seqlens, chunk_size, d_head, d_state, pool_dtype, has_init, seed):
    """A prefill batch over a shuffled state pool, with final-state and recompute
    tracking rows, laid out as the Mamba2 backend builds them."""
    device = get_device()
    gen = torch.Generator(device=device).manual_seed(seed)
    n = len(seqlens)
    num_slots = 3 * n + 4
    perm = torch.randperm(num_slots, generator=gen, device=device)
    slots, track_slots = perm[:n].to(torch.int32), perm[n : 2 * n]
    pool = (
        torch.randn(num_slots, N_HEADS, d_head, d_state, generator=gen, device=device)
        * 0.1
    ).to(pool_dtype)

    starts = [sum(seqlens[:i]) for i in range(n + 1)]
    cu = torch.tensor(starts, dtype=torch.int32, device=device)
    seq_idx = torch.repeat_interleave(
        torch.arange(n, dtype=torch.int32, device=device),
        torch.tensor(seqlens, device=device),
    ).unsqueeze(0)
    has = torch.tensor(has_init, dtype=torch.bool, device=device)
    chunk_indices = chunk_offsets = None
    if any(has_init):
        chunk_indices, chunk_offsets = (
            Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
                cu, chunk_size, starts[-1]
            )
        )

    # Rows whose tracked position is off the chunk grid are recomputed; every
    # other row is tracked at its final state.
    recompute_rows, end_locs = [], []
    for i, ln in enumerate(seqlens):
        aligned = (ln // chunk_size) * chunk_size
        if aligned and aligned != ln and (starts[i] + aligned) % chunk_size:
            recompute_rows.append(i)
            end_locs.append(starts[i] + aligned)
    final_rows = [i for i in range(n) if i not in recompute_rows]
    final_dst_per_row = torch.full((n,), -1, dtype=torch.int32, device=device)
    final_dst_per_row[final_rows] = track_slots[final_rows].to(torch.int32)
    as_dev = lambda v: torch.tensor(v, dtype=torch.int64, device=device)
    return dict(
        args=_inputs(seqlens, d_head, d_state, gen, device),
        chunk_size=chunk_size,
        pool=pool,
        slots=slots,
        has=has,
        cu=cu,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        recompute_rows=as_dev(recompute_rows),
        end_locs=torch.tensor(end_locs, dtype=torch.int32, device=device),
        recompute_dst=track_slots[recompute_rows],
        final_src=slots[final_rows].long(),
        final_dst=track_slots[final_rows],
        final_dst_per_row=final_dst_per_row,
    )


def _run(c, pool, in_pool):
    x, dt, A, B, C, D, dt_bias = c["args"]
    has_prefix = bool(c["has"].any())
    kwargs = {}
    initial_states = None
    if in_pool:
        kwargs = dict(
            state_pool=pool,
            state_indices=c["slots"],
            has_initial_state=c["has"] if has_prefix else None,
            final_track_slots=c["final_dst_per_row"],
            track_out_slots=c["recompute_dst"],
        )
    elif has_prefix:
        initial_states = torch.where(c["has"][:, None, None, None], pool[c["slots"]], 0)
    out = torch.empty_like(x)
    states, varlen, track = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        c["chunk_size"],
        D=D,
        dt_bias=dt_bias,
        seq_idx=c["seq_idx"],
        chunk_indices=c["chunk_indices"],
        chunk_offsets=c["chunk_offsets"],
        cu_seqlens=c["cu"],
        initial_states=initial_states,
        return_varlen_states=True,
        return_track_states=True,
        track_seq_idx=c["recompute_rows"],
        track_end_locs=c["end_locs"],
        dt_softplus=True,
        out=out,
        state_dtype=pool.dtype,
        **kwargs,
    )
    if in_pool:
        assert varlen is None and (track is None or c["recompute_rows"].numel() == 0)
    else:
        # What MambaMixer2 and _track_mamba_state_extend do with the results.
        pool[c["slots"]] = varlen
        if c["recompute_rows"].numel():
            pool[c["recompute_dst"]] = track.squeeze(0).to(pool.dtype)
        pool[c["final_dst"]] = pool[c["final_src"]]
    return out, states


def _assert_pool_mode_matches(c):
    ref_pool, new_pool = c["pool"].clone(), c["pool"].clone()
    ref_out, ref_states = _run(c, ref_pool, in_pool=False)
    new_out, new_states = _run(c, new_pool, in_pool=True)
    assert not torch.equal(ref_pool, c["pool"]), "the case must update the pool"
    assert torch.equal(new_out, ref_out)
    assert torch.equal(new_states, ref_states)
    assert torch.equal(new_pool, ref_pool)


SEQLENS = [
    (64,) * 8,  # every sequence starts and ends inside a chunk
    (300, 500, 128, 1000),
    (17, 256, 1, 700, 129),
]


@pytest.mark.parametrize("seqlens", SEQLENS)
@pytest.mark.parametrize("prefix", ["all", "some", "none"])
@pytest.mark.parametrize("pool_dtype", [torch.float32, torch.bfloat16])
def test_pool_mode_matches_gather_where_and_index_put(seqlens, prefix, pool_dtype):
    if get_device() != "cuda":
        pytest.skip("CUDA only")
    has_init = {
        "all": [True] * len(seqlens),
        "some": [i % 2 == 0 for i in range(len(seqlens))],
        "none": [False] * len(seqlens),
    }[prefix]
    c = _case(seqlens, 128, 64, 128, pool_dtype, has_init, seed=len(seqlens))
    _assert_pool_mode_matches(c)


def test_pool_mode_survives_autotuning_its_first_call():
    """The final-state kernel may read a row's initial state from the slot it then
    overwrites; autotuning that first pool-mode call must not chain its trials."""
    if get_device() != "cuda":
        pytest.skip("CUDA only")
    c = _case((64,) * 8, 64, 32, 64, torch.float32, [True] * 8, seed=7)
    tuner = _chunk_state_varlen_kernel
    saved_cache, saved_disk = dict(tuner.cache), tuner.cache_results
    tuner.cache.clear()
    tuner.cache_results = False
    try:
        new_pool = c["pool"].clone()
        new_out, _ = _run(c, new_pool, in_pool=True)  # tunes, in pool mode
        ref_pool = c["pool"].clone()
        ref_out, _ = _run(c, ref_pool, in_pool=False)  # reuses that config
    finally:
        tuner.cache.clear()
        tuner.cache.update(saved_cache)
        tuner.cache_results = saved_disk
    assert torch.equal(new_out, ref_out)
    assert torch.equal(new_pool, ref_pool)


def test_strided_pool_is_not_in_pool_mode():
    pool = torch.zeros(4, N_HEADS, 64, 128)
    assert _ssm_slots_contiguous(pool)
    assert _ssm_slots_contiguous(torch.zeros(2, 4, N_HEADS, 64, 128)[1])
    assert not _ssm_slots_contiguous(pool.transpose(2, 3))
    assert not _ssm_slots_contiguous(torch.zeros(4, N_HEADS, 64, 256)[..., :128])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
