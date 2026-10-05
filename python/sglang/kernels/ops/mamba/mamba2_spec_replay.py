"""Pinned FlashInfer checkpointing verify with eager accepted-state commit.

Records are physical-slot indexed. Pending history stays zero: verification
never changes live state, and the accepted endpoint is materialized before the
next forward. Compact records are scratch, not transferable radix checkpoints.
"""

from functools import lru_cache
from importlib.metadata import version

from sglang.srt.runtime_context import get_exec


@lru_cache(maxsize=1)
def checkpointing_kernel():
    installed = version("flashinfer-python")
    if installed.split("+")[0] != "0.7.0.post1":
        raise ValueError(
            "--enable-linear-replayssm-spec for Mamba2 requires FlashInfer 0.7.0.post1; "
            f"found {installed}"
        )
    from flashinfer.mamba import checkpointing_ssu

    return checkpointing_ssu


def verify_mamba2_replay(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    *,
    layer_cache,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    pad_slot_id=-1,
    out=None,
    disable_state_update=False,
    intermediate_states_buffer=None,
    cache_steps=None,
    retrieve_parent_token=None,
    intermediate_state_indices=None,
):
    """SSU-compatible verify entry; writes records, never candidate snapshots."""
    assert disable_state_update and intermediate_states_buffer is None
    assert retrieve_parent_token is None
    assert x.shape[1] == cache_steps == 4
    assert layer_cache.intermediate_ssm is None
    # The native materializer needs per-head A, which the old cumulative-decay
    # record encoded implicitly. Copy into graph-stable pool-owned storage.
    layer_cache.mamba2_replay_A.copy_(A[:, 0, 0])
    checkpointing_kernel()(
        state=state,
        x_cache=layer_cache.mamba2_replay_x,
        B_cache=layer_cache.mamba2_replay_B,
        dt_cache=layer_cache.mamba2_replay_dt,
        ring_start=layer_cache.mamba2_replay_ring_start,
        prev_num_accepted_tokens=layer_cache.mamba2_replay_pending,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        out=out,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        pad_slot_id=pad_slot_id,
        # p=0 and ring_len=2*T prevent checkpoint writes. SR is only needed
        # when acceptance materializes a state, not for verification outputs.
        algorithm="monolith",
    )


def commit_mamba2_replay(cache, slots, last, tracks=None, track_steps=None):
    """Materialize all-layer endpoints; convolution rollback remains unchanged."""
    from sglang.kernels.ops.mamba.flashinfer_replay_materialize import (
        materialize_flashinfer_mamba2,
    )

    cfg = get_exec().mamba
    seed = None
    if cfg.enable_mamba_cache_stochastic_rounding:
        # Native all-layer materialization derives independent layer seeds.
        seed = cache.mamba2_replay_seed[:1]
        seed.random_(0, 2**32)
    materialize_flashinfer_mamba2(
        cache.temporal,
        cache.mamba2_replay_x,
        cache.mamba2_replay_B,
        cache.mamba2_replay_dt,
        cache.mamba2_replay_A,
        cache.mamba2_replay_ptrs,
        slots,
        last,
        tracks,
        track_steps,
        seed=seed,
        philox_rounds=(
            (cfg.mamba_cache_philox_rounds or 10)
            if cfg.enable_mamba_cache_stochastic_rounding
            else 0
        ),
    )
