import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"
CACHE_LEN = 16
HISTORY_LEN = 4
TRACK_LEN = 3
LAYERS, SLOTS, H, HV, K, V = 2, 6, 1, 2, 64, 80
FIELDS = ("d_cache", "k_cache", "g_cache", "d_residual_cache", "k_residual_cache")


def _run_commit(
    *,
    base: int,
    mode: str,
    residual: bool,
    field: str | None = None,
    poison: float = 0.0,
    poison_start: int = HISTORY_LEN,
    history_len: int = HISTORY_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(42)

    def rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=DEVICE, dtype=dtype) * 0.125

    state = rand(LAYERS, SLOTS, HV, V, K, dtype=torch.float32)
    initial = state.clone()
    caches = {
        "d_cache": rand(LAYERS, SLOTS, HV, CACHE_LEN, V),
        "k_cache": rand(LAYERS, SLOTS, H, CACHE_LEN, K),
        "g_cache": torch.full((LAYERS, SLOTS, HV, CACHE_LEN), -0.03125, device=DEVICE),
        "d_residual_cache": rand(LAYERS, SLOTS, HV, CACHE_LEN, V) * 0.001,
        "k_residual_cache": rand(LAYERS, SLOTS, H, CACHE_LEN, K) * 0.001,
    }
    physical = (base + torch.arange(CACHE_LEN, device=DEVICE)) % CACHE_LEN
    if field is not None:
        caches[field][:, 2, :, physical[poison_start:]] = poison

    def ints(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, device=DEVICE, dtype=torch.int32)

    state_indices = ints([1, 2, 0])
    replay_indices = ints([2, 1, 0])
    write_pos = ints([history_len, history_len, history_len])
    cache_base = ints([base, base, base])
    is_flush = ints([1, 0, int(mode != "track")])
    accepted = ints([2, 2, 2])
    track_indices = ints([3, 0, 5]) if mode != "active" else None
    track_steps = ints([0, -1, 0]) if mode != "active" else None

    commit_gdn_replayssm_circular(
        checkpoint_state=state,
        d_cache=caches["d_cache"],
        k_cache=caches["k_cache"],
        g_cache=caches["g_cache"],
        d_residual_cache=caches["d_residual_cache"] if residual else None,
        k_residual_cache=caches["k_residual_cache"] if residual else None,
        state_batch_indices=state_indices,
        replay_indices=replay_indices,
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        accept_lens=accepted,
        mamba_track_indices=track_indices,
        mamba_steps_to_track=track_steps,
        null_block_id=0,
    )

    changed = set()
    if mode != "track":
        changed.add(1)
    if mode != "active":
        changed.add(3)
    for slot in set(range(SLOTS)) - changed:
        torch.testing.assert_close(state[:, slot], initial[:, slot], rtol=0, atol=0)
    torch.testing.assert_close(
        write_pos,
        ints([history_len, history_len, history_len if mode == "track" else 0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        cache_base,
        ints(
            [base, base, base if mode == "track" else (base + history_len) % CACHE_LEN]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(is_flush, ints([1, 0, 0]), rtol=0, atol=0)
    return state, initial


@pytest.mark.parametrize(
    "field,residual",
    [(field, True) for field in FIELDS] + [(field, False) for field in FIELDS[:3]],
)
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "mode,base", [("active", 0), ("track", 13), ("both", 0), ("both", 13)]
)
def test_unused_history_cannot_contaminate_checkpoint(
    field: str, residual: bool, poison: float, mode: str, base: int
) -> None:
    clean, _ = _run_commit(base=base, mode=mode, residual=residual)
    reused, _ = _run_commit(
        base=base, mode=mode, residual=residual, field=field, poison=poison
    )
    assert torch.isfinite(reused).all()
    torch.testing.assert_close(reused, clean, rtol=0, atol=0)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("base", [0, 13])
def test_tracked_prefix_excludes_later_valid_nan(field: str, base: int) -> None:
    clean, _ = _run_commit(base=base, mode="both", residual=True)
    poisoned, _ = _run_commit(
        base=base,
        mode="both",
        residual=True,
        field=field,
        poison=float("nan"),
        poison_start=TRACK_LEN,
    )
    assert not torch.isfinite(poisoned[:, 1]).all()
    assert torch.isfinite(poisoned[:, 3]).all()
    torch.testing.assert_close(poisoned[:, 3], clean[:, 3], rtol=0, atol=0)


@pytest.mark.parametrize("field", FIELDS)
def test_empty_history_preserves_checkpoint(field: str) -> None:
    result, initial = _run_commit(
        base=13,
        mode="active",
        residual=True,
        field=field,
        poison=float("nan"),
        poison_start=0,
        history_len=0,
    )
    torch.testing.assert_close(result, initial, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
