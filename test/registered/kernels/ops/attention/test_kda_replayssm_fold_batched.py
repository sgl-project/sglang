"""Parity: layer-batched fold == per-layer loop of commit_kda_replayssm_spec.

The commit-side ReplaySSM fold used to launch commit_kda_replayssm_spec once per
KDA layer (a Python loop -> ~69 tiny eager launches at bs=1, dispatch-bound).
`commit_kda_replayssm_spec_all_layers` packs the layer into the head grid axis so
one launch folds every layer. This must be BIT-IDENTICAL to the loop: each
(layer, head, v-tile) block runs the same per-slot recurrent replay; batching only
fans out the grid. Any layer-offset / stride bug shows up as a per-layer mismatch.

Shapes cover GQA (H != HV), non-pow2 K/V, single accept step, padding (-1) slots,
varied accept_lens (incl. 0 = skip), track on/off, and single-layer (i_layer==0
must match the per-layer entry). GPU-only.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip(
        "KDA ReplaySSM batched-fold parity needs CUDA (triton).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (  # noqa: E402
    commit_kda_replayssm_spec,
    commit_kda_replayssm_spec_all_layers,
)

DEV = "cuda"

# (num_layers, num_slots, HV, H, K, V, L, has_track)
SHAPES = [
    (3, 6, 4, 4, 64, 64, 16, False),  # square heads, no track
    (2, 5, 8, 2, 128, 128, 16, True),  # GQA 4:1, track on
    (4, 7, 6, 3, 96, 80, 16, False),  # non-pow2 K/V, GQA 2:1
    (2, 4, 4, 4, 64, 64, 16, True),  # track on
    (1, 4, 4, 4, 64, 64, 8, False),  # single layer -> i_layer==0 path
]
SHAPE_IDS = ["square", "gqa4-track", "nonpow2", "track", "single-layer"]


def _rand_rings(num_layers, num_slots, HV, H, K, V, L):
    torch.manual_seed(0)
    return dict(
        temporal=torch.randn(
            num_layers, num_slots, HV, V, K, device=DEV, dtype=torch.float32
        ),
        rawv=torch.randn(
            num_layers, num_slots, HV, L, V, device=DEV, dtype=torch.bfloat16
        ),
        rawk=torch.randn(
            num_layers, num_slots, H, L, K, device=DEV, dtype=torch.bfloat16
        ),
        gk=(-5.0)
        * torch.sigmoid(
            torch.randn(
                num_layers, num_slots, HV, L, K, device=DEV, dtype=torch.float32
            )
        ),
        beta=torch.sigmoid(
            torch.randn(num_layers, num_slots, HV, L, device=DEV, dtype=torch.float32)
        ),
    )


@pytest.mark.parametrize(
    "num_layers,num_slots,HV,H,K,V,L,has_track", SHAPES, ids=SHAPE_IDS
)
def test_fold_batched_matches_per_layer(
    num_layers, num_slots, HV, H, K, V, L, has_track
):
    r = _rand_rings(num_layers, num_slots, HV, H, K, V, L)
    B = min(3, num_slots - 1)
    # slots 1..B; one row is a -1 padding slot (must be skipped identically).
    slots = torch.arange(1, B + 1, device=DEV, dtype=torch.int32)
    if B > 1:
        slots[-1] = -1
    # accept_lens: mix 0 (skip), mid, full L.
    accept = torch.tensor(
        [[0, L // 2, L][i % 3] for i in range(B)], device=DEV, dtype=torch.int32
    )
    if has_track:
        track_idx = torch.arange(
            num_slots - B, num_slots, device=DEV, dtype=torch.int32
        )
        track_step = torch.tensor(
            [[-1, L // 2 - 1, 1][i % 3] for i in range(B)],
            device=DEV,
            dtype=torch.int32,
        )
    else:
        track_idx = track_step = None

    # reference: per-layer loop
    ref = r["temporal"].clone()
    for li in range(num_layers):
        commit_kda_replayssm_spec(
            checkpoint_state=ref[li],
            rawv_cache=r["rawv"][li],
            rawk_cache=r["rawk"][li],
            gk_cache=r["gk"][li],
            beta_cache=r["beta"][li],
            ssm_state_indices=slots,
            accept_lens=accept,
            max_cache_len=L,
            num_k_heads=H,
            mamba_track_indices=track_idx,
            mamba_steps_to_track=track_step,
            null_block_id=-1,
        )
    # batched: one launch
    bat = r["temporal"].clone()
    commit_kda_replayssm_spec_all_layers(
        checkpoint_state=bat,
        rawv_cache=r["rawv"],
        rawk_cache=r["rawk"],
        gk_cache=r["gk"],
        beta_cache=r["beta"],
        ssm_state_indices=slots,
        accept_lens=accept,
        max_cache_len=L,
        num_k_heads=H,
        mamba_track_indices=track_idx,
        mamba_steps_to_track=track_step,
        null_block_id=-1,
    )
    assert torch.equal(ref, bat), (
        f"batched fold != per-layer loop; max abs diff "
        f"{(ref - bat).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
