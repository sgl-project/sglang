"""Unit tests for BaseLayerWithLoRA.commit_merged_as_base.

Validates the "merge a fixed-strength LoRA into the base once, then apply the
rest as a dynamic delta" primitive used by LTX-2 original mode. Weight-level
invariants are checked on CPU with a plain nn.Linear base (the parallel forward
path needs a distributed runtime and is covered by the diffusion server tests).
"""

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA


def _make_layer(
    base_weight: torch.Tensor,
    rank: int = 2,
    alpha: int = 2,
    snapshot_base: bool = True,
):
    base = nn.Linear(base_weight.shape[1], base_weight.shape[0], bias=False)
    with torch.no_grad():
        base.weight.copy_(base_weight)
    return LinearWithLoRA(
        base, lora_rank=rank, lora_alpha=alpha, snapshot_base=snapshot_base
    )


def _offload(layer: LinearWithLoRA) -> torch.Tensor:
    """Move the weight into a CPU view and leave a (1,) placeholder, as layerwise
    offload does; returns the view."""
    base = layer.base_layer
    stored = base.weight.detach()
    base._packed_weight_cpu = stored
    base.weight.data = torch.empty(1, dtype=stored.dtype)
    return stored


def test_commit_merged_as_base_promotes_weights_and_resets_state():
    torch.manual_seed(0)
    out_f, in_f, rank = 6, 8, 2
    base_w = torch.randn(out_f, in_f)
    A = torch.randn(rank, in_f)
    B = torch.randn(out_f, rank)
    s1 = 0.25  # alpha == rank below, so scale == strength

    layer = _make_layer(base_w, rank=rank, alpha=rank)
    layer.set_lora_weights(
        A.clone(), B.clone(), strength=s1, clear_existing=True, merge_weights=True
    )
    assert layer.merged

    layer.commit_merged_as_base()

    merged_w = base_w + s1 * (B @ A)
    # Merged weights become the permanent base + restore target.
    assert torch.allclose(layer.base_layer.weight, merged_w, atol=1e-5)
    assert torch.allclose(layer.cpu_weight, merged_w, atol=1e-5)
    # Bookkeeping reset so a later dynamic set_lora adds a delta on top, and
    # deactivate (which only unmerges when merged=True) leaves the base intact.
    assert layer.merged is False
    assert layer.disable_lora is True
    assert layer.lora_weights_list == []
    assert layer.lora_A is None and layer.lora_B is None


def test_lora_output_offset_tracks_dynamic_and_merged_scale():
    layer = _make_layer(torch.zeros(1, 2), rank=1, alpha=2)
    inputs = torch.zeros(3, 2)
    layer.set_lora_weights(
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        output_offset=torch.tensor([3.0]),
        strength=0.5,
        clear_existing=True,
        merge_weights=False,
    )
    torch.testing.assert_close(layer(inputs), torch.full((3, 1), 3.0))
    layer.merge_lora_weights()
    torch.testing.assert_close(layer(inputs), torch.full((3, 1), 3.0))
    with pytest.raises(ValueError, match="constant output offset"):
        layer.commit_merged_as_base()


def test_dynamic_delta_after_commit_does_not_unmerge_base():
    torch.manual_seed(1)
    out_f, in_f, rank = 4, 5, 2
    base_w = torch.randn(out_f, in_f)
    A = torch.randn(rank, in_f)
    B = torch.randn(out_f, rank)
    s1, delta = 0.25, 0.25

    layer = _make_layer(base_w, rank=rank, alpha=rank)
    layer.set_lora_weights(
        A.clone(), B.clone(), strength=s1, clear_existing=True, merge_weights=True
    )
    layer.commit_merged_as_base()
    merged_w = layer.base_layer.weight.detach().clone()

    layer.set_lora_weights(
        A.clone(), B.clone(), strength=delta, clear_existing=True, merge_weights=False
    )
    # Dynamic mode must NOT touch the merged base weights (no unmerge happened).
    assert layer.merged is False
    assert layer.disable_lora is False
    assert torch.allclose(layer.base_layer.weight, merged_w, atol=1e-6)
    assert len(layer.lora_weights_list) == 1
    assert layer.strength == delta
    # Effective transform (base + dynamic delta) equals a single merge at s1+delta.
    effective = layer.base_layer.weight + layer.strength * (B @ A)
    assert torch.allclose(effective, base_w + (s1 + delta) * (B @ A), atol=1e-5)


def test_negative_merge_after_commit_restores_original_base():
    """Restore path: re-merging at the negative strength recovers the base."""
    torch.manual_seed(2)
    out_f, in_f, rank = 5, 7, 3
    base_w = torch.randn(out_f, in_f)
    A = torch.randn(rank, in_f)
    B = torch.randn(out_f, rank)
    s1 = 0.25

    layer = _make_layer(base_w, rank=rank, alpha=rank)
    layer.set_lora_weights(
        A.clone(), B.clone(), strength=s1, clear_existing=True, merge_weights=True
    )
    layer.commit_merged_as_base()
    assert not torch.allclose(layer.base_layer.weight, base_w, atol=1e-5)

    # Subtract the merged delta (as _unmerge_stage1_distilled_from_base does).
    layer.set_lora_weights(
        A.clone(), B.clone(), strength=-s1, clear_existing=True, merge_weights=True
    )
    layer.commit_merged_as_base()

    assert torch.allclose(layer.base_layer.weight, base_w, atol=1e-5)
    assert torch.allclose(layer.cpu_weight, base_w, atol=1e-5)
    assert layer.merged is False


@pytest.mark.parametrize("snapshot_base", [True, False])
@pytest.mark.parametrize("offloaded", [False, True])
def test_commit_snapshots_materialized_weight(offloaded, snapshot_base):
    """The committed snapshot is the real weight, not the layerwise placeholder.

    Replays LTX-2 original mode when a request overrides the stage-1 distilled
    strength: commit s1 into the base, subtract it again, merge the request's
    stage-1 strength, then merge stage 2 (which unmerges stage 1 first).
    """
    torch.manual_seed(3)
    out_f, in_f, rank = 6, 8, 2
    base_w = torch.randn(out_f, in_f)
    A = torch.randn(rank, in_f)
    B = torch.randn(out_f, rank)
    s1, s1_request, s2 = 0.25, 0.5, 0.75

    layer = _make_layer(base_w, rank=rank, alpha=rank, snapshot_base=snapshot_base)
    stored = _offload(layer) if offloaded else None

    def weight():
        return stored if offloaded else layer.base_layer.weight.detach()

    def merge(strength):
        layer.set_lora_weights(
            A.clone(),
            B.clone(),
            strength=strength,
            clear_existing=True,
            merge_weights=True,
        )

    merge(s1)
    layer.commit_merged_as_base()
    torch.testing.assert_close(layer.cpu_weight, base_w + s1 * (B @ A))
    assert not layer._base_is_view

    merge(-s1)
    layer.commit_merged_as_base()
    torch.testing.assert_close(layer.cpu_weight, base_w, atol=1e-5, rtol=0)

    merge(s1_request)
    merge(s2)
    torch.testing.assert_close(weight(), base_w + s2 * (B @ A), atol=1e-5, rtol=0)

    layer.unmerge_lora_weights()
    torch.testing.assert_close(weight(), base_w, atol=1e-5, rtol=0)
