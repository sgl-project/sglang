"""Unit tests for BaseLayerWithLoRA.commit_merged_as_base.

Validates the "merge a fixed-strength LoRA into the base once, then apply the
rest as a dynamic delta" primitive used by LTX-2 original mode. Weight-level
invariants are checked on CPU with a plain nn.Linear base (the parallel forward
path needs a distributed runtime and is covered by the diffusion server tests).
"""

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2TwoStageHQPipeline,
)


def _make_layer(
    base_weight: torch.Tensor,
    rank: int = 2,
    alpha: int = 2,
    snapshot_base: bool = True,
):
    base = nn.Linear(
        base_weight.shape[1],
        base_weight.shape[0],
        bias=False,
        dtype=base_weight.dtype,
    )
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


def _make_bf16_distilled_fixture(seed: int = 7):
    torch.manual_seed(seed)
    out_f, in_f, rank = 16, 16, 4
    base_w = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    A = torch.randn(rank, in_f, dtype=torch.bfloat16) * 0.2
    B = torch.randn(out_f, rank, dtype=torch.bfloat16) * 0.2
    lora_path = "/models/ltx-2.3-22b-distilled-lora-384.safetensors"
    return base_w, A, B, rank, lora_path


def test_bf16_distilled_negative_merge_drifts_while_preserved_snapshot_is_exact():
    """Reproduce the bf16 double-rounding drift from #46130 and verify exact restore."""
    base_w, A, B, rank, lora_path = _make_bf16_distilled_fixture()
    s1 = 0.25

    # Old restore strategy: permanent commit, then merge -s1 and commit again.
    old_layer = _make_layer(base_w.clone(), rank=rank, alpha=rank)
    assert old_layer.base_layer.weight.dtype == torch.bfloat16
    assert old_layer.cpu_weight.dtype == torch.bfloat16
    old_layer.set_lora_weights(
        A.clone(),
        B.clone(),
        lora_path=lora_path,
        strength=s1,
        clear_existing=True,
        merge_weights=True,
    )
    assert not old_layer._should_merge_in_fp32(old_layer._active_lora_list())
    old_layer.commit_merged_as_base()
    old_layer.set_lora_weights(
        A.clone(),
        B.clone(),
        lora_path=lora_path,
        strength=-s1,
        clear_existing=True,
        merge_weights=True,
    )
    old_layer.commit_merged_as_base()
    assert old_layer.base_layer.weight.dtype == torch.bfloat16
    assert not torch.equal(old_layer.base_layer.weight, base_w)

    # Fixed restore strategy: preserve the existing pre-merge cpu_weight snapshot.
    layer = _make_layer(base_w.clone(), rank=rank, alpha=rank)
    assert layer.base_layer.weight.dtype == torch.bfloat16
    assert layer.cpu_weight.dtype == torch.bfloat16
    layer.set_lora_weights(
        A.clone(),
        B.clone(),
        lora_path=lora_path,
        strength=s1,
        clear_existing=True,
        merge_weights=True,
    )
    snapshot_ptr = layer.cpu_weight.data_ptr()
    layer.commit_merged_as_base(preserve_base_snapshot=True)

    assert layer.has_preserved_base_snapshot
    assert layer.cpu_weight.data_ptr() == snapshot_ptr
    assert layer.cpu_weight.dtype == torch.bfloat16
    assert torch.equal(layer.cpu_weight, base_w)
    assert not torch.equal(layer.base_layer.weight, base_w)

    layer.restore_committed_base()

    assert not layer.has_preserved_base_snapshot
    assert layer.merged is False
    assert layer.disable_lora is True
    assert layer.lora_A is None and layer.lora_B is None
    assert layer.base_layer.weight.dtype == torch.bfloat16
    assert layer.cpu_weight.dtype == torch.bfloat16
    assert torch.equal(layer.base_layer.weight, base_w)
    assert torch.equal(layer.cpu_weight, base_w)


@pytest.mark.parametrize("snapshot_base", [True, False])
@pytest.mark.parametrize("offloaded", [False, True])
def test_preserved_commit_restore_exact_across_offload_and_dynamic_stage2(
    offloaded, snapshot_base
):
    """Preserved commit supports dynamic stage-2 deltas, exact restore, and later merges."""
    base_w, A, B, rank, lora_path = _make_bf16_distilled_fixture(seed=11)
    s1, s1_request, s2 = 0.25, 0.5, 0.75
    x = torch.randn(3, base_w.shape[1], dtype=torch.bfloat16)

    layer = _make_layer(
        base_w.clone(), rank=rank, alpha=rank, snapshot_base=snapshot_base
    )
    stored = _offload(layer) if offloaded else None

    def weight():
        return stored if offloaded else layer.base_layer.weight.detach()

    assert weight().dtype == torch.bfloat16
    assert layer.cpu_weight.dtype == torch.bfloat16

    layer.set_lora_weights(
        A.clone(),
        B.clone(),
        lora_path=lora_path,
        strength=s1,
        clear_existing=True,
        merge_weights=True,
    )
    merged_s1 = weight().clone()
    snapshot_ptr = layer.cpu_weight.data_ptr()

    layer.commit_merged_as_base(preserve_base_snapshot=True)
    assert layer.has_preserved_base_snapshot
    assert not layer._base_is_view
    assert layer.cpu_weight.data_ptr() == snapshot_ptr
    assert layer.cpu_weight.dtype == torch.bfloat16
    assert torch.equal(layer.cpu_weight, base_w)
    assert torch.equal(weight(), merged_s1)
    if not offloaded:
        torch.testing.assert_close(layer(x), x @ merged_s1.T, rtol=0, atol=0)

    # Dynamic stage-2 delta runs on top of the merged base without unmerging.
    layer.set_lora_weights(
        A.clone(),
        B.clone(),
        lora_path=lora_path,
        strength=s2 - s1,
        clear_existing=True,
        merge_weights=False,
    )
    assert layer.merged is False
    assert layer.disable_lora is False
    assert torch.equal(weight(), merged_s1)
    assert torch.equal(layer.cpu_weight, base_w)
    if not offloaded:
        expected_stage2 = x @ merged_s1.T + (s2 - s1) * (x @ A.T @ B.T)
        torch.testing.assert_close(layer(x), expected_stage2, atol=2e-2, rtol=2e-2)

    # Restore recovers the exact pre-merge bf16 base and clears dynamic state.
    layer.restore_committed_base()
    assert not layer.has_preserved_base_snapshot
    assert layer.merged is False
    assert layer.disable_lora is True
    assert layer.lora_weights_list == []
    assert weight().dtype == torch.bfloat16
    assert torch.equal(weight(), base_w)
    assert torch.equal(layer.cpu_weight, base_w)
    if not offloaded:
        torch.testing.assert_close(layer(x), x @ base_w.T, rtol=0, atol=0)

    # Subsequent per-request merge -> stage-2 merge -> unmerge recovers exact base.
    layer.set_lora_weights(
        A.clone(),
        B.clone(),
        lora_path=lora_path,
        strength=s1_request,
        clear_existing=True,
        merge_weights=True,
    )
    layer.set_lora_weights(
        A.clone(),
        B.clone(),
        lora_path=lora_path,
        strength=s2,
        clear_existing=True,
        merge_weights=True,
    )
    layer.unmerge_lora_weights()
    assert torch.equal(weight(), base_w)


def test_preserved_commit_restore_handles_inference_mode_and_cached_merge():
    """Preserved restore works cleanly for inference-mode weights and cached merges."""
    base_w, A, B, rank, lora_path = _make_bf16_distilled_fixture(seed=12)

    # Inference-mode base weight: restored parameter must be mutable and unaliased.
    with torch.inference_mode():
        base = nn.Linear(
            base_w.shape[1], base_w.shape[0], bias=False, dtype=torch.bfloat16
        )
        base.weight.copy_(base_w)
    inf_layer = LinearWithLoRA(base, lora_rank=rank, lora_alpha=rank)
    assert inf_layer.base_layer.weight.is_inference()
    with torch.inference_mode(False):
        inf_layer.set_lora_weights(
            A.clone(),
            B.clone(),
            lora_path=lora_path,
            strength=0.25,
            clear_existing=True,
            merge_weights=True,
        )
        inf_layer.commit_merged_as_base(preserve_base_snapshot=True)
        inf_layer.restore_committed_base()
        assert not inf_layer.base_layer.weight.is_inference()
        assert inf_layer.base_layer.weight.dtype == torch.bfloat16
        assert torch.equal(inf_layer.base_layer.weight, base_w)
        assert inf_layer.base_layer.weight.data_ptr() != inf_layer.cpu_weight.data_ptr()
        inf_layer.base_layer.weight.data.zero_()
        assert torch.equal(inf_layer.cpu_weight, base_w)
        # Subsequent merge/unmerge after restore still works and recovers base_w.
        inf_layer.set_lora_weights(
            A.clone(),
            B.clone(),
            lora_path=lora_path,
            strength=0.5,
            clear_existing=True,
            merge_weights=True,
        )
        inf_layer.unmerge_lora_weights()
        assert torch.equal(inf_layer.base_layer.weight, base_w)

    # Cached-merge path: install_merged_weight -> preserved commit owns its snapshot.
    cached_layer = _make_layer(
        base_w.clone(), rank=rank, alpha=rank, snapshot_base=False
    )
    base_view = cached_layer.base_layer.weight.detach()
    cached_layer.install_merged_weight((base_w + 1).clone(), base_view)
    assert cached_layer._base_is_view
    cached_layer.commit_merged_as_base(preserve_base_snapshot=True)
    assert not cached_layer._base_is_view
    assert cached_layer.cpu_weight.data_ptr() != base_view.data_ptr()
    cached_layer.restore_committed_base()
    assert torch.equal(cached_layer.base_layer.weight, base_w)
    assert torch.equal(cached_layer.cpu_weight, base_w)


def test_preserved_commit_guards_against_in_place_merge_and_noop_restore():
    """Merging onto a preserved commit fails cleanly; restore is a no-op otherwise."""
    base_w, A, B, rank, lora_path = _make_bf16_distilled_fixture(seed=13)
    layer = _make_layer(base_w.clone(), rank=rank, alpha=rank)

    # No-op before any commit.
    layer.restore_committed_base()
    assert torch.equal(layer.base_layer.weight, base_w)

    # Permanent commit is not reverted by restore_committed_base.
    perm_layer = _make_layer(base_w.clone(), rank=rank, alpha=rank)
    perm_layer.set_lora_weights(
        A.clone(), B.clone(), lora_path=lora_path, strength=0.25, merge_weights=True
    )
    perm_layer.commit_merged_as_base()
    perm_merged = perm_layer.base_layer.weight.detach().clone()
    assert not perm_layer.has_preserved_base_snapshot
    perm_layer.restore_committed_base()
    assert torch.equal(perm_layer.base_layer.weight, perm_merged)

    # Preserved commit rejects merging or installing merged weights until restored.
    layer.set_lora_weights(
        A.clone(), B.clone(), lora_path=lora_path, strength=0.25, merge_weights=True
    )
    layer.commit_merged_as_base(preserve_base_snapshot=True)
    merged_w = layer.base_layer.weight.detach().clone()
    layer.set_lora_weights(
        A.clone(), B.clone(), lora_path=lora_path, strength=0.25, merge_weights=False
    )

    with pytest.raises(RuntimeError, match="restore_committed_base"):
        layer.set_lora_weights(
            A.clone(), B.clone(), lora_path=lora_path, strength=0.5, merge_weights=True
        )
    with pytest.raises(RuntimeError, match="restore_committed_base"):
        layer.merge_lora_weights()
    with pytest.raises(RuntimeError, match="restore_committed_base"):
        layer.install_merged_weight(merged_w.clone(), base_w.clone())

    # Guard failures must leave weights and dynamic adapter intact.
    assert torch.equal(layer.base_layer.weight, merged_w)
    assert torch.equal(layer.cpu_weight, base_w)
    assert layer.strength == 0.25
    assert layer.disable_lora is False


def _make_ltx2_hq_pipeline(
    matched_layer: LinearWithLoRA,
    unmatched_layer: LinearWithLoRA,
    A: torch.Tensor,
    B: torch.Tensor,
    lora_path: str,
) -> LTX2TwoStageHQPipeline:
    pipeline = object.__new__(LTX2TwoStageHQPipeline)
    pipeline._stages = []
    pipeline.modules = {"transformer": nn.Module()}
    pipeline.server_args = SimpleNamespace(
        lora_alpha=None,
        lora_merge_mode="merge",
        model_path="/models/ltx-2.3",
        use_fsdp_inference=False,
        quantization=None,
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(arch_config=SimpleNamespace())
        ),
    )
    pipeline._ltx2_residency = SimpleNamespace(
        mode="original", enter_phase=lambda _phase: False
    )
    pipeline._use_premerged_stage2_transformer = False
    pipeline._distilled_lora_path = lora_path
    pipeline._stage1_lora_path = None
    pipeline._stage1_lora_scale = 1.0
    pipeline._active_lora_signature = None
    pipeline._stage1_distilled_in_base = False
    pipeline._stage1_distilled_base_strength = None
    pipeline._should_merge_stage2_distilled_lora = staticmethod(lambda _args: True)
    pipeline._temporarily_disable_offload = lambda *args, **kwargs: nullcontext([])
    pipeline.lora_initialized = True
    pipeline.lora_adapters = defaultdict(dict)
    pipeline.loaded_adapter_paths = {
        "ltx2_stage1_distilled": lora_path,
        "ltx2_stage2_distilled": lora_path,
    }
    pipeline.loaded_adapter_files = {}
    pipeline.loaded_adapter_configs = {}
    pipeline.loaded_adapter_fingerprints = {}
    pipeline.loaded_adapter_alphas = {
        "ltx2_stage1_distilled": None,
        "ltx2_stage2_distilled": None,
    }
    pipeline.cur_adapter_name = {}
    pipeline.cur_adapter_path = {}
    pipeline.cur_adapter_strength = {}
    pipeline.cur_adapter_config = {}
    pipeline.auto_merge_rounding_loss = {}
    pipeline.lora_layers = {"matched": matched_layer, "unmatched": unmatched_layer}
    pipeline.lora_layers_transformer_2 = {}
    pipeline.lora_layers_critic = {}
    pipeline.is_lora_merged = {}

    adapter_weights = {"matched.lora_A": A, "matched.lora_B": B}
    pipeline.lora_adapters["ltx2_stage1_distilled"] = adapter_weights
    pipeline.lora_adapters["ltx2_stage2_distilled"] = adapter_weights
    return pipeline


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("override_strength", [0.0, 0.5])
def test_ltx2_hq_stage1_override_restores_exact_bf16_base(offloaded, override_strength):
    """LTX2TwoStageHQPipeline restores the exact bf16 base on stage-1 override."""
    base_w, A, B, rank, lora_path = _make_bf16_distilled_fixture(seed=17)
    unmatched_w = torch.randn_like(base_w)

    matched = _make_layer(
        base_w.clone(), rank=rank, alpha=rank, snapshot_base=not offloaded
    )
    unmatched = _make_layer(
        unmatched_w.clone(), rank=rank, alpha=rank, snapshot_base=not offloaded
    )
    matched_stored = _offload(matched) if offloaded else None
    unmatched_stored = _offload(unmatched) if offloaded else None

    def cur_weight(layer, stored):
        return stored if offloaded else layer.base_layer.weight.detach()

    pipeline = _make_ltx2_hq_pipeline(matched, unmatched, A, B, lora_path)

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        pipeline._maybe_merge_stage1_distilled_into_base(pipeline.server_args)
        assert pipeline._stage1_distilled_in_base is True
        assert pipeline._stage1_distilled_base_strength == 0.25
        assert matched.has_preserved_base_snapshot is True
        assert unmatched.has_preserved_base_snapshot is False
        assert torch.equal(matched.cpu_weight, base_w)
        merged_stage1_w = cur_weight(matched, matched_stored).clone()
        assert not torch.equal(merged_stage1_w, base_w)

        # Normal default request: stage 1 uses merged base, stage 2 uses dynamic delta.
        pipeline.switch_lora_phase("stage1")
        assert matched.disable_lora is True
        assert torch.equal(cur_weight(matched, matched_stored), merged_stage1_w)

        pipeline.switch_lora_phase("stage2")
        assert matched.merged is False
        assert matched.disable_lora is False
        assert matched.strength == pytest.approx(0.25)
        assert torch.equal(cur_weight(matched, matched_stored), merged_stage1_w)

        # Request overriding stage-1 strength restores exact original base first.
        override_req = SimpleNamespace(
            extra={"ltx2_distilled_lora_strength_stage_1": override_strength}
        )
        pipeline.switch_lora_phase("stage1", override_req)
        assert pipeline._stage1_distilled_in_base is False
        assert pipeline._stage1_distilled_base_strength is None
        assert matched.has_preserved_base_snapshot is False
        assert torch.equal(matched.cpu_weight, base_w)
        assert torch.equal(cur_weight(unmatched, unmatched_stored), unmatched_w)

        if override_strength == 0.0:
            assert matched.merged is False
            assert matched.disable_lora is True
            assert cur_weight(matched, matched_stored).dtype == torch.bfloat16
            assert torch.equal(cur_weight(matched, matched_stored), base_w)

            # Stage 2 after a 0.0 override runs full stage-2 strength dynamically
            # on the exact restored base, and a later default request merges cleanly.
            pipeline.switch_lora_phase("stage2", override_req)
            assert matched.merged is False
            assert matched.disable_lora is False
            assert matched.strength == pytest.approx(0.5)
            assert torch.equal(cur_weight(matched, matched_stored), base_w)

            pipeline.switch_lora_phase("stage1")
            assert matched.merged is True
            pipeline.deactivate_lora_weights(target="transformer")
            assert torch.equal(cur_weight(matched, matched_stored), base_w)
        else:
            assert matched.merged is True
            assert not torch.equal(cur_weight(matched, matched_stored), base_w)
            # Switching to stage 2 unmerges stage 1 back to the exact original base
            # and applies stage 2 dynamically.
            pipeline.switch_lora_phase("stage2", override_req)
            assert matched.merged is False
            assert matched.disable_lora is False
            assert matched.strength == pytest.approx(0.5)
            assert cur_weight(matched, matched_stored).dtype == torch.bfloat16
            assert torch.equal(cur_weight(matched, matched_stored), base_w)

            pipeline.deactivate_lora_weights(target="transformer")
            assert matched.disable_lora is True
            assert torch.equal(cur_weight(matched, matched_stored), base_w)


@pytest.mark.parametrize("retry_phase", ["stage1", "stage2"])
def test_ltx2_hq_partial_restore_failure_retries_on_next_phase_switch(retry_phase):
    """If restoring fails mid-loop, a subsequent phase switch completes restore."""
    base_w1, A, B, rank, lora_path = _make_bf16_distilled_fixture(seed=19)
    base_w2 = torch.randn_like(base_w1)
    layer1 = _make_layer(base_w1.clone(), rank=rank, alpha=rank)
    layer2 = _make_layer(base_w2.clone(), rank=rank, alpha=rank)

    pipeline = _make_ltx2_hq_pipeline(layer1, layer2, A, B, lora_path)
    pipeline.lora_layers = {"l1": layer1, "l2": layer2}
    adapter_weights = {
        "l1.lora_A": A,
        "l1.lora_B": B,
        "l2.lora_A": A,
        "l2.lora_B": B,
    }
    pipeline.lora_adapters["ltx2_stage1_distilled"] = adapter_weights
    pipeline.lora_adapters["ltx2_stage2_distilled"] = adapter_weights

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        pipeline._maybe_merge_stage1_distilled_into_base(pipeline.server_args)
        assert pipeline._stage1_distilled_in_base is True
        assert layer1.has_preserved_base_snapshot is True
        assert layer2.has_preserved_base_snapshot is True

        orig_restore = layer2.restore_committed_base
        fail_once = {"fail": True}

        def flaky_restore():
            if fail_once["fail"]:
                fail_once["fail"] = False
                raise RuntimeError("injected restore failure")
            orig_restore()

        layer2.restore_committed_base = flaky_restore

        override_req = SimpleNamespace(
            extra={"ltx2_distilled_lora_strength_stage_1": 0.0}
        )
        with pytest.raises(RuntimeError, match="injected restore failure"):
            pipeline.switch_lora_phase("stage1", override_req)

        # Layer 1 restored, layer 2 still pending; pipeline must not assume a valid
        # stage-1 base on the next request.
        assert torch.equal(layer1.base_layer.weight, base_w1)
        assert not layer1.has_preserved_base_snapshot
        assert layer2.has_preserved_base_snapshot
        assert pipeline._stage1_distilled_in_base is True
        assert pipeline._stage1_distilled_base_strength is None

        # Next phase switch (stage1 or stage2) finishes restoring layer 2 and
        # falls through to the per-request path consistently across both layers.
        pipeline.switch_lora_phase(retry_phase)
        assert pipeline._stage1_distilled_in_base is False
        assert not layer2.has_preserved_base_snapshot
        if retry_phase == "stage1":
            assert layer1.merged is True and layer2.merged is True
            pipeline.deactivate_lora_weights(target="transformer")
        else:
            assert layer1.merged is False and layer2.merged is False
            assert layer1.disable_lora is False and layer2.disable_lora is False
            assert layer1.strength == pytest.approx(0.5)
            assert layer2.strength == pytest.approx(0.5)
        assert torch.equal(layer1.base_layer.weight, base_w1)
        assert torch.equal(layer2.base_layer.weight, base_w2)
