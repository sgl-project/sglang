"""Qwen3-VL ViT position-embedding interpolation must be consistent across paths.

`fast_pos_embed_interpolate()` (used by the CUDA-graph ViT path) honors
`enable_precise_embedding_interpolation` via `_get_interpolation_indices()`.
`fast_pos_embed_interpolate_from_list()` (used by the default eager path)
hardcoded `torch.linspace(...)`, i.e. it always behaved as
`align_corners=True`, so the flag had no effect on the default path.

These tests pin both paths to the same coordinates for both flag values.

Destination in-tree:
    test/registered/unit/models/test_qwen3_vl_pos_embed_interp.py
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_GRID_PER_SIDE = 8
MERGE = 2
HIDDEN = 4

# Grid sizes deliberately != NUM_GRID_PER_SIDE: at h == num_grid_per_side the two
# coordinate formulas coincide, so that case cannot discriminate.
GRIDS = [(1, 4, 6), (1, 6, 10), (1, 12, 4), (2, 4, 4)]


def _make_vision_model(align_corners: bool) -> Qwen3VLMoeVisionModel:
    """Build only the attributes the two interpolation paths touch.

    `__init__` needs a full vision config plus a global exec context, neither of
    which is required here:
      from_list path -> device, dtype, num_grid_per_side, pos_embed, spatial_merge_size
      interpolate path -> pos_embed, num_grid_per_side, align_corners, spatial_merge_size
    """
    model = object.__new__(Qwen3VLMoeVisionModel)
    torch.nn.Module.__init__(
        model
    )  # set up _modules/_parameters for attribute assignment
    model.num_grid_per_side = NUM_GRID_PER_SIDE
    model.spatial_merge_size = MERGE
    model.align_corners = align_corners
    # device/dtype are read-only properties over patch_embed.proj.weight
    model.patch_embed = SimpleNamespace(
        proj=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32))
    )
    embed = torch.nn.Embedding(NUM_GRID_PER_SIDE * NUM_GRID_PER_SIDE, HIDDEN)
    torch.nn.init.normal_(embed.weight, std=1.0)
    model.pos_embed = embed.to(torch.float32)
    return model


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("grid", GRIDS)
def test_from_list_matches_reference_path(align_corners, grid):
    """Both interpolation paths must agree, for either flag value."""
    model = _make_vision_model(align_corners)
    t, h, w = grid

    from_list = model.fast_pos_embed_interpolate_from_list([[t, h, w]])
    reference = model.fast_pos_embed_interpolate(
        torch.tensor([[t, h, w]], dtype=torch.int32)
    )

    assert from_list.shape == reference.shape
    torch.testing.assert_close(from_list, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dim_size", [4, 6, 10, 12])
def test_indices_helpers_agree(dim_size):
    """The torch helper must match the numpy helper it documents itself against."""
    for align_corners in (False, True):
        model = _make_vision_model(align_corners)
        torch_idx = model._torch_interp_indices(dim_size, torch.device("cpu"))
        numpy_idx = torch.from_numpy(model._get_interpolation_indices(dim_size))
        torch.testing.assert_close(torch_idx, numpy_idx, rtol=1e-6, atol=1e-6)


def test_flag_actually_changes_from_list_output():
    """The flag must be observable on the default (from_list) path.

    Before the fix this failed: from_list ignored the flag, so both outputs were
    identical and the flag was a silent no-op on the default code path.
    """
    t, h, w = 1, 4, 6
    model_off = _make_vision_model(align_corners=False)
    model_on = _make_vision_model(align_corners=True)
    # identical weights; the flag is the only difference
    model_on.pos_embed.load_state_dict(model_off.pos_embed.state_dict())

    off = model_off.fast_pos_embed_interpolate_from_list([[t, h, w]])
    on = model_on.fast_pos_embed_interpolate_from_list([[t, h, w]])

    assert not torch.allclose(
        off, on, rtol=1e-5, atol=1e-5
    ), "align_corners had no effect on fast_pos_embed_interpolate_from_list"
