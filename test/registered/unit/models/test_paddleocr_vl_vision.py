"""CPU coverage for the PaddleOCR-VL packed vision-tower fast paths.

The tower encodes a whole (possibly cross-request) batch as one packed
``[total_patches, dim]`` tensor. These tests pin the packed results to the
straightforward per-image reference so the packing stays a pure optimization.
"""

import pytest
import torch
import torch.nn as nn
from einops import rearrange

from sglang.srt.models.paddleocr_vl import (
    Projector,
    SiglipVisionEmbeddings,
    build_packed_2d_position_ids,
    merge_patch_neighbourhoods,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

# Mixes repeated grids (LFU cache hits), an odd aspect ratio, and t > 1.
GRIDS = [(1, 4, 6), (1, 8, 10), (2, 2, 4), (1, 8, 10)]


class _VisionConfig:
    """Minimal stand-in for PaddleOCR-VL's `vision_config`."""

    hidden_size = 32
    image_size = 56
    patch_size = 14
    num_channels = 3


class _TextConfig:
    hidden_size = 48


def _grid_offsets():
    """Start row of each image inside the packed batch."""
    offset = 0
    for t, h, w in GRIDS:
        yield offset
        offset += t * h * w


def _packed_features(dtype=torch.float64) -> torch.Tensor:
    total = sum(t * h * w for t, h, w in GRIDS)
    return torch.randn(total, _VisionConfig.hidden_size, dtype=dtype)


def _reference_projector_output(
    projector: Projector, packed: torch.Tensor
) -> torch.Tensor:
    """Per-image merge + projection, i.e. the pre-packing formulation."""
    m1, m2 = projector.merge_kernel_size
    outputs = []
    offset = 0
    for t, h, w in GRIDS:
        num_patches = t * h * w
        feature = projector.pre_norm(packed[offset : offset + num_patches])
        feature = rearrange(
            feature,
            "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
            t=t,
            h=h // m1,
            p1=m1,
            w=w // m2,
            p2=m2,
        )
        outputs.append(projector.linear_2(projector.act(projector.linear_1(feature))))
        offset += num_patches
    return torch.cat(outputs, dim=0)


def _build_projector() -> Projector:
    torch.manual_seed(0)
    projector = Projector(_TextConfig(), _VisionConfig()).to(torch.float64)
    return projector


# The merge is pure data movement, so it must be bit-exact. The projections that
# follow are not: batching N per-image GEMMs into one changes the blocking, and
# with it the summation order, so the results differ in the last bits (observed
# up to 3e-14 relative in fp64, and it is BLAS-implementation dependent -- equal
# on Apple silicon, unequal on x86). A permutation bug would move values by
# order 1, so this tolerance still catches one decisively.
_GEMM_REORDER_RTOL = 1e-12
_GEMM_REORDER_ATOL = 1e-12


def test_projector_merge_permutation_is_exact():
    """The 2x2 regroup moves data without arithmetic, so it must be bit-exact."""
    torch.manual_seed(1)
    projector = _build_projector()
    packed = _packed_features()
    normed = projector.pre_norm(packed)

    actual = merge_patch_neighbourhoods(normed, GRIDS, projector.merge_kernel_size)

    m1, m2 = projector.merge_kernel_size
    expected = torch.cat(
        [
            rearrange(
                normed[offset : offset + t * h * w],
                "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                t=t,
                h=h // m1,
                p1=m1,
                w=w // m2,
                p2=m2,
            )
            for offset, (t, h, w) in zip(_grid_offsets(), GRIDS)
        ],
        dim=0,
    )

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_projector_packed_merge_matches_per_image_reference():
    torch.manual_seed(1)
    projector = _build_projector()
    packed = _packed_features()

    actual = projector(packed, GRIDS)
    expected = _reference_projector_output(projector, packed)

    assert actual.shape == expected.shape
    assert actual.shape[0] == sum(t * h * w for t, h, w in GRIDS) // 4
    assert actual.shape[1] == _TextConfig.hidden_size
    torch.testing.assert_close(
        actual, expected, rtol=_GEMM_REORDER_RTOL, atol=_GEMM_REORDER_ATOL
    )


def test_projector_is_batch_invariant():
    """Encoding images together must equal encoding them one at a time."""
    torch.manual_seed(2)
    projector = _build_projector()
    packed = _packed_features()

    together = projector(packed, GRIDS)

    apart = []
    offset = 0
    for grid in GRIDS:
        num_patches = grid[0] * grid[1] * grid[2]
        apart.append(projector(packed[offset : offset + num_patches], [grid]))
        offset += num_patches
    apart = torch.cat(apart, dim=0)

    torch.testing.assert_close(
        together, apart, rtol=_GEMM_REORDER_RTOL, atol=_GEMM_REORDER_ATOL
    )


def _build_embeddings() -> SiglipVisionEmbeddings:
    torch.manual_seed(3)
    embeddings = SiglipVisionEmbeddings(_VisionConfig()).to(torch.float64)
    nn.init.normal_(embeddings.position_embedding.weight)
    return embeddings


def _reference_position_embedding_add(
    embeddings: SiglipVisionEmbeddings, patch_embeds: torch.Tensor
) -> torch.Tensor:
    """Uncached interpolation per image, concatenated — the pre-cache formulation."""
    outputs = []
    offset = 0
    for t, h, w in GRIDS:
        num_patches = t * h * w
        image = patch_embeds[offset : offset + num_patches]
        position = embeddings.interpolate_pos_encoding(h, w).squeeze(0).repeat(t, 1)
        outputs.append(image + position)
        offset += num_patches
    return torch.cat(outputs, dim=0)


def test_position_embedding_cache_matches_uncached_interpolation():
    embeddings = _build_embeddings()
    torch.manual_seed(4)
    patch_embeds = torch.randn(
        sum(t * h * w for t, h, w in GRIDS),
        _VisionConfig.hidden_size,
        dtype=torch.float64,
    )
    expected = _reference_position_embedding_add(embeddings, patch_embeds)

    actual = patch_embeds.clone()
    offset = 0
    for t, h, w in GRIDS:
        num_patches = t * h * w
        actual[offset : offset + num_patches].view(
            t, h * w, _VisionConfig.hidden_size
        ).add_(embeddings.fetch_position_embedding_lfu_cache(h, w))
        offset += num_patches

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # (8, 10) appears twice in GRIDS, so it must have been served from the cache.
    assert embeddings.cache_position_count[(8, 10)] == 2
    assert len(embeddings.cache_position_embedding) == 3


def test_position_embedding_cache_evicts_least_frequently_used():
    embeddings = _build_embeddings()

    embeddings.fetch_position_embedding_lfu_cache(4, 4, max_cache=2)
    embeddings.fetch_position_embedding_lfu_cache(4, 4, max_cache=2)
    embeddings.fetch_position_embedding_lfu_cache(6, 6, max_cache=2)
    embeddings.fetch_position_embedding_lfu_cache(8, 8, max_cache=2)

    assert set(embeddings.cache_position_embedding) == {(4, 4), (8, 8)}


def test_patch_embedding_takes_the_matmul_path():
    """kernel == stride and zero padding, so the conv must lower to a matmul."""
    embeddings = _build_embeddings()
    assert embeddings.patch_embedding.enable_linear

    torch.manual_seed(5)
    patch_size = _VisionConfig.patch_size
    pixel_values = torch.randn(7, 3, patch_size, patch_size, dtype=torch.float64)

    actual = embeddings.patch_embedding(pixel_values)
    expected = nn.functional.conv2d(
        pixel_values,
        embeddings.patch_embedding.weight,
        embeddings.patch_embedding.bias,
        stride=(patch_size, patch_size),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    # The tower adds position embeddings in place on this view.
    assert actual.flatten(-2).squeeze(-1).is_contiguous()


def test_packed_2d_position_ids_match_per_image_reference():
    pids, max_grid_size = build_packed_2d_position_ids(GRIDS, torch.device("cpu"))

    expected_hids = []
    expected_wids = []
    for t, h, w in GRIDS:
        image_pids = torch.arange(t * h * w) % (h * w)
        expected_hids.append(image_pids // w)
        expected_wids.append(image_pids % w)
    expected = torch.stack([torch.cat(expected_hids), torch.cat(expected_wids)], dim=-1)

    assert torch.equal(pids, expected)
    # Must match the device-side `pids.max() + 1` it replaces.
    assert max_grid_size == int(expected.max()) + 1


def test_packed_2d_position_ids_single_image_avoids_cat():
    grid = (1, 3, 5)
    pids, max_grid_size = build_packed_2d_position_ids([grid], torch.device("cpu"))

    image_pids = torch.arange(15)
    expected = torch.stack([image_pids // 5, image_pids % 5], dim=-1)

    assert torch.equal(pids, expected)
    assert max_grid_size == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
