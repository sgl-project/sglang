# SPDX-License-Identifier: Apache-2.0
"""Configurable ref2va reference-image short edge.

The released contract resizes every reference image to a 2048px short edge.
That default is pinned here; the override exists only to trade conditioning
fidelity for the memory those tokens cost, so it must keep every other property
of the shape policy (display ratio, 32px grid, upscaling) intact.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
    MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE,
    minimax_h3_resolve_reference_image_shape,
    minimax_h3_validate_reference_image_short_edge,
)


def test_default_short_edge_is_the_released_contract():
    """Omitting the override must resolve exactly as before."""
    shape = minimax_h3_resolve_reference_image_shape(width=1920, height=1080)
    assert shape["base_short_edge"] == MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE == 2048
    assert min(shape["width"], shape["height"]) == 2048
    assert shape["shape_policy_version"] == "reference_image_short_edge_v1"


def test_explicit_default_matches_the_implicit_one():
    implicit = minimax_h3_resolve_reference_image_shape(width=1280, height=720)
    explicit = minimax_h3_resolve_reference_image_shape(
        width=1280, height=720, short_edge=2048
    )
    assert implicit == explicit


@pytest.mark.parametrize("short_edge", (1024, 1280, 1440, 2048))
def test_short_edge_lands_on_the_requested_tier(short_edge):
    shape = minimax_h3_resolve_reference_image_shape(
        width=1920, height=1080, short_edge=short_edge
    )
    assert shape["base_short_edge"] == short_edge
    assert min(shape["width"], shape["height"]) == short_edge
    assert shape["effective_short_edge"] == short_edge
    assert not shape["width"] % MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
    assert not shape["height"] % MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE


@pytest.mark.parametrize("short_edge", (1024, 1440, 2048))
def test_display_ratio_survives_the_override(short_edge):
    shape = minimax_h3_resolve_reference_image_shape(
        width=1920, height=1080, short_edge=short_edge
    )
    assert shape["width"] / shape["height"] == pytest.approx(1920 / 1080, rel=0.02)


def test_upscaling_still_applies_below_the_tier():
    """A small source is still upscaled -- shrinking inputs client-side is not
    a way to save memory, only the server-side tier is."""
    shape = minimax_h3_resolve_reference_image_shape(
        width=256, height=256, short_edge=1024
    )
    assert (shape["width"], shape["height"]) == (1024, 1024)
    assert shape["allow_upscale"] is True


def test_token_count_scales_with_the_square_of_the_tier():
    """The whole point of the knob: halving the edge quarters the tokens."""

    def dit_tokens(shape):
        # VAE spatial_compression_ratio=16, DiT patch_size=(1, 2, 2)
        return (shape["width"] // 32) * (shape["height"] // 32)

    full = minimax_h3_resolve_reference_image_shape(
        width=1024, height=1024, short_edge=2048
    )
    half = minimax_h3_resolve_reference_image_shape(
        width=1024, height=1024, short_edge=1024
    )
    assert dit_tokens(full) == 4096
    assert dit_tokens(half) == 1024


@pytest.mark.parametrize("bad", (0, -32, 1000, 1023, 2049))
def test_off_grid_and_non_positive_values_are_rejected(bad):
    with pytest.raises(ValueError):
        minimax_h3_validate_reference_image_short_edge(bad)
    with pytest.raises(ValueError):
        minimax_h3_resolve_reference_image_shape(
            width=1920, height=1080, short_edge=bad
        )


@pytest.mark.parametrize("bad", ("1024", None, 3.5))
def test_non_integer_values_are_rejected(bad):
    with pytest.raises(ValueError):
        minimax_h3_validate_reference_image_short_edge(bad)


def test_server_args_validation_rejects_off_grid_values():
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = ServerArgs.__new__(ServerArgs)
    args.minimax_h3_reference_image_short_edge = 1000
    with pytest.raises(ValueError):
        args._validate_minimax_h3_reference_image_short_edge()


def test_server_args_validation_accepts_none_and_valid_tiers():
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = ServerArgs.__new__(ServerArgs)
    args.minimax_h3_reference_image_short_edge = None
    args._validate_minimax_h3_reference_image_short_edge()
    assert args.minimax_h3_reference_image_short_edge is None

    args.minimax_h3_reference_image_short_edge = 1440
    args._validate_minimax_h3_reference_image_short_edge()
    assert args.minimax_h3_reference_image_short_edge == 1440
