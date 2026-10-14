# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/moe_plan.py
"""Which MoE ug flags (UGF) and MID rows the ug stage writes and the down stage reads, defined once."""

from sglang.kernels.ops.mono.common_hip.plan import cdiv, imin

TILE = 16  # picks a pick tile: an MFMA's N


def pick_tiles(count):
    """A slot's pick tiles: its ``count`` picks, ``TILE`` a tile."""
    return cdiv(count, TILE)


def ug_parts(groups, ug_groups):
    """Return a unit's ug parts; a part wholly in the padding is not run."""
    return cdiv(groups, ug_groups)


def ug_flag(unit, part, ug_parts):
    """The UGF pair of ug task (``unit``, ``part``), ``ug_parts`` parts a unit."""
    return unit * ug_parts + part


def ug_tasks(units, ug_parts):
    """A step's ug tasks: every part of each of its ``units``."""
    return units * ug_parts


def tile_pick0(slot_first, tile, tile_first):
    """Return pick tile ``tile``'s first pick."""
    return slot_first + (tile - tile_first) * TILE


def polled_tile(tile_first, tiles, j):
    """Return the pick tile a down round polls as a slot's ``j``-th."""
    return tile_first + imin(j, tiles - 1)
