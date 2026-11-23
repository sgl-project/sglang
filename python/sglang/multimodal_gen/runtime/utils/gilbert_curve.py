# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2018 Jakub Červený
# Copyright (c) 2024 abetusk
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Generalized Hilbert ("Gilbert") space-filling curve over a 3D cuboid.

Consecutive curve steps are neighbouring cells, and any run of steps covers a
compact region, for every cuboid shape (the sides need not be powers of two).
"""

import functools

import torch


def _sgn(x: int) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def _generate3d(
    out: list[int],
    x: int,
    y: int,
    z: int,
    ax: int,
    ay: int,
    az: int,
    bx: int,
    by: int,
    bz: int,
    cx: int,
    cy: int,
    cz: int,
) -> None:
    """Append flattened (x, y, z) triples in curve order to out."""
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    dax, day, daz = _sgn(ax), _sgn(ay), _sgn(az)
    dbx, dby, dbz = _sgn(bx), _sgn(by), _sgn(bz)
    dcx, dcy, dcz = _sgn(cx), _sgn(cy), _sgn(cz)

    if h == 1 and d == 1:
        for _ in range(w):
            out.extend((x, y, z))
            x, y, z = x + dax, y + day, z + daz
        return
    if w == 1 and d == 1:
        for _ in range(h):
            out.extend((x, y, z))
            x, y, z = x + dbx, y + dby, z + dbz
        return
    if w == 1 and h == 1:
        for _ in range(d):
            out.extend((x, y, z))
            x, y, z = x + dcx, y + dcy, z + dcz
        return

    ax2, ay2, az2 = ax // 2, ay // 2, az // 2
    bx2, by2, bz2 = bx // 2, by // 2, bz // 2
    cx2, cy2, cz2 = cx // 2, cy // 2, cz // 2

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # Keep the halves even so each sub-curve can end next to the following one.
    if (w2 % 2) and (w > 2):
        ax2, ay2, az2 = ax2 + dax, ay2 + day, az2 + daz
    if (h2 % 2) and (h > 2):
        bx2, by2, bz2 = bx2 + dbx, by2 + dby, bz2 + dbz
    if (d2 % 2) and (d > 2):
        cx2, cy2, cz2 = cx2 + dcx, cy2 + dcy, cz2 + dcz

    if (2 * w > 3 * h) and (2 * w > 3 * d):
        _generate3d(out, x, y, z, ax2, ay2, az2, bx, by, bz, cx, cy, cz)
        _generate3d(
            out,
            x + ax2,
            y + ay2,
            z + az2,
            ax - ax2,
            ay - ay2,
            az - az2,
            bx,
            by,
            bz,
            cx,
            cy,
            cz,
        )
        return

    if 3 * h > 4 * d:
        _generate3d(out, x, y, z, bx2, by2, bz2, cx, cy, cz, ax2, ay2, az2)
        _generate3d(
            out,
            x + bx2,
            y + by2,
            z + bz2,
            ax,
            ay,
            az,
            bx - bx2,
            by - by2,
            bz - bz2,
            cx,
            cy,
            cz,
        )
        _generate3d(
            out,
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2,
            -by2,
            -bz2,
            cx,
            cy,
            cz,
            -(ax - ax2),
            -(ay - ay2),
            -(az - az2),
        )
        return

    if 3 * d > 4 * h:
        _generate3d(out, x, y, z, cx2, cy2, cz2, ax2, ay2, az2, bx, by, bz)
        _generate3d(
            out,
            x + cx2,
            y + cy2,
            z + cz2,
            ax,
            ay,
            az,
            bx,
            by,
            bz,
            cx - cx2,
            cy - cy2,
            cz - cz2,
        )
        _generate3d(
            out,
            x + (ax - dax) + (cx2 - dcx),
            y + (ay - day) + (cy2 - dcy),
            z + (az - daz) + (cz2 - dcz),
            -cx2,
            -cy2,
            -cz2,
            -(ax - ax2),
            -(ay - ay2),
            -(az - az2),
            bx,
            by,
            bz,
        )
        return

    _generate3d(out, x, y, z, bx2, by2, bz2, cx2, cy2, cz2, ax2, ay2, az2)
    _generate3d(
        out,
        x + bx2,
        y + by2,
        z + bz2,
        cx,
        cy,
        cz,
        ax2,
        ay2,
        az2,
        bx - bx2,
        by - by2,
        bz - bz2,
    )
    _generate3d(
        out,
        x + (bx2 - dbx) + (cx - dcx),
        y + (by2 - dby) + (cy - dcy),
        z + (bz2 - dbz) + (cz - dcz),
        ax,
        ay,
        az,
        -bx2,
        -by2,
        -bz2,
        -(cx - cx2),
        -(cy - cy2),
        -(cz - cz2),
    )
    _generate3d(
        out,
        x + (ax - dax) + bx2 + (cx - dcx),
        y + (ay - day) + by2 + (cy - dcy),
        z + (az - daz) + bz2 + (cz - dcz),
        -cx,
        -cy,
        -cz,
        -(ax - ax2),
        -(ay - ay2),
        -(az - az2),
        bx - bx2,
        by - by2,
        bz - bz2,
    )
    _generate3d(
        out,
        x + (ax - dax) + (bx2 - dbx),
        y + (ay - day) + (by2 - dby),
        z + (az - daz) + (bz2 - dbz),
        -bx2,
        -by2,
        -bz2,
        cx2,
        cy2,
        cz2,
        -(ax - ax2),
        -(ay - ay2),
        -(az - az2),
    )


@functools.lru_cache(maxsize=16)
def _curve_cells(depth: int, height: int, width: int) -> tuple[int, ...]:
    """Raster cell index (z * height * width + y * width + x) of each curve step."""
    out: list[int] = []
    if width >= height and width >= depth:
        _generate3d(out, 0, 0, 0, width, 0, 0, 0, height, 0, 0, 0, depth)
    elif height >= width and height >= depth:
        _generate3d(out, 0, 0, 0, 0, height, 0, width, 0, 0, 0, 0, depth)
    else:
        _generate3d(out, 0, 0, 0, 0, 0, depth, width, 0, 0, 0, height, 0)
    return tuple(
        (out[i + 2] * height + out[i + 1]) * width + out[i]
        for i in range(0, len(out), 3)
    )


def gilbert_curve_step_of_cell(depth: int, height: int, width: int) -> torch.Tensor:
    """(depth * height * width,) int64 on CPU: the curve step of each raster cell."""
    cells = torch.tensor(_curve_cells(depth, height, width), dtype=torch.long)
    steps = torch.empty_like(cells)
    steps[cells] = torch.arange(cells.numel(), dtype=torch.long)
    return steps
