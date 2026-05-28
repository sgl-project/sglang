#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Idempotently patch aiter gluon pa_mqa_logits for Triton >= 3.5 MFMA instr_shape.

The base _gluon_deepgemm_fp8_paged_mqa_logits variant hardcoded 2D instr_shape
[16, 16] on older aiter commits. Triton 3.5+ requires 3D [16, 16, 32] when
_Use_2d_instr_shape_mfma_layout is false (GLM-5 NSA / deepgemm path).

Upstream fix: ROCm/aiter a1bdcec (#2575). This hotpatch remains for ROCm images
that still ship an older vendored aiter until images are rebuilt.

Usage:
  python3 patch_aiter_gluon_pa_mqa_logits.py [AITER_REPO_ROOT]
Default AITER_REPO_ROOT: /sgl-workspace/aiter
"""

from __future__ import annotations

import os
import sys

_SENTINEL = "[PATCHED] 3D instr_shape for base gluon variant"

_OLD = """\
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, NumWarps],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )"""

_NEW = """\
    # [PATCHED] 3D instr_shape for base gluon variant
    if _Use_2d_instr_shape_mfma_layout:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
        )
    else:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16, 32],
            transposed=False,
            warps_per_cta=[1, NumWarps],
        )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )"""


def patch_gluon_pa_mqa_logits(aiter_root: str) -> bool:
    target = os.path.join(
        aiter_root, "aiter", "ops", "triton", "gluon", "pa_mqa_logits.py"
    )
    if not os.path.isfile(target):
        print(f"[aiter-hotpatch] {target} not found, skipping")
        return False

    src = open(target, encoding="utf-8").read()
    if _SENTINEL in src:
        print("[aiter-hotpatch] gluon pa_mqa_logits 3D instr_shape already applied")
        return False

    if _OLD not in src:
        print(
            "[aiter-hotpatch] WARN: gluon pa_mqa_logits pattern not found "
            "(aiter may already include ROCm/aiter#2575)"
        )
        return False

    new_src = src.replace(_OLD, _NEW, 1)
    with open(target, "w", encoding="utf-8") as f:
        f.write(new_src)
    print("[aiter-hotpatch] Patched gluon pa_mqa_logits 3D instr_shape (base variant)")
    return True


def main() -> int:
    aiter_root = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/aiter"
    patch_gluon_pa_mqa_logits(aiter_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
