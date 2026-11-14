"""JIT build + load of the four gfx950 DSA-indexer kernels, on the first gated
call -- never in a timed region or a graph capture.

Four modules rather than one because the flags differ per kernel: the logits
kernel wants its own -mllvm set, the qk kernel needs aiter's headers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# The accepted configuration of each kernel.  Changing one of these means
# editing the kernel it belongs to; the sources reject any other value rather
# than mis-launching.
LOGITS_HIST_BITS = 12  # must match logits_hist_m<8, 12> and topk_transform.cuh
# Logits kernel (node 3) variants, chosen per launch by logits_launch():
#   0 logits_hist_m    per row, LOGITS_BLOCKS_PER_ROW blocks per row;
#   1 logits_hist_s    per row, tuned for short batches (rows <= LOGITS_SHORT_MAX_ROWS);
#   2 logits_hist_grp  sibling groups: consecutive rows of one request -- the MTP verify rows -- share its page
#                      table, so each K page is loaded once per group instead of once per row.  The group size is a
#                      template argument of the kernel (HOST_GR in paged_mqa_logits.cuh).
# All three produce bit-identical logits and the same top-k selection.
LOGITS_BLOCKS_PER_ROW = 48
LOGITS_SHORT_BLOCKS_PER_ROW = 32
# Measured, graph timed, 6 rows per request at 120k tokens: rows 12 / 24 run 22 / 40 us on logits_hist_s against
# 26 / 48 us on logits_hist_m; from 36 rows logits_hist_s is the slower one (80 vs 66 us).
LOGITS_SHORT_MAX_ROWS = 24
LOGITS_GROUP_ROWS = 6
# Smallest row count routed to the sibling-group kernel (measured at 120k tokens: 30 / 36 / 48 rows run
# 51 / 54 / 61 us grouped against 55 / 66 / 79 us on logits_hist_m; at 24 rows grouping only wins on long contexts).
LOGITS_GROUP_MIN_ROWS = 30
# topk_transform blocks per row.  The row-barrier and cursor-atomic cost per
# block outweighs the extra fill past ~48 (measured: 48 beats 64 on every case);
# G only repartitions the exact selection, so the top-k set does not change.
# The kernel lowers it further where rows * G blocks would not be co-resident
# for its persistent-rank form.
TOPK_G = 48


def logits_launch(
    rows: int, table_pages: int, rows_per_request: int
) -> Tuple[int, int]:
    """(blocks_per_row, variant) for the logits kernel, in its argument order.

    The sibling-group kernel is used when the rows come ``LOGITS_GROUP_ROWS`` per
    request and there are at least ``LOGITS_GROUP_MIN_ROWS`` of them; it checks on
    the device that the siblings really share their page ids and stays exact if
    they do not.  ``logits_hist_s`` gets ``LOGITS_SHORT_BLOCKS_PER_ROW`` blocks,
    raised so that a wave never walks more than 64 pages (its page ids live one
    per lane) and kept a multiple of 8 so block x of every row lands on XCD x % 8.
    """
    if (
        rows_per_request == LOGITS_GROUP_ROWS
        and rows % LOGITS_GROUP_ROWS == 0
        and rows >= LOGITS_GROUP_MIN_ROWS
    ):
        return 0, 2
    if rows <= LOGITS_SHORT_MAX_ROWS:
        need = -(-table_pages // (8 * 64))
        return max(LOGITS_SHORT_BLOCKS_PER_ROW, -(-need // 8) * 8), 1
    return LOGITS_BLOCKS_PER_ROW, 0


@cache_once
def modules() -> Tuple[Module, Module, Module, Module]:
    """(gemv, qk, logits, topk) modules, or raise. qk goes early: it is the one
    that can fail for a reason outside this tree (a missing or stale aiter)."""
    gemv = load_jit(
        "dsa_gfx950_dual_gemv",
        cuda_files=["dsa_gfx950/dual_gemv_bf16.cuh"],
        cuda_wrappers=[("dual_gemv_bf16", "DualGemvBf16Kernel::run")],
        extra_cuda_cflags=["-O3"],
    )
    # Includes aiter's hip_reduce.h and opus.hpp, so the numeric helpers are the
    # same code aiter's own fused indexer uses. opus.hpp needs the C++20 the JIT
    # defaults already pass.
    qk = load_jit(
        "dsa_gfx950_qk",
        cuda_files=["dsa_gfx950/qk_rope_hadamard_quant.cuh"],
        cuda_wrappers=[
            (
                "indexer_qk_rope_hadamard_quant_and_cache",
                "QkRopeHadamardQuantKernel::run",
            )
        ],
        extra_cuda_cflags=["-O3", "-Wno-unused-result"],
        extra_dependencies=["aiter"],
    )
    logits = load_jit(
        "dsa_gfx950_logits",
        cuda_files=["dsa_gfx950/paged_mqa_logits.cuh"],
        cuda_wrappers=[("logits_hist", "PagedMqaLogitsKernel::run")],
        extra_cuda_cflags=[
            "-O3",
            "-fno-honor-nans",
            "-mllvm",
            "-amdgpu-mfma-vgpr-form",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
            "-Wno-unused-result",
        ],
    )
    # The width is a module argument as well as a -D. The -D already keys the
    # cache; the argument keeps two widths in separate modules, so their exports
    # cannot collide in one process.
    topk = load_jit(
        "dsa_gfx950_topk",
        f"hist_bits{LOGITS_HIST_BITS}",
        cuda_files=["dsa_gfx950/topk_transform.cuh"],
        cuda_wrappers=[
            ("topk_transform", "TopKTransformKernel::run"),
            ("hist_stride", "TopKHistStride::run"),
        ],
        extra_cuda_cflags=["-O3", f"-DDSA_TOPK_HIST_BITS={LOGITS_HIST_BITS}"],
    )
    return gemv, qk, logits, topk


@cache_once
def modules_or_none() -> Optional[Tuple[Module, ...]]:
    """Build once; on failure log and return None. Whether the caller falls back
    or refuses is the gate's decision, so this does not claim either."""
    try:
        return modules()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning("gfx950 fused DSA indexer unavailable, build failed: %s", e)
        return None
