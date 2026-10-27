"""Optimized LiteTopK fused indexer top-k for DSA prefill (SM100).

One persistent kernel scores every visible key of each query row with the DSA
indexer (GLM-5.2's geometry: 32 index heads x 128, FP8 index K) and selects the
row's top-2048 without materializing the ``[rows, keys]`` logits: during the
FP8 scan a per-row gate set from a sampled quantile of the row's scores
(LiteTopK) keeps the candidates, selection runs in-kernel, and blocks whose
candidate count falls outside the exact range are rescanned densely, so the
result is the exact top-k up to ties at the k-th score. The kernel is LiteTopK
(arXiv:2607.11976) as optimized in vLLM PR #60618, with further changes; see
``kernels/jit/csrc/dsa/fused_mqa_topk/README.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    override_jit_cuda_arch,
)
from sglang.kernels.jit.utils.arch import _jit_cuda_version

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Gates and dispatch are tuned and validated on B200 only.
SUPPORTED_CUDA_ARCHS = ((10, 0),)
FUSED_MQA_TOPK_NUM_HEADS = 32
FUSED_MQA_TOPK_HEAD_DIM = 128
# Mirror csrc/dsa/fused_mqa_topk/constants.h (the host entry checks them).
# The only top-k the kernel is compiled for (kTopK).
FUSED_MQA_TOPK_TOPK = 2048
# Fewest keys (rows of k_fp8) the kernel takes (kMinKeys): its gates are tuned
# above 20480 keys.
FUSED_MQA_TOPK_MIN_KEYS = 20481
FUSED_MQA_TOPK_MAX_KEYS = 1 << 20
# rows * 32 is the int32 outer extent of the kernel's Q tensor map.
FUSED_MQA_TOPK_MAX_ROWS = (2**31 - 1) // 32

# Workspace layout (constants.h): one persistent CTA per SM owns two 8-row
# slots, each row a 16384-candidate slab and a 4096-entry rescan slab.
_SLOTS_PER_SM = 2
_CANDIDATES_PER_ROW = 16384
_RETAINED_PER_ROW = 4096


class _Workspace(NamedTuple):
    counter: torch.Tensor  # [2] int32, zero between launches
    values: torch.Tensor  # [8 * slots, 16384] fp16
    indices: torch.Tensor  # [8 * slots, 16384] int32
    retained: torch.Tensor  # [8 * slots, 4096] fp32
    retained_indices: torch.Tensor  # [8 * slots, 4096] int32
    counts: torch.Tensor  # [rows rounded up to 8] int32
    defer: torch.Tensor  # [rows / 8] int32


# One workspace per (device, stream): a launch owns the counter and the slots
# until it retires, and only launches on the same stream are ordered.
_WORKSPACES: Dict[Tuple[int, int], _Workspace] = {}
# Per-row tensors a larger call replaced, never freed:
# graphs captured before that call still launch with them.
_RETIRED_PER_ROW: List[torch.Tensor] = []


@cache_once
def is_fused_mqa_topk_supported() -> bool:
    """Whether the current CUDA device and the JIT's nvcc support the fused kernel."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    return (
        torch.cuda.get_device_capability() in SUPPORTED_CUDA_ARCHS
        # The module is built for sm_100f, which needs a CUDA >= 12.9 nvcc.
        and _jit_cuda_version() >= (12, 9)
    )


@cache_once
def _jit_fused_mqa_topk_module() -> Module:
    if not is_fused_mqa_topk_supported():
        raise RuntimeError(
            "fused_mqa_topk_prefill requires CUDA SM100 with nvcc >= 12.9"
        )
    # sm_100f, not the default sm_100a; see the README.
    with override_jit_cuda_arch(10, 0, suffix="f"):
        return load_jit(
            "fused_mqa_topk",
            cuda_files=["dsa/fused_mqa_topk.cuh"],
            cuda_wrappers=[("fused_mqa_topk_prefill", "fused_mqa_topk_prefill")],
            # CUTLASS, and <deep_gemm/...> from the installed sgl-deep-gemm.
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=["--expt-extended-lambda"],
        )


def can_use_fused_mqa_topk_prefill(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
) -> bool:
    """Whether :func:`fused_mqa_topk_prefill` covers these operands."""
    return (
        topk == FUSED_MQA_TOPK_TOPK
        and q_fp8.dtype == torch.float8_e4m3fn
        and k_fp8.dtype == torch.float8_e4m3fn
        and k_scale.dtype == torch.float32
        and weights.dtype == torch.float32
        and q_fp8.dim() == 3
        and q_fp8.shape[1] == FUSED_MQA_TOPK_NUM_HEADS
        and q_fp8.shape[2] == FUSED_MQA_TOPK_HEAD_DIM
        and k_fp8.dim() == 2
        and k_fp8.shape[1] == FUSED_MQA_TOPK_HEAD_DIM
        and 0 < q_fp8.shape[0] <= FUSED_MQA_TOPK_MAX_ROWS
        and FUSED_MQA_TOPK_MIN_KEYS <= k_fp8.shape[0] <= FUSED_MQA_TOPK_MAX_KEYS
        and k_scale.shape == (k_fp8.shape[0],)
        and weights.shape == (q_fp8.shape[0], FUSED_MQA_TOPK_NUM_HEADS)
        # TMA descriptors need 16-byte aligned contiguous operands.
        and all(
            t.is_contiguous() and t.data_ptr() % 16 == 0
            for t in (q_fp8, k_fp8, k_scale, weights)
        )
        and is_fused_mqa_topk_supported()
    )


def _get_workspace(device: torch.device, rows: int) -> _Workspace:
    rows = (rows + 7) // 8 * 8
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    workspace = _WORKSPACES.get(key)
    if workspace is not None and workspace.counts.numel() >= rows:
        return workspace
    if torch.cuda.is_current_stream_capturing():
        # Allocated under capture, the counter would only be zeroed at replay.
        raise RuntimeError(
            "fused_mqa_topk_prefill allocates its workspace on the first call per "
            "stream; call it eagerly on the capture stream with at least as many "
            "rows before capturing."
        )
    if workspace is None:
        slot_rows = (
            8
            * _SLOTS_PER_SM
            * torch.cuda.get_device_properties(device).multi_processor_count
        )
        workspace = _Workspace(
            counter=torch.zeros(2, dtype=torch.int32, device=device),
            values=torch.empty(
                (slot_rows, _CANDIDATES_PER_ROW), dtype=torch.float16, device=device
            ),
            indices=torch.empty(
                (slot_rows, _CANDIDATES_PER_ROW), dtype=torch.int32, device=device
            ),
            retained=torch.empty(
                (slot_rows, _RETAINED_PER_ROW), dtype=torch.float32, device=device
            ),
            retained_indices=torch.empty(
                (slot_rows, _RETAINED_PER_ROW), dtype=torch.int32, device=device
            ),
            counts=torch.empty(rows, dtype=torch.int32, device=device),
            defer=torch.empty(rows // 8, dtype=torch.int32, device=device),
        )
    else:
        # At least double, so the retired tensors together stay below the live ones.
        rows = max(rows, 2 * workspace.counts.numel())
        _RETIRED_PER_ROW.extend((workspace.counts, workspace.defer))
        workspace = workspace._replace(
            counts=torch.empty(rows, dtype=torch.int32, device=device),
            defer=torch.empty(rows // 8, dtype=torch.int32, device=device),
        )
    _WORKSPACES[key] = workspace
    return workspace


def fused_mqa_topk_prefill(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    *,
    out_offsets: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    row_batch: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused DSA indexer scores + per-row top-k without a logits buffer.

    Row ``r`` scores keys ``j`` in ``[starts[r], ends[r])`` as
    ``k_scale[j] * sum_h weights[r, h] * relu(q_fp8[r, h] . k_fp8[j])`` and
    returns its ``topk`` (2048) best keys as ``j - starts[r] + out_offsets[r]``
    (``out_offsets`` defaults to 0), unordered and ``-1`` padded when fewer keys
    are visible. Ties at the k-th score are broken nondeterministically. With
    ``page_table`` each selected key is returned instead as its KV slot
    ``page_table[row_batch[r], j - starts[r]]``, the DSA PAGED transform.

    Operands: ``q_fp8`` ``[rows, 32, 128]`` and ``k_fp8`` ``[keys, 128]``
    float8_e4m3fn with ``FUSED_MQA_TOPK_MIN_KEYS`` to ``FUSED_MQA_TOPK_MAX_KEYS``
    keys, ``k_scale`` ``[keys]`` and ``weights`` ``[rows, 32]`` float32,
    ``starts`` / ``ends`` / ``out_offsets`` / ``row_batch`` ``[rows]`` int32,
    ``page_table`` ``[requests, width]`` int32 with unit column stride and
    ``width`` at least the keys any row of the request sees, ``out``
    ``[rows, topk]`` int32; check coverage with
    :func:`can_use_fused_mqa_topk_prefill`. ``page_table`` and ``row_batch``
    come together and exclude ``out_offsets``. A persistent workspace of about
    300 MiB is allocated per (device, stream) on first use, its per-row part
    regrown only for a call with more rows than any before (graphs captured
    earlier keep the old one); a call can be captured in a CUDA graph once that
    stream's workspace holds its rows.
    """
    rows = q_fp8.shape[0]
    if out is None:
        out = torch.empty((rows, topk), dtype=torch.int32, device=q_fp8.device)
    workspace = _get_workspace(q_fp8.device, rows)
    _jit_fused_mqa_topk_module().fused_mqa_topk_prefill(
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        starts,
        ends,
        out_offsets,
        page_table,
        row_batch,
        out,
        *workspace,
    )
    return out
