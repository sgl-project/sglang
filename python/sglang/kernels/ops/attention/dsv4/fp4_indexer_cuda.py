"""SM90 CUDA scorer for the paged FP4 candidate indexer.

This entry point implements the H32/D128 ABI without a slots-map allocation.
It uses natural D128 BF16 MMA operands. It preserves
scale FTZ and BF16 rounding sites but changes K accumulation order relative to
the pinned parity-split Triton 3.5.1 reference at commit 349c1b11c7. Numerical
and top-k acceptance require independent oracle gates; bitwise parity is not
promised.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_fp4_candidate_module():
    return load_jit(
        "dpsk_v4_fp4_index_logits_candidates_cuda",
        cuda_files=["deepseek_v4/fp4_indexer_candidates.cuh"],
        cuda_wrappers=[("run", "FP4IndexerCandidatesKernel::run")],
        extra_dependencies=["cutlass"],
    )


def is_fp4_index_logits_candidates_cuda_supported(
    q: torch.Tensor,
    weights: torch.Tensor,
    req: torch.Tensor,
    req_table: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    max_len: int,
    ratio: int,
    blocks: torch.Tensor,
) -> bool:
    """Check host metadata only, including layouts normalized by the wrapper.

    Arbitrary positive page sizes, padded cache/request rows, strided inputs
    and unaligned storage offsets retain the support of the original kernel.
    Request IDs and mapped slots must be valid, as for the Triton scorer.
    """
    if not q.is_cuda or torch.version.hip is not None:
        return False
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[1:] != (32, 128):
        return False
    batch = q.shape[0]
    if weights.shape != (batch, 32) or req.shape != (batch,) or lens.shape != (batch,):
        return False
    if table.dtype != torch.uint8 or table.ndim != 2 or table.stride(1) != 1:
        return False
    if not 0 < page_size <= 0x7FFFFFFF or table.shape[1] != page_size * 68:
        return False
    if req_table.ndim != 2 or req_table.stride(1) != 1:
        return False
    if ratio not in (1, 2) or not 0 <= max_len <= req_table.shape[1] // ratio:
        return False
    if blocks.ndim != 2 or blocks.shape[0] != batch:
        return False
    if any(
        t.dtype not in (torch.int32, torch.int64)
        for t in (req, req_table, lens, blocks)
    ):
        return False
    if any(
        t.device != q.device for t in (weights, req, req_table, lens, table, blocks)
    ):
        return False
    return torch.cuda.get_device_capability(q.device) == (9, 0)


def fp4_index_logits_candidates_cuda(
    q: torch.Tensor,
    weights: torch.Tensor,
    req: torch.Tensor,
    req_table: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    max_len: int,
    ratio: int,
    blocks: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 ``[rows, blocks_per_row * 8]`` scores in candidate order.

    Q has 32 BF16 heads of dimension 128. K stays in its paged E2M1/UE8M0
    representation until this kernel decodes it. Invalid and invisible
    candidates produce ``-inf``. Inputs use the same normalization as the
    Triton scorer; cache and request-table row strides are preserved.
    """
    assert is_fp4_index_logits_candidates_cuda_supported(
        q, weights, req, req_table, lens, table, page_size, max_len, ratio, blocks
    ), "unsupported SM90 FP4 candidate scorer inputs"
    batch = q.shape[0]
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    req, lens, blocks = req.contiguous(), lens.contiguous(), blocks.contiguous()
    width = blocks.shape[1] * 8
    out = torch.empty((batch, width), device=q.device, dtype=torch.float32)
    if batch and width:
        _jit_fp4_candidate_module().run(
            q,
            weights,
            req,
            req_table,
            lens,
            table,
            blocks,
            out,
            page_size,
            max_len,
            ratio,
        )
    return out
