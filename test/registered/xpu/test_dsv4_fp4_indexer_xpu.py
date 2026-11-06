"""C4Indexer.compute_q's XPU FP4 path against the real sgl_kernel op.

Kernel numerics are tested in sgl-kernel-xpu; this guards the sglang side of
the contract: the XPU branch must return the same layout as the CUDA
``fused_q_indexer_rope_hadamard_fp4_quant`` (int8 payload, packed int32
scales, float32 fused weights), fed compute_q's real tensors.
"""

import pytest
import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")

DEVICE = "xpu"
ROWS, HEADS, HEAD_DIM, MAX_POS = 3, 4, 128, 16


def test_compute_q_fp4_allocations_accepted_by_kernel():
    from sglang.srt.layers.attention.dsv4.indexer import C4Indexer

    torch.manual_seed(0)
    indexer = object.__new__(C4Indexer)
    indexer.use_fp4_indexer = True
    indexer.n_local_heads = HEADS
    indexer.head_dim = HEAD_DIM
    indexer.weight_scale = 0.25
    # The kernel reads the rope cache as real float32 [MAX_POS, 64].
    indexer.freqs_cis = torch.randn(
        MAX_POS, 32, dtype=torch.complex64, device=DEVICE
    )
    q = torch.randn(
        ROWS, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
    )
    indexer.wq_b = lambda q_lora: (q, None)
    weight = torch.randn(ROWS, HEADS, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, MAX_POS, (ROWS,), device=DEVICE)

    # The real kernel wrapper allocates the outputs and TORCH_CHECKs dtypes;
    # with the earlier uint8 q_fp4 regression this raised "q_fp4 must be int8".
    (q_fp4, q_scale), weights = indexer.compute_q(object(), positions, weight)

    # Callers consume the CUDA kernel's layout: int8 payload and one packed
    # int32 ue8m0 scale per (token, head).
    assert q_fp4.dtype == torch.int8
    assert q_fp4.shape == (ROWS, HEADS, HEAD_DIM // 2)
    assert q_scale.dtype == torch.int32
    assert q_scale.shape == (ROWS, HEADS)
    assert weights.dtype == torch.float32
    assert weights.shape == (ROWS, HEADS, 1)
    # The kernel actually wrote: ue8m0 exponent bytes are never all zero for
    # gaussian inputs, and the fused weights must be finite.
    assert q_scale.view(torch.uint8).count_nonzero() > 0
    assert torch.isfinite(weights).all()
