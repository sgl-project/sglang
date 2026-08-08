"""The chunked-SGMV kernel cache must key on every codegen constexpr.

Bug regression (found 2026-07-27 by the MoE-LoRA campaign's admission
gates): ``_chunked_lora_shrink_kernel`` is wrapped in
``@cached_triton_kernel`` keyed on ``(K, NUM_SLICES, BLOCK_M)`` while
``N``/``BLOCK_N``/``BLOCK_K`` are ``tl.constexpr`` parameters baked into
the compiled binary.  On a key hit the wrapper re-launches the CACHED
binary with the new arguments, so a process that first ran rank 16
(N=32) and then rank 64 (N=128) with the same (K, NUM_SLICES, BLOCK_M)
reused the narrow kernel and left output columns 32..127 uninitialized —
silently wrong results, NaN only by allocator luck.  Production is
shielded today because a server runs one fixed N, which is exactly why
this must be pinned: nothing else fails when the key is wrong.

The expand sibling has the same shape (``MAX_RANK`` and its block sizes
are constexprs absent from its key), so both keys are exercised through
the shrink entry point at two ranks in one process, second-call output
checked column-complete against a torch reference.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

from sglang.kernels.ops.gemm.chunked_sgmv_expand import (
    chunked_sgmv_lora_expand_forward,
)
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
    chunked_sgmv_lora_shrink_forward,
)
from sglang.srt.lora.utils import LoRABatchInfo

NUM_TOKENS = 64
HIDDEN = 2048
NUM_SLICES = 2
CHUNK = 16


def _batch_info(rank: int, device: torch.device) -> LoRABatchInfo:
    """All tokens on adapter 0, adapter runs chunked like the backend."""
    seg_indptr = list(range(0, NUM_TOKENS + 1, CHUNK))
    num_segments = len(seg_indptr) - 1
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=num_segments,
        num_segments=num_segments,
        seg_indptr=torch.tensor(seg_indptr, dtype=torch.int32, device=device),
        weight_indices=torch.zeros(num_segments, dtype=torch.int32, device=device),
        lora_ranks=torch.full((1,), rank, dtype=torch.int32, device=device),
        scalings=torch.ones(1, dtype=torch.float, device=device),
        max_len=CHUNK,
        seg_lens=None,
        permutation=torch.arange(NUM_TOKENS, dtype=torch.int32, device=device),
    )


def _reference(x: torch.Tensor, weights: torch.Tensor, rank: int) -> torch.Tensor:
    columns = min(weights.shape[1], rank * NUM_SLICES)
    out = torch.zeros(x.shape[0], weights.shape[1], dtype=x.dtype, device=x.device)
    out[:, :columns] = (x.float() @ weights[0, :columns].float().T).to(x.dtype)
    return out


class TestChunkedSgmvCacheKey(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def test_two_ranks_in_one_process_are_both_column_complete(self):
        torch.manual_seed(41)
        device = self.device
        x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        for rank in (16, 64):
            weights = (
                torch.randn(
                    1,
                    NUM_SLICES * rank,
                    HIDDEN,
                    dtype=torch.float32,
                    device=device,
                )
                * HIDDEN**-0.5
            ).to(torch.bfloat16)
            got = chunked_sgmv_lora_shrink_forward(
                x, weights, _batch_info(rank, device), NUM_SLICES
            )
            want = _reference(x, weights, rank)
            torch.testing.assert_close(
                got.float(),
                want.float(),
                rtol=3e-2,
                atol=3e-2,
                msg=f"rank {rank}: output not column-complete — a stale "
                "cached kernel compiled for a different N wrote fewer "
                "columns than this call's constexpr N",
            )

    def test_two_dtypes_in_one_process_do_not_share_a_binary(self):
        """Sixth S3 review: pointer dtypes specialize the binary
        (``x.dtype.element_ty`` casts), so a BF16-first-then-FP16 call at
        identical dimensions must not replay the BF16 compilation."""
        torch.manual_seed(43)
        device = self.device
        rank = 16
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(NUM_TOKENS, HIDDEN, dtype=dtype, device=device)
            weights = (
                torch.randn(
                    1, NUM_SLICES * rank, HIDDEN, dtype=torch.float32, device=device
                )
                * HIDDEN**-0.5
            ).to(dtype)
            got = chunked_sgmv_lora_shrink_forward(
                x, weights, _batch_info(rank, device), NUM_SLICES
            )
            want = _reference(x, weights, rank)
            torch.testing.assert_close(
                got.float(),
                want.float(),
                rtol=3e-2,
                atol=3e-2,
                msg=f"{dtype}: wrong values — a cached binary compiled for "
                "a different pointer dtype was replayed",
            )

    def test_expand_two_ranks_in_one_process_are_both_column_complete(self):
        """The expand sibling's own key (MAX_RANK feeds constexpr strides):
        rank 16 then rank 64 in one process, output checked against a torch
        reference — previously claimed by the module docstring but only the
        shrink entry point was exercised (sixth S3 review)."""
        torch.manual_seed(44)
        device = self.device
        output_dim = 256
        slice_offsets = torch.tensor(
            [0, output_dim // 2, output_dim], dtype=torch.int32, device=device
        )
        for rank in (16, 64):
            x = (
                torch.randn(
                    NUM_TOKENS, NUM_SLICES * rank, dtype=torch.float32, device=device
                )
                * rank**-0.5
            ).to(torch.bfloat16)
            weights = (
                torch.randn(1, output_dim, rank, dtype=torch.float32, device=device)
                * rank**-0.5
            ).to(torch.bfloat16)
            got = chunked_sgmv_lora_expand_forward(
                x,
                weights,
                _batch_info(rank, device),
                slice_offsets,
                output_dim // 2,
                None,
            )
            want = torch.zeros(
                NUM_TOKENS, output_dim, dtype=torch.float32, device=device
            )
            for s in range(NUM_SLICES):
                lo, hi = int(slice_offsets[s]), int(slice_offsets[s + 1])
                want[:, lo:hi] = (
                    x[:, s * rank : (s + 1) * rank].float()
                    @ weights[0, lo:hi].float().T
                )
            torch.testing.assert_close(
                got.float(),
                want,
                rtol=3e-2,
                atol=3e-2,
                msg=f"expand rank {rank}: output wrong — a cached binary "
                "compiled for a different MAX_RANK was replayed",
            )


if __name__ == "__main__":
    unittest.main()
