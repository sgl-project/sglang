# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Kernel-level correctness test: paged LoRA kernels vs flat LoRA kernels.

Verifies that chunked_sgmv_lora_{shrink,expand}_forward_paged produces
tensor outputs matching chunked_sgmv_lora_{shrink,expand}_forward (flat)
to within floating-point tolerance.

The test creates identical LoRA weights in both flat and paged formats,
runs both kernel paths on the same input, and compares outputs with
torch.allclose.

Usage:
    python -m pytest test/registered/lora/test_paged_kernel_correctness.py -v
"""


def _patch_kernels_revision():
    """Patch kernels LayerRepository to default revision='main'."""
    try:
        from kernels.layer.func import FuncRepository as _FR
        from kernels.layer.layer import LayerRepository as _LR

        _lr_orig = _LR.__init__

        def _lr_patched(
            self, repo_id, *, layer_name, revision=None, version=None, **kw
        ):
            if revision is None and version is None:
                revision = "main"
            _lr_orig(
                self,
                repo_id,
                layer_name=layer_name,
                revision=revision,
                version=version,
                **kw,
            )

        _LR.__init__ = _lr_patched

        _fr_orig = _FR.__init__

        def _fr_patched(self, repo_id, *, func_name, revision=None, version=None, **kw):
            if revision is None and version is None:
                revision = "main"
            _fr_orig(
                self,
                repo_id,
                func_name=func_name,
                revision=revision,
                version=version,
                **kw,
            )

        _FR.__init__ = _fr_patched
    except ImportError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"patch_kernels failed: {e}")
        pass


_patch_kernels_revision()

from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.gemm.chunked_sgmv_expand import chunked_sgmv_lora_expand_forward
from sglang.kernels.ops.gemm.chunked_sgmv_expand_paged import (
    chunked_sgmv_lora_expand_forward_paged,
)
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import chunked_sgmv_lora_shrink_forward
from sglang.kernels.ops.gemm.chunked_sgmv_shrink_paged import (
    chunked_sgmv_lora_shrink_forward_paged,
)

PAGE_RANK_SIZE = 8
INPUT_DIM = 128
OUTPUT_DIM = 64
RANK = 8
NUM_SLICES = 1
S = 16  # number of tokens


def _make_batch_info(device, num_adapters=1):
    """Create a minimal LoRABatchInfo for 1 segment per adapter."""
    seg_indptr = torch.tensor([0, S], dtype=torch.int32, device=device)
    weight_indices = torch.tensor([0], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([RANK], dtype=torch.int32, device=device)
    scalings = torch.tensor([1.0], dtype=torch.float32, device=device)
    permutation = torch.arange(S, dtype=torch.int32, device=device)
    return LoRABatchInfo(
        bs=1,
        num_segments=1,
        max_len=S,
        use_cuda_graph=False,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        permutation=permutation,
        seg_lens=None,
    )


class TestPagedShrinkMatchesFlat(CustomTestCase):
    """Paged shrink kernel output must match flat shrink kernel output."""

    def test_shrink_single_adapter(self):
        device = "cuda"
        torch.manual_seed(42)

        x = torch.randn(S, INPUT_DIM, device=device, dtype=torch.float32)

        flat_A = torch.randn(
            1, NUM_SLICES * RANK, INPUT_DIM, device=device, dtype=torch.float32
        )
        paged_A = flat_A.clone()
        page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        max_pages = 1

        bi = _make_batch_info(device)

        flat_out = chunked_sgmv_lora_shrink_forward(
            x=x, weights=flat_A, batch_info=bi, num_slices=NUM_SLICES
        )
        paged_out = chunked_sgmv_lora_shrink_forward_paged(
            x=x,
            A_pages=paged_A,
            batch_info=bi,
            num_slices=NUM_SLICES,
            page_table=page_table,
            max_pages_per_lora=max_pages,
            page_rank_size=PAGE_RANK_SIZE,
        )

        self.assertEqual(flat_out.shape, paged_out.shape)
        self.assertTrue(
            torch.allclose(flat_out, paged_out, atol=1e-5, rtol=1e-5),
            f"Shrink mismatch: max diff={ (flat_out - paged_out).abs().max().item()}",
        )

    def test_shrink_multiple_adapters(self):
        device = "cuda"
        torch.manual_seed(123)

        num_adapters = 3
        S_per = 8
        S_total = S_per * num_adapters

        x = torch.randn(S_total, INPUT_DIM, device=device, dtype=torch.float32)
        flat_A = torch.randn(
            num_adapters,
            NUM_SLICES * RANK,
            INPUT_DIM,
            device=device,
            dtype=torch.float32,
        )
        paged_A = flat_A.clone()

        page_table = torch.tensor(
            [[i] for i in range(num_adapters)], dtype=torch.int32, device=device
        )
        max_pages = 1

        seg_indptr = torch.tensor(
            [0] + [S_per * (i + 1) for i in range(num_adapters)],
            dtype=torch.int32,
            device=device,
        )
        weight_indices = torch.tensor(
            list(range(num_adapters)), dtype=torch.int32, device=device
        )
        lora_ranks = torch.tensor(
            [RANK] * num_adapters, dtype=torch.int32, device=device
        )
        scalings = torch.tensor(
            [1.0] * num_adapters, dtype=torch.float32, device=device
        )
        permutation = torch.arange(S_total, dtype=torch.int32, device=device)

        bi = LoRABatchInfo(
            bs=num_adapters,
            num_segments=num_adapters,
            max_len=S_per,
            use_cuda_graph=False,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
            seg_lens=None,
        )

        flat_out = chunked_sgmv_lora_shrink_forward(
            x=x, weights=flat_A, batch_info=bi, num_slices=NUM_SLICES
        )
        paged_out = chunked_sgmv_lora_shrink_forward_paged(
            x=x,
            A_pages=paged_A,
            batch_info=bi,
            num_slices=NUM_SLICES,
            page_table=page_table,
            max_pages_per_lora=max_pages,
            page_rank_size=PAGE_RANK_SIZE,
        )

        self.assertEqual(flat_out.shape, paged_out.shape)
        self.assertTrue(
            torch.allclose(flat_out, paged_out, atol=1e-5, rtol=1e-5),
            f"Shrink multi-adapter mismatch: max diff={ (flat_out - paged_out).abs().max().item()}",
        )


class TestPagedExpandMatchesFlat(CustomTestCase):
    """Paged expand kernel output must match flat expand kernel output."""

    def test_expand_single_adapter(self):
        device = "cuda"
        torch.manual_seed(42)

        x = torch.randn(S, NUM_SLICES * RANK, device=device, dtype=torch.float32)

        flat_B = torch.randn(1, OUTPUT_DIM, RANK, device=device, dtype=torch.float32)
        paged_B = flat_B.clone()
        page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        max_pages = 1
        slice_offsets = torch.tensor([0, OUTPUT_DIM], dtype=torch.int32, device=device)

        bi = _make_batch_info(device)

        flat_out = chunked_sgmv_lora_expand_forward(
            x=x,
            weights=flat_B,
            batch_info=bi,
            slice_offsets=slice_offsets,
            max_slice_size=OUTPUT_DIM,
            base_output=None,
        )
        paged_out = chunked_sgmv_lora_expand_forward_paged(
            x=x,
            B_pages=paged_B,
            batch_info=bi,
            slice_offsets=slice_offsets,
            max_slice_size=OUTPUT_DIM,
            base_output=None,
            page_table=page_table,
            max_pages_per_lora=max_pages,
            page_rank_size=PAGE_RANK_SIZE,
        )

        self.assertEqual(flat_out.shape, paged_out.shape)
        self.assertTrue(
            torch.allclose(flat_out, paged_out, atol=1e-5, rtol=1e-5),
            f"Expand mismatch: max diff={ (flat_out - paged_out).abs().max().item()}",
        )


class TestPagedShrinkExpandChain(CustomTestCase):
    """Full shrink→expand chain: paged output must match flat output."""

    def test_shrink_expand_chain(self):
        device = "cuda"
        torch.manual_seed(42)

        x = torch.randn(S, INPUT_DIM, device=device, dtype=torch.float32)

        flat_A = torch.randn(
            1, NUM_SLICES * RANK, INPUT_DIM, device=device, dtype=torch.float32
        )
        flat_B = torch.randn(1, OUTPUT_DIM, RANK, device=device, dtype=torch.float32)
        paged_A = flat_A.clone()
        paged_B = flat_B.clone()
        page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        max_pages = 1
        slice_offsets = torch.tensor([0, OUTPUT_DIM], dtype=torch.int32, device=device)

        bi = _make_batch_info(device)

        flat_shrink = chunked_sgmv_lora_shrink_forward(
            x=x, weights=flat_A, batch_info=bi, num_slices=NUM_SLICES
        )
        flat_out = chunked_sgmv_lora_expand_forward(
            x=flat_shrink,
            weights=flat_B,
            batch_info=bi,
            slice_offsets=slice_offsets,
            max_slice_size=OUTPUT_DIM,
            base_output=None,
        )

        paged_shrink = chunked_sgmv_lora_shrink_forward_paged(
            x=x,
            A_pages=paged_A,
            batch_info=bi,
            num_slices=NUM_SLICES,
            page_table=page_table,
            max_pages_per_lora=max_pages,
            page_rank_size=PAGE_RANK_SIZE,
        )
        paged_out = chunked_sgmv_lora_expand_forward_paged(
            x=paged_shrink,
            B_pages=paged_B,
            batch_info=bi,
            slice_offsets=slice_offsets,
            max_slice_size=OUTPUT_DIM,
            base_output=None,
            page_table=page_table,
            max_pages_per_lora=max_pages,
            page_rank_size=PAGE_RANK_SIZE,
        )

        self.assertEqual(flat_out.shape, paged_out.shape)
        self.assertTrue(
            torch.allclose(flat_out, paged_out, atol=1e-4, rtol=1e-4),
            f"Chain mismatch: max diff={ (flat_out - paged_out).abs().max().item()}",
        )


class TestPagedShrinkMultiPageAndEviction(CustomTestCase):
    """Tests multi-page adapters and evicted-page (-1) handling."""

    def test_shrink_multi_page(self):
        """rank=16, page_rank_size=8 → 2 pages per adapter."""
        device = "cuda"
        torch.manual_seed(99)

        rank = 16
        num_pages = rank // PAGE_RANK_SIZE  # 2
        S_local = 16

        x = torch.randn(S_local, INPUT_DIM, device=device, dtype=torch.float32)
        flat_A = torch.randn(
            1, NUM_SLICES * rank, INPUT_DIM, device=device, dtype=torch.float32
        )
        paged_A = flat_A.clone()
        page_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)

        seg_indptr = torch.tensor([0, S_local], dtype=torch.int32, device=device)
        weight_indices = torch.tensor([0], dtype=torch.int32, device=device)
        lora_ranks = torch.tensor([rank], dtype=torch.int32, device=device)
        scalings = torch.tensor([1.0], dtype=torch.float32, device=device)
        permutation = torch.arange(S_local, dtype=torch.int32, device=device)

        bi = LoRABatchInfo(
            bs=1,
            num_segments=1,
            max_len=S_local,
            use_cuda_graph=False,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
            seg_lens=None,
        )

        flat_out = chunked_sgmv_lora_shrink_forward(
            x=x, weights=flat_A, batch_info=bi, num_slices=NUM_SLICES
        )
        paged_out = chunked_sgmv_lora_shrink_forward_paged(
            x=x,
            A_pages=paged_A,
            batch_info=bi,
            num_slices=NUM_SLICES,
            page_table=page_table,
            max_pages_per_lora=num_pages,
            page_rank_size=PAGE_RANK_SIZE,
        )

        self.assertEqual(flat_out.shape, paged_out.shape)
        self.assertTrue(
            torch.allclose(flat_out, paged_out, atol=1e-5, rtol=1e-5),
            f"Multi-page mismatch: max diff={ (flat_out - paged_out).abs().max().item()}",
        )

    def test_shrink_evicted_page(self):
        """Page with -1 in page_table → output columns for that page must be zero."""
        device = "cuda"
        torch.manual_seed(77)

        rank = 16
        num_pages = 2
        S_local = 16

        x = torch.randn(S_local, INPUT_DIM, device=device, dtype=torch.float32)
        paged_A = torch.randn(
            num_pages,
            PAGE_RANK_SIZE * NUM_SLICES,
            INPUT_DIM,
            device=device,
            dtype=torch.float32,
        )
        # page_table: page 0 resident, page 1 evicted (-1)
        page_table = torch.tensor([[0, -1]], dtype=torch.int32, device=device)

        seg_indptr = torch.tensor([0, S_local], dtype=torch.int32, device=device)
        weight_indices = torch.tensor([0], dtype=torch.int32, device=device)
        lora_ranks = torch.tensor([rank], dtype=torch.int32, device=device)
        scalings = torch.tensor([1.0], dtype=torch.float32, device=device)
        permutation = torch.arange(S_local, dtype=torch.int32, device=device)

        bi = LoRABatchInfo(
            bs=1,
            num_segments=1,
            max_len=S_local,
            use_cuda_graph=False,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
            seg_lens=None,
        )

        paged_out = chunked_sgmv_lora_shrink_forward_paged(
            x=x,
            A_pages=paged_A,
            batch_info=bi,
            num_slices=NUM_SLICES,
            page_table=page_table,
            max_pages_per_lora=num_pages,
            page_rank_size=PAGE_RANK_SIZE,
        )

        # Output shape: (S, max_pages * page_rank_size * num_slices) = (16, 16)
        self.assertEqual(
            paged_out.shape, (S_local, num_pages * PAGE_RANK_SIZE * NUM_SLICES)
        )

        # Columns for page 0 (indices 0..7) should have non-zero values
        page0_cols = paged_out[:, :PAGE_RANK_SIZE]
        self.assertFalse(
            torch.allclose(page0_cols, torch.zeros_like(page0_cols)),
            "Page 0 output should be non-zero",
        )

        # Columns for page 1 (indices 8..15) should be all zeros (evicted)
        page1_cols = paged_out[:, PAGE_RANK_SIZE:]
        self.assertTrue(
            torch.allclose(page1_cols, torch.zeros_like(page1_cols)),
            f"Evicted page output should be zero, got max={page1_cols.abs().max().item()}",
        )


if __name__ == "__main__":
    unittest.main()
