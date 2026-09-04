import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa_backend import DSAIndexerMetadata, DSAMetadata
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=2, suite="stage-b-test-1-gpu-xpu")


@unittest.skipIf(not torch.xpu.is_available(), "Test requires XPU")
class TestTopKTransformXPU(CustomTestCase):
    def _make_tie_free_logits(
        self, batch_size: int, max_score_len: int, device: torch.device
    ) -> torch.Tensor:
        # Build tie-free rows so expected top-k is deterministic.
        perm = torch.argsort(
            torch.randn(batch_size, max_score_len, dtype=torch.float32, device=device),
            dim=-1,
        )
        return torch.gather(
            torch.arange(max_score_len, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1),
            dim=1,
            index=perm,
        )

    def test_dsa_use_fast_topk_v2(self):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(
            torch.ops.sgl_kernel, "fast_topk"
        ):
            self.skipTest("sgl_kernel.fast_topk op is not available on this runtime")

        device = torch.device("xpu")
        backend = DSATopKBackend.SGL_KERNEL
        batch_size = 4
        max_score_len = 4096
        topk = 2048

        logits = self._make_tie_free_logits(batch_size, max_score_len, device)

        row_starts = None
        lengths = torch.randint(
            1,
            max_score_len,
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )

        out = backend.topk_func(logits, lengths, topk, row_starts=row_starts)

        self.assertEqual(out.shape, (batch_size, topk))
        self.assertEqual(out.dtype, torch.int32)

        expected_valid = torch.minimum(lengths, torch.full_like(lengths, topk))
        actual_valid = (out >= 0).sum(dim=-1).to(torch.int32)
        self.assertTrue(torch.equal(actual_valid, expected_valid))

        starts = (
            row_starts.to(torch.int32)
            if row_starts is not None
            else torch.zeros((batch_size,), dtype=torch.int32, device=device)
        )

        for row in range(batch_size):
            valid = out[row][out[row] >= 0]
            expected_k = int(expected_valid[row].item())
            self.assertEqual(valid.numel(), expected_k)
            if expected_k == 0:
                continue

            start = int(starts[row].item())
            row_len = int(lengths[row].item())
            self.assertTrue(torch.all((valid >= 0) & (valid < row_len)))
            ref = torch.topk(
                logits[row, start : start + row_len], expected_k, dim=-1, sorted=False
            ).indices
            self.assertTrue(
                torch.equal(
                    torch.sort(valid.to(torch.int32)).values,
                    torch.sort(ref.to(torch.int32)).values,
                )
            )

    def test_dsa_use_fast_topk_transform_fused(self):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(
            torch.ops.sgl_kernel, "fast_topk_transform_fused"
        ):
            self.skipTest(
                "sgl_kernel.fast_topk_transform_fused op is not available on this runtime"
            )

        device = torch.device("xpu")
        backend = DSATopKBackend.SGL_KERNEL
        batch_size = 2
        max_score_len = 4096
        topk = 2048

        logits = self._make_tie_free_logits(batch_size, max_score_len, device)
        lengths_list = [
            int(torch.randint(1, max_score_len, (), device=device).item())
            for _ in range(batch_size)
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)

        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        dsa_cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        dsa_cu_seqlens_k[1:] = torch.cumsum(lengths, dim=0)
        page_table_1 = (
            torch.arange(max_score_len, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        metadata = DSAIndexerMetadata(
            attn_metadata=DSAMetadata(
                page_size=1,
                cache_seqlens_int32=lengths.clone(),
                max_seq_len_q=1,
                max_seq_len_k=max_score_len,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q.clone(),
                page_table_1=page_table_1,
                real_page_table=page_table_1,
                dsa_cache_seqlens_int32=lengths.clone(),
                dsa_cu_seqlens_q=cu_seqlens_q.clone(),
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                dsa_extend_seq_lens_list=lengths_list,
                dsa_seqlens_expanded=lengths,
            ),
            topk_transform_method=TopkTransformMethod.PAGED,
            topk_backend=backend,
        )

        # Force legacy fused transform path so we can verify the XPU fused
        # op hook (`fast_topk_transform_fused`) is reachable.
        with envs.SGLANG_DSA_FUSE_TOPK.override(
            True
        ), envs.SGLANG_OPT_USE_TOPK_V2.override(False):
            out = metadata.topk_transform(logits, topk)

        self.assertEqual(out.shape, (batch_size, topk))
        self.assertEqual(out.dtype, torch.int32)

        expected_valid = torch.minimum(lengths, torch.full_like(lengths, topk))
        actual_valid = (out >= 0).sum(dim=-1).to(torch.int32)
        self.assertTrue(torch.equal(actual_valid, expected_valid))

        for row in range(batch_size):
            valid = out[row][out[row] >= 0]
            expected_k = int(expected_valid[row].item())
            self.assertEqual(valid.numel(), expected_k)
            if expected_k == 0:
                continue

            row_len = int(lengths[row].item())
            self.assertTrue(torch.all((valid >= 0) & (valid < row_len)))

            ref = torch.topk(
                logits[row, :row_len], expected_k, dim=-1, sorted=False
            ).indices
            self.assertTrue(
                torch.equal(
                    torch.sort(valid.to(torch.int32)).values,
                    torch.sort(ref.to(torch.int32)).values,
                )
            )

    def test_dsa_use_fast_topk_transform_ragged_fused(self):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(
            torch.ops.sgl_kernel, "fast_topk_transform_ragged_fused"
        ):
            self.skipTest(
                "sgl_kernel.fast_topk_transform_ragged_fused op is not available on this runtime"
            )

        device = torch.device("xpu")
        backend = DSATopKBackend.SGL_KERNEL
        batch_size = 3
        max_score_len = 4096
        topk = 2048

        logits = self._make_tie_free_logits(batch_size, max_score_len, device)
        lengths_list = [
            int(torch.randint(1, max_score_len, (), device=device).item())
            for _ in range(batch_size)
        ]
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)

        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        dsa_cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        dsa_cu_seqlens_k[1:] = torch.cumsum(lengths, dim=0)
        page_table_1 = (
            torch.arange(max_score_len, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        topk_indices_offset = (
            torch.arange(batch_size, dtype=torch.int32, device=device) * max_score_len
        )

        metadata = DSAIndexerMetadata(
            attn_metadata=DSAMetadata(
                page_size=1,
                cache_seqlens_int32=lengths.clone(),
                max_seq_len_q=1,
                max_seq_len_k=max_score_len,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q.clone(),
                page_table_1=page_table_1,
                real_page_table=page_table_1,
                dsa_cache_seqlens_int32=lengths.clone(),
                dsa_cu_seqlens_q=cu_seqlens_q.clone(),
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                dsa_extend_seq_lens_list=lengths_list,
                dsa_seqlens_expanded=lengths,
                topk_indices_offset=topk_indices_offset,
            ),
            topk_transform_method=TopkTransformMethod.RAGGED,
            topk_backend=backend,
        )

        with envs.SGLANG_DSA_FUSE_TOPK.override(
            True
        ), envs.SGLANG_OPT_USE_TOPK_V2.override(False):
            out = metadata.topk_transform(logits, topk)

        self.assertEqual(out.shape, (batch_size, topk))
        self.assertEqual(out.dtype, torch.int32)

        expected_valid = torch.minimum(lengths, torch.full_like(lengths, topk))
        actual_valid = (out >= 0).sum(dim=-1).to(torch.int32)
        self.assertTrue(torch.equal(actual_valid, expected_valid))

        for row in range(batch_size):
            valid = out[row][out[row] >= 0]
            expected_k = int(expected_valid[row].item())
            self.assertEqual(valid.numel(), expected_k)
            if expected_k == 0:
                continue

            row_len = int(lengths[row].item())
            offset = int(topk_indices_offset[row].item())
            self.assertTrue(torch.all((valid >= offset) & (valid < offset + row_len)))

            ref_local = torch.topk(
                logits[row, :row_len], expected_k, dim=-1, sorted=False
            ).indices.to(torch.int32)
            ref_global = ref_local + offset
            self.assertTrue(
                torch.equal(
                    torch.sort(valid.to(torch.int32)).values,
                    torch.sort(ref_global).values,
                )
            )


if __name__ == "__main__":
    unittest.main()
