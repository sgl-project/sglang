import unittest
from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.dsa.transform_index as transform_index_module
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_fast,
    transform_index_page_table_prefill_fast,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

TOPK = 2048
KPOOL_TAILS = 3
TOPK_KPOOL = TOPK + KPOOL_TAILS  # native GLM 5.3 Flash MTP width = 2051


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDSATransformIndex(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _make_page_table(self, rows: int, context_length: int) -> torch.Tensor:
        columns = torch.arange(context_length, dtype=torch.int32, device=self.device)
        row_bias = (
            torch.arange(rows, dtype=torch.int32, device=self.device).unsqueeze(1) * 17
        )
        return columns.unsqueeze(0) + row_bias

    def _make_topk(
        self, rows: int, context_length: int, width: int = TOPK
    ) -> torch.Tensor:
        topk = (
            torch.arange(width, dtype=torch.int64, device=self.device)
            .remainder(context_length)
            .repeat(rows, 1)
        )
        if rows > 0:
            topk[:, 0] = 0
            topk[:, 1] = context_length - 1
            topk[:, 257::257] = -1
            if width > TOPK:
                # KPool tail tokens are live, not padding.
                topk[:, TOPK:] = (
                    context_length
                    - KPOOL_TAILS
                    + torch.arange(KPOOL_TAILS, dtype=torch.int64, device=self.device)
                ).clamp(max=context_length - 1)
        return topk

    def _expected(
        self,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
        extend_lens_cpu: list[int],
        output_num_tokens: int,
        page_table_is_expanded: bool,
    ) -> torch.Tensor:
        real_num_tokens = sum(extend_lens_cpu)
        expected = torch.full(
            (output_num_tokens, topk_indices.shape[1]),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        if real_num_tokens == 0:
            return expected

        if page_table_is_expanded:
            source_rows = page_table[:real_num_tokens]
        else:
            request_ids = torch.repeat_interleave(
                torch.arange(
                    len(extend_lens_cpu), dtype=torch.int64, device=self.device
                ),
                torch.tensor(extend_lens_cpu, dtype=torch.int64, device=self.device),
            )
            source_rows = page_table[request_ids]

        real_topk = topk_indices[:real_num_tokens]
        torch.gather(
            source_rows,
            dim=1,
            index=real_topk.clamp(min=0),
            out=expected[:real_num_tokens],
        )
        expected[:real_num_tokens][real_topk < 0] = -1
        return expected

    def _check_decode_case(
        self,
        batch_size: int,
        context_length: int,
        *,
        zero_row_stride: bool = False,
        provide_result: bool = False,
    ) -> None:
        if zero_row_stride:
            page_table = self._make_page_table(1, context_length).expand(batch_size, -1)
        else:
            page_table = self._make_page_table(batch_size, context_length)
        topk_indices = self._make_topk(batch_size, context_length)
        expected = torch.empty(
            (batch_size, topk_indices.shape[1]), dtype=torch.int32, device=self.device
        )
        torch.gather(
            page_table,
            dim=1,
            index=topk_indices.clamp(min=0),
            out=expected,
        )
        expected[topk_indices < 0] = -1
        result = torch.empty_like(expected) if provide_result else None

        actual = transform_index_page_table_decode_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            result=result,
        )
        torch.cuda.synchronize()
        if result is not None:
            self.assertIs(actual, result)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _check_case(
        self,
        extend_lens_cpu: list[int],
        context_length: int,
        *,
        page_table_is_expanded: bool,
        topk_padding: int = 0,
        output_padding: int = 0,
        topk_width: int = TOPK,
    ) -> None:
        real_num_tokens = sum(extend_lens_cpu)
        page_table_rows = (
            real_num_tokens if page_table_is_expanded else len(extend_lens_cpu)
        )
        topk_num_tokens = real_num_tokens + topk_padding
        output_num_tokens = topk_num_tokens + output_padding
        page_table = self._make_page_table(page_table_rows, context_length)
        topk_indices = self._make_topk(
            topk_num_tokens, context_length, width=topk_width
        )
        expected = self._expected(
            page_table,
            topk_indices,
            extend_lens_cpu,
            output_num_tokens,
            page_table_is_expanded,
        )

        actual = transform_index_page_table_prefill_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            extend_lens_cpu=extend_lens_cpu,
            output_num_tokens=output_num_tokens,
            page_table_is_expanded=page_table_is_expanded,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_uses_dedicated_kernel(self):
        extend_lens_cpu = [2, 1]
        context_length = 4096
        page_table = self._make_page_table(len(extend_lens_cpu), context_length)
        topk_indices = self._make_topk(sum(extend_lens_cpu), context_length)

        with patch.object(
            transform_index_module,
            "transform_index_page_table_decode_fast",
            side_effect=AssertionError("prefill must not launch decode per request"),
        ):
            transform_index_page_table_prefill_fast(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
            )

    def test_prefill_uses_device_cu_seqlens(self):
        extend_lens_cpu = [2, 1]
        context_length = 4096
        page_table = self._make_page_table(len(extend_lens_cpu), context_length)
        topk_indices = self._make_topk(sum(extend_lens_cpu), context_length)
        cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32, device=self.device)

        with patch.object(
            transform_index_module.torch,
            "tensor",
            side_effect=AssertionError("must reuse device-side metadata"),
        ):
            transform_index_page_table_prefill_fast(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
                cu_seqlens_q=cu_seqlens_q,
            )

    def test_prefill_dynamic_page_table_row_strides(self):
        context_lengths = (4096, 4160, 4224)
        kernel = transform_index_module.transform_index_page_table_prefill_kernel

        self._check_case(
            [2, 1],
            context_lengths[0],
            page_table_is_expanded=True,
        )
        kernel_cache = kernel.device_caches[torch.cuda.current_device()][0]
        specialization_count = len(kernel_cache)

        for context_length in context_lengths[1:]:
            with self.subTest(context_length=context_length):
                self._check_case(
                    [2, 1],
                    context_length,
                    page_table_is_expanded=True,
                )
                self.assertEqual(len(kernel_cache), specialization_count)

    def test_mixed_lengths_padding_and_empty_batch(self):
        self._check_case(
            [0, 3, 1, 0, 4],
            8192,
            page_table_is_expanded=False,
            topk_padding=5,
            output_padding=7,
        )
        self._check_case(
            [0, 0],
            16,
            page_table_is_expanded=False,
            output_padding=8,
        )

    def test_large_batch_size(self):
        self._check_case(
            [1] * 8192,
            4096,
            page_table_is_expanded=False,
        )

    def test_large_context_lengths(self):
        for context_length, page_table_is_expanded in (
            (640_000, True),
            (1_000_000, False),
        ):
            with self.subTest(
                context_length=context_length,
                page_table_is_expanded=page_table_is_expanded,
            ):
                self._check_case(
                    [2, 1],
                    context_length,
                    page_table_is_expanded=page_table_is_expanded,
                )

    def test_decode_fast_correctness_and_strides(self):
        self._check_decode_case(17, 8192, provide_result=True)
        self._check_decode_case(17, 8192, zero_row_stride=True)

    def test_decode_fast_extreme_shapes(self):
        self._check_decode_case(8192, 4096)
        self._check_decode_case(2, 1_000_000)

    def test_prefill_kpool_2051_tail_width(self):
        for expanded in (True, False):
            with self.subTest(page_table_is_expanded=expanded):
                self._check_case(
                    [2, 1],
                    8192,
                    page_table_is_expanded=expanded,
                    topk_padding=4,
                    output_padding=5,
                    topk_width=TOPK_KPOOL,
                )

    def test_decode_kpool_2051_tail_width(self):
        self._check_decode_case(3, 8192)
        topk_indices = self._make_topk(3, 8192, width=TOPK_KPOOL)
        page_table = self._make_page_table(3, 8192)
        actual = transform_index_page_table_decode_fast(
            page_table=page_table,
            topk_indices=topk_indices,
        )
        torch.cuda.synchronize()
        expected = torch.full(
            (3, TOPK_KPOOL), -1, dtype=torch.int32, device=self.device
        )
        torch.gather(page_table, dim=1, index=topk_indices.clamp(min=0), out=expected)
        expected[topk_indices < 0] = -1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_ordinary_2048_preserved(self):
        self._check_case(
            [1, 4],
            4096,
            page_table_is_expanded=True,
            topk_padding=2,
            output_padding=3,
            topk_width=TOPK,
        )

    def test_prefill_kpool_2051_expanded_graph_replay_576(self):
        rows = 96 * 6
        context_length = 8192
        extend_lens_cpu = [6] * 96
        page_table = self._make_page_table(rows, context_length)
        topk_indices = self._make_topk(rows, context_length, width=TOPK_KPOOL)
        cu_seqlens_q = torch.arange(
            0, rows + 1, 6, dtype=torch.int32, device=self.device
        )

        # Warm up/JIT before capture.
        transform_index_page_table_prefill_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            extend_lens_cpu=extend_lens_cpu,
            output_num_tokens=rows,
            page_table_is_expanded=True,
            cu_seqlens_q=cu_seqlens_q,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = transform_index_page_table_prefill_fast(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
                output_num_tokens=rows,
                page_table_is_expanded=True,
                cu_seqlens_q=cu_seqlens_q,
            )

        for phase in range(3):
            tail = (context_length - KPOOL_TAILS + phase * 2) % context_length
            topk_indices[:, TOPK:] = (
                tail + torch.arange(KPOOL_TAILS, dtype=torch.int64, device=self.device)
            ) % context_length
            graph.replay()
            torch.cuda.synchronize()
            expected = self._expected(
                page_table,
                topk_indices,
                extend_lens_cpu,
                rows,
                page_table_is_expanded=True,
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            # Keep the masked padding columns invalid after the replay.
            self.assertTrue(torch.all(topk_indices[0, 257::257] == -1))


if __name__ == "__main__":
    unittest.main()
