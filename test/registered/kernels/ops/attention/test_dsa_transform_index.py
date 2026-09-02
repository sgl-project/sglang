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

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

TOPK = 2048
# k-pool appends up to index_kpool - 1 open-tail tokens to index_topk, so the
# width the indexer hands over is not a power of two: 2048 + 4 - 1 for
# GLM-5.3-Flash. See get_dsa_mtp_topk_width() in srt/configs/model_config.py.
KPOOL_TOPK = 2051


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
        self, rows: int, context_length: int, topk_width: int = TOPK
    ) -> torch.Tensor:
        topk = (
            torch.arange(topk_width, dtype=torch.int64, device=self.device)
            .remainder(context_length)
            .repeat(rows, 1)
        )
        if rows > 0:
            topk[:, 0] = 0
            topk[:, 1] = context_length - 1
            topk[:, 257::257] = -1
        return topk

    def _expected(
        self,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
        extend_lens_cpu: list[int],
        output_num_tokens: int,
        page_table_is_expanded: bool,
        topk_width: int = TOPK,
    ) -> torch.Tensor:
        real_num_tokens = sum(extend_lens_cpu)
        expected = torch.full(
            (output_num_tokens, topk_width),
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
        topk_width: int = TOPK,
    ) -> None:
        if zero_row_stride:
            page_table = self._make_page_table(1, context_length).expand(batch_size, -1)
        else:
            page_table = self._make_page_table(batch_size, context_length)
        topk_indices = self._make_topk(batch_size, context_length, topk_width)
        expected = torch.empty(
            (batch_size, topk_width), dtype=torch.int32, device=self.device
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
        topk_indices = self._make_topk(topk_num_tokens, context_length, topk_width)
        expected = self._expected(
            page_table,
            topk_indices,
            extend_lens_cpu,
            output_num_tokens,
            page_table_is_expanded,
            topk_width,
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

    def test_decode_fast_kpool_widths(self):
        # 2051 is the GLM-5.3-Flash k-pool width; the others cover a partial
        # trailing tile and a width below one tile.
        for topk_width in (KPOOL_TOPK, 515, 257):
            with self.subTest(topk_width=topk_width):
                self._check_decode_case(17, 8192, topk_width=topk_width)
                self._check_decode_case(
                    17, 8192, topk_width=topk_width, provide_result=True
                )

    def test_prefill_kpool_widths(self):
        for topk_width in (KPOOL_TOPK, 515, 257):
            with self.subTest(topk_width=topk_width):
                self._check_case(
                    [2, 1],
                    4096,
                    page_table_is_expanded=False,
                    topk_width=topk_width,
                )


if __name__ == "__main__":
    unittest.main()
