"""DeepSeek-V4 Q8KV8 sparse-prefill KV adapter tests.

The adapter consumes the existing DeepSeek-V4 packed paged KV cache layout and
emits the flat FP8 workspace required by the SM90 Q8KV8 sparse prefill kernel.
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    cast_q_fp8_for_q8kv8_prefill,
    dequantize_k_cache_paged,
    fp8_dtype,
    gather_dequant_requant_fp8_paged_ref,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import SetKAndS
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large"
)


class _Pool:
    def __init__(self, page_size: int):
        self.page_size = page_size


def _make_v4_paged_kv_cache(
    *,
    num_pages: int,
    page_size: int,
    num_written_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # DeepSeek-V4 packed layout: per page stores all token data first
    # (448 fp8 nope + 64 bf16 rope = 576 bytes/token), then 8 scale bytes/token.
    bytes_per_token = 448 + 64 * 2 + 8
    bytes_per_page = page_size * bytes_per_token
    quant_k_cache = torch.zeros(
        num_pages, bytes_per_page, dtype=torch.uint8, device=device
    )

    k_bf16 = (torch.randn(num_written_tokens, 512, device=device) * 0.25).to(
        torch.bfloat16
    )
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16)
    loc = torch.randperm(num_pages * page_size, device=device)[:num_written_tokens].to(
        torch.int32
    )
    SetKAndS.torch(_Pool(page_size), quant_k_cache, loc, pack)
    return quant_k_cache, loc


def _fp8_workspace_to_bf16(workspace: torch.Tensor) -> torch.Tensor:
    return workspace.to(torch.bfloat16)


class TestDeepSeekV4Q8KV8KVAdapter(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.device = torch.device("cuda")

    def test_cast_q_fp8_for_q8kv8_prefill_returns_kernel_inputs(self):
        q = (
            torch.randn(5, 64, 512, dtype=torch.bfloat16, device=self.device) * 0.05
        )

        q_fp8, q_scale = cast_q_fp8_for_q8kv8_prefill(q)

        self.assertEqual(q_fp8.shape, q.shape)
        self.assertEqual(q_fp8.dtype, fp8_dtype)
        self.assertTrue(q_fp8.is_contiguous())
        self.assertEqual(q_scale.shape, torch.Size([]))
        self.assertEqual(q_scale.dtype, torch.float32)
        self.assertEqual(q_scale.device, q.device)
        self.assertEqual(q_scale.item(), 1.0)
        torch.testing.assert_close(
            q_fp8.to(torch.bfloat16).float(),
            q.float(),
            atol=3e-2,
            rtol=2e-1,
        )

    def test_gather_dequant_requant_fp8_paged_matches_bf16_workspace(self):
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            gather_dequant_requant_fp8_paged,
        )

        page_size = 64
        quant_k_cache, loc = _make_v4_paged_kv_cache(
            num_pages=5,
            page_size=page_size,
            num_written_tokens=173,
            seed=1,
            device=self.device,
        )
        gather_ids = loc[
            torch.tensor([0, 7, 19, 42, 88, 129, 172], device=self.device)
        ]

        actual = gather_dequant_requant_fp8_paged(
            quant_k_cache,
            gather_ids,
            page_size=page_size,
        )
        expected_bf16 = dequantize_k_cache_paged(
            quant_k_cache,
            gather_ids,
            page_size=page_size,
        )
        expected_ref = gather_dequant_requant_fp8_paged_ref(
            quant_k_cache,
            gather_ids,
            page_size=page_size,
        )

        self.assertEqual(actual.shape, expected_bf16.shape)
        self.assertEqual(actual.dtype, fp8_dtype)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected_ref.view(torch.uint8))
        )
        torch.testing.assert_close(
            _fp8_workspace_to_bf16(actual).float(),
            expected_bf16.float(),
            atol=3e-2,
            rtol=2e-1,
        )

    def test_gather_dequant_requant_fp8_paged_appends_zero_extra_rows(self):
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            gather_dequant_requant_fp8_paged,
        )

        page_size = 32
        extra_rows = 11
        quant_k_cache, loc = _make_v4_paged_kv_cache(
            num_pages=4,
            page_size=page_size,
            num_written_tokens=79,
            seed=2,
            device=self.device,
        )
        gather_ids = loc[:13].contiguous()

        actual = gather_dequant_requant_fp8_paged(
            quant_k_cache,
            gather_ids,
            page_size=page_size,
            extra_rows=extra_rows,
        )
        expected_active_bf16 = dequantize_k_cache_paged(
            quant_k_cache,
            gather_ids,
            page_size=page_size,
        )
        expected_active_ref = gather_dequant_requant_fp8_paged_ref(
            quant_k_cache,
            gather_ids,
            page_size=page_size,
        )

        self.assertEqual(actual.shape, (gather_ids.shape[0] + extra_rows, 1, 512))
        self.assertTrue(
            torch.equal(
                actual[: gather_ids.shape[0]].view(torch.uint8),
                expected_active_ref.view(torch.uint8),
            )
        )
        torch.testing.assert_close(
            _fp8_workspace_to_bf16(actual[: gather_ids.shape[0]]).float(),
            expected_active_bf16.float(),
            atol=3e-2,
            rtol=2e-1,
        )
        extra = _fp8_workspace_to_bf16(actual[gather_ids.shape[0] :]).float()
        self.assertTrue(torch.equal(extra, torch.zeros_like(extra)))


if __name__ == "__main__":
    unittest.main()
