import unittest

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_score import quest_page_scores
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _reference_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    history_page_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    in_range = (physical_pages >= 0) & (physical_pages < page_valid.shape[0])
    safe_pages = physical_pages.clamp(0, page_valid.shape[0] - 1)
    k_min = page_k_min[safe_pages].to(torch.float32)
    k_max = page_k_max[safe_pages].to(torch.float32)
    valid = in_range & page_valid[safe_pages]
    batch_size, query_heads, head_dim = queries.shape
    kv_heads = k_min.shape[-2]
    group_size = query_heads // kv_heads
    query = queries.reshape(batch_size, kv_heads, group_size, head_dim)
    query = query.mean(dim=2).to(torch.float32).unsqueeze(1)
    scores = torch.where(
        query >= 0,
        query * k_max,
        query * k_min,
    ).sum(dim=(2, 3))
    if active_mask is not None:
        page_idx = torch.arange(physical_pages.shape[1], device=physical_pages.device)
        valid = (
            valid
            & active_mask.unsqueeze(1)
            & (page_idx.unsqueeze(0) < history_page_counts.unsqueeze(1))
        )
    return torch.where(valid, scores, torch.full_like(scores, float("-inf")))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestScoreKernel(unittest.TestCase):
    def test_matches_reference_for_gqa_and_noncontiguous_queries(self):
        torch.manual_seed(42)
        device = torch.device("cuda")
        batch_size = 3
        num_pool_pages = 19
        num_pages = 11
        query_heads = 16
        kv_heads = 8
        head_dim = 128

        page_k_min = torch.randn(
            num_pool_pages,
            kv_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        page_k_max = page_k_min + torch.rand_like(page_k_min)
        page_valid = torch.ones(num_pool_pages, dtype=torch.bool, device=device)
        page_valid[[2, 13]] = False
        physical_page_storage = torch.empty(
            (batch_size, num_pages * 2), dtype=torch.int64, device=device
        )
        pool_indices = torch.arange(num_pool_pages, device=device)
        for batch_idx in range(batch_size):
            physical_page_storage[batch_idx, ::2] = torch.roll(
                pool_indices, shifts=batch_idx * 3
            )[:num_pages]
        physical_page_storage[0, 0] = -1
        physical_page_storage[2, 2] = num_pool_pages
        physical_pages = physical_page_storage[:, ::2]
        active_mask = torch.tensor([True, False, True], device=device)
        history_page_counts = torch.tensor(
            [num_pages - 1, num_pages, num_pages], device=device
        )

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                query_storage = torch.randn(
                    batch_size,
                    query_heads,
                    head_dim * 2,
                    dtype=dtype,
                    device=device,
                )
                queries = query_storage[:, :, ::2]
                expected = _reference_scores(
                    queries,
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                    active_mask,
                    history_page_counts,
                )

                actual_3d = quest_page_scores(
                    queries,
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                    active_mask=active_mask,
                    history_page_counts=history_page_counts,
                )
                flat_storage = torch.empty(
                    batch_size,
                    query_heads * head_dim * 2,
                    dtype=dtype,
                    device=device,
                )
                flat_storage[:, ::2] = queries.reshape(batch_size, -1)
                actual_2d = quest_page_scores(
                    flat_storage[:, ::2],
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                    active_mask=active_mask,
                    history_page_counts=history_page_counts,
                )

                torch.testing.assert_close(actual_3d, expected, rtol=2e-4, atol=2e-3)
                torch.testing.assert_close(actual_2d, expected, rtol=2e-4, atol=2e-3)
                topk = min(5, num_pages)
                for row in (0, 2):
                    self.assertTrue(
                        torch.equal(
                            actual_3d[row].topk(topk).indices,
                            expected[row].topk(topk).indices,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
