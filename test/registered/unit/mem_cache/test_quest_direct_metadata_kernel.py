import unittest

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_finalize import (
    quest_finalize_selected_pages,
)
from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
    quest_finalize_to_flashattention_metadata_,
    quest_update_flashattention_metadata_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _reference_metadata(inputs, page_size):
    batch_size, topk_width = inputs["topk_scores"].shape
    output_width = topk_width + inputs["recent_indices"].shape[1]
    page_table = torch.full(
        (batch_size, output_width),
        -99,
        dtype=torch.int32,
        device=inputs["topk_scores"].device,
    )
    valid_lengths = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    cache_seqlens = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")

    cumulative = 0
    for row in range(batch_size):
        selected = []
        for col in range(int(inputs["k_per_req"][row].item())):
            score = inputs["topk_scores"][row, col]
            if torch.isfinite(score).item():
                selected.append(int(inputs["topk_indices"][row, col].item()))
        for col in range(inputs["recent_indices"].shape[1]):
            if inputs["recent_valid"][row, col].item():
                selected.append(int(inputs["recent_indices"][row, col].item()))

        selected.sort()
        valid_lengths[row] = len(selected)
        if inputs["sparse_mask"][row].item() and selected:
            req_idx = int(inputs["req_pool_indices"][row].item())
            for out_col, logical_page in enumerate(selected):
                first_token = inputs["req_to_token"][req_idx, logical_page * page_size]
                page_table[row, out_col] = first_token // page_size

        seq_len = int(inputs["seq_lens"][row].item())
        last_page_len = (seq_len - 1) % page_size + 1
        sparse_len = (len(selected) - 1) * page_size + last_page_len
        cache_len = (
            sparse_len if inputs["sparse_mask"][row].item() and selected else seq_len
        )
        cache_seqlens[row] = cache_len
        cu_seqlens[row] = cumulative
        cumulative += cache_len
    cu_seqlens[batch_size] = cumulative
    return page_table, valid_lengths, cache_seqlens, cu_seqlens


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestDirectMetadataKernel(unittest.TestCase):
    page_size = 4

    def _make_inputs(self):
        device = torch.device("cuda")
        return {
            "topk_scores": torch.tensor(
                [
                    [5.0, float("nan"), 2.0, float("-inf")],
                    [float("inf"), 9.0, 8.0, 7.0],
                    [float("-inf")] * 4,
                ],
                dtype=torch.float32,
                device=device,
            ),
            "topk_indices": torch.tensor(
                [[6, 1, 3, 0], [5, 4, 0, 2], [1, 2, 3, 4]],
                dtype=torch.int32,
                device=device,
            ),
            "k_per_req": torch.tensor([3, 4, 0], dtype=torch.int32, device=device),
            "recent_indices": torch.tensor(
                [[7, 8], [6, 7], [4, 5]], dtype=torch.int32, device=device
            ),
            "recent_valid": torch.tensor(
                [[True, True], [True, False], [False, False]], device=device
            ),
            "sparse_mask": torch.tensor([True, True, False], device=device),
            "seq_lens": torch.tensor([35, 31, 9], dtype=torch.int64, device=device),
            "req_pool_indices": torch.tensor(
                [2, 0, 1], dtype=torch.int64, device=device
            ),
            "req_to_token": torch.arange(
                3 * 64, dtype=torch.int32, device=device
            ).reshape(3, 64)
            + 16,
        }

    @staticmethod
    def _make_outputs(batch_size, output_width):
        return (
            torch.full(
                (batch_size, output_width), -99, dtype=torch.int32, device="cuda"
            ),
            torch.empty(batch_size, dtype=torch.int32, device="cuda"),
            torch.empty(batch_size, dtype=torch.int32, device="cuda"),
            torch.empty(batch_size + 1, dtype=torch.int32, device="cuda"),
        )

    def _run(self, inputs, outputs):
        page_table, valid_lengths, cache_seqlens, cu_seqlens = outputs
        quest_finalize_to_flashattention_metadata_(
            **inputs,
            valid_lengths=valid_lengths,
            page_table=page_table,
            cache_seqlens_int32=cache_seqlens,
            cu_seqlens_k=cu_seqlens,
            page_size=self.page_size,
            update_lengths=True,
        )

    @staticmethod
    def _assert_outputs(actual, expected):
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_finalize_to_metadata_matches_reference(self):
        inputs = self._make_inputs()
        batch_size, topk_width = inputs["topk_scores"].shape
        output_width = topk_width + inputs["recent_indices"].shape[1]
        outputs = self._make_outputs(batch_size, output_width)

        self._run(inputs, outputs)
        torch.cuda.synchronize()
        self._assert_outputs(outputs, _reference_metadata(inputs, self.page_size))

    def test_finalize_rejects_unsafe_shapes_and_output_stride(self):
        inputs = self._make_inputs()
        batch_size, topk_width = inputs["topk_scores"].shape
        output_width = topk_width + inputs["recent_indices"].shape[1]
        page_table, valid_lengths, cache_seqlens, cu_seqlens = self._make_outputs(
            batch_size, output_width
        )
        base_kwargs = {
            **inputs,
            "valid_lengths": valid_lengths,
            "page_table": page_table,
            "cache_seqlens_int32": cache_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "page_size": self.page_size,
            "update_lengths": True,
        }
        noncontiguous_lengths = torch.empty(
            (batch_size, 2), dtype=torch.int32, device="cuda"
        )[:, 0]
        invalid_values = {
            "recent_valid": inputs["recent_valid"][:, :1],
            "k_per_req": inputs["k_per_req"][:1],
            "req_to_token": inputs["req_to_token"].reshape(-1),
            "seq_lens": inputs["seq_lens"].to(torch.float32),
            "req_pool_indices": inputs["req_pool_indices"].to(torch.float32),
            "valid_lengths": noncontiguous_lengths,
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                quest_finalize_to_flashattention_metadata_(
                    **{**base_kwargs, name: value}
                )

    def test_selected_pages_cuda_graph_replay_reads_updated_input(self):
        device = torch.device("cuda")
        selected_indices = torch.tensor(
            [[2, 0], [1, -1]], dtype=torch.int32, device=device
        )
        valid_lengths = torch.tensor([2, 1], dtype=torch.int32, device=device)
        sparse_mask = torch.tensor([True, True], device=device)
        seq_lens = torch.tensor([10, 7], dtype=torch.int64, device=device)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
        req_to_token = torch.arange(24, dtype=torch.int32, device=device).view(2, 12)
        page_table = torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device=device
        )
        cache_seqlens = torch.empty(2, dtype=torch.int32, device=device)
        cu_seqlens = torch.empty(3, dtype=torch.int32, device=device)

        def run():
            quest_update_flashattention_metadata_(
                selected_indices=selected_indices,
                valid_lengths=valid_lengths,
                sparse_mask=sparse_mask,
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                page_table=page_table,
                cache_seqlens_int32=cache_seqlens,
                cu_seqlens_k=cu_seqlens,
                page_size=self.page_size,
                update_lengths=True,
            )

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            run()
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        selected_indices.copy_(
            torch.tensor([[1, 0], [2, -1]], dtype=torch.int32, device=device)
        )
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(page_table.tolist(), [[1, 0, 12], [5, 21, 22]])
        self.assertEqual(cache_seqlens.tolist(), [6, 3])
        self.assertEqual(cu_seqlens.tolist(), [0, 6, 9])

    def test_finalize_selected_pages_matches_reference(self):
        inputs = self._make_inputs()
        selected, lengths = quest_finalize_selected_pages(
            inputs["topk_scores"],
            inputs["topk_indices"],
            inputs["k_per_req"],
            inputs["recent_indices"],
            inputs["recent_valid"],
        )
        torch.cuda.synchronize()

        expected = torch.tensor(
            [
                [3, 6, 7, 8, -1, -1],
                [0, 2, 4, 6, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
            device="cuda",
        )
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        self.assertEqual(lengths.tolist(), [4, 4, 0])


if __name__ == "__main__":
    unittest.main()
