from collections import Counter
from types import SimpleNamespace
import unittest

import torch
from torch.overrides import TorchFunctionMode

from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference_retrieve_topk(
    algo,
    queries: torch.Tensor,
    layer_id: int,
    req_pool_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    forward_batch,
):
    bs, device = queries.shape[0], queries.device
    seq_lens = forward_batch.seq_lens.to(device)
    req_to_token = algo.req_to_token_pool.req_to_token
    max_req_tokens = req_to_token.shape[1]

    per_request_indices = []
    per_request_lengths = []

    for i in range(bs):
        if not sparse_mask[i]:
            per_request_indices.append(torch.empty(0, device=device, dtype=torch.int32))
            per_request_lengths.append(0)
            continue

        num_pages = int((seq_lens[i].item() + algo.page_size - 1) // algo.page_size)
        if num_pages <= algo.num_recent_pages:
            per_request_indices.append(torch.empty(0, device=device, dtype=torch.int32))
            per_request_lengths.append(0)
            continue

        page_idx = torch.arange(num_pages, device=device)
        page_start_token = req_to_token[
            req_pool_indices[i],
            (page_idx * algo.page_size).clamp(0, max_req_tokens - 1),
        ]
        phys_pages = (page_start_token // algo.page_size).unsqueeze(0)

        scores = algo._retrieve_page_scores(
            layer_id,
            phys_pages,
            req_pool_indices[i : i + 1],
            queries[i : i + 1],
        )

        recent_start = max(num_pages - algo.num_recent_pages, 0)
        scores = scores.clone()
        scores[:, recent_start:] = float("-inf")

        history_pages = max(recent_start, 1)
        k = max(int(history_pages * algo.sparsity_ratio), 1)
        k = min(k, history_pages)
        topk_idx = torch.topk(scores, k=k, dim=1, sorted=False)[1].squeeze(0)

        recent_idx = torch.arange(
            recent_start, recent_start + algo.num_recent_pages, device=device
        )
        recent_idx = recent_idx[recent_idx < num_pages]

        combined = torch.cat([topk_idx, recent_idx], dim=0).sort()[0].to(torch.int32)
        per_request_indices.append(combined)
        per_request_lengths.append(int(combined.numel()))

    max_len = max(max(per_request_lengths, default=0), 1)
    out_indices = torch.full((bs, max_len), -1, dtype=torch.int32, device=device)
    out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

    for i, selected in enumerate(per_request_indices):
        length = per_request_lengths[i]
        if length == 0:
            continue
        out_indices[i, :length] = selected
        out_lengths[i] = length

    return out_indices, out_lengths


class _SyncCounterMode(TorchFunctionMode):
    SYNC_OPS = {"item", "__bool__", "__int__", "__index__", "tolist"}

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in self.SYNC_OPS:
            self.counts[name] += 1
        return func(*args, **(kwargs or {}))


class TestSparseAlgorithmRetrieveTopK(CustomTestCase):
    def _build_quest_algo(
        self,
        bs: int = 8,
        max_seq_len: int = 512,
        page_size: int = 16,
        sparsity_ratio: float = 0.5,
        num_recent_pages: int = 4,
    ) -> QuestAlgorithm:
        torch.manual_seed(42)
        device = torch.device("cpu")
        config = SimpleNamespace(
            page_size=page_size,
            sparse_extra_config={
                "sparsity_ratio": sparsity_ratio,
                "num_recent_pages": num_recent_pages,
            },
        )
        algo = QuestAlgorithm(config, device)
        total_tokens = bs * max_seq_len
        total_pages = total_tokens // page_size
        head_num, head_dim = 4, 8

        token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: torch.zeros(
                (total_tokens, head_num, head_dim), dtype=torch.float32, device=device
            )
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(total_tokens, dtype=torch.int32, device=device).view(
                bs, max_seq_len
            )
        )
        states = SimpleNamespace()
        algo.initialize_representation_pool(
            0, 1, token_to_kv_pool, req_to_token_pool, states
        )
        algo.page_k_min[0] = torch.randn(
            (total_pages, head_num, head_dim), dtype=torch.float32, device=device
        )
        algo.page_k_max[0] = algo.page_k_min[0] + torch.rand(
            (total_pages, head_num, head_dim), dtype=torch.float32, device=device
        )
        algo.page_valid[0] = torch.ones(total_pages, dtype=torch.bool, device=device)
        return algo

    def test_retrieve_topk_matches_reference_mixed_batch(self):
        algo = self._build_quest_algo(bs=6, max_seq_len=512, page_size=16)
        queries = torch.randn(6, 4, 8)
        req_pool_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        sparse_mask = torch.tensor([True, False, True, True, True, False])
        # Includes: active heterogeneous lengths, masked-out row, and num_pages <= num_recent_pages
        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([256, 300, 64, 480, 144, 16], dtype=torch.int64)
        )

        ref_indices, ref_lengths = _reference_retrieve_topk(
            algo,
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
        )
        got_indices, got_lengths = algo.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
        )

        self.assertTrue(torch.equal(got_lengths, ref_lengths))
        self.assertTrue(torch.equal(got_indices, ref_indices))

    def test_retrieve_topk_all_inactive_returns_default_padding(self):
        algo = self._build_quest_algo(bs=4, max_seq_len=128, page_size=16)
        queries = torch.randn(4, 4, 8)
        req_pool_indices = torch.arange(4, dtype=torch.int64)
        sparse_mask = torch.tensor([False, True, False, True])
        # Active rows have num_pages <= num_recent_pages (4)
        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([128, 48, 128, 64], dtype=torch.int64)
        )

        got_indices, got_lengths = algo.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
        )

        self.assertEqual(got_indices.shape, (4, 1))
        self.assertTrue(torch.all(got_indices == -1))
        self.assertTrue(torch.all(got_lengths == 0))

    def test_retrieve_topk_constant_sync_count_across_batch_sizes(self):
        for bs in (1, 8, 32):
            algo = self._build_quest_algo(bs=bs, max_seq_len=256, page_size=16)
            queries = torch.randn(bs, 4, 8)
            req_pool_indices = torch.arange(bs, dtype=torch.int64)
            sparse_mask = torch.ones(bs, dtype=torch.bool)
            forward_batch = SimpleNamespace(
                seq_lens=torch.full((bs,), 256, dtype=torch.int64)
            )

            with _SyncCounterMode() as counter:
                algo.retrieve_topk(
                    queries=queries,
                    layer_id=0,
                    req_pool_indices=req_pool_indices,
                    sparse_mask=sparse_mask,
                    forward_batch=forward_batch,
                )

            self.assertEqual(counter.counts["item"], 0)
            self.assertEqual(counter.counts["__bool__"], 0)
            self.assertEqual(counter.counts["tolist"], 1)


if __name__ == "__main__":
    unittest.main()
