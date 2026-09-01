from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
    load_optional_quest_kernel,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ScoreAlgorithm(BaseSparseAlgorithmImpl):
    def _retrieve_page_scores(self, layer_id, phys_pages, req_pool_indices, queries):
        del layer_id, req_pool_indices, queries
        self.score_calls = getattr(self, "score_calls", 0) + 1
        return phys_pages.to(torch.float32)

    def _finalize_topk_with_recent(self, topk_scores, topk_idx, plan, **kwargs):
        self.finalize_calls = getattr(self, "finalize_calls", 0) + 1
        return super()._finalize_topk_with_recent(topk_scores, topk_idx, plan, **kwargs)


def _config(**extra):
    return SimpleNamespace(
        page_size=4,
        sparse_extra_config={
            "sparsity_ratio": 0.5,
            "num_recent_pages": 2,
            **extra,
        },
    )


def _score_algorithm(batch_size: int) -> _ScoreAlgorithm:
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int64).repeat(batch_size, 1)
    )
    return algorithm


def _reference_selected(seq_len: int, sparse: bool) -> list[int]:
    num_pages = (seq_len + 3) // 4
    if not sparse or num_pages <= 2:
        return []
    history_pages = num_pages - 2
    k = min(max(int(history_pages * 0.5), 1), history_pages)
    return list(range(history_pages - k, num_pages))


def test_batched_retrieval_matches_ragged_per_request_reference():
    algorithm = _score_algorithm(5)
    seq_lens = torch.tensor([40, 12, 28, 36, 8], dtype=torch.int64)
    sparse_mask = torch.tensor([True, True, False, True, True])
    forward_batch = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.tolist(),
        sparse_mask_cpu=sparse_mask.tolist(),
    )

    selected, lengths = algorithm.retrieve_topk(
        queries=torch.zeros((5, 1)),
        layer_id=0,
        req_pool_indices=torch.arange(5),
        sparse_mask=sparse_mask,
        forward_batch=forward_batch,
    )

    for row, (seq_len, sparse) in enumerate(zip(seq_lens.tolist(), sparse_mask)):
        expected = _reference_selected(seq_len, bool(sparse))
        length = int(lengths[row])
        assert length == len(expected)
        assert selected[row, :length].tolist() == expected
        assert (selected[row, length:] == -1).all()
    assert algorithm.finalize_calls == 1


def test_single_request_uses_host_seq_mirror_and_exact_host_mask():
    algorithm = _score_algorithm(1)
    seq_lens = torch.tensor([40], dtype=torch.int64)
    forward_batch = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=[40],
        sparse_mask_cpu=[True],
    )

    with patch.object(
        algorithm,
        "_build_topk_plan",
        side_effect=AssertionError("single-request path built a batched plan"),
    ):
        selected, lengths = algorithm.retrieve_topk(
            queries=torch.zeros((1, 1)),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([True]),
            forward_batch=forward_batch,
        )

    assert lengths.tolist() == [6]
    assert selected.tolist() == [[4, 5, 6, 7, 8, 9]]

    forward_batch.sparse_mask_cpu = [False]
    selected, lengths = algorithm.retrieve_topk(
        queries=torch.zeros((1, 1)),
        layer_id=0,
        req_pool_indices=torch.tensor([0]),
        sparse_mask=torch.tensor([False]),
        forward_batch=forward_batch,
    )
    assert algorithm.score_calls == 1
    assert lengths.tolist() == [0]
    assert selected.tolist() == [[-1]]


def test_single_request_without_host_seq_mirror_uses_batched_fallback():
    algorithm = _score_algorithm(1)
    seq_lens = torch.tensor([40], dtype=torch.int64)

    with patch.object(
        algorithm,
        "_retrieve_topk_single",
        side_effect=AssertionError("single-request path read device seq_lens"),
    ):
        selected, lengths = algorithm.retrieve_topk(
            queries=torch.zeros((1, 1)),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([True]),
            forward_batch=SimpleNamespace(seq_lens=seq_lens),
        )

    assert lengths.tolist() == [6]
    assert selected.tolist() == [[4, 5, 6, 7, 8, 9]]


def test_missing_host_mask_never_reads_a_device_scalar():
    algorithm = _score_algorithm(1)
    seq_lens = torch.tensor([40], dtype=torch.int64)
    forward_batch = SimpleNamespace(seq_lens=seq_lens, seq_lens_cpu=[40])

    with (
        patch.object(algorithm, "_get_bool_mask_cpu", return_value=None),
        patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("device sparse_mask scalar was read"),
        ),
    ):
        selected, lengths = algorithm.retrieve_topk(
            queries=torch.zeros((1, 1)),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([False]),
            forward_batch=forward_batch,
        )

    assert algorithm.score_calls == 1
    assert lengths.tolist() == [0]
    assert (selected == -1).all()


def test_full_page_fast_path_matches_reference():
    algorithm = QuestAlgorithm(_config(), torch.device("cpu"))
    assert algorithm.enable_cuda_graph_retrieval
    assert algorithm.supports_fixed_cuda_graph_capacity
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(12, dtype=torch.int64).unsqueeze(0)
    )
    algorithm.page_k_min[0] = torch.zeros((3, 1, 2), dtype=torch.float32)
    algorithm.page_k_max[0] = torch.zeros((3, 1, 2), dtype=torch.float32)
    algorithm.page_valid[0] = torch.zeros(3, dtype=torch.bool)
    keys = torch.arange(24, dtype=torch.float32).reshape(12, 1, 2)

    algorithm._compute_page_representations(
        0, torch.tensor([0]), torch.tensor([12]), 0, torch.tensor([3]), keys
    )
    expected = keys.reshape(3, 4, 1, 2)
    torch.testing.assert_close(algorithm.page_k_min[0], expected.amin(dim=1))
    torch.testing.assert_close(algorithm.page_k_max[0], expected.amax(dim=1))


def test_host_page_boundary_skip_is_conservative_for_multi_token_decode():
    algorithm = _score_algorithm(1)

    single_token = SimpleNamespace(
        seq_lens_cpu=[17],
        num_token_non_padded_cpu=1,
        positions=torch.tensor([16]),
        spec_info=None,
    )
    assert not algorithm.should_update_representations(single_token)

    page_boundary = SimpleNamespace(
        seq_lens_cpu=[16],
        num_token_non_padded_cpu=1,
        positions=torch.tensor([15]),
        spec_info=None,
    )
    assert algorithm.should_update_representations(page_boundary)

    multi_token = SimpleNamespace(
        seq_lens_cpu=[18],
        num_token_non_padded_cpu=3,
        positions=torch.tensor([15, 16, 17]),
        spec_info=object(),
    )
    assert algorithm.should_update_representations(multi_token)


def test_fixed_capacity_plan_is_stable_and_skips_dynamic_single_request_path():
    algorithm = _score_algorithm(1)
    algorithm.cuda_graph_max_num_pages = 8

    plans = [
        algorithm._build_topk_plan(
            SimpleNamespace(seq_lens=torch.tensor([seq_len])),
            torch.tensor([0]),
            torch.tensor([True]),
            torch.device("cpu"),
            fixed_capacity=64,
        )
        for seq_len in (12, 28)
    ]
    assert [plan.max_num_pages for plan in plans] == [8, 8]
    assert [plan.max_k for plan in plans] == [3, 3]
    assert [plan.physical_pages.shape for plan in plans] == [(1, 8), (1, 8)]
    assert all(plan.fixed_capacity for plan in plans)

    forward_batch = SimpleNamespace(seq_lens=torch.tensor([28]))
    algorithm.begin_forward(
        forward_batch,
        torch.tensor([0]),
        torch.tensor([True]),
        torch.device("cpu"),
        fixed_capacity=64,
    )
    with patch.object(
        algorithm,
        "_retrieve_topk_single",
        side_effect=AssertionError("fixed path used dynamic single-request retrieval"),
    ):
        _, lengths = algorithm.retrieve_topk(
            queries=torch.zeros((1, 1)),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([True]),
            forward_batch=forward_batch,
        )

    assert algorithm._topk_plan.max_num_pages == 8
    assert lengths.tolist() == [4]

    algorithm.begin_forward(
        forward_batch,
        torch.tensor([0]),
        torch.tensor([True]),
        torch.device("cpu"),
    )
    assert algorithm._topk_plan is None


def test_optional_kernel_discovery_falls_back_but_does_not_hide_import_errors():
    module = "sglang.srt.mem_cache.sparsity.kernels.quest_missing"
    assert (
        load_optional_quest_kernel(
            "sglang.srt.mem_cache.sparsity.missing_parent.quest_kernel"
        )
        is None
    )
    with patch(
        "sglang.srt.mem_cache.sparsity.algorithms.base_algorithm.find_spec",
        return_value=None,
    ):
        assert load_optional_quest_kernel(module) is None

    with (
        patch(
            "sglang.srt.mem_cache.sparsity.algorithms.base_algorithm.find_spec",
            return_value=object(),
        ),
        patch(
            "sglang.srt.mem_cache.sparsity.algorithms.base_algorithm.import_module",
            side_effect=ImportError("broken provider"),
        ),
        pytest.raises(ImportError, match="broken provider"),
    ):
        load_optional_quest_kernel(module)


def test_present_score_provider_is_dispatched():
    algorithm = QuestAlgorithm(_config(), torch.device("cpu"))
    algorithm.page_k_min[0] = torch.zeros((2, 1, 2))
    algorithm.page_k_max[0] = torch.ones((2, 1, 2))
    algorithm.page_valid[0] = torch.ones(2, dtype=torch.bool)
    expected = torch.tensor([[1.0, 2.0]])
    score = Mock(return_value=expected)
    plan = SimpleNamespace(
        physical_pages=torch.tensor([[0, 1]]),
        active_mask=torch.tensor([True]),
        recent_start=torch.tensor([2]),
    )

    with patch.object(
        algorithm,
        "_optional_score_kernel",
        return_value=SimpleNamespace(quest_page_scores=score),
    ):
        result = algorithm._retrieve_page_scores_batched(
            0, torch.zeros((1, 1, 2)), plan
        )

    assert result is expected
    score.assert_called_once()


def test_direct_metadata_requires_explicit_runtime_opt_in():
    algorithm = _score_algorithm(1)
    plan = SimpleNamespace(
        max_k=1,
        max_num_pages=2,
        k_per_req=torch.tensor([1]),
        recent_idx=torch.tensor([[1]]),
        recent_valid=torch.tensor([[True]]),
    )
    topk_scores = torch.tensor([[1.0]])
    topk_idx = torch.tensor([[0]])
    prepared = (torch.tensor([[0, 1]]), torch.tensor([2]), True)

    with patch.object(
        algorithm, "_try_finalize_to_flashattention_metadata", return_value=prepared
    ) as direct:
        eager = algorithm._finalize_topk_with_recent(topk_scores, topk_idx, plan)
        direct.assert_not_called()
        captured = algorithm._finalize_topk_with_recent(
            topk_scores,
            topk_idx,
            plan,
            allow_prepared_metadata=True,
            attn_metadata=object(),
        )

    assert len(eager) == 2
    assert captured is prepared
