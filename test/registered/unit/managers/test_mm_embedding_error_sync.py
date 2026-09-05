from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.utils import (
    complete_mm_embedding_validations,
    has_mm_embedding_failures,
    merge_mm_embedding_error_tensors,
    synchronize_mm_embedding_errors,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu-intel")


def _single_rank_parallel():
    group = SimpleNamespace(world_size=1, device_group=None)
    return SimpleNamespace(attn_tp_group=group, attn_cp_group=group)


def test_valid_mm_validation_allows_early_publication_and_completes():
    with patch(
        "sglang.srt.managers.utils.get_parallel", return_value=_single_rank_parallel()
    ):
        result = synchronize_mm_embedding_errors(None, 2, torch.device("cpu"), [0])

    assert result is not None
    assert not has_mm_embedding_failures(result)
    reqs = [
        SimpleNamespace(mm_embedding_validation_count=1),
        SimpleNamespace(mm_embedding_validation_count=0),
    ]
    complete_mm_embedding_validations(reqs, result)
    assert [req.mm_embedding_validation_count for req in reqs] == [0, 0]


def test_failed_mm_validation_blocks_early_publication():
    with patch(
        "sglang.srt.managers.utils.get_parallel", return_value=_single_rank_parallel()
    ):
        result = synchronize_mm_embedding_errors(
            [(1, 3, 2)], 2, torch.device("cpu"), [0, 1]
        )

    assert has_mm_embedding_failures(result)


def test_pp_merge_preserves_request_scoped_failure_and_validations():
    incoming = torch.tensor([[1, 0, 0, 0], [0, 1, 3, 2]])
    local = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]])

    result = merge_mm_embedding_error_tensors(incoming, local)

    assert torch.equal(result, torch.tensor([[1, 0, 0, 0], [1, 1, 3, 2]]))
