from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.tp_sharded_greedy import (
    can_use_tp_sharded_greedy,
    select_global_argmax_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class _ForwardMode:
    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False


def _batch(**overrides):
    sampling_info = SimpleNamespace(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        grammars=None,
        grammar_mask=None,
        logit_bias=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        return_sampling_masks=[False, False],
    )
    defaults = dict(
        sampling_info=sampling_info,
        return_logprob=False,
        top_logprobs_nums=[0, 0],
        token_ids_logprobs=[None, None],
        is_prefill_only=False,
        spec_info=None,
        forward_mode=_ForwardMode(),
        next_token_logits_buffer=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_select_global_candidates_tie_uses_lowest_token_id():
    values = torch.tensor([[4.0, -torch.inf], [4.0, -torch.inf]])
    token_ids = torch.tensor([[17, 9], [3, 2]], dtype=torch.int32)
    assert torch.equal(
        select_global_argmax_candidates(values, token_ids),
        torch.tensor([3, 2], dtype=torch.int32),
    )


def test_select_global_candidates_per_row_winners():
    values = torch.tensor([[1.0, 8.0, -2.0], [2.0, 7.0, 5.0]])
    token_ids = torch.tensor([[1, 2, 3], [10, 11, 12]], dtype=torch.int32)
    assert torch.equal(
        select_global_argmax_candidates(values, token_ids),
        torch.tensor([10, 2, 12], dtype=torch.int32),
    )


def test_select_global_candidates_nan_matches_first_global_argmax():
    values = torch.tensor([[1.0, torch.nan], [torch.nan, torch.nan]])
    token_ids = torch.tensor([[2, 3], [11, 7]], dtype=torch.int32)
    assert torch.equal(
        select_global_argmax_candidates(values, token_ids),
        torch.tensor([11, 3], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: setattr(b, "return_logprob", True),
        lambda b: setattr(b, "top_logprobs_nums", [1, 0]),
        lambda b: setattr(b, "token_ids_logprobs", [[7], None]),
        lambda b: setattr(b, "spec_info", object()),
        lambda b: setattr(b, "next_token_logits_buffer", torch.empty(1)),
        lambda b: setattr(b.sampling_info, "is_all_greedy", False),
        lambda b: setattr(b.sampling_info, "has_custom_logit_processor", True),
        lambda b: setattr(b.sampling_info, "grammars", [object(), None]),
        lambda b: setattr(b.sampling_info, "grammar_mask", object()),
        lambda b: setattr(b.sampling_info, "logit_bias", torch.empty(1)),
        lambda b: setattr(b.sampling_info, "acc_additive_penalties", torch.empty(1)),
        lambda b: setattr(b.sampling_info, "acc_scaling_penalties", torch.empty(1)),
        lambda b: setattr(b.sampling_info.penalizer_orchestrator, "is_required", True),
        lambda b: setattr(b.sampling_info, "return_sampling_masks", [False, True]),
    ],
)
def test_gate_falls_back_for_full_logits_features(mutation):
    batch = _batch()
    mutation(batch)
    with envs.SGLANG_ENABLE_TP_SHARDED_GREEDY.override(True):
        assert not can_use_tp_sharded_greedy(batch)


def test_gate_requires_explicit_opt_in():
    batch = _batch()
    with envs.SGLANG_ENABLE_TP_SHARDED_GREEDY.override(False):
        assert not can_use_tp_sharded_greedy(batch)
    with envs.SGLANG_ENABLE_TP_SHARDED_GREEDY.override(True):
        assert can_use_tp_sharded_greedy(batch)
