from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.prompt_lookup import GreedyPromptLookup, PromptSuffixIndex
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def test_lookup_requires_exact_suffix_and_complete_proposal():
    index = PromptSuffixIndex(list(range(30)))
    assert index.propose() is None  # No tokens beyond prompt end.
    index.suffix = list(range(8))
    assert index.propose() == (8, 9, 10)
    index.accept([99])
    assert index.propose() is None


def test_verified_history_overrides_rejected_proposals():
    index = PromptSuffixIndex(list(range(50)))
    index.suffix = list(range(8))
    assert index.propose() == (8, 9, 10)
    # Accept one draft then a target correction. Rejected drafts never enter history.
    index.accept([8, 42])
    assert index.suffix == [2, 3, 4, 5, 6, 7, 8, 42]
    assert index.propose() is None


@pytest.mark.parametrize(
    "reason", ["sampling", "grammar", "multimodal", "batch", "idle"]
)
def test_unsupported_requests_fall_back_without_reading_gpu(reason):
    lookup = GreedyPromptLookup.__new__(GreedyPromptLookup)
    lookup.pending = lookup.request = lookup.index = None
    lookup.hits = lookup.attempts = 0
    req = SimpleNamespace(
        multimodal_inputs=object() if reason == "multimodal" else None
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: reason == "idle"),
        reqs=[req] * (2 if reason == "batch" else 1),
        sampling_info=SimpleNamespace(is_all_greedy=reason != "sampling"),
        has_grammar=reason == "grammar",
    )
    assert lookup.draft(batch, None) is None


def test_last_matching_occurrence_is_used():
    prompt = list(range(8)) + [11, 12, 13] + list(range(8)) + [21, 22, 23]
    index = PromptSuffixIndex(prompt)
    index.suffix = list(range(8))
    assert index.propose() == (21, 22, 23)


def test_request_opt_out_does_not_read_bonus_or_track_history():
    lookup = GreedyPromptLookup.__new__(GreedyPromptLookup)
    lookup.pending = lookup.request = lookup.index = None
    lookup.hits = lookup.attempts = 0
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                multimodal_inputs=None,
                sampling_params=SimpleNamespace(custom_params={"prompt_lookup": False}),
            )
        ],
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        has_grammar=False,
    )
    assert lookup.draft(batch, None) is None
    assert lookup.request is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_history_uses_verified_tokens_and_resets_requests():
    lookup = GreedyPromptLookup(3, "cuda")
    req = SimpleNamespace(
        rid="one",
        origin_input_ids=list(range(40)),
        multimodal_inputs=None,
        sampling_params=SimpleNamespace(custom_params=None),
    )
    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        has_grammar=False,
    )
    lookup.draft(batch, torch.tensor([0], dtype=torch.int32, device="cuda"))
    lookup.observe(
        batch,
        torch.tensor([[1, 2, 999, 999]], dtype=torch.int32, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda"),
    )
    lookup._drain()
    assert lookup.index.suffix[-3:] == [0, 1, 2]
    assert 999 not in lookup.index.suffix
    req2 = SimpleNamespace(
        rid="two",
        origin_input_ids=list(range(100, 140)),
        multimodal_inputs=None,
        sampling_params=SimpleNamespace(custom_params=None),
    )
    batch.reqs = [req2]
    lookup.draft(batch, torch.tensor([140], dtype=torch.int32, device="cuda"))
    assert lookup.index.suffix == list(range(133, 141))
