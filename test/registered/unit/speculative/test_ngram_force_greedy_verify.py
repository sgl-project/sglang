"""The NGRAM greedy-verify flag must reach the verify branch predicate.

SGLANG_NGRAM_FORCE_GREEDY_VERIFY stayed registered and documented, but the
NGRAM refactor onto the shared eagle_sample helper dropped the read: the
greedy predicate only saw sampling_info.is_all_greedy. eagle_sample takes a
force_greedy_verify override that the NGRAM worker feeds from the flag, so
these tests pin the plumbing at the predicate seam. EAGLE callers never pass
it and are unchanged.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PredicateHit(Exception):
    pass


def _capture_predicate_kwargs(monkeypatch, *, is_all_greedy, force_greedy_verify):
    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        raise _PredicateHit

    monkeypatch.setattr(eagle_utils, "_verify_uses_greedy", spy)
    monkeypatch.setattr(
        eagle_utils,
        "get_spec",
        lambda: SimpleNamespace(speculative_use_rejection_sampling=False),
    )

    bs, draft_token_num, vocab = 1, 2, 16
    batch = SimpleNamespace(
        device=torch.device("cpu"),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=[4],
        sampling_info=SimpleNamespace(
            is_all_greedy=is_all_greedy,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            has_watermark_candidates=False,
        ),
    )
    verify_input = SimpleNamespace(
        draft_token=torch.zeros(bs * draft_token_num, dtype=torch.int64),
        draft_token_num=draft_token_num,
        max_tree_depth=3,
    )
    logits_output = SimpleNamespace(
        next_token_logits=torch.zeros(bs * draft_token_num, vocab)
    )

    with pytest.raises(_PredicateHit):
        eagle_utils.eagle_sample(
            verify_input,
            batch,
            logits_output,
            force_greedy_verify=force_greedy_verify,
        )
    return captured


def test_force_greedy_verify_reaches_predicate(monkeypatch):
    captured = _capture_predicate_kwargs(
        monkeypatch, is_all_greedy=False, force_greedy_verify=True
    )
    assert captured["is_all_greedy"] is True


def test_default_keeps_sampling_predicate(monkeypatch):
    captured = _capture_predicate_kwargs(
        monkeypatch, is_all_greedy=False, force_greedy_verify=False
    )
    assert captured["is_all_greedy"] is False


def test_all_greedy_batch_unaffected_by_default(monkeypatch):
    captured = _capture_predicate_kwargs(
        monkeypatch, is_all_greedy=True, force_greedy_verify=False
    )
    assert captured["is_all_greedy"] is True
