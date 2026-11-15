"""Speculative decoding must keep the sampling penalty state lossless.

Two defects are covered:

A. ``ScheduleBatch.cumulate_penalty_output_tokens`` fed only
   ``output_ids[-1]``, so every accepted token of a speculative step except the
   last never entered the repetition / presence / frequency state.
B. ``eagle_sample`` repeated the committed-prefix penalty row to every draft
   position, so a draft position was not penalised for the tokens on its own
   path.

CPU only.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")


# ---------------------------------------------------------------------------
# A: every committed token must reach the penalizers
# ---------------------------------------------------------------------------


class _Algo:
    """Stand-in for SpeculativeAlgorithm; only is_none() is consulted."""

    def __init__(self, none):
        self._none = none

    def is_none(self):
        return self._none


def _a_fake_self(reqs, spec=True):
    calls = []

    def _rec(tokens, rows):
        calls.append((tokens.tolist(), rows.tolist()))

    def _rec_single(tokens):
        calls.append((tokens.tolist(), None))

    orch = SimpleNamespace(
        cumulate_output_tokens_ragged=_rec,
        cumulate_output_tokens=_rec_single,
    )
    fake = SimpleNamespace(
        reqs=reqs,
        device=torch.device("cpu"),
        spec_algorithm=_Algo(none=not spec),
        sampling_info=SimpleNamespace(penalizer_orchestrator=orch),
    )
    return fake, calls


def _a_run(reqs, spec=True):
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    fake, calls = _a_fake_self(reqs, spec=spec)
    ScheduleBatch.cumulate_penalty_output_tokens(fake)
    return calls


def _a_req(origin, output=None, cursor=0):
    return SimpleNamespace(
        origin_input_ids=list(origin),
        output_ids=list(output or []),
        penalty_cumulated_len=cursor,
    )


def test_first_decode_seeds_prompt_tail():
    req = _a_req([10, 11, 12])
    calls = _a_run([req])
    assert calls == [([12], [0])], calls


def test_all_committed_tokens_are_fed():
    req = _a_req([10], [20, 21, 22])
    calls = _a_run([req])
    assert calls == [([20, 21, 22], [0, 0, 0])], calls
    assert req.penalty_cumulated_len == 3


def test_cursor_prevents_double_counting():
    req = _a_req([10], [20, 21], cursor=0)
    _a_run([req])
    calls = _a_run([req])
    assert calls == [], f"second call must feed nothing, got {calls}"


def test_new_tokens_only_after_cursor():
    req = _a_req([10], [20, 21, 22, 23], cursor=2)
    calls = _a_run([req])
    assert calls == [([22, 23], [0, 0])], calls


def test_retraction_does_not_overflow():
    # output_ids shrank below the cursor (retraction); must not raise.
    req = _a_req([10], [20, 21], cursor=5)
    calls = _a_run([req])
    assert calls == [], calls


def test_mixed_batch_rows():
    reqs = [
        _a_req([7], []),
        _a_req([8], [30, 31]),
        _a_req([9], [40], cursor=1),
    ]
    calls = _a_run(reqs)
    assert calls == [([7, 30, 31], [0, 1, 1])], calls


def test_non_spec_path_is_unchanged():
    # Ordinary decoding commits one token per step; it must keep using the
    # original single-token call so the decode hot path is untouched.
    req = _a_req([10], [20, 21, 22])
    calls = _a_run([req], spec=False)
    assert calls == [([22], None)], calls
    assert req.penalty_cumulated_len == 0


# ---------------------------------------------------------------------------
# Ragged cumulation must equal token-by-token cumulation, all penalizers
# ---------------------------------------------------------------------------

VOCAB_M = 32


def _m_req(rp=1.2, pp=0.5, fp=0.5, mnt=0):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            repetition_penalty=rp,
            presence_penalty=pp,
            frequency_penalty=fp,
            min_new_tokens=mnt,
            stop_token_ids={1},
        ),
        eos_token_ids=set(),
        tokenizer=SimpleNamespace(additional_stop_token_ids=set(), eos_token_id=None),
    )


class _Orch:
    """BatchedPenalizerOrchestrator stand-in.

    Must be a real class: _BatchedPenalizer.__init__ calls weakref.ref() on it,
    and SimpleNamespace is not weak-referenceable.
    """

    def __init__(self, reqs):
        self._reqs = reqs
        self.vocab_size = VOCAB_M
        self.device = torch.device("cpu")

    def reqs(self):
        return self._reqs


def _m_orch(n):
    reqs = [_m_req() for _ in range(n)]
    return _Orch(reqs), reqs


def _pair(cls, n=1):
    o1, r1 = _m_orch(n)
    o2, r2 = _m_orch(n)
    p1, p2 = cls(o1), cls(o2)
    p1.prepare()
    p2.prepare()
    # keep orchestrators alive (weakref)
    return (p1, o1, r1), (p2, o2, r2)


def _assert_same(attr, cls, steps):
    """steps: list of per-step token lists, one entry per batch row."""
    n = len(steps[0])
    (p1, o1, r1), (p2, o2, r2) = _pair(cls, n)
    for step in steps:
        p1._cumulate_output_tokens(torch.tensor(step, dtype=torch.long))

    tokens, rows = [], []
    for row in range(n):
        for step in steps:
            tokens.append(step[row])
            rows.append(row)
    p2._cumulate_output_tokens_ragged(
        torch.tensor(tokens, dtype=torch.long), torch.tensor(rows, dtype=torch.long)
    )
    a = getattr(p1, attr)
    b = getattr(p2, attr)
    assert torch.equal(a, b), f"{attr} differs\nseq=\n{a}\nragged=\n{b}"
    # hold weakref targets alive until the assertions are done
    assert o1 is not None and o2 is not None and r1 is not None and r2 is not None


def test_repetition_ragged_matches_sequential():
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        BatchedRepetitionPenalizer,
    )

    _assert_same(
        "cumulated_repetition_penalties",
        BatchedRepetitionPenalizer,
        [[3, 3], [3, 5], [7, 5], [7, 7]],
    )


def test_presence_ragged_matches_sequential():
    from sglang.srt.sampling.penaltylib.presence_penalty import (
        BatchedPresencePenalizer,
    )

    _assert_same(
        "cumulated_presence_penalties",
        BatchedPresencePenalizer,
        [[3, 3], [3, 5], [7, 5]],
    )


def test_frequency_ragged_matches_sequential():
    from sglang.srt.sampling.penaltylib.frequency_penalty import (
        BatchedFrequencyPenalizer,
    )

    # repeated token 3 must accumulate 3 times on row 0
    _assert_same(
        "cumulated_frequency_penalties",
        BatchedFrequencyPenalizer,
        [[3, 3], [3, 5], [3, 5]],
    )


def test_min_new_tokens_ragged_counts_tokens():
    from sglang.srt.sampling.penaltylib.min_new_tokens import (
        BatchedMinNewTokensPenalizer,
    )

    o1, r1 = _m_orch(1)
    o2, r2 = _m_orch(1)
    for r in r1 + r2:
        r.sampling_params.min_new_tokens = 8
    p1, p2 = BatchedMinNewTokensPenalizer(o1), BatchedMinNewTokensPenalizer(o2)
    p1.prepare()
    p2.prepare()

    # one spec step committing 4 tokens
    p1._cumulate_output_tokens(torch.tensor([3]))
    p1._cumulate_output_tokens(torch.tensor([4]))
    p1._cumulate_output_tokens(torch.tensor([5]))
    p1._cumulate_output_tokens(torch.tensor([6]))
    p2._cumulate_output_tokens_ragged(
        torch.tensor([3, 4, 5, 6]), torch.tensor([0, 0, 0, 0])
    )
    assert p1.len_output_tokens.item() == 4
    assert torch.equal(p1.len_output_tokens, p2.len_output_tokens)


# ---------------------------------------------------------------------------
# B: every draft position must get the penalty row of its own context
# ---------------------------------------------------------------------------

RP = 1.2
VOCAB_B = 16


def _child_lists(parent: torch.Tensor):
    """parent[bs, n], -1 or self for the root -> (next_token, next_sibling)."""
    bs, n = parent.shape
    nxt = torch.full((bs, n), -1, dtype=torch.long)
    sib = torch.full((bs, n), -1, dtype=torch.long)
    last = [dict() for _ in range(bs)]
    for b in range(bs):
        for i in range(n):
            p = int(parent[b, i])
            if p < 0 or p == i:
                continue
            if p in last[b]:
                sib[b, last[b][p]] = i
            else:
                nxt[b, p] = i
            last[b][p] = i
    return nxt, sib


def _vi(draft_token, parent, n, spec_steps=None):
    draft = torch.tensor(draft_token, dtype=torch.long)
    bs = draft.numel() // n
    if parent is None:
        next_token = next_sibling = None
        index = torch.zeros((bs, n), dtype=torch.long)
    else:
        p = torch.tensor(parent, dtype=torch.long)
        next_token, next_sibling = _child_lists(p)
        index = torch.arange(n, dtype=torch.long).expand(bs, n).clone()
    return SimpleNamespace(
        draft_token=draft,
        draft_token_num=n,
        retrieve_index=index,
        retrieve_next_token=next_token,
        retrieve_next_sibling=next_sibling,
        spec_steps=spec_steps,
    )


def _b_run(vi, bs):
    from sglang.srt.speculative.eagle_utils import add_draft_prefix_penalties

    penalties = torch.ones(bs * vi.draft_token_num, VOCAB_B)
    rp = torch.full((bs, 1), RP)
    add_draft_prefix_penalties(penalties, vi, rp)
    return penalties


def _expected(bs, n, paths):
    """paths[bs][n] = set of token ids that must carry RP on that row."""
    out = torch.ones(bs * n, VOCAB_B)
    for r in range(bs):
        for node in range(n):
            for tok in paths[r][node]:
                out[r * n + node, tok] = RP
    return out


def test_single_request_tree():
    # 0 root -> 1,3 ; 1 -> 2
    vi = _vi([7, 8, 7, 9], [[-1, 0, 1, 0]], n=4, spec_steps=3)
    got = _b_run(vi, bs=1)
    want = _expected(1, 4, [[{7}, {7, 8}, {7, 8}, {7, 9}]])
    assert torch.equal(got, want), f"\ngot=\n{got}\nwant=\n{want}"


def test_flat_draft_token_layout():
    # draft_token handed over flat, row-major: two requests, 3 nodes each.
    vi = _vi([1, 2, 3, 1, 1, 4], [[-1, 0, 0], [-1, 0, 1]], n=3, spec_steps=2)
    got = _b_run(vi, bs=2)
    want = _expected(
        2,
        3,
        [
            [{1}, {1, 2}, {1, 3}],
            [{1}, {1}, {1, 4}],
        ],
    )
    assert torch.equal(got, want), f"\ngot=\n{got}\nwant=\n{want}"


def test_repeated_token_on_path_is_penalised():
    # node0 and node1 both "the": row 1 must scale "the".
    vi = _vi([5, 5], [[-1, 0]], n=2, spec_steps=1)
    got = _b_run(vi, bs=1)
    # compare against the float32 value actually stored, not the Python 1.2
    assert got[0, 5] == torch.tensor(RP, dtype=got.dtype)
    assert got[1, 5] == torch.tensor(RP, dtype=got.dtype)


def test_idempotent():
    vi = _vi([7, 8, 7, 9], [[-1, 0, 1, 0]], n=4, spec_steps=3)
    a = _b_run(vi, bs=1)
    b = _b_run(vi, bs=1)
    assert torch.equal(a, b)


def test_no_parent_marks_self_only():
    # When the child lists are absent the chain cannot be walked, so only the
    # node's own token is marked. (NGRAM without the fields degrades this way.)
    vi = _vi([1, 2, 3], None, n=3, spec_steps=None)
    got = _b_run(vi, bs=1)
    want = _expected(1, 3, [[{1}, {2}, {3}]])
    assert torch.equal(got, want), f"\ngot=\n{got}\nwant=\n{want}"


def test_padding_ids_are_skipped():
    # Unused tree slots carry -1 / padding; those must not become column
    # indices (negative wraps, >= vocab aborts the kernel).
    vi = _vi([1, -1, 2], [[-1, 0, 0]], n=3, spec_steps=1)
    got = _b_run(vi, bs=1)
    want = _expected(1, 3, [[{1}, {1}, {1, 2}]])
    assert torch.equal(got, want), f"\ngot=\n{got}\nwant=\n{want}"


def test_none_penalties_is_noop():
    from sglang.srt.speculative.eagle_utils import add_draft_prefix_penalties

    vi = _vi([1, 2], [[-1, 0]], n=2, spec_steps=1)
    penalties = torch.ones(2, VOCAB_B)
    add_draft_prefix_penalties(penalties, vi, None)
    assert torch.equal(penalties, torch.ones(2, VOCAB_B))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
