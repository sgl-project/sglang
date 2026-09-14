"""The decode result loop skips ``Req.update_finish_state`` for requests that the
batch-level candidate test proves cannot finish this step. The skip must be exact:
whenever the pre-filter says "no check needed", running the real
``update_finish_state`` must leave the request untouched."""

import random
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    _finish_check_needed,
    spec_finish_candidates,
)
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EOS_ID = 2
STOP_ID = 1
VOCAB = 10_000
ID_TO_TEXT = {
    STOP_ID: "STOP",
    EOS_ID: "",
    **{i: chr(ord("a") + i % 26) for i in range(10, 10_000)},
}


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(ID_TO_TEXT.get(int(i), "?") for i in ids)


class _MockTokenizerForNormalize:
    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))


def _make_req(
    output_ids, *, max_new_tokens, stop=None, eos_token_ids=frozenset({EOS_ID})
):
    sp = SamplingParams(max_new_tokens=max_new_tokens, stop=stop)
    sp.normalize(tokenizer=_MockTokenizerForNormalize())
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=set(eos_token_ids),
        vocab_size=VOCAB,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


def _snapshot(req):
    return (
        req.finished_reason,
        req.finished_len,
        list(req.output_ids),
        req.to_finish,
    )


class TestSpecFinishCandidates(CustomTestCase):
    def test_padded_rows_beyond_accept_len_are_ignored(self):
        # bs=2, stride=4; the EOS sits in a padded slot of row 0 and in an
        # accepted slot of row 1.
        ids = torch.tensor([10, 11, EOS_ID, 12, 13, EOS_ID, 14, 15], dtype=torch.int64)
        accept = torch.tensor([2, 3], dtype=torch.int64)
        out = spec_finish_candidates(
            ids, accept, stride=4, candidate_tokens={EOS_ID}, vocab_size=VOCAB
        )
        self.assertEqual(out, [False, True])

    def test_out_of_vocab_and_negative_tokens_are_candidates(self):
        ids = torch.tensor([10, VOCAB, 11, -1], dtype=torch.int64)
        accept = torch.tensor([2, 2], dtype=torch.int64)
        self.assertEqual(
            spec_finish_candidates(
                ids, accept, stride=2, candidate_tokens=set(), vocab_size=VOCAB
            ),
            [True, True],
        )
        # Without a vocab size only negative ids count as out of range.
        self.assertEqual(
            spec_finish_candidates(
                ids, accept, stride=2, candidate_tokens=set(), vocab_size=None
            ),
            [False, True],
        )

    def test_no_candidates_no_hits(self):
        ids = torch.arange(10, 22, dtype=torch.int64)
        accept = torch.tensor([4, 4, 4], dtype=torch.int64)
        self.assertEqual(
            spec_finish_candidates(
                ids, accept, stride=4, candidate_tokens=set(), vocab_size=VOCAB
            ),
            [False, False, False],
        )


class TestFinishCheckNeeded(CustomTestCase):
    def _plain(self, **kw):
        params = SimpleNamespace(stop_strs=[], stop_regex_strs=[], max_new_tokens=100)
        params.__dict__.update(kw.pop("params", {}))
        base = dict(
            to_finish=None, grammar=None, sampling_params=params, output_ids=[1] * 5
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_plain_request_without_candidate_skips(self):
        self.assertFalse(_finish_check_needed(self._plain(), False))

    def test_each_finish_route_forces_the_check(self):
        self.assertTrue(_finish_check_needed(self._plain(), True))
        self.assertTrue(_finish_check_needed(self._plain(to_finish=object()), False))
        self.assertTrue(_finish_check_needed(self._plain(grammar=object()), False))
        self.assertTrue(
            _finish_check_needed(self._plain(params={"stop_strs": ["x"]}), False)
        )
        self.assertTrue(
            _finish_check_needed(self._plain(params={"stop_regex_strs": ["x"]}), False)
        )
        self.assertTrue(
            _finish_check_needed(self._plain(params={"max_new_tokens": 5}), False)
        )


class TestSkipIsExact(CustomTestCase):
    """Property check against the real Req: when the pre-filter says the check is
    not needed, update_finish_state is a no-op; when a finish is possible, the
    pre-filter says the check is needed."""

    def _cases(self):
        rng = random.Random(1234)
        for _ in range(300):
            accept = rng.randint(1, 6)
            history = [rng.randint(10, 500) for _ in range(rng.randint(0, 12))]
            run = [rng.randint(10, 500) for _ in range(accept)]
            kind = rng.choice(["plain", "plain", "eos", "stop_str", "oov", "cap"])
            stop = None
            if kind == "eos":
                run[rng.randrange(accept)] = EOS_ID
            elif kind == "stop_str":
                run[rng.randrange(accept)] = STOP_ID
                stop = ["STOP"]
            elif kind == "oov":
                run[rng.randrange(accept)] = rng.choice([-1, VOCAB, VOCAB + 7])
            max_new = len(history) + accept + rng.randint(1, 4)
            if kind == "cap":
                max_new = len(history) + rng.randint(1, accept)
            yield history + run, accept, max_new, stop

    def test_skip_matches_update_finish_state(self):
        checked = skipped = 0
        for output_ids, accept, max_new, stop in self._cases():
            req = _make_req(output_ids, max_new_tokens=max_new, stop=stop)
            run = output_ids[-accept:]
            ids = torch.tensor(run + [0] * (6 - accept), dtype=torch.int64)
            cand = spec_finish_candidates(
                ids,
                torch.tensor([accept], dtype=torch.int64),
                stride=6,
                candidate_tokens={EOS_ID}
                | set(req.sampling_params.stop_token_ids or ()),
                vocab_size=VOCAB,
            )[0]
            needed = _finish_check_needed(req, cand)
            before = _snapshot(req)
            req.update_finish_state(accept)
            after = _snapshot(req)
            if not needed:
                skipped += 1
                self.assertEqual(before, after, f"skip changed state for {output_ids}")
                self.assertFalse(req.finished())
            else:
                checked += 1
        self.assertGreater(skipped, 50)
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()
