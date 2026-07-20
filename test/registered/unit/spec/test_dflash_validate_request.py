"""Unit tests for dflash_utils.validate_dflash_request grammar gating.

The shared DFLASH/DSPARK admission check gates grammar-constrained decoding
(json_schema / regex / ebnf / structural_tag). DSPARK enforces the grammar by
masking the target verify logits along its linear verify chain, so those
requests must be admitted; DFLASH has no such masking and must keep rejecting
them. The guarded failure mode is a predicate that degrades to always-reject
(DSPARK grammar silently 400s, losing the feature) or always-admit (DFLASH
grammar slips through and returns unconstrained output). Real SamplingParams
objects are used so the four grammar field names stay pinned to the request API.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_GRAMMAR_KINDS = (
    {"json_schema": '{"type": "object"}'},
    {"regex": "[0-9]+"},
    {"ebnf": 'root ::= "a"'},
    {"structural_tag": '{"type": "structural_tag"}'},
)

DFLASH = SpeculativeAlgorithm.from_string("DFLASH")
DSPARK = SpeculativeAlgorithm.from_string("DSPARK")


def _make_req(**sampling_kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SamplingParams(**sampling_kwargs),
        return_logprob=False,
        return_hidden_states=False,
    )


class TestValidateDflashRequestGrammarGating(CustomTestCase):
    def test_dspark_admits_every_grammar_kind(self):
        for kind in _GRAMMAR_KINDS:
            with self.subTest(grammar=next(iter(kind))):
                req = _make_req(**kind)
                self.assertIsNone(
                    validate_dflash_request(
                        req, enable_overlap=False, spec_algorithm=DSPARK
                    )
                )

    def test_dflash_rejects_every_grammar_kind(self):
        for kind in _GRAMMAR_KINDS:
            with self.subTest(grammar=next(iter(kind))):
                req = _make_req(**kind)
                error = validate_dflash_request(
                    req, enable_overlap=False, spec_algorithm=DFLASH
                )
                self.assertIsNotNone(error)
                self.assertIn("grammar", error.lower())

    def test_ungrammared_request_admitted_on_both(self):
        for algo in (DFLASH, DSPARK):
            with self.subTest(algo=algo):
                self.assertIsNone(
                    validate_dflash_request(
                        _make_req(), enable_overlap=False, spec_algorithm=algo
                    )
                )

    def test_non_grammar_rejections_survive_for_dspark(self):
        # The grammar carve-out must not swallow the earlier guards: DSPARK still
        # rejects return_logprob and (under overlap) return_hidden_states.
        logprob_req = _make_req()
        logprob_req.return_logprob = True
        self.assertIsNotNone(
            validate_dflash_request(
                logprob_req, enable_overlap=False, spec_algorithm=DSPARK
            )
        )

        hidden_req = _make_req()
        hidden_req.return_hidden_states = True
        self.assertIsNotNone(
            validate_dflash_request(
                hidden_req, enable_overlap=True, spec_algorithm=DSPARK
            )
        )


if __name__ == "__main__":
    unittest.main()
