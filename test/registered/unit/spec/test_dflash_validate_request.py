"""Admission gating for grammar-constrained requests on the DFLASH family.

The gate must track supports_grammar_overlap() rather than a hardcoded algorithm
list, so an algorithm that grows the verify-time bitmask does not need a second
edit here -- and one that has not grown it cannot silently admit requests it
would answer unconstrained. Real SamplingParams objects keep the four grammar
field names pinned to the request API.
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

_ALGORITHMS = (
    SpeculativeAlgorithm.from_string("DFLASH"),
    SpeculativeAlgorithm.from_string("DSPARK"),
)


def _make_req(**sampling_kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SamplingParams(**sampling_kwargs),
        return_logprob=False,
        return_hidden_states=False,
    )


class TestValidateDflashRequestGrammarGating(CustomTestCase):
    def test_grammar_admission_follows_capability(self):
        for algo in _ALGORITHMS:
            for kind in _GRAMMAR_KINDS:
                with self.subTest(algo=algo.name, grammar=next(iter(kind))):
                    error = validate_dflash_request(
                        _make_req(**kind), enable_overlap=False, spec_algorithm=algo
                    )
                    if algo.supports_grammar_overlap():
                        self.assertIsNone(error)
                    else:
                        self.assertIn("grammar", error.lower())

    def test_ungrammared_request_admitted(self):
        for algo in _ALGORITHMS:
            with self.subTest(algo=algo.name):
                self.assertIsNone(
                    validate_dflash_request(
                        _make_req(), enable_overlap=False, spec_algorithm=algo
                    )
                )

    def test_non_grammar_rejections_survive(self):
        # The grammar carve-out must not swallow the earlier guards.
        for algo in _ALGORITHMS:
            with self.subTest(algo=algo.name):
                logprob_req = _make_req()
                logprob_req.return_logprob = True
                self.assertIsNotNone(
                    validate_dflash_request(
                        logprob_req, enable_overlap=False, spec_algorithm=algo
                    )
                )

                hidden_req = _make_req()
                hidden_req.return_hidden_states = True
                self.assertIsNotNone(
                    validate_dflash_request(
                        hidden_req, enable_overlap=True, spec_algorithm=algo
                    )
                )


if __name__ == "__main__":
    unittest.main()
