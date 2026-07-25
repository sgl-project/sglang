"""Admission checks for DFLASH-family requests.

Grammar-constrained decoding (json_schema / regex / ebnf / structural_tag) is
enforced in verify by the vocab bitmask, so admission must let it through -- a
reinstated blanket reject would 400 every structured-output request on this
family. Real SamplingParams objects keep the four grammar field names pinned to
the request API.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_GRAMMAR_KINDS = (
    {"json_schema": '{"type": "object"}'},
    {"regex": "[0-9]+"},
    {"ebnf": 'root ::= "a"'},
    {"structural_tag": '{"type": "structural_tag"}'},
)


def _make_req(**sampling_kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SamplingParams(**sampling_kwargs),
        return_logprob=False,
        return_hidden_states=False,
    )


class TestValidateDflashRequest(CustomTestCase):
    def test_every_grammar_kind_admitted(self):
        for kind in _GRAMMAR_KINDS:
            with self.subTest(grammar=next(iter(kind))):
                self.assertIsNone(
                    validate_dflash_request(_make_req(**kind), enable_overlap=False)
                )

    def test_ungrammared_request_admitted(self):
        self.assertIsNone(validate_dflash_request(_make_req(), enable_overlap=False))

    def test_non_grammar_rejections_survive(self):
        logprob_req = _make_req()
        logprob_req.return_logprob = True
        self.assertIsNotNone(validate_dflash_request(logprob_req, enable_overlap=False))

        hidden_req = _make_req()
        hidden_req.return_hidden_states = True
        self.assertIsNotNone(validate_dflash_request(hidden_req, enable_overlap=True))


if __name__ == "__main__":
    unittest.main()
