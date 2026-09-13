"""Unit tests for health-check probe classification — no server, no model loading."""

import unittest

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.io_struct import (
    AbortReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.utils import (
    is_health_check_generate_req,
    is_health_check_probe,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEALTH_CHECK_RID = f"{HEALTH_CHECK_RID_PREFIX}_0123456789abcdef"


def _generate_probe(rid: str) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=None,
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def _embedding_probe(rid: str) -> TokenizedEmbeddingReqInput:
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text=None,
        input_ids=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0),
    )


class TestIsHealthCheckProbe(CustomTestCase):
    def test_matches_generate_and_embedding_probes(self):
        self.assertTrue(is_health_check_probe(_generate_probe(HEALTH_CHECK_RID)))
        self.assertTrue(is_health_check_probe(_embedding_probe(HEALTH_CHECK_RID)))

    def test_ignores_ordinary_requests(self):
        self.assertFalse(is_health_check_probe(_generate_probe("ordinary-rid")))
        self.assertFalse(is_health_check_probe(_embedding_probe("ordinary-rid")))

    def test_ignores_abort_of_a_probe(self):
        """The scheduler drops probes while busy; it must never drop their aborts."""
        abort = AbortReq(rid=HEALTH_CHECK_RID)

        self.assertTrue(is_health_check_generate_req(abort))
        self.assertFalse(is_health_check_probe(abort))


if __name__ == "__main__":
    unittest.main()
