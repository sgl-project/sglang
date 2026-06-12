"""Regression test for PR #24082.

Covers the specific trigger that the v0.5.11 Spec V1 grammar-finish fix does not
reach: PD disaggregation + overlap scheduling + EAGLE Spec V2 (topk=1) + a grammar
constraint.

Spec V2 proposes several tokens per decode step. With a grammar constraint, the
request can reach grammar completion partway through an accepted list, so:

  * the decode result processor must accept the proposed tokens one at a time and
    stop at grammar completion (no tokens emitted past the closing of the grammar),
    trimming the over-proposed tokens and aligning logprob bookkeeping; and
  * the disaggregated overlap decode loop must process the previous batch result
    (advancing the grammar) before launching the next Spec V2 grammar decode batch.

Notes for whoever runs this on GPU hardware:
  * Spec V2 is gated behind ``SGLANG_ENABLE_SPEC_V2`` and only supports
    ``--speculative-eagle-topk 1``. The env override below is inherited by the
    launched prefill/decode subprocesses.
  * Overlap scheduling is on by default, so the decode side runs
    ``event_loop_overlap_disagg_decode`` (the loop modified by this PR).
"""

import json
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationSpecV2Grammar(PDDisaggregationServerBase):
    # Minimal delta from the known-good PD spec config (TestDisaggregationMooncakeSpec):
    # switch topk 4 -> 1 and enable Spec V2 so the Spec-V2 grammar path is exercised.
    model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3
    spec_algorithm = "EAGLE"
    spec_steps = 3
    spec_topk = 1  # Spec V2 only supports topk=1
    spec_draft_tokens = 4
    grammar_backend = "xgrammar"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        spec_args = [
            "--speculative-algorithm",
            cls.spec_algorithm,
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-steps",
            str(cls.spec_steps),
            "--speculative-eagle-topk",
            str(cls.spec_topk),
            "--speculative-num-draft-tokens",
            str(cls.spec_draft_tokens),
            "--grammar-backend",
            cls.grammar_backend,
            "--cuda-graph-max-bs",
            "8",
            "--dtype=float16",
        ]
        cls.extra_prefill_args = spec_args
        cls.extra_decode_args = spec_args
        with (
            envs.SGLANG_ENABLE_SPEC_V2.override(True),
            # The EAGLE3 draft model config derives a 2048 context length, which is
            # shorter than the Llama-3.1 target's 131072. The Spec V2 draft worker
            # (eagle_worker_v2.py) builds its own ModelConfig and rejects this
            # mismatch unless overriding longer context is allowed. Outputs here are
            # well under 2048 tokens, so allowing the override is safe.
            envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True),
        ):
            cls.launch_all()

    @staticmethod
    def _json_schema() -> str:
        return json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                    "country": {"type": "string", "pattern": "^[\\w ]+$"},
                    "capital": {"type": "string", "pattern": "^[\\w ]+$"},
                },
                "required": ["name", "population", "country", "capital"],
            }
        )

    def _generate(self, return_logprob: bool):
        # max_new_tokens is generous so completion is driven by grammar termination,
        # not the length cap, and the output spans multiple decode iterations.
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "Here is the information of the capital of France in the JSON format.\n",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                    "json_schema": self._json_schema(),
                },
                "return_logprob": return_logprob,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_structured_output_no_trailing_tokens(self):
        """Output is valid JSON with nothing emitted past grammar completion."""
        out = self._generate(return_logprob=False)
        text = out["text"]

        # json.loads rejects trailing non-whitespace content, so a clean parse of
        # the raw text means no stray tokens leaked after the grammar terminated.
        parsed = json.loads(text)
        for key in ("name", "population", "country", "capital"):
            self.assertIn(key, parsed)

        # Belt and suspenders: the decoded text should end exactly at the JSON
        # object close, not be followed by extra generated content.
        self.assertTrue(
            text.strip().endswith("}"), f"unexpected trailing tokens: {text!r}"
        )

    def test_spec_v2_actually_ran(self):
        """The accepted-length stat confirms Spec V2 verification took place."""
        out = self._generate(return_logprob=False)
        spec_verify_ct = out["meta_info"]["spec_verify_ct"]
        self.assertGreater(
            spec_verify_ct,
            0,
            f"expected Spec V2 to run (spec_verify_ct > 0), got {spec_verify_ct}",
        )

    def test_logprob_count_matches_completion_tokens(self):
        """Trimmed Spec V2 tokens must keep logprob count == completion token count."""
        out = self._generate(return_logprob=True)
        meta = out["meta_info"]
        completion_tokens = meta["completion_tokens"]
        output_logprobs = meta["output_token_logprobs"]
        self.assertEqual(
            len(output_logprobs),
            completion_tokens,
            "output logprobs must align with retained (trimmed) tokens: "
            f"got {len(output_logprobs)} logprobs vs {completion_tokens} completion tokens",
        )
        # And the constrained output is still valid structured JSON.
        json.loads(out["text"])


if __name__ == "__main__":
    unittest.main()
