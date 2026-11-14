"""Manual single-Hopper STANDALONE PD token-equivalence regression.

Run from the repository root (Mooncake CUDA transport, NVML and FA3 required)::

    python test/manual/spec/test_spec_standalone_disaggregation.py -f

Defaults are public Qwen3-4B / Qwen3-0.6B checkpoints. Set
SGLANG_TEST_STANDALONE_PD_TARGET and SGLANG_TEST_STANDALONE_PD_DRAFT to local
checkpoint paths when offline. Both checkpoints must have the same vocabulary.
Set SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR to a fresh directory to retain raw
HTTP requests/responses, health/metrics observations, and server logs.
This special same-GPU setup is not registered in an existing CI suite.
"""

import unittest

from standalone_pd_fixture import StandalonePDServerBase
from transformers import AutoTokenizer


class TestStandaloneDisaggregation(StandalonePDServerBase):
    def test_three_greedy_completions_match_ordinary(self):
        target = AutoTokenizer.from_pretrained(self.target_model)
        draft = AutoTokenizer.from_pretrained(self.draft_model)
        self.assertEqual(target.get_vocab(), draft.get_vocab())
        self.assertEqual(target.all_special_ids, draft.all_special_ids)
        prompts = [
            "Explain why the sky appears blue during the day in one short paragraph.",
            "Continue this list of numbers in words: one, two, three, four, five,",
            "Write a brief story about a friendly cat exploring a quiet garden on a sunny morning.",
        ]
        inputs = [target.encode(text, add_special_tokens=False) for text in prompts]
        self.assertTrue(all(4 <= len(ids) <= 256 for ids in inputs))
        references = []
        self.start_server("ordinary")
        for ids in inputs:
            self.assert_idle("ordinary")
            request = self.new_payload(ids, pd=False)
            response = self.http("ordinary", "POST", "/generate", request)
            references.append(self.assert_completion(response, request)[0])
            self.assert_idle("ordinary")
        # stop_server includes the process/port/NVML release gate. A failed gate
        # raises here; neither PD peer may start against a dying ordinary engine.
        self.stop_server("ordinary")
        self.start_server("prefill")
        self.start_server("decode")
        accepted = 0
        observations = []
        for ids, expected in zip(inputs, references):
            self.assert_idle("prefill")
            self.assert_idle("decode")
            request = self.new_payload(ids)
            responses = self.post_pair(request)
            first, _ = self.assert_completion(
                responses["prefill"], request, prefill=True
            )
            actual, meta = self.assert_completion(responses["decode"], request)
            self.assertEqual(first, expected[:1])
            self.assertEqual(
                actual, expected, f"Wrong greedy output for RID {request['rid']}"
            )
            self.assertGreaterEqual(meta["spec_verify_ct"], 2)
            self.assertGreaterEqual(meta["spec_num_correct_drafts"], 0)
            accepted += meta["spec_num_correct_drafts"]
            self.assert_idle("prefill")
            self.assert_idle("decode")
            observations.append(
                {
                    "rid": request["rid"],
                    "output_ids": actual,
                    "verify_count": meta["spec_verify_ct"],
                    "accepted_draft_tokens": meta["spec_num_correct_drafts"],
                }
            )
        # Exact target output alone could silently pass with speculative decoding
        # bypassed. Require real multi-round verification and accepted drafts.
        self.assertGreater(accepted, 0)
        self.record(
            "normal-summary.json",
            {
                "accepted_draft_tokens": accepted,
                "cases": observations,
                "exact_ordinary_token_match": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
