import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.deferred_draft_prefill_request import (
    validate_deferred_draft_prefill_request,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeferredDraftPrefillRequest(unittest.TestCase):
    def request(self, length=8192, **sampling):
        return SimpleNamespace(
            origin_input_ids=[1] * length,
            sampling_params=SimpleNamespace(**sampling),
        )

    def test_text_prompt_on_budget_is_accepted(self):
        self.assertIsNone(
            validate_deferred_draft_prefill_request(
                self.request(), SimpleNamespace(), 8192
            )
        )

    def test_oversized_and_empty_prompts_are_rejected_before_allocation(self):
        for length in (0, 8193):
            with self.subTest(length=length):
                self.assertIn(
                    "input tokens",
                    validate_deferred_draft_prefill_request(
                        self.request(length), SimpleNamespace(), 8192
                    ),
                )

    def test_each_grammar_constraint_is_rejected_before_compilation(self):
        for name in ("json_schema", "regex", "ebnf", "structural_tag"):
            with self.subTest(name=name):
                self.assertIn(
                    "constrained decoding",
                    validate_deferred_draft_prefill_request(
                        self.request(**{name: "constraint"}), SimpleNamespace(), 8192
                    ),
                )

    def test_unsupported_tokenized_inputs_are_rejected(self):
        for fields, message in (
            ({"mm_inputs": object()}, "text prompts"),
            ({"input_embeds": []}, "input embeddings"),
            ({"session_params": object()}, "sessions"),
            ({"lora_id": "adapter"}, "LoRA"),
            ({"custom_logit_processor": "processor"}, "custom logit"),
        ):
            with self.subTest(fields=fields):
                self.assertIn(
                    message,
                    validate_deferred_draft_prefill_request(
                        self.request(), SimpleNamespace(**fields), 8192
                    ),
                )


if __name__ == "__main__":
    unittest.main()
