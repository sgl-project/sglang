"""Unit tests for the adapter-weight embedding check under EAGLE-family spec.

Embedding-module adapters are supported, not rejected: the draft is handed
base weights for both modules, so only the accept rate is affected. This
weight-level check catches what the CLI check cannot see -- an adapter that
ships embedding weights via a PEFT shorthand or through the update endpoints.
"""

import logging
import unittest

from sglang.srt.lora.utils import warn_if_adapter_targets_embeddings
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LM_HEAD_KEY = "base_model.model.model.lm_head.lora_A"
EMBED_KEY = "base_model.model.model.embed_tokens.lora_A"


class TestLoRASpecAdapterWeights(CustomTestCase):
    def test_embedding_weights_warn_once_under_eagle_family(self):
        """Real case: opherlie/lora-test-case-Qwen3.5-35B-A3B declares
        target_modules='all-linear' but ships unembed_tokens weights, which
        load rewrites to lm_head. An adapter touching both modules is still
        one situation, so it must not log twice."""
        for algo in ["EAGLE", "EAGLE3"]:
            for keys in [[LM_HEAD_KEY], [EMBED_KEY], [LM_HEAD_KEY, EMBED_KEY]]:
                with self.subTest(algo=algo, keys=len(keys)):
                    with self.assertLogs("sglang.srt.lora.utils", "WARNING") as logs:
                        warn_if_adapter_targets_embeddings(
                            lora_name="case",
                            embedding_layer_names=keys,
                            speculative_algorithm=algo,
                        )
                    self.assertEqual(len(logs.records), 1)
                    self.assertIn("accept rate", logs.output[0])

    def test_silent_when_there_is_nothing_to_warn_about(self):
        """Both no-op branches, so neither predicate can degrade to
        always-true: an adapter with no embedding weights (the common
        all-linear case), and an algorithm whose draft never shares the
        target's lm_head."""
        cases = [([], "EAGLE"), ([LM_HEAD_KEY], None), ([LM_HEAD_KEY], "NGRAM")]
        for names, algo in cases:
            with self.subTest(names=len(names), algo=algo):
                with self.assertNoLogs("sglang.srt.lora.utils", logging.WARNING):
                    warn_if_adapter_targets_embeddings(
                        lora_name="case",
                        embedding_layer_names=names,
                        speculative_algorithm=algo,
                    )


if __name__ == "__main__":
    unittest.main()
