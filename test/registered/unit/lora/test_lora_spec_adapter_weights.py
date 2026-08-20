"""The adapter-weight embedding check under EAGLE-family spec.

Weight-level, so it catches what the CLI check cannot see: an adapter shipping
embedding weights via a PEFT shorthand or the update endpoints.
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
    def test_embedding_weights_warn_once_and_only_when_they_matter(self):
        """all-linear adapters ship unembed_tokens weights that load rewrites
        to lm_head; one adapter touching both modules is still one situation.
        The silent cases keep either predicate from degrading to always-true.
        """
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

        for names, algo in [
            ([], "EAGLE"),
            ([LM_HEAD_KEY], None),
            ([LM_HEAD_KEY], "NGRAM"),
        ]:
            with self.subTest(names=len(names), algo=algo):
                with self.assertNoLogs("sglang.srt.lora.utils", logging.WARNING):
                    warn_if_adapter_targets_embeddings(
                        lora_name="case",
                        embedding_layer_names=names,
                        speculative_algorithm=algo,
                    )


if __name__ == "__main__":
    unittest.main()
