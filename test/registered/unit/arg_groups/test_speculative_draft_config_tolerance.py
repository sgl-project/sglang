"""Draft-config parse failures in speculative-algorithm resolution must not be
fatal: the resolution only probes the draft config to detect Gemma4 assistant
drafts, and a config that cannot be parsed (e.g. a speculators-style draft
config without a ``model_type`` key) can never be one."""

from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _resolve_speculative_algorithm_alias
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest


class TestDraftConfigTolerance(unittest.TestCase):
    def test_unparseable_draft_config_is_not_fatal(self):
        """A draft config that raises on load (no model_type) resolves normally."""
        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            side_effect=ValueError("No model_type found in config.json"),
        ):
            self.assertEqual(
                _resolve_speculative_algorithm_alias(
                    speculative_algorithm="DFLASH",
                    speculative_draft_model_path="/tmp/some-dflash-draft",
                ),
                "DFLASH",
            )

    def test_gemma4_draft_still_promotes(self):
        """Parseable Gemma4 draft configs keep the FROZEN_KV_MTP promotion."""
        cfg = type("Cfg", (), {"architectures": ["Gemma4AssistantForCausalLM"]})()
        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=cfg,
        ):
            self.assertEqual(
                _resolve_speculative_algorithm_alias(
                    speculative_algorithm="NEXTN",
                    speculative_draft_model_path="/tmp/some-gemma4-draft",
                ),
                "FROZEN_KV_MTP",
            )

    def test_parseable_non_gemma4_draft_unaffected(self):
        """Parseable non-Gemma4 drafts resolve exactly as before the guard."""
        cfg = type("Cfg", (), {"architectures": ["DFlashDraftModel"]})()
        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=cfg,
        ):
            self.assertEqual(
                _resolve_speculative_algorithm_alias(
                    speculative_algorithm="EAGLE",
                    speculative_draft_model_path="/tmp/some-dflash-draft",
                ),
                "EAGLE",
            )


if __name__ == "__main__":
    unittest.main()
