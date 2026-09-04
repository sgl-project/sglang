"""Unit tests for the --enable-linear-replayssm-spec + NGRAM validation.

NGRAM always builds a tree mask (retrieve_parent_token is set), but the
fold-every-commit verify path requires a strict linear chain
(retrieve_parent_token is None). The validation must reject the combination
up front instead of letting the first forward crash with the gdn_backend
assert.
"""

import unittest
from unittest.mock import patch

from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Force the CUDA path so resolution runs the same hooks on CPU-only CI
# runners, matching the pattern used in test_server_args.py.
patch("sglang.srt.arg_groups.serving_hook.is_cuda", return_value=True).start()


class TestNgramReplayssmValidation(CustomTestCase):
    def _args(self, **kw):
        base = [
            "--model-path",
            "dummy",
            "--enable-linear-replayssm-spec",
            "--linear-attn-decode-backend",
            "triton",
        ]
        for key, value in kw.items():
            base.append(f"--{key.replace('_', '-')}")
            if value is not None:
                base.append(str(value))
        return prepare_server_args(base)

    def test_ngram_replayssm_rejected(self):
        args = self._args(speculative_algorithm="NGRAM")
        with self.assertRaisesRegex(ValueError, "does not support NGRAM"):
            args.resolve_once()

    def test_dflash_replayssm_not_rejected_by_ngram_check(self):
        args = self._args(speculative_algorithm="DFLASH")
        # Should not raise the NGRAM-specific error during resolution.
        try:
            args.resolve_once()
        except ValueError as exc:
            self.assertNotIn("does not support NGRAM", str(exc))

    def test_ngram_without_replayssm_not_rejected(self):
        args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--speculative-algorithm",
                "NGRAM",
                "--linear-attn-decode-backend",
                "triton",
            ]
        )
        try:
            args.resolve_once()
        except ValueError as exc:
            self.assertNotIn("does not support NGRAM", str(exc))


if __name__ == "__main__":
    unittest.main()
