import unittest
from unittest.mock import patch

from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTokenizerPostLoadRevision(CustomTestCase):
    def _load(self, **kwargs):
        sentinel = object()
        with (
            patch(
                "sglang.srt.utils.hf_transformers.tokenizer.is_bare_tekken_checkpoint",
                return_value=False,
            ),
            patch(
                "sglang.srt.utils.hf_transformers.tokenizer._auto_tokenizer_from_pretrained",
                return_value=sentinel,
            ),
            patch(
                "sglang.srt.utils.hf_transformers.tokenizer._apply_post_load_fixes",
                side_effect=lambda tok, name, rev: tok,
            ) as fixes,
        ):
            got = get_tokenizer("org/model", **kwargs)
        return got, sentinel, fixes

    def test_revision_kwarg_reaches_post_load_fixes(self):
        got, sentinel, fixes = self._load(revision="abc123")
        self.assertIs(got, sentinel)
        self.assertEqual(fixes.call_args.args[2], "abc123")

    def test_explicit_tokenizer_revision_wins_over_revision_kwarg(self):
        _got, _sentinel, fixes = self._load(
            tokenizer_revision="pinned-sha", revision="other-sha"
        )
        self.assertEqual(fixes.call_args.args[2], "pinned-sha")


if __name__ == "__main__":
    unittest.main()
