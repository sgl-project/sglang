"""The draft's ServerArgs is a copy; the target's stays as the launcher left it.

Regression: the v2 spec workers wrote the draft's context_length (and the
scheduler the draft's load_format) onto the ServerArgs instance they share with
the target worker, so every later reader of that instance saw draft values.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.draft_worker_common import (
    draft_server_args_copy,
    draft_server_args_overrides,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TARGET_MODEL_CONFIG = SimpleNamespace(context_len=4096)


class TestDraftServerArgsCopy(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        server_args = override.install()
        self.addCleanup(override.restore)
        return server_args

    def test_the_draft_context_length_follows_the_target(self):
        target = self._seed(context_length=None)
        draft = draft_server_args_copy(target, TARGET_MODEL_CONFIG)
        self.assertEqual(draft.context_length, 4096)

    def test_the_target_instance_is_left_alone(self):
        target = self._seed(context_length=None, load_format="auto")
        draft = draft_server_args_copy(target, TARGET_MODEL_CONFIG)
        self.assertIsNot(draft, target)
        self.assertIsNone(target.context_length)
        self.assertEqual(target.load_format, "auto")

    def test_the_draft_load_format_applies_only_when_configured(self):
        target = self._seed(load_format="auto", speculative_draft_load_format="dummy")
        self.assertEqual(
            draft_server_args_copy(target, TARGET_MODEL_CONFIG).load_format, "dummy"
        )
        self.assertEqual(target.load_format, "auto")

        target = self._seed(load_format="auto")
        self.assertEqual(
            draft_server_args_copy(target, TARGET_MODEL_CONFIG).load_format, "auto"
        )

    def test_load_time_overrides_reach_the_draft(self):
        target = self._seed(disable_chunked_prefix_cache=False)
        # What the target runner resolved before the draft is built — e.g. the
        # chunked-prefix gate for an attention backend that cannot serve it.
        get_context().override("test.gate", disable_chunked_prefix_cache=True)

        draft = draft_server_args_copy(target, TARGET_MODEL_CONFIG)
        self.assertTrue(draft.disable_chunked_prefix_cache)
        self.assertFalse(target.disable_chunked_prefix_cache)

    def test_the_draft_specific_fields_win_over_the_resolved_ones(self):
        target = self._seed(context_length=None, load_format="auto")
        get_context().override("test.late", context_length=128, load_format="npcache")

        draft = draft_server_args_copy(target, TARGET_MODEL_CONFIG)
        self.assertEqual(draft.context_length, 4096)

    def test_the_built_draft_overrides_carry_the_load_format_too(self):
        self._seed(speculative_draft_load_format="dummy")
        fields = draft_server_args_overrides(TARGET_MODEL_CONFIG, "triton")
        self.assertEqual(fields["load_format"], "dummy")

        self._seed()
        self.assertNotIn(
            "load_format", draft_server_args_overrides(TARGET_MODEL_CONFIG, "triton")
        )


if __name__ == "__main__":
    unittest.main()
