import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.model_executor.model_runner import ModelRunner


def _runner(*, speculative: bool, draft: bool) -> SimpleNamespace:
    return SimpleNamespace(
        is_draft_worker=draft,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: speculative),
    )


class TestNonIntrusiveDumperTags(CustomTestCase):
    """ModelRunner tags its dumps only under speculative decoding."""

    def test_no_tags_without_speculative_decoding(self):
        for draft in (False, True):
            runner = _runner(speculative=False, draft=draft)
            self.assertIsNone(ModelRunner._non_intrusive_dumper_tags(runner))

    def test_target_runner_is_tagged_target(self):
        runner = _runner(speculative=True, draft=False)
        self.assertEqual(
            ModelRunner._non_intrusive_dumper_tags(runner), {"runner": "target"}
        )

    def test_draft_runner_is_tagged_draft(self):
        runner = _runner(speculative=True, draft=True)
        self.assertEqual(
            ModelRunner._non_intrusive_dumper_tags(runner), {"runner": "draft"}
        )


if __name__ == "__main__":
    unittest.main()
