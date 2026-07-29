import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_canary.e2e_base import (
    _LONG_PROMPT_BODY,
    _UNIQUE_PROMPT_FIRST_CHARS,
    _make_unique_prompts,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-b-test-cpu")


class TestCanaryE2EBase(CustomTestCase):
    def test_make_unique_prompts_have_distinct_first_characters(self) -> None:
        """Verify generated prompts use distinct first characters and all end with the shared long body."""
        prompts = _make_unique_prompts(8)

        self.assertEqual(len({prompt[0] for prompt in prompts}), len(prompts))
        self.assertTrue(all(prompt.endswith(_LONG_PROMPT_BODY) for prompt in prompts))

    def test_make_unique_prompts_rejects_more_prompts_than_distinct_first_characters(
        self,
    ) -> None:
        """Verify prompt generation rejects requests beyond the unique prefix budget."""
        with self.assertRaisesRegex(ValueError, "unique prompt count"):
            _make_unique_prompts(len(_UNIQUE_PROMPT_FIRST_CHARS) + 1)


if __name__ == "__main__":
    unittest.main()
