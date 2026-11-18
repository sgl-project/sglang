import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_mmmu_vlm import MMMUVLMEval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMMMUPromptImages(CustomTestCase):
    def test_comparison_symbols_are_preserved_without_placeholder(self):
        prompt = "Given x < 2 and y > 3, which inequality is true?\nA. x < y\n"
        messages = MMMUVLMEval.build_chat_messages_from_prompt(prompt, "fixture-image")
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": prompt})
        self.assertEqual(
            content[1], {"type": "image_url", "image_url": {"url": "fixture-image"}}
        )
        self.assertEqual(len(content), 2)

    def test_only_image_placeholder_is_replaced(self):
        for tag in ("<image>", "<image 1>", "<image 2>"):
            with self.subTest(tag=tag):
                prefix = "For x < 2 and y > 3, inspect "
                suffix = " then decide whether x < y."
                messages = MMMUVLMEval.build_chat_messages_from_prompt(
                    prefix + tag + suffix, "fixture-image"
                )
                self.assertEqual(
                    messages[0]["content"],
                    [
                        {"type": "text", "text": prefix},
                        {"type": "image_url", "image_url": {"url": "fixture-image"}},
                        {"type": "text", "text": suffix},
                    ],
                )


if __name__ == "__main__":
    unittest.main()
