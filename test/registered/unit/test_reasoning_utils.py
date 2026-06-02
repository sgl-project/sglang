import unittest

from sglang.srt.reasoning_utils import resolve_require_reasoning
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestResolveRequireReasoning(unittest.TestCase):
    def test_no_parser_disables_reasoning(self):
        self.assertFalse(resolve_require_reasoning(None))

    def test_qwen3_thinks_by_default(self):
        self.assertTrue(resolve_require_reasoning("qwen3"))

    def test_qwen3_enable_thinking_false_disables_reasoning(self):
        self.assertFalse(
            resolve_require_reasoning(
                "qwen3", chat_template_kwargs={"enable_thinking": False}
            )
        )

    def test_qwen3_enable_thinking_true_requires_reasoning(self):
        self.assertTrue(
            resolve_require_reasoning(
                "qwen3", chat_template_kwargs={"enable_thinking": True}
            )
        )

    def test_deepseek_requires_explicit_thinking(self):
        self.assertFalse(resolve_require_reasoning("deepseek-v3"))
        self.assertTrue(
            resolve_require_reasoning(
                "deepseek-v3", chat_template_kwargs={"thinking": True}
            )
        )

    def test_hunyuan_uses_reasoning_effort(self):
        self.assertFalse(resolve_require_reasoning("hunyuan"))
        self.assertFalse(resolve_require_reasoning("hunyuan", reasoning_effort="none"))
        self.assertTrue(resolve_require_reasoning("hunyuan", reasoning_effort="high"))

    def test_unknown_parser_defaults_to_reasoning(self):
        self.assertTrue(resolve_require_reasoning("custom-parser"))

    def test_reasoning_config_default_enabled_toggle(self):
        class Config:
            special_case = None
            toggle_param = "enable_thinking"
            default_enabled = True

        self.assertTrue(resolve_require_reasoning("qwen3", reasoning_config=Config()))
        self.assertFalse(
            resolve_require_reasoning(
                "qwen3",
                chat_template_kwargs={"enable_thinking": False},
                reasoning_config=Config(),
            )
        )

    def test_reasoning_config_default_disabled_toggle(self):
        class Config:
            special_case = None
            toggle_param = "thinking"
            default_enabled = False

        self.assertFalse(
            resolve_require_reasoning("deepseek-v3", reasoning_config=Config())
        )
        self.assertTrue(
            resolve_require_reasoning(
                "deepseek-v3",
                chat_template_kwargs={"thinking": True},
                reasoning_config=Config(),
            )
        )

    def test_reasoning_default_detector_mode(self):
        self.assertFalse(
            resolve_require_reasoning(
                "poolside_v1",
                chat_template_kwargs={},
                reasoning_default="explicit_enable_thinking",
            )
        )
        self.assertTrue(
            resolve_require_reasoning(
                "poolside_v1",
                chat_template_kwargs={"enable_thinking": True},
                reasoning_default="explicit_enable_thinking",
            )
        )


if __name__ == "__main__":
    unittest.main()
