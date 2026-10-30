import unittest

from transformers import PretrainedConfig

from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.nemotron_parse_chat_template import (
    render_nemotron_parse_segments,
)
from sglang.srt.parser.python_chat_template import (
    NEMOTRON_PARSE,
    resolve_python_chat_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNemotronParseChatTemplate(CustomTestCase):
    def test_image_is_retained_and_task_prompt_is_unchanged(self):
        images = []
        prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
        message = process_content_for_template_format(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/page.png"},
                    },
                ],
            },
            NEMOTRON_PARSE.content_format,
            images,
            [],
            [],
            [],
        )
        self.assertEqual("".join(render_nemotron_parse_segments([message])), prompt)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://example.com/page.png")

    def test_selection_is_limited_to_parse_architecture_and_model_type(self):
        for architecture, model_type, expected in [
            ("RadioMBartPrefixForCausalLM", "nemotron_parse", NEMOTRON_PARSE),
            ("RadioMBartPrefixForCausalLM", "other", None),
            ("LlamaForCausalLM", "nemotron_parse", None),
        ]:
            config = PretrainedConfig(architectures=[architecture])
            config.model_type = model_type
            with self.subTest(architecture=architecture, model_type=model_type):
                self.assertIs(resolve_python_chat_template(config, None), expected)

    def test_unsupported_media_and_tools_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "text and document images"):
            render_nemotron_parse_segments([{"content": [{"type": "video"}]}])
        with self.assertRaisesRegex(ValueError, "tool calls"):
            render_nemotron_parse_segments([{"content": "parse"}], tools=[{}])


if __name__ == "__main__":
    unittest.main()
