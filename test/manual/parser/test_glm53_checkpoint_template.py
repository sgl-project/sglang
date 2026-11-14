"""GLM-5.3 / GLM-5.3-Flash Python chat templates against the checkpoints' Jinja.

Set SGLANG_TEST_GLM53_CHECKPOINT / SGLANG_TEST_GLM53_FLASH_CHECKPOINT to model
directories holding chat_template.jinja and the tokenizer, then run:
    python3 test/manual/parser/test_glm53_checkpoint_template.py
"""

import os
import sys
import unittest
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

from sglang.srt.entrypoints.openai.parallel_prompt_encode import encode_prompt_segments
from sglang.srt.parser.glm53_chat_template import render_glm53, render_glm53_flash
from sglang.srt.parser.python_chat_template import (
    GLM53,
    GLM53_FLASH,
    render_chat_segments,
)
from sglang.test.test_utils import CustomTestCase

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "registered" / "unit" / "parser")
)
from test_glm53_chat_template import CASES, _render  # noqa: E402


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_GLM53_CHECKPOINT")
    or os.environ.get("SGLANG_TEST_GLM53_FLASH_CHECKPOINT"),
    "set SGLANG_TEST_GLM53_CHECKPOINT / SGLANG_TEST_GLM53_FLASH_CHECKPOINT",
)
class TestAgainstCheckpointTemplates(CustomTestCase):
    def test_matches_checkpoint_jinja(self):
        for env, render, template in (
            ("SGLANG_TEST_GLM53_CHECKPOINT", render_glm53, GLM53),
            ("SGLANG_TEST_GLM53_FLASH_CHECKPOINT", render_glm53_flash, GLM53_FLASH),
        ):
            model_dir = os.environ.get(env)
            if not model_dir:
                continue
            with open(
                os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8"
            ) as f:
                compiled = _compile_jinja_template(f.read())
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            for name, (kwargs, _, _) in CASES.items():
                for add_generation_prompt in (False, True):
                    with self.subTest(
                        model=model_dir, case=name, gen=add_generation_prompt
                    ):
                        kwargs = {
                            **kwargs,
                            "add_generation_prompt": add_generation_prompt,
                        }
                        prompt = compiled.render(**kwargs)
                        self.assertEqual(_render(render, kwargs), prompt)
                        variables = dict(kwargs)
                        segments = render_chat_segments(
                            tokenizer,
                            template,
                            variables.pop("messages"),
                            tools=variables.pop("tools", None),
                            add_generation_prompt=variables.pop(
                                "add_generation_prompt"
                            ),
                            template_kwargs=variables,
                        )
                        self.assertEqual(
                            encode_prompt_segments(tokenizer, segments),
                            tokenizer.encode(prompt),
                        )


if __name__ == "__main__":
    unittest.main()
