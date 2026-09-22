"""Regenerate the array-only template regression using SGLang's content processor.

Run: PYTHONPATH=../../python python tests/scripts/generate_array_content_fixture.py
"""

import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from sglang.srt.parser.jinja_template_utils import (
    detect_jinja_template_content_format,
    process_content_for_template_format,
)

ROOT = Path(__file__).resolve().parents[1] / "fixtures"
TEMPLATE = (
    "{% for message in messages %}{{ message.role }}:"
    "{% for part in message.content %}"
    "{% if part.type == 'text' %}{{ part.text }}"
    "{% elif part.type == 'image' %}<image>{% endif %}"
    "{% endfor %};{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


def main():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(ROOT / "tiny_tokenizer.json")
    )
    tokenizer.chat_template = TEMPLATE
    content_format = detect_jinja_template_content_format(TEMPLATE)
    assert content_format == "openai"
    cases = []
    for content in ["hello", [{"type": "text", "text": "hello"}]]:
        message = process_content_for_template_format(
            {"role": "user", "content": content}, content_format, [], [], [], []
        )
        ids = tokenizer.apply_chat_template(
            [message], return_dict=False, add_generation_prompt=True
        )
        cases.append(json.dumps({"content": content, "engine_token_ids": ids}))
    cases_json = ",\n".join(cases)
    (ROOT / "array_content_rendering.json").write_text(
        f'{{"chat_template": {json.dumps(TEMPLATE)}, "cases": [\n{cases_json}\n]}}\n'
    )


if __name__ == "__main__":
    main()
