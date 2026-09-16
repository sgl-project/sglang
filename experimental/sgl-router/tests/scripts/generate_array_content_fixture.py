"""Record SGLang's content processing for a minimal array-only VLM template.

Run from experimental/sgl-router with the repository's Python dependencies:
    PYTHONPATH=../../python python tests/scripts/generate_array_content_fixture.py

This fixture records a known string-content mismatch, not general VLM parity.
The template deliberately has no string-content branch, like array-only VLM
templates. The tiny byte-level tokenizer keeps the Rust regression offline.
"""

import copy
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
    for shape, content in [
        ("string", "hello"),
        ("text_array", [{"type": "text", "text": "hello"}]),
    ]:
        messages = [{"role": "user", "content": content}]
        processed = [
            process_content_for_template_format(
                copy.deepcopy(message), content_format, [], [], [], []
            )
            for message in messages
        ]
        prompt = tokenizer.apply_chat_template(
            processed, tokenize=False, add_generation_prompt=True, tools=None
        )
        cases.append(
            {
                "shape": shape,
                "request": {"messages": messages},
                "engine_prompt": prompt,
                "engine_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
            }
        )
    fixture = {
        "chat_template": TEMPLATE,
        "engine_content_format": content_format,
        "cases": cases,
    }
    (ROOT / "array_content_rendering.json").write_text(
        json.dumps(fixture, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
