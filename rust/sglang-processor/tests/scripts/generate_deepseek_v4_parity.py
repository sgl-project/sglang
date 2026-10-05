"""Fill tests/fixtures/deepseek_v4_0731.json from SGLang's real serving pipeline.

Run from rust/sglang-processor in a SGLang Python environment with the pinned
snapshot cached:
    python tests/scripts/generate_deepseek_v4_parity.py
"""

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

os.environ["SGLANG_DEFAULT_THINKING"] = "false"
os.environ["SGLANG_DSV4_REASONING_EFFORT"] = ""

from sglang.srt.entrypoints.openai import chat_encoding
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/deepseek_v4_0731.json"


class RecordingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = []

    def encode(self, text):
        self.texts.append(text)
        return self.tokenizer.encode(text)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def main():
    fixture = json.loads(FIXTURE.read_text())
    path = snapshot_download(
        fixture["model"],
        revision=fixture["revision"],
        local_files_only=True,
        allow_patterns=["config.json", "tokenizer*.json", "encoding/*"],
    )
    tok = RecordingTokenizer(AutoTokenizer.from_pretrained(path, local_files_only=True))
    server = object.__new__(OpenAIServingChat)
    server.chat_encoding_spec = "dsv4"
    server._dsv4_reasoning_effort_profile = (
        chat_encoding._detect_dsv4_reasoning_effort_profile(path)
    )
    assert server._dsv4_reasoning_effort_profile == "official"
    server.tokenizer_manager = SimpleNamespace(tokenizer=tok)
    server.template_manager = SimpleNamespace(jinja_template_content_format="string")
    for case in fixture["cases"]:
        tok.texts.clear()
        request = ChatCompletionRequest(**copy.deepcopy(case["request"]))
        # _convert_to_internal_request: kwargs effort replaces the request effort.
        if request.chat_template_kwargs:
            effort = request.chat_template_kwargs.pop("reasoning_effort", None)
            if effort is not None:
                request.reasoning_effort = effort
        ids = server._apply_jinja_template(
            request, tools=None, is_multimodal=False
        ).prompt_ids
        case.update(
            prompt="".join(tok.texts),
            token_count=len(ids),
            token_sha256=hashlib.sha256(
                b"".join(i.to_bytes(4, "little") for i in ids)
            ).hexdigest(),
        )
    metadata = {k: v for k, v in fixture.items() if k != "cases"}
    FIXTURE.write_text(
        json.dumps(metadata)[:-1]
        + ', "cases": [\n'
        + ",\n".join(
            json.dumps(c, ensure_ascii=False, separators=(",", ":"))
            for c in fixture["cases"]
        )
        + "\n]}\n"
    )
    print(f"wrote {len(fixture['cases'])} cases")


if __name__ == "__main__":
    main()
