"""Regenerate native DeepSeek fixtures with SGLang's actual serving pipeline.

Run from experimental/sgl-router in a SGLang Python environment:
    python tests/scripts/generate_deepseek_parity.py

The Rust tests always compare rendered text, even without cached model files.
With the pinned HF snapshot cached they also compare exact token-ID digests.
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

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

ROOT = Path(__file__).resolve().parents[1] / "fixtures/deepseek"
MODELS = {
    "v41": (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "dba1be0a40aa45a94ad051997016db3960a90277",
    ),
    "v4": ("deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1"),
}


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
    ROOT.mkdir(exist_ok=True)
    for family, (model, revision) in MODELS.items():
        path = snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
            allow_patterns=["config.json", "tokenizer*.json"],
        )
        tok = RecordingTokenizer(
            AutoTokenizer.from_pretrained(path, local_files_only=True)
        )
        server = object.__new__(OpenAIServingChat)
        server.chat_encoding_spec = "ds" + family
        server._dsv41_default_reasoning_effort = "high"
        server.tokenizer_manager = SimpleNamespace(tokenizer=tok)
        server.template_manager = SimpleNamespace(
            jinja_template_content_format="string"
        )
        fixture_path = ROOT / (family + ".json")
        fixture = json.loads(fixture_path.read_text())
        for case in fixture["cases"]:
            server._dsv4_reasoning_effort_profile = case["profile"]
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
        fixture_path.write_text(
            json.dumps(metadata)[:-1]
            + ', "cases": [\n'
            + ",\n".join(
                json.dumps(c, ensure_ascii=False, separators=(",", ":"))
                for c in fixture["cases"]
            )
            + "\n]}\n"
        )
        print(f"wrote {family}: {len(fixture['cases'])} cases")


if __name__ == "__main__":
    main()
