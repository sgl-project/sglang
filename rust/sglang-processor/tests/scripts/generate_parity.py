"""Fill tests/fixtures/parity/*.json from SGLang's own serving path.

Each fixture names a checkpoint (`model`, `revision`, optionally the server's
`tool_call_parser`) and request `cases`. This renders every case with
`OpenAIServingChat` on that checkpoint and records the prompt and its token ids,
or the error SGLang raises, plus the `config.json` fields the processor reads.

Run from rust/sglang-processor in a SGLang Python environment with the
checkpoints cached:
    python tests/scripts/generate_parity.py [tests/fixtures/parity/<model>.json ...]
"""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# Fixtures must not depend on the generating machine's environment.
os.environ["SGLANG_DEFAULT_THINKING"] = "false"
os.environ["SGLANG_DSV4_REASONING_EFFORT"] = ""

from sglang.srt.entrypoints.openai import chat_encoding
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/parity"
DSV4_PROFILE = chat_encoding.DSV4_REASONING_EFFORT_PROFILE_OVERRIDE


class RecordingTokenizer:
    """Records the text the serving path encodes, which is the rendered prompt."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = []

    def encode(self, text):
        self.texts.append(text)
        return self.tokenizer.encode(text)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def serving_chat(path, revision, config, tokenizer, tool_call_parser):
    """The encoder selection `OpenAIServingChat.__init__` performs for this checkpoint."""
    server = object.__new__(OpenAIServingChat)
    server.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
    server.template_manager = SimpleNamespace(jinja_template_content_format="string")
    server.chat_encoding_spec = chat_encoding.resolve_chat_encoding_spec(
        hf_config=SimpleNamespace(
            model_type=config.get("model_type"),
            architectures=config.get("architectures"),
        ),
        tokenizer=tokenizer,
        tool_call_parser=tool_call_parser,
    )
    server._dsv4_reasoning_effort_profile = (
        chat_encoding.resolve_dsv4_reasoning_effort_profile(
            model_path=path, revision=revision, override=config.get(DSV4_PROFILE)
        )
        if server.chat_encoding_spec == "dsv4"
        else None
    )
    return server


def generate(fixture_path):
    fixture = json.loads(fixture_path.read_text())
    path = snapshot_download(
        fixture["model"], revision=fixture["revision"], local_files_only=True
    )
    config = json.loads(Path(path, "config.json").read_text())
    tok = RecordingTokenizer(AutoTokenizer.from_pretrained(path, local_files_only=True))
    server = serving_chat(
        path, fixture["revision"], config, tok, fixture.get("tool_call_parser")
    )
    # What the processor reads from config.json; the resolved profile stands in
    # for the checkpoint's encoder file so the Rust test runs offline.
    fixture["config"] = {
        key: config[key] for key in ("model_type", "architectures") if key in config
    }
    if server._dsv4_reasoning_effort_profile:
        fixture["config"][DSV4_PROFILE] = server._dsv4_reasoning_effort_profile
    fixture["bos_token_id"] = tok.bos_token_id
    fixture["cases"] = [{k: c[k] for k in ("name", "request")} for c in fixture["cases"]]
    for case in fixture["cases"]:
        tok.texts.clear()
        request = ChatCompletionRequest(**copy.deepcopy(case["request"]))
        # _convert_to_internal_request: kwargs effort replaces the request effort.
        if request.chat_template_kwargs:
            effort = request.chat_template_kwargs.pop("reasoning_effort", None)
            if effort is not None:
                request.reasoning_effort = effort
        try:
            ids = server._apply_jinja_template(
                request, tools=None, is_multimodal=False
            ).prompt_ids
        except Exception as error:
            case["error"] = f"{type(error).__name__}: {error}"
            continue
        case.update(
            prompt="".join(tok.texts),
            token_count=len(ids),
            token_sha256=hashlib.sha256(
                b"".join(i.to_bytes(4, "little") for i in ids)
            ).hexdigest(),
        )
    keys = ("model", "revision", "tool_call_parser", "config", "bos_token_id")
    head = {k: fixture[k] for k in keys if k in fixture}
    fixture_path.write_text(
        json.dumps(head)[:-1]
        + ', "cases": [\n'
        + ",\n".join(
            json.dumps(c, ensure_ascii=False, separators=(",", ":"))
            for c in fixture["cases"]
        )
        + "\n]}\n"
    )
    print(f"{fixture_path.name}: {len(fixture['cases'])} cases")


if __name__ == "__main__":
    for path in sys.argv[1:] or sorted(FIXTURES.glob("*.json")):
        generate(Path(path))
