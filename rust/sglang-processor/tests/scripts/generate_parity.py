"""Fill tests/fixtures/parity/*.json from SGLang's own serving path.

Each fixture names a checkpoint (`model`, `revision`, optionally the server's
`tool_call_parser`) and request `cases`, each optionally with the server's
`default_chat_template_kwargs`. This validates and renders every case with
`OpenAIServingChat` on that checkpoint and records the prompt and its token ids,
or the error SGLang raises, plus the `config.json` fields the processor reads
and the Jinja template a checkpoint renders with.

Run from rust/sglang-processor in a SGLang Python environment with the
checkpoints cached:
    python tests/scripts/generate_parity.py [tests/fixtures/parity/<model>.json ...]
"""

import copy
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# Fixtures must not depend on the generating machine's environment.
os.environ["SGLANG_DEFAULT_THINKING"] = "false"
os.environ["SGLANG_DSV4_REASONING_EFFORT"] = ""
os.environ["SGLANG_DSV41_REASONING_EFFORT"] = ""

from sglang.srt.entrypoints.openai import chat_encoding
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/parity"
DSV4_PROFILE = chat_encoding.DSV4_REASONING_EFFORT_PROFILE_OVERRIDE


class RecordingTokenizer:
    """Records the text the serving path encodes, which is the rendered prompt."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = []

    def encode(self, text, **kwargs):
        self.texts.append(text)
        return self.tokenizer.encode(text, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        # Kimi K3 tokenizes inside the template; record its text instead.
        if kwargs.get("tokenize"):
            text = self.tokenizer.apply_chat_template(
                *args, **kwargs | {"tokenize": False}
            )
            self.texts.append(text)
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def serving_chat(model, path, revision, config, tokenizer, tool_call_parser):
    """The state `OpenAIServingChat.__init__` resolves for this checkpoint, without
    a reasoning parser. Served text-only: the processor leaves media to the engine."""
    server = object.__new__(OpenAIServingChat)
    server.tokenizer_manager = SimpleNamespace(
        tokenizer=tokenizer,
        processor=None,
        model_config=SimpleNamespace(is_multimodal=False),
    )
    server.template_manager = TemplateManager()
    server.template_manager.load_chat_template(server.tokenizer_manager, None, model)
    server.tool_call_parser = tool_call_parser
    server.reasoning_parser = None
    server._reasoning_detector = None
    server.is_gpt_oss = server.is_gemma4 = False
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
    server._dsv41_default_reasoning_effort = (
        chat_encoding.default_dsv41_reasoning_effort_from_env(None)
        if server.chat_encoding_spec == "dsv41"
        else None
    )
    server._tokenizer_auto_adds_specials = len(tokenizer.encode("")) > 0
    server._prompt_text_round_trip_is_lossy = server._probe_prompt_text_round_trip()
    server._chat_template_cache = OrderedDict()
    return server


def generate(fixture_path):
    fixture = json.loads(fixture_path.read_text())
    path = snapshot_download(
        fixture["model"],
        revision=fixture["revision"],
        local_files_only=True,
        allow_patterns=["*.json", "*.jinja", "*.py", "*.model", "*.tiktoken"],
    )
    config = json.loads(Path(path, "config.json").read_text())
    # The server args `_validate_request` reads.
    reset_context()
    publish(ServerArgs(model_path=path), role="tokenizer")
    tok = RecordingTokenizer(
        AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=True
        )
    )
    server = serving_chat(
        fixture["model"],
        path,
        fixture["revision"],
        config,
        tok,
        fixture.get("tool_call_parser"),
    )
    # What the processor reads from config.json; the resolved profile stands in
    # for the checkpoint's encoder file so the Rust test runs offline.
    fixture["config"] = {
        key: config[key] for key in ("model_type", "architectures") if key in config
    }
    if server._dsv4_reasoning_effort_profile:
        fixture["config"][DSV4_PROFILE] = server._dsv4_reasoning_effort_profile
    fixture["bos_token_id"] = tok.bos_token_id
    # Jinja checkpoints render with their template; keep it so the test runs offline.
    if server.chat_encoding_spec is None:
        fixture["chat_template"] = tok.chat_template
    inputs = ("name", "request", "default_chat_template_kwargs", "known_gap")
    fixture["cases"] = [{k: c[k] for k in inputs if k in c} for c in fixture["cases"]]
    for case in fixture["cases"]:
        tok.texts.clear()
        server.default_chat_template_kwargs = case.get("default_chat_template_kwargs")
        try:
            request = ChatCompletionRequest(**copy.deepcopy(case["request"]))
            # Validation also rewrites tool schemas the prompt shows.
            invalid = server._validate_request(request)
            if invalid:
                raise ValueError(invalid)
            # _convert_to_internal_request: kwargs effort replaces the request effort.
            if request.chat_template_kwargs:
                effort = request.chat_template_kwargs.pop("reasoning_effort", None)
                if effort is not None:
                    request.reasoning_effort = effort
            ids = server._process_messages(request, is_multimodal=False).prompt_ids
        except AttributeError:
            raise  # The stub server lacks what this model's path reads; extend it.
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
    keys = (
        "model",
        "revision",
        "tool_call_parser",
        "config",
        "bos_token_id",
        "chat_template",
    )
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
