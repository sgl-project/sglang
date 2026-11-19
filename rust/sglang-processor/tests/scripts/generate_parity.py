"""Fill tests/fixtures/parity/*.json from SGLang's own serving path.

Each fixture names a checkpoint (`model`, `revision`, optionally the server's
`tool_call_parser`) and request `cases`, each optionally with the server's
`default_chat_template_kwargs`. This validates and renders every case with
`OpenAIServingChat` on that checkpoint and records the prompt and its token ids,
or the error SGLang raises, plus the `config.json` fields the processor reads
and the Jinja template a checkpoint renders with, with the special tokens its
context carries. The tokenizer comes from SGLang's `get_tokenizer`.

MiniMax cases may also supply `chat_template` for a server-side template override.

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

# Fixtures must not depend on the generating machine's environment.
os.environ["SGLANG_DEFAULT_THINKING"] = "false"
os.environ["SGLANG_DSV4_REASONING_EFFORT"] = ""
os.environ["SGLANG_DSV41_REASONING_EFFORT"] = ""

from sglang.srt.entrypoints.openai import chat_encoding
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.hf_transformers.tokenizer import get_tokenizer
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/parity"


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
    hf_config = SimpleNamespace(
        model_type=config.get("model_type"),
        architectures=config.get("architectures"),
    )
    server.tokenizer_manager = SimpleNamespace(
        tokenizer=tokenizer,
        processor=None,
        model_path=path,
        model_config=SimpleNamespace(is_multimodal=False, hf_config=hf_config),
    )
    server.template_manager = TemplateManager()
    server.template_manager.load_chat_template(server.tokenizer_manager, None, model)
    server.tool_call_parser = tool_call_parser
    server.reasoning_parser = None
    server._reasoning_detector = None
    server.is_gpt_oss = server.is_gemma4 = False
    server.is_minimax_m3 = config.get("model_type") == "minimax_m3_vl"
    server.chat_encoding_spec = chat_encoding.resolve_chat_encoding_spec(
        hf_config=hf_config,
        tokenizer=tokenizer,
        tool_call_parser=tool_call_parser,
    )
    server._encoding_adapter = server._build_chat_encoding_adapter()
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
    # SGLang's loader, whose fixes for transformers v5 the engine's tokenizer gets.
    tok = RecordingTokenizer(get_tokenizer(path, trust_remote_code=True))
    server = serving_chat(
        fixture["model"],
        path,
        fixture["revision"],
        config,
        tok,
        fixture.get("tool_call_parser"),
    )
    # What the processor reads from config.json.
    fixture["config"] = {
        key: config[key] for key in ("model_type", "architectures") if key in config
    }
    fixture["bos_token_id"] = tok.bos_token_id
    # Jinja checkpoints render with their template; keep it so the test runs offline,
    # with the special tokens the template context carries.
    if server.chat_encoding_spec is None:
        fixture["chat_template"] = tok.chat_template
        fixture["special_tokens"] = {
            name: getattr(tok, name)
            for name in ("bos_token", "eos_token", "unk_token")
            if getattr(tok, name) is not None
        }
    inputs = (
        "name",
        "request",
        "default_chat_template_kwargs",
        "known_gap",
        "chat_template",
    )
    fixture["cases"] = [{k: c[k] for k in inputs if k in c} for c in fixture["cases"]]
    for case in fixture["cases"]:
        tok.texts.clear()
        # A server-side --chat-template override keeps the checkpoint's model
        # type, but content normalization follows the replacement template.
        if "chat_template" in case and not server.is_minimax_m3:
            raise ValueError(
                "Per-case chat templates are currently supported for MiniMax-M3"
            )
        if server.is_minimax_m3:
            template = case.get("chat_template", fixture["chat_template"])
            tok.tokenizer.chat_template = template
            server.template_manager._jinja_template_content_format = (
                detect_jinja_template_content_format(template)
            )
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
            # An encoding adapter tokenizes with its own library, not `encode`.
            prompt=(
                tok.decode(ids, skip_special_tokens=False)
                if server._encoding_adapter is not None
                else "".join(tok.texts)
            ),
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
        "special_tokens",
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
