"""Regenerate Kimi IDs with SGLang and the pinned checkpoint tokenizer.

Run from experimental/sgl-router in a SGLang Python environment.
"""

import base64
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

from huggingface_hub import hf_hub_download
from tokenizers import AddedToken

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
    ThinkingMode,
    normalize_assistant_tool_call_arguments,
)

REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
for name in ("encoding_k3.py", "tokenization_kimi.py"):
    source = hf_hub_download("moonshotai/Kimi-K3", name, revision=REVISION)
sys.path.insert(0, str(pathlib.Path(source).parent))
from tokenization_kimi import TikTokenTokenizer  # noqa: E402

fixture = pathlib.Path(__file__).resolve().parents[1] / "fixtures/kimi_k3"
config = json.loads((fixture / "tokenizer_config.json").read_text())
config["added_tokens_decoder"] = {
    int(k): AddedToken(**v) for k, v in config["added_tokens_decoder"].items()
}
tokens = [bytes([b]) for b in range(256)]
tokens += [s.encode() for s in (fixture / "merges.txt").read_text().split()]
vocab = "".join(
    f"{base64.b64encode(token).decode()} {rank}\n" for rank, token in enumerate(tokens)
)
with tempfile.NamedTemporaryFile(suffix=".model", mode="w+") as model:
    model.write(vocab)
    model.flush()
    tokenizer = TikTokenTokenizer(model.name, **config)
server = object.__new__(OpenAIServingChat)
server.chat_encoding_spec = "kimi_k3"
server.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
cases = json.loads((fixture / "prompts.json").read_text())
for case in cases:
    data = copy.deepcopy(case["request"])
    if "repeat" in case:
        data["messages"][0]["content"] *= case["repeat"]
    request = ChatCompletionRequest(**data)
    messages = [message.model_dump() for message in request.messages]
    for message in messages:
        normalize_assistant_tool_call_arguments(message, strict=False)
    ids = server._encode_messages(messages, request, ThinkingMode.THINKING)
    case["token_count"] = len(ids)
    case["sha256"] = hashlib.sha256(
        b"".join(token.to_bytes(4, "little") for token in ids)
    ).hexdigest()
(fixture / "prompts.json").write_text(
    "[\n"
    + ",\n".join(
        json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cases
    )
    + "\n]\n"
)
