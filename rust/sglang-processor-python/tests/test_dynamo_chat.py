import asyncio
import json
from types import SimpleNamespace

import pytest
from dynamo.llm.exceptions import InvalidArgument
from sglang_processor import ChatProcessor
from sglang_processor.dynamo import DynamoChat
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


def processor(tmp_path, template, model_type="llama"):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE({ch: i for i, ch in enumerate(alphabet)}, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template})
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return ChatProcessor(str(tmp_path))


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("marker", "requested", "output", "expected"),
    [
        ("<think>", False, "private</think>public", ("private", "public")),
        ("</think>", True, "public", ("", "public")),
    ],
)
def test_glm_uses_rendered_reasoning_state(
    tmp_path, stream, marker, requested, output, expected
):
    # Even whitespace spanning multiple decode windows can follow the marker.
    chat = processor(tmp_path, "{{ messages[0].content }}" + marker + "{{ ' ' * 80 }}")
    request = {
        "model": "glm",
        "messages": [{"role": "user", "content": "Hi"}],
        "chat_template_kwargs": {"enable_thinking": requested},
        "stream": stream,
    }
    _, parser = chat.encode_reasoning_chat(json.dumps(request).encode(), "glm45")
    assert parser.push(output, True) == expected


def test_dynamo_declines_nonassistant_continuation(tmp_path):
    chat = DynamoChat(processor(tmp_path, "{{ messages[0].content }}"), None, [])
    request = {
        "model": "m",
        "messages": [{"role": "user", "content": "partial"}],
        "add_generation_prompt": False,
        "continue_final_message": True,
    }
    with pytest.raises(InvalidArgument, match="final assistant"):
        asyncio.run(anext(chat.generate(request)))


def test_kimi_public_usage_excludes_stub_on_worker_cache_hit(tmp_path):
    chat = processor(tmp_path, "", model_type="kimi_k3")
    request = {"model": "kimi", "messages": [{"role": "user", "content": "Hi"}]}
    prompt_len = len(chat.encode_chat(json.dumps(request).encode()))

    async def generate(worker, context):
        async def frames():
            yield SimpleNamespace(
                is_error=lambda: False,
                data=lambda: {
                    "token_ids": [],
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": len(worker["token_ids"]),
                        "prompt_tokens_details": {"cached_tokens": prompt_len},
                    },
                },
            )

        return frames()

    engine = DynamoChat(chat, SimpleNamespace(generate=generate), [])
    frame = asyncio.run(anext(engine.generate(request)))
    assert frame["usage"]["prompt_tokens"] == prompt_len - 3
    assert frame["usage"]["prompt_tokens_details"]["cached_tokens"] == prompt_len - 3
    assert frame["usage"]["total_tokens"] == prompt_len - 3
    assert frame["llm_metrics"]["input_tokens"] == prompt_len
