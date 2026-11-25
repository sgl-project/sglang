import asyncio
import json
from types import SimpleNamespace

import pytest
from dynamo.llm.exceptions import InvalidArgument
from sglang_processor import ChatProcessor
from sglang_processor.dynamo import DynamoChat, _load
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors


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


@pytest.fixture
def turn_tokenizer(tmp_path):
    # One turn-end spelling exists only in the base vocabulary. The other
    # two are added tokens, one of which is intentionally marked ordinary.
    words = ["[UNK]", "<|ifm|im_end|>", "prompt", "think", "response", "answer"]
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, "[UNK]")
    )
    tokenizer.decoder = decoders.Fuse()
    tokenizer.add_special_tokens(["<|open|>", "<|close|>", "<|sep|>", "<|eom_id|>"])
    tokenizer.add_tokens(["<|content_model_end_sampling|>"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages[0].content }}"})
    )
    return tokenizer


def run_generation(chat, eos, parser, batches, **options):
    workers = []

    async def generate(worker, context):
        workers.append(worker)

        async def frames():
            for i, ids in enumerate(batches):
                frame = {
                    "token_ids": ids,
                    "finish_reason": "length" if i == len(batches) - 1 else None,
                }
                yield SimpleNamespace(
                    is_error=lambda: False, data=lambda frame=frame: frame
                )

        return frames()

    async def run():
        adapter = DynamoChat(
            chat, SimpleNamespace(generate=generate), eos, reasoning_parser=parser
        )
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": "prompt"}],
            **options,
        }
        return [chunk["choices"][0] async for chunk in adapter.generate(request)]

    choices = asyncio.run(run())
    return workers[0], choices


def test_added_turn_end_tokens_stop_generation(tmp_path, turn_tokenizer):
    chat, eos = _load(str(tmp_path), None)
    token_id = turn_tokenizer.token_to_id
    ids = [token_id(w) for w in ("answer", "<|eom_id|>", "response")]
    worker, choices = run_generation(chat, eos, None, [ids])
    assert worker["stop_conditions"]["stop_token_ids_hidden"] == sorted(
        [token_id("<|eom_id|>"), token_id("<|content_model_end_sampling|>")]
    )
    assert choices[-1]["delta"]["content"] == "answer"
    assert choices[-1]["finish_reason"] == "stop"
    assert worker["require_reasoning"] is False
    worker, choices = run_generation(chat, eos, None, [ids], ignore_eos=True)
    assert "stop_token_ids_hidden" not in worker["stop_conditions"]
    assert choices[-1]["delta"]["content"] == "answerresponse"
    assert choices[-1]["finish_reason"] == "length"

    # A checkpoint can declare a turn end in tokenizer_config.json alone.
    config_file = tmp_path / "tokenizer_config.json"
    config = json.loads(config_file.read_text())
    config["added_tokens_decoder"] = {
        str(token_id("<|ifm|im_end|>")): {
            "content": "<|ifm|im_end|>",
            "special": True,
            "lstrip": False,
            "rstrip": False,
            "single_word": False,
            "normalized": False,
        }
    }
    config_file.write_text(json.dumps(config))
    _, configured_eos = _load(str(tmp_path), None)
    assert set(configured_eos) == {*eos, token_id("<|ifm|im_end|>")}


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_worker_receives_thinking_mode(tmp_path, enabled, separate):
    chat = processor(tmp_path, "{{ messages[0].content }}")
    worker, choices = run_generation(
        chat,
        [],
        "qwen3",
        [[chat.token_id("x")]],
        chat_template_args={"enable_thinking": enabled},
        separate_reasoning=separate,
    )
    assert worker["require_reasoning"] is enabled
    delta = choices[-1]["delta"]
    assert delta.get("reasoning_content", "") == ("x" if separate and enabled else "")
    assert delta["content"] == ("" if separate and enabled else "x")


@pytest.mark.parametrize("stream", [False, True])
def test_truncated_kimi_channel_name_stays_hidden(tmp_path, turn_tokenizer, stream):
    chat, eos = _load(str(tmp_path), None)
    # A literal "response" in the answer must survive; only the channel name
    # after the final <|close|> belongs to the truncated framing.
    output = [
        "think",
        "<|close|>",
        "think",
        "<|sep|>",
        "<|open|>",
        "response",
        "<|sep|>",
        "answer",
        "response",
        "<|close|>",
        "response",
    ]
    batches = [[turn_tokenizer.token_to_id(token)] for token in output]
    _, choices = run_generation(chat, eos, "kimi_k3", batches, stream=stream)
    assert "".join(c["delta"].get("reasoning_content", "") for c in choices) == "think"
    assert "".join(c["delta"]["content"] for c in choices) == "answerresponse"
    assert choices[-1]["finish_reason"] == "length"


def test_named_checkpoint_template_precedence(tmp_path):
    processor(tmp_path, "embedded")
    named = tmp_path / "additional_chat_templates"
    named.mkdir()
    (named / "rag.jinja").write_text(r"named\n{{ messages[0].content }}")
    request = json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
    ).encode()
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))

    def render(template=None):
        ids = ChatProcessor(str(tmp_path), template).encode_chat(request)
        return tokenizer.decode(ids)

    assert render() == r"named\nHi"
    (tmp_path / "chat_template.jinja").write_text("root")
    assert render() == "root"
    (named / "default.jinja").write_text("named default")
    assert render() == "named default"
    assert render(str(named / "rag.jinja")) == "named\nHi"


@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_marker_with_automatic_bos(tmp_path, turn_tokenizer, stream):
    turn_tokenizer.add_special_tokens(["[BOS]", "</think>"])
    turn_tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A",
        special_tokens=[("[BOS]", turn_tokenizer.token_to_id("[BOS]"))],
    )
    turn_tokenizer.save(str(tmp_path / "tokenizer.json"))
    chat = ChatProcessor(str(tmp_path))
    tokens = ["think", "</think>", "answer"]
    _, choices = run_generation(
        chat, [], "qwen3", [[chat.token_id(t)] for t in tokens], stream=stream
    )
    assert "".join(c["delta"].get("reasoning_content", "") for c in choices) == "think"
    assert "".join(c["delta"]["content"] for c in choices) == "answer"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("hidden_controls", [False, True])
def test_kimi_channel_segmentation_and_truncation(
    tmp_path, turn_tokenizer, stream, hidden_controls
):
    if not hidden_controls:
        # The official K3 tokenizer exposes the XTML controls in decoded text.
        config = json.loads(turn_tokenizer.to_str())
        for token in config["added_tokens"]:
            token["special"] = False
        turn_tokenizer = Tokenizer.from_str(json.dumps(config))
    turn_tokenizer.add_tokens(["res", "ponse"])
    turn_tokenizer.save(str(tmp_path / "tokenizer.json"))
    chat = ChatProcessor(str(tmp_path))
    tokens = [
        "think",
        "<|close|>",
        "think",
        "<|sep|>",
        "<|open|>",
        "res",
        "ponse",
        "<|sep|>",
        "answer",
    ]
    for tail in (
        ["<|close|>"],
        ["<|close|>", "response"],
        ["<|close|>", "response", "<|sep|>"],
    ):
        batches = [[turn_tokenizer.token_to_id(t)] for t in tokens + tail]
        _, choices = run_generation(chat, [], "kimi_k3", batches, stream=stream)
        assert (
            "".join(c["delta"].get("reasoning_content", "") for c in choices) == "think"
        )
        assert "".join(c["delta"]["content"] for c in choices) == "answer"
        assert choices[-1]["finish_reason"] == "length"


@pytest.mark.parametrize(
    ("stop", "expected"), [("[CUT]", "€"), ("€[CUT]", ""), ("[CUT]!", "€")]
)
def test_stop_preserves_buffered_utf8(tmp_path, stop, expected):
    words = ["[UNK]", "prompt", "<0xE2>", "<0x82>", "<0xAC>", "[CUT]", "!"]
    tokenizer = Tokenizer(
        models.BPE({s: i for i, s in enumerate(words)}, [], byte_fallback=True)
    )
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    tokenizer.add_special_tokens(["[CUT]"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "prompt"}')
    chat = ChatProcessor(str(tmp_path))
    batches = [[tokenizer.token_to_id(t)] for t in words[2:]]
    _, choices = run_generation(chat, [], None, batches, stop=stop)
    assert "".join(c["delta"]["content"] for c in choices) == expected
    assert choices[-1]["finish_reason"] == "stop"


def test_standalone_template_without_tokenizer_config(tmp_path):
    processor(tmp_path, "embedded")
    (tmp_path / "tokenizer_config.json").unlink()
    (tmp_path / "chat_template.jinja").write_text(
        "standalone {{ messages[0].content }}"
    )
    request = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
    body = json.dumps(request).encode()
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert (
        ChatProcessor(str(tmp_path)).encode_chat(body)
        == tokenizer.encode("standalone Hi").ids
    )


def test_tool_parser_selects_native_encoder_over_checkpoint_template(tmp_path):
    processor(tmp_path, "checkpoint")
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    request = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
    native = ChatProcessor(str(tmp_path), tool_call_parser="deepseekv32")
    ids = native.encode_chat(json.dumps(request).encode())
    assert "<｜User｜>Hi<｜Assistant｜>" in tokenizer.decode(ids)
