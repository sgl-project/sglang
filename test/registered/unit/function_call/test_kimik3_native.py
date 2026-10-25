"""Native framing regressions using synthetic inputs, including every split."""

import json
from types import SimpleNamespace

import pytest

from sglang.srt.function_call.kimik3_native import (
    KimiK3HeaderValidator,
    KimiK3NativeParser,
    KimiK3ParseError,
    restore_kimik3_stop,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

O, C, S = "<|open|>", "<|close|>", "<|sep|>"


def argument(key="command", kind="string", value="echo synthetic"):
    return f'{O}argument key="{key}" type="{kind}"{S}{value}{C}argument{S}'


def call(body=None, attrs='tool="bash" index="1"'):
    return f"{O}call {attrs}{S}{argument() if body is None else body}{C}call{S}"


def section(body=None):
    return f"{O}tools{S}{call() if body is None else body}{C}tools{S}"


def schedules(text):
    yield [text]
    yield list(text)
    yield [text[i : i + 7] for i in range(0, len(text), 7)]
    for split in range(len(text) + 1):
        yield [text[:split], text[split:]]


def parse(chunks, *, allow_incomplete=False):
    parser = KimiK3NativeParser()
    text, calls = "", []
    for chunk in chunks:
        result = parser.feed(chunk)
        text += result.normal_text
        calls.extend(result.calls)
    tail = parser.finish(allow_incomplete=allow_incomplete)
    return text + tail.normal_text, calls + tail.calls


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("", ""),
        (f".{C}think{S}", "."),
        (f"{O}response{S}ready{C}response{S}", "ready"),
        ("normal prose. ", "normal prose. "),
    ],
)
def test_valid_calls_at_every_split(prefix, expected):
    for chunks in schedules(prefix + section() + C + "message" + S):
        text, calls = parse(chunks)
        assert text == expected
        assert [(c.name, c.arguments) for c in calls] == [
            ("bash", {"command": "echo synthetic"})
        ]


def test_parallel_calls_and_multiple_complete_sections():
    for text in [section(call() + call()), section() + section()]:
        for chunks in schedules(text):
            _, calls = parse(chunks)
            assert len(calls) == 2


def test_unseparated_reasoning_preserves_text_without_native_wrappers():
    for close in [f"{C}think{S}", ""]:
        text = f"{O}think{S}thought" + close + section()
        for chunks in schedules(text):
            normal, calls = parse(chunks)
            assert normal == "thought"
            assert len(calls) == 1


@pytest.mark.parametrize(
    "text", ["plain < prose", "trailing <", "literal <|op", "think0 is text."]
)
def test_plain_text_and_literal_prefixes(text):
    for chunks in schedules(text):
        assert parse(chunks) == (text, [])


_BAD = [
    (f".{C}think0,60:080\u21923433/te" + section(), "invalid_header"),
    (f"{C}think unexpected{S}" + section(), "invalid_header"),
    (f"{O}tools{S}" + call(), "unclosed_channel"),
    (f"{O}tools{S}" + call() + f'{O}call tool="bash" index="2"{S}', "unclosed_call"),
    (section() + call(), "invalid_boundary"),
    (section(""), "empty_tools"),
    (section(section()), "invalid_boundary"),
    (section("garbage" + call()), "invalid_boundary"),
    (section(call("garbage" + argument())), "invalid_boundary"),
    (section(call(argument() + argument())), "duplicate_argument"),
    (section(call(attrs='tool="first" tool="bash" index="1"')), "invalid_attributes"),
    (section(call(attrs='garbage tool="bash" index="1"')), "invalid_attributes"),
    (section(call(attrs='tool="bash" index="0"')), "invalid_attributes"),
    (section(call(argument(kind="integer", value="1.5"))), "invalid_argument_type"),
    (section(call(argument(kind="integer", value="true"))), "invalid_argument_type"),
    (section(call(argument(kind="object", value="{bad"))), "invalid_json"),
    (
        section(call(argument(kind="object", value='{"x":1,"x":2}'))),
        "duplicate_argument",
    ),
    (section(call(argument(kind="number", value="NaN"))), "invalid_json"),
    (section(call(argument(kind="object", value='{"n":1e999}'))), "invalid_json"),
    (section(call(argument(kind="unknown"))), "invalid_argument_type"),
    (f"{C}tools{S}", "invalid_boundary"),
    (section() + f"{O}tools{S}", "unclosed_channel"),
    (f"{O}response{S}" + section(), "invalid_boundary"),
    (f"{O}response{S}unclosed", "unclosed_channel"),
]


@pytest.mark.parametrize("text,reason", _BAD)
def test_malformed_output_fails_at_every_split(text, reason):
    for chunks in schedules(text):
        with pytest.raises(KimiK3ParseError) as caught:
            parse(chunks)
        assert caught.value.reason == reason
        assert "echo synthetic" not in str(caught.value)


@pytest.mark.parametrize(
    "kind,raw,expected",
    [
        ("integer", "2", 2),
        ("number", "2.5", 2.5),
        ("boolean", "true", True),
        ("null", "null", None),
        ("array", '[1,"text"]', [1, "text"]),
        ("object", '{"x":1}', {"x": 1}),
        ("string", "&amp; raw < text", "&amp; raw < text"),
    ],
)
def test_typed_arguments(kind, raw, expected):
    for chunks in schedules(section(call(argument(kind=kind, value=raw)))):
        _, calls = parse(chunks)
        assert calls[0].arguments == {"command": expected}


def test_escaped_attributes_and_empty_call():
    _, calls = parse([section(call(argument(key="a&amp;&quot;b", value="")))])
    assert calls[0].arguments == {'a&"b': ""}
    _, calls = parse([section(call(""))])
    assert calls[0].arguments == {}


@pytest.mark.parametrize("marker", [C + "argument" + S, C + "think0", O + "tools" + S])
def test_json_string_markers_are_data_for_parser_and_header_guard(marker):
    expected = {"text": 'escaped quote " and slash \\ then ' + marker}
    text = section(call(argument(kind="object", value=json.dumps(expected))))
    for chunks in schedules(text):
        _, calls = parse(chunks)
        assert calls[0].arguments == {"command": expected}
        validator = KimiK3HeaderValidator()
        for chunk in chunks:
            validator.feed(chunk)
        validator.finish()


def test_raw_string_preserves_native_looking_data_at_every_split():
    value = f"{C}call{S}{C}tools{S}{O}tools{S}{C}think0 malformed < text"
    text = section(call(argument(value=value)))
    for chunks in schedules(text):
        _, calls = parse(chunks)
        assert calls[0].arguments == {"command": value}
        validator = KimiK3HeaderValidator()
        for chunk in chunks:
            validator.feed(chunk)
        validator.finish()


def test_argument_bound_is_enforced_for_utf8_bytes_and_failure_is_sticky():
    parser = KimiK3NativeParser(stream_arguments=True)
    parser.feed(
        f'{O}tools{S}{O}call tool="bash" index="1"{S}'
        f'{O}argument key="command" type="string"{S}'
    )
    with pytest.raises(KimiK3ParseError, match="argument_too_large"):
        parser.feed("中" * (4 * 1024 * 1024 // 3 + 1))
    with pytest.raises(KimiK3ParseError, match="argument_too_large"):
        parser.finish(allow_incomplete=True)


def test_complete_call_is_emitted_before_section_end():
    parser = KimiK3NativeParser()
    first = parser.feed(f"{O}tools{S}" + call())
    assert len(first.calls) == 1
    assert not parser.feed(f"{C}tools{S}").calls
    assert not parser.finish().calls


def test_failure_is_sticky_after_previously_emitted_call():
    parser = KimiK3NativeParser()
    assert len(parser.feed(f"{O}tools{S}" + call()).calls) == 1
    with pytest.raises(KimiK3ParseError):
        parser.feed("invalid")
    for action in [lambda: parser.feed(f"{C}tools{S}"), parser.finish]:
        with pytest.raises(KimiK3ParseError):
            action()


@pytest.mark.parametrize(
    "suffix",
    [
        f"{O}tools{S}",
        f"{O}tools{S}" + call(),
        f"{O}tools{S}" + call()[:-5],
        f"{O}response",
    ],
)
def test_known_length_truncation_does_not_fabricate_incomplete_call(suffix):
    for chunks in schedules(suffix):
        text, calls = parse(chunks, allow_incomplete=True)
        assert "<|" not in text
        assert len(calls) == (1 if suffix.endswith(C + "call" + S) else 0)


def test_length_does_not_excuse_invalid_complete_framing():
    with pytest.raises(KimiK3ParseError, match="invalid_header"):
        parse([f"{C}think0" + section()], allow_incomplete=True)


@pytest.mark.parametrize(
    "prefix", ["", f"{C}think{S}", f"{O}think{S}thought{C}think{S}"]
)
def test_raw_header_guard_accepts_prefilled_and_explicit_thinking(prefix):
    for chunks in schedules(prefix + section()):
        validator = KimiK3HeaderValidator()
        for chunk in chunks:
            validator.feed(chunk)
        validator.finish()


def test_raw_header_guard_catches_corruption_in_reasoning():
    for chunks in schedules(f"thought{C}think0" + section()):
        validator = KimiK3HeaderValidator()
        with pytest.raises(KimiK3ParseError, match="invalid_header"):
            for chunk in chunks:
                validator.feed(chunk)
            validator.finish()


def test_header_guard_distinguishes_length_from_unclosed_header():
    validator = KimiK3HeaderValidator()
    validator.feed("thought" + O + "think")
    validator.finish(allow_incomplete=True)
    with pytest.raises(KimiK3ParseError, match="unclosed_channel"):
        validator.finish()


@pytest.mark.parametrize("kind", ["stop", "length", "abort", None])
@pytest.mark.parametrize("matched", [C + "tools" + S, 123, None])
def test_only_proven_stop_suffix_can_be_restored(kind, matched):
    text = section()[: -len(S)]
    tokenizer = SimpleNamespace(decode=lambda *a, **kw: S)
    restored = restore_kimik3_stop(text, {"type": kind, "matched": matched}, tokenizer)
    expected = text
    if kind == "stop" and matched == 123:
        expected += S
    elif kind == "stop" and matched == C + "tools" + S:
        expected += (
            C + "tools" + S
        )  # matched full string means that entire string was trimmed
    assert restored == expected
    assert (
        restore_kimik3_stop(section(), {"type": kind, "matched": matched}, tokenizer)
        == section()
    )


def guard(chunks, *, allow_incomplete=False):
    validator = KimiK3HeaderValidator()
    for chunk in chunks:
        validator.feed(chunk)
    validator.finish(allow_incomplete=allow_incomplete)


@pytest.mark.parametrize(
    "text,reason",
    [
        (f"{O}response{S}hello" + section(), "invalid_boundary"),
        (f"thought{C}think{S}{O}response{S}hello" + section(), "invalid_boundary"),
        (f"{O}response{S}hello", "unclosed_channel"),
        (f"{O}response{S}a{O}response{S}b{C}response{S}", "invalid_boundary"),
        (f"{O}response{S}a{C}message{S}", "invalid_boundary"),
        (f"{O}response{S}a{C}response{S}{C}message{S}" + section(), "invalid_boundary"),
    ],
)
def test_raw_header_guard_keeps_response_boundaries_cleanup_removes(text, reason):
    # The raw parser rejects these inputs itself; reasoning cleanup would strip
    # the response/message markers before the tool parser could see them.
    with pytest.raises(KimiK3ParseError, match=reason):
        parse([text])
    for chunks in schedules(text):
        with pytest.raises(KimiK3ParseError, match=reason):
            guard(chunks)


@pytest.mark.parametrize(
    "text",
    [
        f"{O}response{S}hello{C}response{S}" + section(),
        f"thought{O}response{S}hi{C}response{S}{C}message{S}",
        f"thought{C}think{S}{O}response{S}hi{C}response{S}{C}message{S}",
        f"skipped think answer{C}response{S}{C}message{S}",
        f"{O}response{S}a{C}response{S}{O}response{S}b{C}response{S}",
        section() + f"{O}response{S}after tools{C}response{S}",
    ],
)
def test_raw_header_guard_accepts_supported_response_transitions(text):
    parse([text])
    for chunks in schedules(text):
        guard(chunks)


def test_raw_header_guard_allows_length_inside_response():
    guard([f"{O}response{S}partial"], allow_incomplete=True)
    with pytest.raises(KimiK3ParseError, match="unclosed_channel"):
        guard([f"{O}response{S}partial"])
