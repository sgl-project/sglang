import sys
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.kimi_token_provenance import KimiTokenProvenance
from sglang.srt.function_call.kimik3_format import RESPONSE_OPEN, THINK_CLOSE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def ledger(pieces):
    tokenizer = SimpleNamespace(
        model=SimpleNamespace(decode_single_token_bytes=pieces.__getitem__)
    )
    result = KimiTokenProvenance(tokenizer)
    result.append(range(len(pieces)))
    return result


def ids(value):
    return value[0]["token_ids"]


def test_one_batch_thinking_and_answer_have_distinct_tokens():
    source = ledger([b"same", THINK_CLOSE.encode(), RESPONSE_OPEN.encode(), b"same"])
    assert ids(source.take("same", text=True, reasoning=True)) == [0]
    assert ids(source.take("same", text=True)) == [3]


def test_utf8_split_and_repeated_token_positions():
    source = ledger([b"\xe4\xb8", b"\xad", b"!", b"!"])
    assert ids(source.take("中", text=True)) == [0, 1]
    assert ids(source.take("!!", text=True)) == [2, 3]


def test_a_token_can_contribute_to_two_increments():
    source = ledger([b"abcdef"])
    assert ids(source.take("abc", text=True)) == [0]
    assert ids(source.take("def", text=True)) == [0]


def test_markers_in_argument_value_are_not_stripped():
    source = ledger([RESPONSE_OPEN.encode()])
    assert ids(source.take(RESPONSE_OPEN)) == [0]


def test_replayed_reasoning_uses_its_original_tokens():
    source = ledger([b"first", b" second", b" tail"])
    assert ids(source.take("first", text=True, reasoning=True)) == [0]
    assert ids(source.take(" second", text=True, reasoning=True)) == [1]
    assert ids(source.take("first second tail", text=True, replay_chars=12)) == [
        0,
        1,
        2,
    ]


def test_text_spanning_removed_wrappers():
    source = ledger([b"a", RESPONSE_OPEN.encode(), b"b"])
    assert ids(source.take("ab", text=True)) == [0, 2]


def test_same_token_id_generated_twice_is_not_deduplicated():
    source = ledger([b"a"])
    source.append([0])
    assert ids(source.take("aa", text=True)) == [0, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
