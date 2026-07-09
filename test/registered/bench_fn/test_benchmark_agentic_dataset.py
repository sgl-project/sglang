"""Offline unit tests for the agentic multi-turn benchmark dataset: HF
streams are injected or patched, sizing runs on a deterministic character
tokenizer, and prebuilt mode uses the checked-in fixture."""

import argparse
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import sglang.benchmark.datasets.agentic as agentic_mod
from sglang.benchmark.datasets import DATASET_MAPPING
from sglang.benchmark.datasets.agentic import (
    PAD_SIZING_MAX_DEFICIT_TOKENS,
    AgenticDataset,
    SciencePadPool,
    _extract_messages,
    _sanitize_r1,
    _truncate_to_bare_tokens,
    build_agentic_conversations,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "agentic_prebuilt.json"


class CharTokenizer:
    """Zero-drift character tokenizer: 1 token == 1 char, decode is exact."""

    name_or_path = "char-tokenizer-test"

    def __init__(self):
        self.all_special_ids = [0]

    def get_added_vocab(self):
        return {"<extra>": 1114000}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            chr(i) for i in ids if not (skip_special_tokens and i in (0, 1114000))
        )

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, return_dict=True
    ):
        # Mimics transformers >= 5: tokenize=True returns a dict unless
        # return_dict=False, so len() over the raw result counts dict keys.
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        ids = self.encode(text)
        return (
            {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids
        )


class DriftTokenizer(CharTokenizer):
    """Decoding a slice re-encodes longer (emulates byte-fallback drift)."""

    def decode(self, ids, skip_special_tokens=True):
        text = super().decode(ids, skip_special_tokens)
        return "".join(c + c if i % 10 == 5 else c for i, c in enumerate(text))


def make_source_rows(n=10, with_blocks=True, field="trajectory"):
    """OpenHands-style trajectories: content blocks, tool role, assistants."""
    rows = []
    for i in range(n):
        first_user = (
            [{"type": "text", "text": f"issue {i} needs fixing " * 3}]
            if with_blocks
            else f"issue {i} needs fixing " * 3
        )
        rows.append(
            {
                field: [
                    {"role": "system", "content": "source system prompt"},
                    {"role": "user", "content": first_user},
                    {"role": "assistant", "content": "SOURCE_ASSISTANT_TEXT"},
                    {"role": "tool", "content": f"tool observation {i}"},
                    {"role": "weird", "content": "dropped role"},
                    {"role": "user", "content": ""},
                ],
                "resolved": 1 if i % 2 == 0 else 0,
            }
        )
    return rows


def make_pad_rows(n=20):
    text = (
        "<think>chain of thought</think> the quick brown fox jumps over "
        "the lazy dog near the river bank. <|control|> and <｜ctrl｜> gone. " * 200
    )
    return [{"output": text} for _ in range(n)]


BUILD_KW = dict(
    model_path="test-model",
    source_dataset="fake/source",
    source_split="train",
    source_field="trajectory",
    only_resolved=False,
    first_turn_len=64,
    subsequent_turn_len=16,
    num_turns=4,
    num_conversations=3,
    pad_dataset="fake/pads",
    pad_split="train",
    pad_text_field="output",
    max_real_first_turn_frac=0.5,
    output_len=20,
    seed=42,
)


def build(tok, pad_source, **overrides):
    return build_agentic_conversations(
        tok,
        pad_source=pad_source,
        source_rows=iter(make_source_rows()),
        pad_rows=iter(make_pad_rows()),
        **{**BUILD_KW, **overrides},
    )


def bare_turn_lens(tok, conversation):
    return [
        sum(len(tok.encode(m["content"])) for m in turn["messages"])
        for turn in conversation
    ]


def make_dataset(**overrides):
    fields = dict(
        num_prompts=2,
        dataset_path="",
        source_dataset="fake/source",
        source_split="train",
        source_field="trajectory",
        only_resolved=False,
        first_turn_len=64,
        subsequent_turn_len=16,
        num_turns=4,
        num_conversations=3,
        pad_source="openscience",
        pad_dataset="fake/pads",
        pad_split="train",
        pad_text_field="output",
        max_real_first_turn_frac=0.5,
        offset=0,
        output_len=20,
        cache_path="",
        rebuild=False,
        seed=42,
    )
    fields.update(overrides)
    return AgenticDataset(**fields)


class FakeStreams:
    """Patches the HF streaming hook and counts how often it is hit."""

    def __init__(self):
        self.calls = 0

    def __call__(self, dataset_name, split):
        self.calls += 1
        return iter(
            make_source_rows() if dataset_name == "fake/source" else make_pad_rows()
        )


class TestRegistration(CustomTestCase):
    def test_registered_and_from_args_maps_flags(self):
        self.assertIs(DATASET_MAPPING["agentic"], AgenticDataset)
        args = Namespace(
            backend="sglang-oai-chat",
            num_prompts=4,
            dataset_path="",
            seed=7,
            agentic_source_dataset="a/b",
            agentic_source_split="tool",
            agentic_source_field="messages",
            agentic_only_resolved=True,
            agentic_first_turn_len=100,
            agentic_subsequent_turn_len=10,
            agentic_num_turns=5,
            agentic_num_conversations=8,
            agentic_pad_source="random",
            agentic_pad_dataset="c/d",
            agentic_pad_split="train",
            agentic_pad_text_field="output",
            agentic_max_real_first_turn_frac=0.25,
            agentic_offset=2,
            agentic_output_len=64,
            agentic_cache_path="/tmp/x.json",
            agentic_rebuild=True,
        )
        ds = AgenticDataset.from_args(args)
        self.assertEqual(
            (ds.source_split, ds.source_field, ds.offset, ds.output_len, ds.rebuild),
            ("tool", "messages", 2, 64, True),
        )

    def test_backend_validation(self):
        from sglang.benchmark.serving import _validate_parsed_agentic_args

        with self.assertRaisesRegex(ValueError, "multi-turn chat backend"):
            AgenticDataset.from_args(Namespace(backend="sglang"))

        parser = argparse.ArgumentParser()
        with self.assertRaises(SystemExit):
            _validate_parsed_agentic_args(
                parser,
                Namespace(dataset_name="agentic", backend="sglang", agentic_offset=0),
            )
        # Other datasets are not affected by the agentic validator.
        _validate_parsed_agentic_args(
            parser, Namespace(dataset_name="sharegpt", backend="sglang")
        )


class TestNormalization(CustomTestCase):
    def test_roles_blocks_and_control_markup(self):
        msgs = _extract_messages(make_source_rows(1)[0]["trajectory"])
        self.assertEqual(
            [m["role"] for m in msgs], ["system", "user", "assistant", "user"]
        )
        self.assertIn("issue 0", msgs[1]["content"])
        self.assertEqual(msgs[3]["content"], "tool observation 0")

        cleaned = _sanitize_r1("<think>x</think> a <|sys|> b <｜user｜> c < think > d")
        for marker in ("<think>", "</think>", "<|sys|>", "<｜user｜>", "< think >"):
            self.assertNotIn(marker, cleaned)
        for kept in ("a", "b", "c", "d"):
            self.assertIn(kept, cleaned)


class TestBuilderShapes(CustomTestCase):
    def test_exact_shape_and_pad_hygiene_both_sources(self):
        tok = CharTokenizer()
        shapes, sample_pads = {}, {}
        for pad_source in ("openscience", "random"):
            payload = build(tok, pad_source)
            self.assertEqual(len(payload["conversations"]), 3)
            sys_pads = []
            for conv in payload["conversations"]:
                self.assertEqual(bare_turn_lens(tok, conv), [64, 16, 16, 16])
                sys_pads.append(conv[0]["messages"][0]["content"])
                for turn in conv:
                    for m in turn["messages"]:
                        for forbidden in (
                            "SOURCE_ASSISTANT_TEXT",
                            "<think>",
                            "</think>",
                            "<|control|>",
                            "<｜ctrl｜>",
                        ):
                            self.assertNotIn(forbidden, m["content"])
            # Pads are unique per conversation (disjoint spans / per-conv RNG).
            self.assertEqual(len(set(sys_pads)), len(sys_pads))
            shapes[pad_source] = [
                bare_turn_lens(tok, c) for c in payload["conversations"]
            ]
            sample_pads[pad_source] = sys_pads[0]
        self.assertEqual(shapes["openscience"], shapes["random"])
        self.assertNotEqual(sample_pads["openscience"], sample_pads["random"])

    def test_turn_structure_and_first_turn_cap(self):
        tok = CharTokenizer()
        payload = build(tok, "random", max_real_first_turn_frac=0.25)
        for conv in payload["conversations"]:
            self.assertEqual(
                [m["role"] for m in conv[0]["messages"]], ["system", "user"]
            )
            for turn in conv[1:]:
                self.assertEqual([m["role"] for m in turn["messages"]], ["user"])
            real_len = len(tok.encode(conv[0]["messages"][1]["content"]))
            self.assertLessEqual(real_len, int(64 * 0.25))

    def test_random_pads_reproducible_per_seed(self):
        tok = CharTokenizer()
        self.assertEqual(
            build(tok, "random")["conversations"],
            build(tok, "random")["conversations"],
        )
        self.assertNotEqual(
            build(tok, "random")["conversations"][0][0]["messages"][0]["content"],
            build(tok, "random", seed=7)["conversations"][0][0]["messages"][0][
                "content"
            ],
        )

    def test_turn_synthesis_skipping_and_exhaustion(self):
        tok = CharTokenizer()
        # Unusable rows are skipped; a 1-user-message trajectory still yields
        # all turns (synthesized, never skipped).
        rows = [
            {"trajectory": [{"role": "assistant", "content": "no user"}]},
            {"trajectory": []},
            {"no_field": True},
            {
                "trajectory": [
                    {"role": "user", "content": [{"type": "text", "text": "only one"}]}
                ]
            },
        ]
        payload = build_agentic_conversations(
            tok,
            pad_source="random",
            source_rows=iter(rows),
            pad_rows=None,
            **{**BUILD_KW, "num_conversations": 1},
        )
        self.assertEqual(
            bare_turn_lens(tok, payload["conversations"][0]), [64, 16, 16, 16]
        )

        with self.assertRaisesRegex(ValueError, "ran out"):
            build_agentic_conversations(
                tok,
                pad_source="random",
                source_rows=iter(make_source_rows(2)),
                pad_rows=None,
                **{**BUILD_KW, "num_conversations": 5},
            )

    def test_source_variants(self):
        tok = CharTokenizer()
        # only_resolved keeps rows with resolved == 1 (even indices).
        payload = build_agentic_conversations(
            tok,
            pad_source="random",
            source_rows=iter(make_source_rows()),
            pad_rows=None,
            **{**BUILD_KW, "only_resolved": True, "num_conversations": 2},
        )
        first_users = [c[0]["messages"][1]["content"] for c in payload["conversations"]]
        self.assertIn("issue 0", first_users[0])
        self.assertIn("issue 2", first_users[1])

        # SWE-smith style: plain-string content in a "messages" column that is
        # a JSON string; invalid JSON rows are skipped.
        rows = [
            {"messages": json.dumps(r["messages"]), "resolved": r["resolved"]}
            for r in make_source_rows(3, with_blocks=False, field="messages")
        ]
        rows.insert(0, {"messages": "{not valid json"})
        payload = build_agentic_conversations(
            tok,
            pad_source="random",
            source_rows=iter(rows),
            pad_rows=None,
            **{**BUILD_KW, "source_field": "messages", "num_conversations": 2},
        )
        for conv in payload["conversations"]:
            self.assertEqual(bare_turn_lens(tok, conv), [64, 16, 16, 16])

    def test_prompt_tokens_metadata(self):
        tok = CharTokenizer()
        conv = build(tok, "random")["conversations"][0]
        expected_t1 = len(
            tok.apply_chat_template(
                conv[0]["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        # Guards against counting dict keys when a tokenizer returns a mapping
        # (the transformers >= 5 default for tokenize=True).
        self.assertGreater(expected_t1, 60)
        self.assertEqual(conv[0]["prompt_tokens"], expected_t1)
        for k in range(1, 4):
            self.assertEqual(conv[k]["prompt_tokens"], expected_t1 + k * (16 + 20))


class TestExactSizing(CustomTestCase):
    def test_trim_never_exceeds_and_deficit_bounded(self):
        tok = CharTokenizer()
        self.assertEqual(
            _truncate_to_bare_tokens("x" * 500, 100, tok), ("x" * 100, 100)
        )
        self.assertEqual(_truncate_to_bare_tokens("abc", 100, tok), ("abc", 3))

        drift = DriftTokenizer()
        for target in (10, 33, 100, 257):
            sized, realized = _truncate_to_bare_tokens(
                "x" * (target * 3), target, drift
            )
            self.assertEqual(realized, len(drift.encode(sized)))
            self.assertLessEqual(realized, target)
            self.assertLessEqual(target - realized, PAD_SIZING_MAX_DEFICIT_TOKENS)

    def test_pool_spans_disjoint_and_exhaustion_fallback(self):
        tok = CharTokenizer()
        pool = SciencePadPool(
            tok,
            dataset_name="fake/pads",
            split="train",
            text_field="output",
            needed_tokens=100,
            rows=iter(make_pad_rows(2)),
        )
        a, b = pool.next_pad(50), pool.next_pad(50)
        self.assertEqual((len(tok.encode(a)), len(tok.encode(b))), (50, 50))
        self.assertNotEqual(a, b)
        self.assertIsNone(pool.next_pad(10**6))

        # A too-small pool falls back to random padding, preserving shape.
        payload = build_agentic_conversations(
            tok,
            pad_source="openscience",
            source_rows=iter(make_source_rows()),
            pad_rows=iter([{"output": "tiny"}]),
            **BUILD_KW,
        )
        for conv in payload["conversations"]:
            self.assertEqual(bare_turn_lens(tok, conv), [64, 16, 16, 16])


class TestCaching(CustomTestCase):
    def setUp(self):
        self.tok = CharTokenizer()
        self.cache_file = Path(tempfile.mkdtemp()) / "agentic-cache.json"
        self.streams = FakeStreams()
        patcher = patch.object(agentic_mod, "_stream_hf_rows", self.streams)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _dataset(self, **overrides):
        return make_dataset(cache_path=str(self.cache_file), **overrides)

    def test_build_reuse_rebuild_and_row_contract(self):
        rows = self._dataset().load(self.tok, model_id="test-model")
        self.assertEqual(len(rows), 2)
        # Emitted rows carry the multi-turn contract fields.
        row = rows[0]
        self.assertEqual(len(row.prompt), 4)
        self.assertEqual(row.prompt[0][0]["role"], "system")
        self.assertEqual(row.prompt_len, row.prompt_lens[0])
        self.assertEqual(row.output_len, 20)
        self.assertEqual(
            [b - a for a, b in zip(row.prompt_lens, row.prompt_lens[1:])],
            [16 + 20] * 3,
        )
        self.assertEqual(
            json.loads(self.cache_file.read_text())["metadata"]["tokenizer_path"],
            "char-tokenizer-test",
        )

        calls_after_build = self.streams.calls
        self._dataset().load(self.tok, model_id="test-model")
        self.assertEqual(self.streams.calls, calls_after_build)  # cache reused

        self._dataset(seed=7).load(self.tok, model_id="test-model")
        calls_after_mismatch = self.streams.calls
        self.assertGreater(calls_after_mismatch, calls_after_build)  # rebuilt

        self._dataset(seed=7, rebuild=True).load(self.tok, model_id="test-model")
        self.assertGreater(self.streams.calls, calls_after_mismatch)  # forced

    def test_cache_smaller_than_needed_auto_expands(self):
        self._dataset().load(self.tok, model_id="test-model")
        self._dataset(offset=3, num_prompts=2).load(self.tok, model_id="test-model")
        payload = json.loads(self.cache_file.read_text())
        self.assertEqual(payload["metadata"]["num_conversations"], 5)

    def test_offset_slices_are_disjoint(self):
        first = self._dataset(num_prompts=1).load(self.tok, model_id="test-model")
        second = self._dataset(offset=1, num_prompts=1).load(
            self.tok, model_id="test-model"
        )
        self.assertNotEqual(first[0].prompt[0], second[0].prompt[0])

    def test_corrupt_cache_warns_and_rebuilds(self):
        self._dataset().load(self.tok, model_id="test-model")
        self.cache_file.write_text("{corrupt")
        calls_before = self.streams.calls
        rows = self._dataset().load(self.tok, model_id="test-model")
        self.assertGreater(self.streams.calls, calls_before)
        self.assertEqual(len(rows), 2)

    def test_output_len_change_reuses_cache_and_recomputes_lens(self):
        self._dataset().load(self.tok, model_id="test-model")
        calls_after_build = self.streams.calls
        rows = self._dataset(output_len=100).load(self.tok, model_id="test-model")
        self.assertEqual(self.streams.calls, calls_after_build)
        self.assertEqual(rows[0].prompt_lens[1], rows[0].prompt_lens[0] + 16 + 100)


class TestPrebuilt(CustomTestCase):
    """Prebuilt mode must be fully offline: the stream hook must not be hit."""

    def setUp(self):
        def fail(*a, **k):
            raise AssertionError("prebuilt mode must not stream from HF")

        patcher = patch.object(agentic_mod, "_stream_hf_rows", fail)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tok = CharTokenizer()

    def _dataset(self, **overrides):
        overrides.setdefault("dataset_path", str(FIXTURE_PATH))
        return make_dataset(**overrides)

    def test_legacy_fixture_loads_with_offset(self):
        rows = self._dataset(num_prompts=3, num_turns=3).load(self.tok)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].prompt_len, 40)
        # File metadata wins (subsequent_turn_length=8 from the fixture);
        # the requested output_len drives the per-round arithmetic.
        self.assertEqual(rows[0].prompt_lens, [40, 40 + 8 + 20, 40 + 2 * (8 + 20)])

        offset_rows = self._dataset(offset=1, num_prompts=2).load(self.tok)
        self.assertIn("repo two", offset_rows[0].prompt[0][1]["content"])

    def test_over_consumption_hard_errors(self):
        with self.assertRaisesRegex(ValueError, "requires 4"):
            self._dataset(offset=2, num_prompts=2).load(self.tok)

    def test_invalid_files_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{corrupt")
        with self.assertRaises(json.JSONDecodeError):
            self._dataset(dataset_path=f.name).load(self.tok)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"something": "else"}, f)
        with self.assertRaisesRegex(ValueError, "metadata"):
            self._dataset(dataset_path=f.name).load(self.tok)


if __name__ == "__main__":
    unittest.main()
