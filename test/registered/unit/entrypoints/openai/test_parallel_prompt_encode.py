"""Chunked prompt encoding must produce the ids the single encode produces.

`parallel_prompt_encode` cuts a rendered prompt and sends the pieces through
`backend.encode_batch`, so its whole contract is that the concatenated ids are
identical to `tokenizer.encode(text, **encode_kwargs)` — an off-by-one cut
point is not a crash, it is a silently different prompt. These cases pin the
identity on a ByteLevel-BPE tokenizer built in memory with the exact
Split-regex + ByteLevel + BPE pipeline the intra-gap cuts were verified for,
and pin the refusals: every tokenizer configuration outside that pipeline, and
every gate, must reach the original encode instead.

The tests never assert that the rayon pool actually ran the chunks in
parallel. The helper probes for a serial `encode_batch` once and then disables
itself for the process, which is a correctness-preserving decision a CPU
runner is free to make; `_parallel_ok` is pinned per test so the id
comparisons do not depend on it.
"""

import random
import re
import sys
import unittest
import unittest.mock
import weakref
from functools import lru_cache

import pytest
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai import parallel_prompt_encode as ppe
from sglang.srt.environ import envs
from sglang.srt.parser.glm53_chat_template import (
    OrdinaryText,
    render_glm53_flash,
    render_glm53_flash_segments,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=22, suite="base-a-test-cpu")

_TRAIN_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world, this is a test of the tokenizer",
    "def main(argv):\n    return 0\n",
    "it's they're I'm we'll you'd she's",
    "中文文本 日本語 한국어 émigré naïve Ωmega",
    "numbers 123 4567 89 1,234.56 <tag/> [bracket]",
]
_ALL_KWARGS = ({}, {"add_special_tokens": True}, {"add_special_tokens": False})
_SPECIALS = [
    "<|user|>",
    "<|assistant|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "[gMASK]",
    "<sop>",
    "/nothink",
]


@lru_cache(maxsize=1)
def _safe_backend_json():
    """A GLM / Llama-3 shaped backend: the pipeline `_intra_cut_safe` accepts.

    Only the pre-tokenizer pipeline and the model type decide which cut points
    the helper takes, so a 900-token vocabulary exercises the same code path a
    real checkpoint does. The ByteLevel alphabet is seeded explicitly so every
    byte of the test corpus has a token and nothing is silently dropped.
    """
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(ppe._INTRA_SPLIT_REGEX), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        _TRAIN_CORPUS * 20,
        trainers.BpeTrainer(
            vocab_size=900,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
        ),
    )
    backend.add_tokens(
        [AddedToken(t, normalized=False, special=True) for t in _SPECIALS]
    )
    return backend.to_str()


def _tokenizer(mutate=None, **kwargs):
    backend = Tokenizer.from_str(_safe_backend_json())
    if mutate is not None:
        mutate(backend)
    return PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)


def _chunk_ids(tokenizer, text, all_cuts=False):
    """Encode `text` chunk by chunk, the way the helper splits it.

    Each chunk goes through the backend on this thread, so the comparison is
    the plan's cut points against a single encode — not `encode_batch`'s
    threading.
    """
    backend, _, pattern, npattern, intra = ppe._plan_for(tokenizer)
    if all_cuts:
        # Target size 1: take every candidate cut point, not one per 2 KiB.
        with unittest.mock.patch.object(ppe, "_MIN_CHUNK_CHARS", 1):
            chunks = ppe.split_prompt(
                text, pattern, npattern, intra, max_chunks=1 << 30
            )
    else:
        chunks = ppe.split_prompt(text, pattern, npattern, intra)
    ids = []
    for chunk in chunks:
        ids.extend(backend.encode(chunk, add_special_tokens=False).ids)
    return ids, chunks


_EDGE_TEXTS = [
    "",
    "a",
    " ",
    "\n",
    "abc def",
    "no specials at all\nline two\r\nline3 foo",
    "<|user|>",
    "<|user|><|assistant|>",
    "<|user|>x<|assistant|>",
    "<|user|> hello world",
    "hello world <|assistant|>",
    "[gMASK]<sop><|user|>\nhi<|assistant|>\n<think>reasoning</think>answer",
    "/nothinking",
    "/nothinker",
    "<think",
    "think>",
    "<<think>>",
    "中文文本，没有空格。日本語のテキスト 한국어",
    "emoji 🙂🙃 x 👩‍👩‍👧 é é \u200b x",
    "tabs\tand  double  spaces\n\n\nX\n \nY \n  Z",
    "it's they're I'M we'll 'd x's",
    "\r\nA\rB\n\rC",
    "123456789 abc123def 1,234.56",
    "word Line Para",
    "Ωmega ΣΙΓΜΑ ßtraße İi",
    "a" * 5000 + " b",
    "x y\n" * 3000,
]


def _fuzz_texts():
    """Random texts over the atoms that sit on either side of a cut point."""
    rng = random.Random(0)
    atoms = list("aZ é中🙂'1 \n\r\t.,;:{}()<>/_-\"\\") + [
        "\u0301",
        " ",
        "'s",
        "  ",
        "\r\n",
        " \n",
        "\n ",
    ]
    atoms += _SPECIALS
    texts = [
        "".join(rng.choice(atoms) for _ in range(rng.randint(0, 400)))
        for _ in range(4000)
    ]
    texts += [
        "".join(rng.choice(atoms) for _ in range(rng.randint(5000, 60000)))
        for _ in range(30)
    ]
    return texts


class _EncodeCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        # `_plans` and `_parallel_ok` are process-wide and sticky: a plan
        # cached for a mutated tokenizer, or a probe that fell back, would
        # leak into whatever case runs next.
        self.addCleanup(setattr, ppe, "_plans", ppe._plans)
        ppe._plans = weakref.WeakKeyDictionary()
        self.addCleanup(setattr, ppe, "_parallel_ok", ppe._parallel_ok)
        ppe._parallel_ok = True

    def _set_min_chars(self, value):
        override = envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.override(value)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)

    def _no_min_chars(self):
        self._set_min_chars(0)

    def _assert_matches_encode(self, tokenizer, text, kwargs_variants=_ALL_KWARGS):
        for encode_kwargs in kwargs_variants:
            self.assertEqual(
                ppe.parallel_prompt_encode(tokenizer, text, encode_kwargs),
                tokenizer.encode(text, **encode_kwargs),
                f"{encode_kwargs} on {text[:60]!r}",
            )


class TestSafePipelineIdsAreIdentical(_EncodeCase):
    def setUp(self):
        super().setUp()
        self._no_min_chars()
        self.tokenizer = _tokenizer()

    def test_the_plan_takes_every_kind_of_cut_point(self):
        """Without this the id comparisons below could pass on no cuts at all."""
        plan = ppe._plan_for(self.tokenizer)
        self.assertIsNotNone(plan)
        self.assertIsNotNone(plan[2], "added tokens must be matched")
        self.assertIs(plan[4], ppe._INTRA_CUT, "intra-gap cuts must be enabled")
        for text, expected in (
            ("<think>abcd", ["<think>", "abcd"]),
            ("abcd efgh", ["abcd", " efgh"]),
            ("abcd\nefgh", ["abcd\n", "efgh"]),
        ):
            self.assertEqual(
                _chunk_ids(self.tokenizer, text, all_cuts=True)[1], expected
            )

    def test_edge_cases_encode_the_same_at_every_cut_point(self):
        texts = list(_EDGE_TEXTS)
        for text in _EDGE_TEXTS:
            texts += [
                "<think>" + text,
                text + "</think>",
                "<think>" + text + "</think>",
            ]
        for text in texts:
            reference = self.tokenizer.encode(text)
            self.assertEqual(
                _chunk_ids(self.tokenizer, text, all_cuts=True)[0], reference
            )
            self._assert_matches_encode(self.tokenizer, text)

    def test_fuzzed_texts_encode_the_same_at_every_cut_point(self):
        for text in _fuzz_texts():
            reference = self.tokenizer.encode(text)
            self.assertEqual(
                _chunk_ids(self.tokenizer, text, all_cuts=True)[0],
                reference,
                repr(text[:80]),
            )
            self.assertEqual(
                ppe.parallel_prompt_encode(self.tokenizer, text, {}), reference
            )

    def test_a_long_prompt_uses_the_default_chunking(self):
        text = (
            "<|user|>\n" + "the quick brown fox jumps over the lazy dog\n" * 40
        ) * 60
        ids, chunks = _chunk_ids(self.tokenizer, text)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks), ppe._MAX_CHUNKS + 1)
        self.assertEqual(ids, self.tokenizer.encode(text))
        self._assert_matches_encode(self.tokenizer, text)

    def test_the_serial_probe_leaves_the_ids_unchanged(self):
        """The first call times `encode_batch`; whichever way that goes, the
        ids are the ones `encode` returns."""
        ppe._parallel_ok = None
        text = "hello world, this is a test\n" * 4000
        self.assertEqual(
            ppe.parallel_prompt_encode(self.tokenizer, text, {}),
            self.tokenizer.encode(text),
        )
        self.assertIn(ppe._parallel_ok, (True, False))

    def _run_probe(self, process_cpu, thread_cpu):
        ppe._parallel_ok = None
        text = "hello world, this is a test\n" * 4000
        with (
            unittest.mock.patch.object(
                ppe.time, "perf_counter", side_effect=(1.0, 1.01)
            ),
            unittest.mock.patch.object(
                ppe.time, "thread_time", side_effect=(1.0, 1.0 + thread_cpu)
            ),
            unittest.mock.patch.object(
                ppe.time,
                "process_time",
                side_effect=(1.0, 1.0 + process_cpu),
                create=True,
            ),
        ):
            ppe.parallel_prompt_encode(self.tokenizer, text, {})
        return ppe._parallel_ok

    def test_the_probe_counts_rayon_worker_cpu(self):
        self.assertTrue(self._run_probe(process_cpu=0.04, thread_cpu=0.006))

    def test_the_probe_rejects_one_cpu(self):
        self.assertFalse(self._run_probe(process_cpu=0.01, thread_cpu=0.0))

    def test_a_forked_process_rechecks_parallelism(self):
        old_pid = getattr(ppe, "_parallel_pid", None)
        self.addCleanup(setattr, ppe, "_parallel_pid", old_pid)
        ppe._parallel_ok = False
        ppe._parallel_pid = 100
        text = "hello world, this is a test\n" * 4000
        with (
            unittest.mock.patch.object(ppe.os, "getpid", return_value=101),
            unittest.mock.patch.object(
                ppe.time, "perf_counter", side_effect=(1.0, 1.01)
            ),
            unittest.mock.patch.object(
                ppe.time, "process_time", side_effect=(1.0, 1.04)
            ),
        ):
            ppe.parallel_prompt_encode(self.tokenizer, text, {})
        self.assertEqual(ppe._parallel_pid, 101)
        self.assertTrue(ppe._parallel_ok)


def _metaspace(backend):
    backend.pre_tokenizer = pre_tokenizers.Metaspace()


def _other_split_regex(backend):
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\s+|\w+|[^\w\s]+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def _prefix_space(backend):
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(ppe._INTRA_SPLIT_REGEX), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False),
        ]
    )


def _lowercase_normalizer(backend):
    backend.add_tokens([AddedToken("NORM", normalized=True)])
    backend.normalizer = normalizers.Lowercase()


def _post_processor_adds_bos(backend):
    backend.post_processor = processors.TemplateProcessing(
        single="<think> $A",
        special_tokens=[("<think>", backend.token_to_id("<think>"))],
    )


def _post_processor_duplicates_sequence(backend):
    backend.post_processor = processors.TemplateProcessing(
        single="$A $A", special_tokens=[]
    )


def _post_processor_sequence(backend, duplicate=False):
    template = "$A $A" if duplicate else "<think> $A"
    backend.post_processor = processors.Sequence(
        [
            processors.ByteLevel(),
            processors.TemplateProcessing(
                single=template,
                special_tokens=[("<think>", backend.token_to_id("<think>"))],
            ),
        ]
    )


class _WrappedTokenizer(PreTrainedTokenizerFast):
    def encode(self, text, **kwargs):
        return super().encode(text, **kwargs)


class _SlowWordTokenizer(PreTrainedTokenizer):
    """A whitespace tokenizer on the python backend, i.e. `is_fast` False."""

    def __init__(self, **kwargs):
        self._vocab = {
            w: i for i, w in enumerate(["<unk>", "hello", "world", "a", "b"])
        }
        super().__init__(unk_token="<unk>", **kwargs)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text, **kwargs):
        return text.split()

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab["<unk>"])

    def _convert_id_to_token(self, index):
        return list(self._vocab)[index]


class TestUnsafeTokenizersFallBack(_EncodeCase):
    def setUp(self):
        super().setUp()
        self._no_min_chars()
        self.text = (
            "<think>Hello NORM world\nsecond line, it's fine </think> tail " * 200
        )

    def _assert_original_encode(self, tokenizer, kwargs_variants=_ALL_KWARGS):
        with unittest.mock.patch.object(
            ppe, "split_prompt", side_effect=AssertionError("chunked path taken")
        ):
            self._assert_matches_encode(tokenizer, self.text, kwargs_variants)

    def test_configurations_without_a_plan(self):
        cases = {
            "Metaspace pre-tokenizer": _tokenizer(_metaspace),
            "lstrip added token": _tokenizer(
                lambda b: b.add_tokens(
                    [AddedToken("<a>", lstrip=True, normalized=False)]
                )
            ),
            "rstrip added token": _tokenizer(
                lambda b: b.add_tokens(
                    [AddedToken("<a>", rstrip=True, normalized=False)]
                )
            ),
            "single_word added token": _tokenizer(
                lambda b: b.add_tokens(
                    [AddedToken("<a>", single_word=True, normalized=False)]
                )
            ),
            "split_special_tokens": _tokenizer(split_special_tokens=True),
            "wrapped encode": _WrappedTokenizer(
                tokenizer_object=Tokenizer.from_str(_safe_backend_json())
            ),
            "slow tokenizer": _SlowWordTokenizer(),
        }
        for name, tokenizer in cases.items():
            with self.subTest(name):
                self.assertIsNone(ppe._plan_for(tokenizer))
                self._assert_original_encode(tokenizer)

    def test_configurations_that_keep_only_the_added_token_cuts(self):
        """A safe-but-unverified pre-tokenizer still cuts at added tokens: the
        backend restarts the pipeline there whatever the pre-tokenizer is."""
        cases = {
            "different Split regex": _tokenizer(_other_split_regex),
            "ByteLevel add_prefix_space": _tokenizer(_prefix_space),
            "normalizer present": _tokenizer(_lowercase_normalizer),
        }
        for name, tokenizer in cases.items():
            with self.subTest(name):
                plan = ppe._plan_for(tokenizer)
                self.assertIsNotNone(plan)
                self.assertIsNone(plan[4], "intra-gap cuts are not verified here")
                self.assertEqual(
                    _chunk_ids(tokenizer, self.text, all_cuts=True)[0],
                    tokenizer.encode(self.text, add_special_tokens=False),
                )
                self._assert_matches_encode(tokenizer, self.text)

    def test_a_post_processor_that_adds_specials_falls_back_when_asked_for_them(self):
        tokenizer = _tokenizer(_post_processor_adds_bos)
        plan = ppe._plan_for(tokenizer)
        self.assertIsNotNone(plan)
        self.assertEqual(plan[1], 1)
        with unittest.mock.patch.object(
            ppe, "split_prompt", side_effect=AssertionError("chunked path taken")
        ):
            for encode_kwargs in ({}, {"add_special_tokens": True}):
                self.assertEqual(
                    ppe.parallel_prompt_encode(tokenizer, self.text, encode_kwargs),
                    tokenizer.encode(self.text, **encode_kwargs),
                )
        # Asked not to add them, the chunked path is exact again.
        encode_kwargs = {"add_special_tokens": False}
        self.assertEqual(
            ppe.parallel_prompt_encode(tokenizer, self.text, encode_kwargs),
            tokenizer.encode(self.text, **encode_kwargs),
        )
        self.assertEqual(
            _chunk_ids(tokenizer, self.text, all_cuts=True)[0],
            tokenizer.encode(self.text, add_special_tokens=False),
        )

    def test_a_post_processor_that_duplicates_ids_falls_back(self):
        tok = _tokenizer(_post_processor_duplicates_sequence)
        text = "hello world " + "<think>hello world</think> " * 2000
        self.assertIsNone(ppe._plan_for(tok))
        with unittest.mock.patch.object(
            ppe, "split_prompt", side_effect=AssertionError("chunked path taken")
        ):
            assert ppe.parallel_prompt_encode(
                tok, text, {"add_special_tokens": False}
            ) == tok.encode(text, add_special_tokens=False)

    def test_sequence_post_processors_are_checked_recursively(self):
        safe = _tokenizer(lambda b: _post_processor_sequence(b))
        unsafe = _tokenizer(lambda b: _post_processor_sequence(b, duplicate=True))
        self.assertIsNotNone(ppe._plan_for(safe))
        self.assertIsNone(ppe._plan_for(unsafe))
        self._assert_original_encode(unsafe)

    def test_backend_state_that_rewrites_the_ids_falls_back(self):
        """These three live on the backend, not in the plan, so the helper has
        to look at them per call.

        The mutation is applied after the wrapper is built (it copies the
        backend), and each case gets one call: falling back runs
        `tokenizer.encode`, which clears the backend's truncation and padding.
        """
        cases = {
            "truncation": lambda b: b.enable_truncation(50),
            "padding": lambda b: b.enable_padding(length=64),
            "encode_special_tokens": lambda b: setattr(
                b, "encode_special_tokens", True
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                tokenizer = _tokenizer()
                mutate(tokenizer._tokenizer)
                self.assertIsNotNone(ppe._plan_for(tokenizer))
                self._assert_original_encode(tokenizer, ({},))


class TestGatesAndArgumentAllowlist(_EncodeCase):
    threshold = 4096

    def setUp(self):
        super().setUp()
        # Pinned rather than inherited: the ambient environment may carry its
        # own threshold, and the default is 32 KiB of text per case.
        self._set_min_chars(self.threshold)
        self.tokenizer = _tokenizer()
        self.text = "the quick brown fox jumps over the lazy dog\n" * 200
        self.assertGreater(len(self.text), self.threshold)

    def _assert_chunked(self, encode_kwargs):
        with unittest.mock.patch.object(
            ppe, "split_prompt", wraps=ppe.split_prompt
        ) as split:
            ids = ppe.parallel_prompt_encode(self.tokenizer, self.text, encode_kwargs)
        self.assertEqual(split.call_count, 1)
        self.assertEqual(ids, self.tokenizer.encode(self.text, **(encode_kwargs or {})))

    def _assert_original(self, text, encode_kwargs):
        with unittest.mock.patch.object(
            ppe, "split_prompt", side_effect=AssertionError("chunked path taken")
        ):
            ids = ppe.parallel_prompt_encode(self.tokenizer, text, encode_kwargs)
        self.assertEqual(ids, self.tokenizer.encode(text, **(encode_kwargs or {})))

    def test_a_long_prompt_takes_the_chunked_path(self):
        self._assert_chunked(None)
        self._assert_chunked({"add_special_tokens": True})
        self._assert_chunked({"add_special_tokens": False})

    def test_a_prompt_below_the_char_threshold_uses_the_original_encode(self):
        self._assert_original(self.text[: self.threshold - 1], None)
        with envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.override(len(self.text) + 1):
            self._assert_original(self.text, None)

    def test_the_env_switch_disables_the_chunked_path(self):
        with envs.SGLANG_PARALLEL_PROMPT_ENCODE.override(False):
            self._assert_original(self.text, None)

    def test_encode_kwargs_outside_the_allowlist_use_the_original_encode(self):
        self._assert_original(self.text, {"truncation": False})
        self._assert_original(
            self.text, {"add_special_tokens": False, "truncation": False}
        )

    def test_a_process_with_serial_encode_batch_stays_on_the_original_encode(self):
        ppe._parallel_ok = False
        self._assert_original(self.text, None)


def _fuzz_segments():
    """Prompts as a renderer hands them over: specials on their own, between
    text that may end or start with part of a special, or hold whole ones."""
    rng = random.Random(1)
    partials = [s[:i] for s in _SPECIALS for i in range(1, len(s))]
    partials += [s[i:] for s in _SPECIALS for i in range(1, len(s))]
    atoms = list("aZ é中🙂'1 \n\r\t.,;:{}()<>/|[]_-") + _SPECIALS + partials

    def text():
        return "".join(rng.choice(atoms) for _ in range(rng.randint(0, 40)))

    return [
        [
            rng.choice(_SPECIALS) if rng.random() < 0.4 else text()
            for _ in range(rng.randint(0, 30))
        ]
        for _ in range(1500)
    ]


class TestPromptSegments(_EncodeCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, ppe, "_segment_encoders", ppe._segment_encoders)
        ppe._segment_encoders = weakref.WeakKeyDictionary()
        self.tokenizer = _tokenizer()

    def _assert_segments_match_encode(
        self, tokenizer, segments, kwargs_variants=_ALL_KWARGS
    ):
        text = "".join(segments)
        for encode_kwargs in kwargs_variants:
            # The second call reads the ids the first one cached.
            for _ in range(2):
                self.assertEqual(
                    ppe.encode_prompt_segments(tokenizer, segments, encode_kwargs),
                    tokenizer.encode(text, **encode_kwargs),
                    f"{encode_kwargs} on {segments[:8]!r}",
                )

    def test_ids_equal_one_encode_of_the_joined_prompt(self):
        edge_cases = [
            [],
            [""],
            ["<|user|>"],
            ["<|user|>", "<|assistant|>"],
            ["/no", "<think>", "think>"],
            ["<|user|", "<|user|>", "|>"],
            ["[gMASK]", "<sop>", "<|user|>", "hello <think> world", "<|assistant|>"],
        ]
        for segments in edge_cases + _fuzz_segments():
            self._assert_segments_match_encode(self.tokenizer, segments)

    def test_text_between_cut_points_is_encoded_once_whatever_the_length(self):
        self._set_min_chars(1 << 20)
        first = ["<|user|>", "hello world", "<|assistant|>"]
        second = first + ["<think>", "fresh text"]
        with unittest.mock.patch.object(
            ppe._SegmentEncoder,
            "_encode",
            autospec=True,
            side_effect=ppe._SegmentEncoder._encode,
        ) as encode:
            self._assert_segments_match_encode(self.tokenizer, first, ({},))
            self._assert_segments_match_encode(self.tokenizer, second, ({},))
        self.assertEqual(
            [call.args[1] for call in encode.call_args_list],
            [["hello world"], ["fresh text"]],
        )

    def test_long_text_between_cut_points_is_split(self):
        self._set_min_chars(4096)
        long_text = "the quick brown fox jumps over the lazy dog\n" * 300
        segments = ["<|user|>", long_text, "<|assistant|>", "<think>"]
        with unittest.mock.patch.object(
            ppe, "split_prompt", wraps=ppe.split_prompt
        ) as split:
            self._assert_segments_match_encode(self.tokenizer, segments, ({},))
        self.assertEqual([call.args[0] for call in split.call_args_list], [long_text])

    def test_a_token_another_added_token_can_overlap_is_not_a_cut_point(self):
        tokenizer = _tokenizer(
            lambda b: b.add_tokens(
                [
                    AddedToken(token, normalized=False)
                    for token in ("x<|user|>", "ab<th", "<sop>!")
                ]
            )
        )
        encoder = ppe._segment_encoder_for(tokenizer)
        for token in ("<|user|>", "<think>", "<sop>"):
            self.assertIsNone(encoder.cut_id(token), token)
        self.assertIsNotNone(encoder.cut_id("<|assistant|>"))
        for segments in (
            ["x", "<|user|>", "q"],
            ["ab", "<think>", "ink"],
            ["<sop>", "!", "<|assistant|>"],
        ):
            self._assert_segments_match_encode(tokenizer, segments)

    def _assert_falls_back(self, tokenizer, segments, encode_kwargs):
        with (
            unittest.mock.patch.object(
                ppe._SegmentEncoder,
                "encode",
                side_effect=AssertionError("segment path taken"),
            ),
            unittest.mock.patch.object(
                ppe, "parallel_prompt_encode", wraps=ppe.parallel_prompt_encode
            ) as fallback,
        ):
            ids = ppe.encode_prompt_segments(tokenizer, segments, encode_kwargs)
        fallback.assert_called_once()
        self.assertEqual(ids, tokenizer.encode("".join(segments), **encode_kwargs))

    def test_falls_back_where_parallel_prompt_encode_decides(self):
        segments = ["<|user|>", "hello world " * 50, "<|assistant|>", "<think>"]
        cases = {
            "no plan": (_tokenizer(_metaspace), {}),
            "specials asked for": (_tokenizer(_post_processor_adds_bos), {}),
            "encode kwargs outside the allowlist": (
                self.tokenizer,
                {"truncation": False},
            ),
        }
        for name, mutate in {
            "truncation": lambda b: b.enable_truncation(50),
            "padding": lambda b: b.enable_padding(length=64),
            "encode_special_tokens": lambda b: setattr(
                b, "encode_special_tokens", True
            ),
        }.items():
            tokenizer = _tokenizer()
            mutate(tokenizer._tokenizer)
            cases[name] = (tokenizer, {})
        for name, (tokenizer, encode_kwargs) in cases.items():
            with self.subTest(name):
                self._assert_falls_back(tokenizer, segments, encode_kwargs)
        with envs.SGLANG_PARALLEL_PROMPT_ENCODE.override(False):
            self._assert_falls_back(self.tokenizer, segments, {})

    def test_the_cache_keeps_to_its_token_budget(self):
        with (
            unittest.mock.patch.object(ppe, "_SEGMENT_CACHE_MAX_TOKENS", 40),
            unittest.mock.patch.object(ppe, "_SEGMENT_CACHE_MAX_ENTRY_TOKENS", 20),
        ):
            for i in range(30):
                self._assert_segments_match_encode(
                    self.tokenizer, ["<|user|>", f"turn {i} hello world"], ({},)
                )
            self._assert_segments_match_encode(
                self.tokenizer, ["<|user|>", "hello world " * 30], ({},)
            )
        encoder = ppe._segment_encoder_for(self.tokenizer)
        self.assertLessEqual(encoder.cached_tokens, 40)
        self.assertEqual(encoder.cached_tokens, sum(map(len, encoder.cache.values())))
        self.assertNotIn((False, "hello world " * 30), encoder.cache)


# The other added tokens the GLM-5.3 templates write.
_GLM_TOKENS = [
    "<|system|>",
    "<|observation|>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<|begin_of_image|>",
    "<|image|>",
    "<|end_of_image|>",
]
_MARK = "\ufdd0"
_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns <tool_call> text.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
_ORDINARY_TEXT_CASES = {
    "plain": ([{"role": "user", "content": "hello world"}], None),
    "spellings": (
        [{"role": "user", "content": "quote <tool_call> and </think> here"}],
        None,
    ),
    "exact spelling": ([{"role": "user", "content": "<|assistant|>"}], None),
    "adjacent spellings": (
        [{"role": "user", "content": "<|user|><|assistant|><think>"}],
        None,
    ),
    "partial spellings": (
        [{"role": "user", "content": "<think <|user| think> |>"}],
        None,
    ),
    "system and media": (
        [
            {"role": "system", "content": "Be <|assistant|> brief."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "is <|image|> one?"},
                ],
            },
        ],
        None,
    ),
    "history": (
        [
            {"role": "user", "content": "q <sop>"},
            {
                "role": "assistant",
                "content": "a <tool_call>x</tool_call> </think>",
                "reasoning_content": "r <|observation|>",
            },
            {"role": "user", "content": "q2"},
        ],
        None,
    ),
    "tools": (
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "<arg_value>Paris", "n": 2},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "Sunny </tool_response>"},
            {"role": "tool", "content": [{"tool_call_id": "b", "output": "[gMASK]"}]},
        ],
        [_WEATHER],
    ),
}


def _glm_tokenizer():
    return _tokenizer(
        lambda b: b.add_tokens(
            [AddedToken(t, normalized=False, special=True) for t in _GLM_TOKENS]
        )
    )


def _ordinary_text_reference(tokenizer, messages, tools):
    """The ordinary-text encoding, built another way: break the added-token
    spellings in the caller's strings with a mark, render, and encode with a
    backend that drops the mark after it has matched the added tokens."""
    spelling = re.compile(
        "|".join(
            re.escape(token)
            for token in sorted(tokenizer.get_added_vocab(), key=len, reverse=True)
        )
    )

    def mark(value):
        if isinstance(value, str):
            return spelling.sub(lambda m: m.group()[0] + _MARK + m.group()[1:], value)
        if isinstance(value, dict):
            return {mark(key): mark(item) for key, item in value.items()}
        if isinstance(value, list):
            return [mark(item) for item in value]
        return value

    prompt = render_glm53_flash(
        mark(messages), tools=mark(tools), add_generation_prompt=True
    )
    backend = Tokenizer.from_str(tokenizer._tokenizer.to_str())
    backend.normalizer = normalizers.Replace(_MARK, "")
    return backend.encode(prompt, add_special_tokens=False).ids


def _fuzz_conversations():
    rng = random.Random(0)
    atoms = list("aZ é中 \n.<>|/_") + ["think", "user", "  "] + _SPECIALS + _GLM_TOKENS

    def text():
        return "".join(rng.choice(atoms) for _ in range(rng.randint(0, 30)))

    return [
        [
            {"role": "system", "content": text()},
            {"role": "user", "content": text()},
            # A string reasoning_content keeps the content from being split
            # at `</think>`, which the reference cannot mark.
            {"role": "assistant", "content": text(), "reasoning_content": text()},
            {"role": "user", "content": text()},
        ]
        for _ in range(300)
    ]


class TestOrdinaryTextSegments(_EncodeCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, ppe, "_segment_encoders", ppe._segment_encoders)
        ppe._segment_encoders = weakref.WeakKeyDictionary()
        self.tokenizer = _glm_tokenizer()
        self.added_ids = set(self.tokenizer.get_added_vocab().values())

    def _id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def _encode(self, messages, tools=None):
        segments = render_glm53_flash_segments(
            messages, tools=tools, add_generation_prompt=True
        )
        return ppe.encode_prompt_segments(self.tokenizer, segments)

    def test_ids_equal_the_ordinary_text_reference(self):
        cases = list(_ORDINARY_TEXT_CASES.values())
        cases += [(messages, None) for messages in _fuzz_conversations()]
        for messages, tools in cases:
            expected = _ordinary_text_reference(self.tokenizer, messages, tools)
            # The second call reads the ids the first one cached.
            for _ in range(2):
                self.assertEqual(self._encode(messages, tools), expected, messages)

    def test_quoted_spelling_is_encoded_as_text(self):
        messages = [{"role": "user", "content": "quote <tool_call> here"}]
        ids = self._encode(messages)
        self.assertEqual(ids, _ordinary_text_reference(self.tokenizer, messages, None))
        self.assertNotIn(self._id("<tool_call>"), ids)
        self.assertEqual(
            [i for i in ids if i in self.added_ids],
            [self._id(t) for t in ("[gMASK]", "<sop>", "<|system|>", "<|user|>")]
            + [self._id("<|assistant|>"), self._id("<think>")],
        )

    def test_ordinary_text_never_yields_added_token_ids(self):
        for token in self.tokenizer.get_added_vocab():
            # A segment that is exactly a spelling is not cut to its id either.
            segments = [
                "<|user|>",
                OrdinaryText(token),
                OrdinaryText(f"x{token}y {token}"),
                "<|assistant|><think>",
            ]
            ids = ppe.encode_prompt_segments(self.tokenizer, segments)
            self.assertEqual(
                [i for i in ids if i in self.added_ids],
                [self._id(t) for t in ("<|user|>", "<|assistant|>", "<think>")],
                token,
            )

    def test_ordinary_and_template_runs_are_cached_apart(self):
        text = "x <think> y"
        encoder = ppe._segment_encoder_for(self.tokenizer)
        for _ in range(2):
            ordinary = ppe.encode_prompt_segments(self.tokenizer, [OrdinaryText(text)])
            template = ppe.encode_prompt_segments(self.tokenizer, [text])
            self.assertNotIn(self._id("<think>"), ordinary)
            self.assertEqual(template, self.tokenizer.encode(text))
        self.assertIn((True, text), encoder.cache)
        self.assertIn((False, text), encoder.cache)

    def test_switches(self):
        messages, tools = _ORDINARY_TEXT_CASES["tools"]
        segments = render_glm53_flash_segments(
            messages, tools=tools, add_generation_prompt=True
        )
        with envs.SGLANG_PARALLEL_PROMPT_ENCODE.override(False):
            self.assertEqual(
                ppe.encode_prompt_segments(self.tokenizer, segments),
                _ordinary_text_reference(self.tokenizer, messages, tools),
            )
        with envs.SGLANG_ENABLE_ORDINARY_TEXT_CONTENT.override(False):
            self.assertEqual(
                ppe.encode_prompt_segments(self.tokenizer, segments),
                self.tokenizer.encode("".join(segments)),
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
