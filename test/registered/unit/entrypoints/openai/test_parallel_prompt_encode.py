"""Chunked prompt encoding must produce the ids the single encode produces.

`parallel_prompt_encode` cuts a rendered prompt and sends the pieces through
`backend.encode_batch`, so its whole contract is that the concatenated ids are
identical to `tokenizer.encode(text, **encode_kwargs)` — an off-by-one cut
point is not a crash, it is a silently different prompt. These cases pin the
identity on ByteLevel-BPE tokenizers built in memory with each exact
Split-regex + ByteLevel + BPE pipeline the intra-gap cuts were verified for
(GLM / Llama-3, Qwen2 / Qwen3 and Qwen3.5 under NFC, o200k, DeepSeek-V3), on
the real tokenizer files of those pipelines when they are on this machine,
and pin the refusals: every tokenizer configuration outside those pipelines,
and every gate, must reach the original encode instead.

The tests never assert that the rayon pool actually ran the chunks in
parallel. The helper probes for a serial `encode_batch` once and then disables
itself for the process, which is a correctness-preserving decision a CPU
runner is free to make; `_parallel_ok` is pinned per test so the id
comparisons do not depend on it.
"""

import os
import random
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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

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
# Matched on normalized text in the backend's second pass, as DeepSeek-V3's
# role markers are: a cut point only under the identity normalizer, where that
# text is the input. Added to the deepseek3 backend below.
_NORMALIZED_SPECIAL = "<|norm|>"

# The verified pipelines: Split regexes ahead of ByteLevel, and the normalizer.
_PIPELINES = {
    "llama3": ((ppe._LLAMA3_SPLIT_REGEX,), None),
    "qwen2": ((ppe._QWEN2_SPLIT_REGEX,), normalizers.NFC()),
    "qwen35": ((ppe._QWEN35_SPLIT_REGEX,), normalizers.NFC()),
    "o200k": ((ppe._O200K_SPLIT_REGEX,), None),
    "deepseek3": (ppe._DEEPSEEK3_SPLIT_REGEXES, normalizers.Sequence([])),
}
# The pipelines verified after #41259; llama3 has its own cases below.
_OTHER_PIPELINES = ("qwen2", "qwen35", "o200k", "deepseek3")

# Real tokenizer files for the verified pipelines, used when present:
# SGLANG_TEST_TOKENIZER_JSONS="qwen2=/path/tokenizer.json,o200k=..." or the
# checkpoint's tokenizer.json in the Hugging Face cache (downloaded when the
# hub is reachable). A file that cannot be found skips, it never fails.
_REAL_CHECKPOINTS = {
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2": "Qwen/Qwen3-0.6B",
    "qwen35": "Qwen/Qwen3.5-0.8B",
    "o200k": "openai/gpt-oss-20b",
    "deepseek3": "deepseek-ai/DeepSeek-V3",
}


@lru_cache(maxsize=None)
def _backend_json(pipeline="llama3"):
    """A backend with the pre-tokenizer pipeline `_intra_cut_safe` accepts.

    Only the pre-tokenizer pipeline, the normalizer and the model type decide
    which cut points the helper takes, so a 900-token vocabulary exercises
    the same code path a real checkpoint does. The ByteLevel alphabet is
    seeded explicitly so every byte of the test corpus has a token and
    nothing is silently dropped.
    """
    splits, normalizer = _PIPELINES[pipeline]
    backend = Tokenizer(models.BPE())
    if normalizer is not None:
        backend.normalizer = normalizer
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(split), behavior="isolated", invert=False)
            for split in splits
        ]
        + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
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
    if pipeline == "deepseek3":
        backend.add_tokens(
            [AddedToken(_NORMALIZED_SPECIAL, normalized=True, special=True)]
        )
    return backend.to_str()


def _tokenizer(mutate=None, pipeline="llama3", **kwargs):
    backend = Tokenizer.from_str(_backend_json(pipeline))
    if mutate is not None:
        mutate(backend)
    return PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)


def _real_tokenizer_file(pipeline):
    for item in os.environ.get("SGLANG_TEST_TOKENIZER_JSONS", "").split(","):
        name, _, path = item.partition("=")
        if name.strip() == pipeline and os.path.isfile(path.strip()):
            return path.strip()
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from huggingface_hub.constants import HF_HUB_OFFLINE
    except ImportError:
        return None
    repo = _REAL_CHECKPOINTS[pipeline]
    cached = try_to_load_from_cache(repo, "tokenizer.json")
    if isinstance(cached, str):
        return cached
    if HF_HUB_OFFLINE:
        return None
    try:
        return hf_hub_download(repo, "tokenizer.json")
    except Exception:
        return None


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
    # Digits, CJK and punctuation after a newline; "/" after a newline is a
    # cut only for a letter, since o200k keeps "\n/" in a punctuation run.
    "line\n123 456\n中文。\n文本\n/path x\n'q'\n#c\n鿿\n一",
    "1\n2\n3\n4 5\n中\n文\n\n中\r\n文\n \n中",
    "abcDEF ghi\nJKL mno\nPQRstu\n123abc\nabc123\nHTTPServer",
    # Marks and NFC: composable pairs, a letter that decomposes (U+0958),
    # Hangul jamo and syllables, and marks right after a cut.
    "é é\nf́ g\nåb\nक़/ क़\nक़ क़",
    "ab\ncd́ ef\nx́y\nz̵ w\n1́\n中́ x",
    "한글\n한글 abc\n가 x\n각 b\ncᄀ",
    # Emoji with variation selectors, skin tones and ZWJ sequences.
    "❤️ x\n❤️\n👍🏽 y\n👍\n🇨🇳\n中\n👩‍👩‍👧 a\n❤️ b",
    # Added tokens next to every kind of cut.
    "a <|user|>b\nc<|user|>\nd\n<|user|>e <|user|> f\n<|user|>\n中<|user|>1",
    "x\n<|user|>\n<|norm|>\n1<|norm|>a <|norm|> b\n<|norm|><|user|>",
]


# Chunks the all-cuts split must produce: one case per kind of cut point.
_CUT_CASES = (
    ("<think>abcd", ["<think>", "abcd"]),
    ("abcd efgh", ["abcd", " efgh"]),
    ("abcd\nefgh", ["abcd\n", "efgh"]),
    ("abcd\n1234", ["abcd\n", "1234"]),
    ("abcd\n中文", ["abcd\n", "中文"]),
    ("abcd\n/efgh ijkl", ["abcd\n/efgh", " ijkl"]),
)


def _edge_texts(opening="<think>", closing="</think>"):
    """The edge texts, bare and wrapped in a pair of special tokens."""
    texts = list(_EDGE_TEXTS)
    for text in _EDGE_TEXTS:
        texts += [opening + text, text + closing, opening + text + closing]
    return texts


def _fuzz_texts(specials=_SPECIALS, seed=0, short=4000, long=30):
    """Random texts over the atoms that sit on either side of a cut point."""
    rng = random.Random(seed)
    atoms = list("aZ é中🙂'1 \n\r\t.,;:{}()<>/_-\"\\") + [
        "\u0301",
        " ",
        "'s",
        "  ",
        "\r\n",
        " \n",
        "\n ",
        # Words, contractions, digit groups, a newline before "/" or "#".
        "the",
        "Hello",
        "WORLD",
        "it's",
        "123",
        "4567",
        "\n/",
        ".\n/",
        "\n#",
        # CJK ideographs (U+9FA5 ends DeepSeek's split range, U+9FFF the cut
        # range), fullwidth punctuation, kana with a combining dakuten,
        # Hangul jamo and syllables.
        "文",
        "龥",
        "鿿",
        "。",
        "，",
        "中文文本",
        "の",
        "ガ",
        "한",
        "가",
        "ᆨ",
        "가",
        # NFC: a composable pair, a mark that reorders, U+0958 both ways.
        "é",
        "\u0335",
        "क़",
        "क़",
        # No-break, ideographic and zero width spaces.
        "\u00a0",
        "\u3000",
        "\u200b",
        # Emoji with a skin tone, a variation selector (also bare) and ZWJ.
        "👍🏽",
        "❤️",
        "\ufe0f",
        "👩‍👩‍👧",
        # Non-ASCII lower and upper case letters.
        "ß",
        "İ",
    ]
    atoms += list(specials)
    texts = [
        "".join(rng.choice(atoms) for _ in range(rng.randint(0, 400)))
        for _ in range(short)
    ]
    texts += [
        "".join(rng.choice(atoms) for _ in range(rng.randint(5000, 60000)))
        for _ in range(long)
    ]
    return texts


def _long_prompt(specials):
    """A chat-shaped prompt: role markers around prose, code, Chinese and a
    numbered log, long enough for the default chunking."""
    user, assistant = specials[:2]
    body = (
        "The quick brown fox jumps over the lazy dog. It's a test, isn't it?\n"
        "def main(argv):\n    return 0  # trailing\n\n"
        "中文文本，没有空格。日本語のテキスト 한국어\n"
        "2024-01-02 12:34:56 INFO request 1234 took 56ms /api/v1\n"
        "é ❤️ 👍🏽 क़/ HTTPServer abc123def\n"
    )
    return (user + "\n" + body * 30 + assistant + "\n") * 20


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

    def _assert_every_cut_point_is_exact(self, tokenizer, texts, label=""):
        for text in texts:
            reference = tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(
                _chunk_ids(tokenizer, text, all_cuts=True)[0],
                reference,
                f"{label} {text[:80]!r}",
            )

    def _assert_cuts(self, tokenizer, cases):
        for text, expected in cases:
            self.assertEqual(
                _chunk_ids(tokenizer, text, all_cuts=True)[1], expected, repr(text)
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
        self._assert_cuts(self.tokenizer, _CUT_CASES)

    def test_edge_cases_encode_the_same_at_every_cut_point(self):
        for text in _edge_texts():
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


class TestVerifiedPipelinesIdsAreIdentical(_EncodeCase):
    """The other verified pipelines, built in memory: the same cut points,
    under NFC (Qwen) and under the empty normalizer Sequence (DeepSeek)."""

    def setUp(self):
        super().setUp()
        self._no_min_chars()

    def test_each_pipeline_takes_every_kind_of_cut_point(self):
        for pipeline in _OTHER_PIPELINES:
            with self.subTest(pipeline):
                tokenizer = _tokenizer(pipeline=pipeline)
                plan = ppe._plan_for(tokenizer)
                self.assertIsNotNone(plan)
                self.assertIs(plan[4], ppe._INTRA_CUT, "intra-gap cuts expected")
                self._assert_cuts(tokenizer, _CUT_CASES)
                if pipeline == "deepseek3":
                    # The identity normalizer makes the normalized added token
                    # a cut point as well, and keeps cuts out of it.
                    self.assertIsNotNone(plan[3])
                    self._assert_cuts(
                        tokenizer,
                        (
                            ("ab<|norm|>cd", ["ab<|norm|>", "cd"]),
                            ("ab\n<|norm|>\ncd", ["ab\n<|norm|>", "\n", "cd"]),
                        ),
                    )
                else:
                    self.assertIsNone(plan[3])

    def test_edge_cases_encode_the_same_at_every_cut_point(self):
        for pipeline in _OTHER_PIPELINES:
            with self.subTest(pipeline):
                tokenizer = _tokenizer(pipeline=pipeline)
                texts = _edge_texts()
                self._assert_every_cut_point_is_exact(tokenizer, texts, pipeline)
                for text in texts:
                    self._assert_matches_encode(tokenizer, text)

    def test_fuzzed_texts_encode_the_same_at_every_cut_point(self):
        for pipeline in _OTHER_PIPELINES:
            with self.subTest(pipeline):
                tokenizer = _tokenizer(pipeline=pipeline)
                specials = list(_SPECIALS)
                if pipeline == "deepseek3":
                    specials.append(_NORMALIZED_SPECIAL)
                texts = _fuzz_texts(specials, seed=pipeline, short=2000, long=10)
                self._assert_every_cut_point_is_exact(tokenizer, texts, pipeline)
                # A sample through the entry point, whose chunking is coarser.
                for text in texts[::10]:
                    self.assertEqual(
                        ppe.parallel_prompt_encode(tokenizer, text, {}),
                        tokenizer.encode(text),
                        f"{pipeline} {text[:80]!r}",
                    )

    def test_a_long_prompt_uses_the_default_chunking(self):
        for pipeline in _OTHER_PIPELINES:
            with self.subTest(pipeline):
                tokenizer = _tokenizer(pipeline=pipeline)
                text = _long_prompt(_SPECIALS)
                ids, chunks = _chunk_ids(tokenizer, text)
                self.assertGreater(len(chunks), 1)
                self.assertLessEqual(len(chunks), ppe._MAX_CHUNKS + 1)
                self.assertEqual(ids, tokenizer.encode(text))
                self._assert_matches_encode(tokenizer, text)


class TestRealTokenizerFiles(_EncodeCase):
    """The verified pipelines on real tokenizer files: the full vocabulary,
    the real added tokens, every cut point. Skipped where no file is found."""

    def setUp(self):
        super().setUp()
        self._no_min_chars()

    def test_real_tokenizers_encode_the_same_at_every_cut_point(self):
        found = False
        for pipeline in _PIPELINES:
            path = _real_tokenizer_file(pipeline)
            if path is None:
                continue
            found = True
            with self.subTest(pipeline):
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
                plan = ppe._plan_for(tokenizer)
                self.assertIsNotNone(plan, f"{pipeline}: {path}")
                self.assertIs(plan[4], ppe._INTRA_CUT, f"{pipeline}: {path}")
                # The file's own markers, as wrappers and as fuzz atoms.
                specials = [
                    t.content
                    for t in tokenizer._tokenizer.get_added_tokens_decoder().values()
                    if t.content and not t.content.isalnum()
                ][:8]
                texts = _edge_texts(specials[0], specials[1])
                texts += _fuzz_texts(specials, seed=pipeline, short=2000, long=10)
                self._assert_every_cut_point_is_exact(tokenizer, texts, pipeline)
                text = _long_prompt(specials)
                ids, chunks = _chunk_ids(tokenizer, text)
                self.assertGreater(len(chunks), 1)
                self.assertEqual(ids, tokenizer.encode(text, add_special_tokens=False))
                self._assert_matches_encode(tokenizer, text)
        if not found:
            self.skipTest(
                "no tokenizer.json in SGLANG_TEST_TOKENIZER_JSONS or the HF cache"
            )


def _metaspace(backend):
    backend.pre_tokenizer = pre_tokenizers.Metaspace()


def _other_split_regex(backend):
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\s+|\w+|[^\w\s]+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def _removed_split(backend):
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(ppe._LLAMA3_SPLIT_REGEX), behavior="removed", invert=True
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def _reordered_splits(backend):
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(split), behavior="isolated")
            for split in reversed(ppe._DEEPSEEK3_SPLIT_REGEXES)
        ]
        + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )


def _prefix_space(backend):
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(ppe._LLAMA3_SPLIT_REGEX), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False),
        ]
    )


def _lowercase_normalizer(backend):
    backend.add_tokens([AddedToken("NORM", normalized=True)])
    backend.normalizer = normalizers.Lowercase()


def _nfd_normalizer(backend):
    backend.normalizer = normalizers.NFD()


def _normalized_added_token(backend):
    backend.add_tokens([AddedToken("NORM", normalized=True)])


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
                tokenizer_object=Tokenizer.from_str(_backend_json())
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
            "Split with Removed behavior": _tokenizer(_removed_split),
            "Split sequence in another order": _tokenizer(
                _reordered_splits, pipeline="deepseek3"
            ),
            "ByteLevel add_prefix_space": _tokenizer(_prefix_space),
            "normalizer present": _tokenizer(_lowercase_normalizer),
            "NFD normalizer": _tokenizer(_nfd_normalizer, pipeline="qwen2"),
            "NFC with a normalized added token": _tokenizer(
                _normalized_added_token, pipeline="qwen2"
            ),
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
