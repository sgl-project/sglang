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

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

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

    def _no_min_chars(self):
        self.enterContext(envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.override(0))

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
        self.enterContext(
            envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.override(self.threshold)
        )
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
