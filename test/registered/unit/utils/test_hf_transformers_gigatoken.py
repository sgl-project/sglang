"""Verification that --tokenizer-backend=gigatoken keeps tokenization
byte-identical to the HuggingFace baseline and preserves the rest of the
tokenizer object.

gigatoken replaces only encode/decode; anything it has not been verified
byte-identical on must fall back to transformers. Both directions are silent
when broken -- a wrong fast path mistokenizes prompts, and a wrong fallback
drops padding or token_type_ids -- so each is pinned here against a baseline
tokenizer loaded through the default backend.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

TOKENIZER_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


try:
    import gigatoken  # noqa: F401

    HAS_GIGATOKEN = True
except ImportError:
    HAS_GIGATOKEN = False

# Text shapes the serving path actually feeds the tokenizer: plain prose, chat
# markup carrying special tokens, code, CJK, emoji (multi-byte boundaries the
# incremental detokenizer trips on), and whitespace runs.
TEXTS = [
    "Hello, world!",
    "",
    " ",
    "\n\n\t ",
    "The quick brown fox jumps over the lazy dog. " * 40,
    "<|im_start|>system\nYou are helpful.<|im_end|>\n"
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "def f(x: int) -> int:\n    return x**2  # comment\n",
    "日本語のテキスト、中文文本，한국어 텍스트",
    "emoji 🚀🔥👨‍👩‍👧‍👦 and 🇺🇸 flags",
    "https://example.com/p?q=1&r=2#frag",
]


@unittest.skipUnless(HAS_GIGATOKEN, "gigatoken package not installed")
class TestGigatokenBackend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        cls.baseline = get_tokenizer(TOKENIZER_MODEL)
        cls.accel = get_tokenizer(TOKENIZER_MODEL, tokenizer_backend="gigatoken")

    def test_backend_is_applied(self):
        from sglang.srt.tokenizer.gigatoken_tokenizer import _GigatokenMethods

        self.assertIsInstance(self.accel, _GigatokenMethods)
        # The class swap must keep the original type in the MRO, since SGLang
        # and transformers both branch on isinstance(tokenizer, ...).
        self.assertIsInstance(self.accel, type(self.baseline))

    def test_encode_matches_huggingface(self):
        """Ids must be identical, including the post-processor's affixes."""
        for text in TEXTS:
            for add_special in (True, False):
                with self.subTest(text=text[:30], add_special_tokens=add_special):
                    self.assertEqual(
                        self.accel.encode(text, add_special_tokens=add_special),
                        self.baseline.encode(text, add_special_tokens=add_special),
                    )
                    self.assertEqual(
                        self.accel(text, add_special_tokens=add_special)["input_ids"],
                        self.baseline(text, add_special_tokens=add_special)[
                            "input_ids"
                        ],
                    )

    def test_batch_encode_matches_huggingface(self):
        """The dynamic-batch tokenizer path passes a list of prompts."""
        expected = self.baseline(TEXTS)
        got = self.accel(TEXTS)
        self.assertEqual(got["input_ids"], expected["input_ids"])
        self.assertEqual(got["attention_mask"], expected["attention_mask"])

    def test_incremental_decode_matches_huggingface(self):
        """DetokenizerManager decodes overlapping [surr:read] windows, so every
        window -- including ones cutting a multi-byte character in half -- must
        decode exactly as HF does, or streamed text drifts."""
        ids = self.baseline.encode(
            "Streaming 日本語 🚀 output tokens", add_special_tokens=False
        )
        for read in range(1, len(ids) + 1):
            for surr in range(read):
                window = ids[surr:read]
                with self.subTest(window=(surr, read)):
                    self.assertEqual(
                        self.accel.decode(window, skip_special_tokens=False),
                        self.baseline.decode(window, skip_special_tokens=False),
                    )

    def test_byte_fallback_decode_is_gated_off(self):
        """A byte-fallback (SentencePiece) vocabulary emits one U+FFFD per
        undecodable byte in HF but one per truncated sequence in gigatoken, so
        the startup probe must take those tokenizers off the decode fast path.

        Without the probe, streaming a Llama-2-family model produces a
        different number of replacement characters mid-multibyte-character than
        the HuggingFace reference. Guards against the probe degrading to
        always-true.
        """
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        # Llama 2's vocabulary: SentencePiece BPE with byte_fallback.
        sp = get_tokenizer(
            "hf-internal-testing/llama-tokenizer", tokenizer_backend="gigatoken"
        )
        self.assertFalse(sp._gigatoken_decode_ok)
        # Encoding is unaffected -- that is the point of gating decode only.
        reference = get_tokenizer("hf-internal-testing/llama-tokenizer")
        ids = reference.encode("Streaming 日本語 🚀 output", add_special_tokens=False)
        self.assertEqual(
            sp.encode("Streaming 日本語 🚀 output", add_special_tokens=False), ids
        )
        for read in range(1, len(ids) + 1):
            for surr in range(read):
                window = ids[surr:read]
                with self.subTest(window=(surr, read)):
                    self.assertEqual(sp.decode(window), reference.decode(window))

    def test_single_int_decode_honors_skip_special_tokens(self):
        """decode() accepts a bare int, and skip_special_tokens must still drop
        it when it is special.

        An earlier revision wrapped the int and skipped the filter in the same
        if/elif, so decode(eos_id, skip_special_tokens=True) returned
        "<|im_end|>" instead of "".
        """
        eos = self.baseline.eos_token_id
        for skip in (True, False):
            with self.subTest(skip_special_tokens=skip):
                self.assertEqual(
                    self.accel.decode(eos, skip_special_tokens=skip),
                    self.baseline.decode(eos, skip_special_tokens=skip),
                )

    def test_batch_decode_matches_huggingface(self):
        ids = [self.baseline.encode(text) for text in TEXTS if text.strip()]
        for skip in (True, False):
            with self.subTest(skip_special_tokens=skip):
                self.assertEqual(
                    self.accel.batch_decode(ids, skip_special_tokens=skip),
                    self.baseline.batch_decode(ids, skip_special_tokens=skip),
                )

    def test_unsupported_call_shapes_fall_back(self):
        """Call shapes gigatoken does not drive must reach transformers.

        Each of these produces output the fast path cannot build (padded rows,
        tensors, truncation, segment ids, pair encoding). If the guard ever
        degrades to always-true, the fast path answers instead and the result
        silently loses padding/tensors rather than raising.
        """
        self.accel.pad_token = self.accel.eos_token
        self.baseline.pad_token = self.baseline.eos_token
        pair = ["short", "a much longer second sequence here"]
        cases = [
            ("padding", dict(padding=True)),
            ("padding_max_length", dict(padding="max_length", max_length=32)),
            ("truncation", dict(truncation=True, max_length=8)),
            ("token_type_ids", dict(return_token_type_ids=True)),
        ]
        for name, kwargs in cases:
            with self.subTest(case=name):
                self.assertEqual(
                    dict(self.accel(pair, **kwargs)),
                    dict(self.baseline(pair, **kwargs)),
                )

        # A sequence pair is rejected by gigatoken outright; falling back means
        # transformers' own error/behavior surfaces, not gigatoken's.
        self.assertEqual(
            self.accel("query", text_pair="document")["input_ids"],
            self.baseline("query", text_pair="document")["input_ids"],
        )

    def test_positional_args_fall_back(self):
        """`encode(text, pair)` binds the 2nd positional to text_pair in
        transformers; the override only accepts `text` positionally, so such a
        call must be handed over rather than re-bound to a different meaning.
        """
        self.assertEqual(
            self.accel.encode("query", "document"),
            self.baseline.encode("query", "document"),
        )

    def test_cold_path_surface_survives(self):
        """Everything off the encode/decode path must behave as before.

        The whole reason this integration patches methods instead of swapping
        the tokenizer for gigatoken's HFCompat shim is that HFCompat has no
        chat template, no added-token bookkeeping and no save_pretrained; a
        future refactor toward wholesale replacement would break these.
        """
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(
            self.accel.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            self.baseline.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
        )
        # TemplateManager assigns chat_template onto the tokenizer, and
        # apply_chat_template must then use the assigned value.
        self.accel.chat_template = "{{ 'patched:' }}{{ messages[0]['content'] }}"
        self.assertEqual(
            self.accel.apply_chat_template(messages, tokenize=False), "patched:hi"
        )

        self.assertEqual(self.accel.get_vocab(), self.baseline.get_vocab())
        self.assertEqual(
            self.accel.added_tokens_decoder.keys(),
            self.baseline.added_tokens_decoder.keys(),
        )
        self.assertEqual(self.accel.eos_token_id, self.baseline.eos_token_id)
        self.assertEqual(
            self.accel.additional_stop_token_ids,
            self.baseline.additional_stop_token_ids,
        )
        self.assertEqual(
            self.accel.convert_tokens_to_ids("hello"),
            self.baseline.convert_tokens_to_ids("hello"),
        )

    def test_xgrammar_tokenizer_info_still_builds(self):
        """Constrained decoding builds xgrammar's TokenizerInfo straight from
        the tokenizer object, reading the backend vocab and decoder. gigatoken's
        own HFCompat shim cannot satisfy that; patching methods on the real
        tokenizer must, so this pins the difference.
        """
        from xgrammar import TokenizerInfo

        info = TokenizerInfo.from_huggingface(
            self.accel, vocab_size=self.accel.vocab_size
        )
        expected = TokenizerInfo.from_huggingface(
            self.baseline, vocab_size=self.baseline.vocab_size
        )
        self.assertEqual(info.vocab_size, expected.vocab_size)
        self.assertEqual(info.decoded_vocab, expected.decoded_vocab)

    def test_deepcopy_keeps_working(self):
        """The accelerated tokenizer must survive `copy.deepcopy`.

        `MultimodalProcessorExecutor.__init__` deepcopies the whole processor to
        get one clone per worker. The gigatoken backend is a Rust object that
        cannot be pickled, so before `__deepcopy__` was added this raised
        `TypeError: cannot pickle 'builtins.BPETokenizer' object`, sglang caught
        it, and every multimodal server silently dropped to synchronous
        processing — observed on a live server, not hypothetical.
        """
        import copy

        clone = copy.deepcopy(self.accel)
        text = "Hello, world! 日本語 🚀"
        self.assertEqual(clone.encode(text), self.baseline.encode(text))
        self.assertEqual(
            clone.decode(clone.encode(text)),
            self.baseline.decode(self.baseline.encode(text)),
        )
        # The clone shares the backend rather than building a second cache.
        self.assertIs(clone._gigatoken, self.accel._gigatoken)

    def test_idempotent(self):
        """get_tokenizer applies acceleration once; re-applying must not stack
        another subclass (which would recurse through super() forever)."""
        from sglang.srt.tokenizer.gigatoken_tokenizer import (
            accelerate_with_gigatoken,
        )

        before = type(self.accel)
        self.assertIs(type(accelerate_with_gigatoken(self.accel)), before)


if __name__ == "__main__":
    unittest.main()
