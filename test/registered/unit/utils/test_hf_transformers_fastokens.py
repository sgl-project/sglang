"""End-to-end verification that --tokenizer-backend=fastokens swaps the
backend of the loaded tokenizer with fastokens' _TokenizerShim.
"""

import json
import os
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

TOKENIZER_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

register_cpu_ci(est_time=14, suite="base-a-test-cpu")


try:
    import fastokens

    HAS_FASTOKENS = True
except ImportError:
    HAS_FASTOKENS = False


@unittest.skipUnless(HAS_FASTOKENS, "fastokens package not installed")
class TestFastokensBackend(CustomTestCase):
    @classmethod
    def tearDownClass(cls):
        # The patch is process-global; later tests in this process load with plain HF.
        from sglang.srt.utils.hf_transformers import tokenizer

        fastokens.unpatch_transformers()
        tokenizer._fastokens_patched = False
        super().tearDownClass()

    def test_shim_is_applied(self):
        # `_TokenizerShim` is fastokens' private compat shim. SGLang's
        # integration relies on `tokenizer._tokenizer` being an instance of
        # this class to confirm fastokens is wired up. If fastokens renames
        # or restructures it, update both this assertion and any code in
        # SGLang that depends on the same private name.
        from fastokens._compat import _TokenizerShim

        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            TOKENIZER_MODEL,
            tokenizer_backend="fastokens",
        )
        backend = getattr(tokenizer, "_tokenizer", None)
        self.assertIsInstance(
            backend,
            _TokenizerShim,
            f"Expected tokenizer._tokenizer to be _TokenizerShim, "
            f"got {type(backend).__name__}",
        )

    def test_encode_decode_roundtrip(self):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            TOKENIZER_MODEL,
            tokenizer_backend="fastokens",
        )
        text = "Hello, world!"
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), text)

    def test_v5_rebuilt_pre_tokenizer_loads_and_matches_tokenizer_json(self):
        """A ByteLevel tokenizer.json declared as LlamaTokenizerFast must load.

        The v5 class rebuilds a Metaspace pre-tokenizer, which fastokens rejects
        when it shims the backend before sglang restores tokenizer.json's own.
        """
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        raw = Tokenizer(models.BPE())
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<s>", "</s>", "<PAGE>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        raw.train_from_iterator(["Multi page parsing.", "中文 OCR"] * 20, trainer)
        with tempfile.TemporaryDirectory() as path:
            raw.save(os.path.join(path, "tokenizer.json"))
            with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
                json.dump(
                    {
                        "tokenizer_class": "LlamaTokenizerFast",
                        "bos_token": "<s>",
                        "eos_token": "</s>",
                    },
                    f,
                )
            tokenizer = get_tokenizer(path, tokenizer_backend="fastokens")

        from fastokens._compat import _TokenizerShim

        self.assertIsInstance(tokenizer._tokenizer, _TokenizerShim)
        text = "<PAGE>中文 OCR page parsing"
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(ids, raw.encode(text, add_special_tokens=False).ids)
        self.assertEqual(
            tokenizer.decode(ids), raw.decode(ids, skip_special_tokens=False)
        )


if __name__ == "__main__":
    unittest.main()
