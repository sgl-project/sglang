"""End-to-end check of Nemotron-3-Embed support against a Hugging Face reference.

Builds a tiny random checkpoint with the same config shape as
nvidia/Nemotron-3-Embed-1B (bare ``Ministral3Model``, ``is_causal=false``,
``pooling=avg``) so the test does not download the real weights.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Ministral3Config, Ministral3Model, PreTrainedTokenizerFast

from sglang import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

PROMPTS = ["t3 t4", "t5 t6 t7 t8 t9", " ".join(["t10", "t11"] * 24)]


class TestNemotron3Embedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        config = Ministral3Config(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=128,
            original_max_position_embeddings=128,
            initializer_range=0.1,
            architectures=["Ministral3Model"],
            is_causal=False,
            pooling="avg",
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        hf_model = Ministral3Model(config).eval()

        backend = Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0"))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="t0",
            pad_token="t0",
            bos_token="t1",
            eos_token="t2",
        )

        cls.references = []
        with torch.no_grad():
            for prompt in PROMPTS:
                tokens = tokenizer(prompt, return_tensors="pt")
                hidden = hf_model(**tokens).last_hidden_state
                cls.references.append(F.normalize(hidden.mean(dim=1), dim=-1)[0])

        cls.model_dir = tempfile.TemporaryDirectory()
        hf_model.save_pretrained(cls.model_dir.name)
        tokenizer.save_pretrained(cls.model_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.model_dir.cleanup()

    def _check(self, *, dtype, engine_kwargs, assert_close):
        # No --is-embedding: the checkpoint config alone must select the model.
        engine = Engine(
            model_path=self.model_dir.name,
            dtype=dtype,
            mem_fraction_static=0.15,
            # Schedule each encode() call as one packed batch.
            enable_tokenizer_batch_encode=True,
            **engine_kwargs,
        )
        try:
            model_config = engine.tokenizer_manager.model_config
            self.assertEqual(
                model_config.hf_config.architectures, ["EmbeddingNemotron3Model"]
            )
            self.assertFalse(model_config.is_generation)
            # Batches in different orders exercise the packed mean pooling.
            for order in ([0], [1], [2], [0, 1, 2], [2, 1, 0]):
                results = engine.encode([PROMPTS[i] for i in order])
                self.assertEqual(len(results), len(order))
                for result, index in zip(results, order):
                    actual = torch.tensor(result["embedding"])
                    assert_close(actual, self.references[index])
        finally:
            engine.shutdown()

    def test_fp32_torch_native_matches_hf(self):
        def assert_close(actual, expected):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

        with patch.dict(os.environ, {"SGLANG_FORCE_FUSED_OP_BACKEND": "torch"}):
            self._check(
                dtype="float32",
                engine_kwargs={"attention_backend": "torch_native"},
                assert_close=assert_close,
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "requires a GPU with bf16 support",
    )
    def test_bf16_default_backend_matches_hf(self):
        def assert_close(actual, expected):
            cosine = F.cosine_similarity(actual, expected, dim=0).item()
            self.assertGreaterEqual(cosine, 0.999)

        self._check(dtype="bfloat16", engine_kwargs={}, assert_close=assert_close)


if __name__ == "__main__":
    unittest.main()
