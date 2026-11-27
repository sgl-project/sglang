"""End-to-end check of Nemotron-3-Embed support against a Hugging Face reference.

Builds a tiny random checkpoint with the same config shape as
nvidia/Nemotron-3-Embed-1B (bare ``Ministral3Model``, ``is_causal=false``,
``pooling=avg``) so the test does not download the real weights.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Ministral3Config, Ministral3Model, PreTrainedTokenizerFast

from sglang import Engine
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_cuda_ci, register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-small")
register_npu_ci(est_time=120, suite="extra-b-test-1-npu-a3")

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

        with torch.no_grad():
            tokens = tokenizer(PROMPTS, padding=True, return_tensors="pt")
            mask = tokens.attention_mask.unsqueeze(-1)
            hidden = hf_model(**tokens).last_hidden_state
            pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
            cls.references = F.normalize(pooled, dim=-1)

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

    @unittest.skipIf(is_npu(), "the Ascend runtime uses BF16 attention")
    def test_fp32_torch_native_matches_hf(self):
        def assert_close(actual, expected):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

        with patch.dict(os.environ, {"SGLANG_FORCE_FUSED_OP_BACKEND": "torch"}):
            self._check(
                dtype="float32",
                engine_kwargs={"attention_backend": "torch_native"},
                assert_close=assert_close,
            )

    def _assert_bf16_close(self, actual, expected):
        self.assertAlmostEqual(actual.norm().item(), 1.0, places=5)
        cosine = F.cosine_similarity(actual, expected, dim=0).item()
        self.assertGreaterEqual(cosine, 0.999)

    @unittest.skipUnless(
        is_npu() or (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "requires a BF16 GPU or an Ascend NPU",
    )
    def test_bf16_default_backend_matches_hf(self):
        self._check(
            dtype="bfloat16",
            engine_kwargs={"max_total_tokens": 2048},
            assert_close=self._assert_bf16_close,
        )

    @unittest.skipUnless(is_npu(), "requires an Ascend NPU")
    def test_bf16_fia_backend_matches_hf(self):
        with patch.dict(os.environ, {"ASCEND_USE_FIA": "1"}):
            self._check(
                dtype="bfloat16",
                engine_kwargs={"max_total_tokens": 2048},
                assert_close=self._assert_bf16_close,
            )

    @unittest.skipUnless(is_npu(), "requires an Ascend NPU")
    def test_fp32_mean_pooling_on_npu(self):
        from sglang.srt.layers.pooler import PoolingType
        from sglang.srt.models.nemotron3_embedding import _FP32Pooler

        # The last two tokens follow a long packed sequence.
        hidden = torch.arange(514 * 4).reshape(514, 4).remainder(17).bfloat16()
        lengths = [512, 2]
        batch = SimpleNamespace(
            extend_seq_lens=torch.tensor(lengths, device="npu"), dimensions=None
        )
        pooler = _FP32Pooler(pooling_type=PoolingType.MEAN, normalize=True)
        actual = pooler(hidden.to("npu"), batch).embeddings.cpu()
        expected = torch.stack([seq.float().mean(0) for seq in hidden.split(lengths)])
        torch.testing.assert_close(actual, F.normalize(expected, dim=-1))


if __name__ == "__main__":
    unittest.main()
