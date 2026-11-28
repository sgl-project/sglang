"""Reusing a LoRA slot must stop applying the evicted adapter's embedding weights."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import LoRAType
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, HIDDEN, RANK, MAX_ADDED = 8, 4, 2, 4
EMBED_A = "model.embed_tokens.lora_embedding_A"
EMBED_B = "model.embed_tokens.lora_embedding_B"
HEAD_A = "lm_head.lora_A.weight"
HEAD_B = "lm_head.lora_B.weight"


def _embedding_weights(added_tokens, value):
    return {
        EMBED_A: torch.full((RANK, VOCAB + added_tokens), value),
        EMBED_B: torch.full((HIDDEN, RANK), value),
    }


def _head_weights():
    return {HEAD_A: torch.ones(RANK, HIDDEN), HEAD_B: torch.ones(VOCAB, RANK)}


class TestLoRAEmbeddingSlotReuse(CustomTestCase):
    def setUp(self):
        config = SimpleNamespace(
            num_hidden_layers=0,
            num_attention_heads=1,
            hidden_size=HIDDEN,
            vocab_size=VOCAB,
        )
        model = torch.nn.Linear(HIDDEN, HIDDEN)
        model.config = config
        with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1):
            self.pool = LoRAMemoryPool(
                base_hf_config=config,
                max_loras_per_batch=1,
                dtype=torch.float32,
                max_lora_rank=RANK,
                target_modules={"embed_tokens", "lm_head"},
                base_model=model,
                eviction_policy="lru",
                lora_added_tokens_size=MAX_ADDED,
                lora_layer_ids=[],
                has_embed_tokens_lora=True,
                has_lm_head_lora=True,
            )

    def _load(self, uid, weights, added_tokens=0, value=1.0):
        added_embeddings = (
            {"input_embeddings": torch.full((added_tokens, HIDDEN), value)}
            if added_tokens
            else {}
        )
        adapter = SimpleNamespace(
            config=SimpleNamespace(r=RANK, lora_added_tokens_size=added_tokens),
            layers=[],
            embedding_layers=weights,
            pinned_embedding_layers={},
            added_tokens_embeddings=added_embeddings,
            pinned_added_tokens_embeddings={},
        )
        self.pool.prepare_lora_batch(
            cur_uids={uid},
            lora_adapters={uid: adapter},
            lora_modules=[],
            lora_refs={},
            lora_embed_tokens_module=object(),
            lora_lm_head_module=object(),
        )
        self.assertEqual(self.pool.uid_to_buffer_id, {uid: 0})

    def _slot(self, module, lora_type=LoRAType.LORA_A):
        return self.pool.get_embedding_tensor(module, lora_type)[0]

    def _assert_zero(self, tensor):
        torch.testing.assert_close(tensor, torch.zeros_like(tensor))

    def test_head_only_evicts_embedding_and_added_tokens(self):
        self._load("embed", _embedding_weights(MAX_ADDED, 1.0), MAX_ADDED)
        weights = _head_weights()
        self._load("head", weights)

        for module, lora_type in (
            ("embed_tokens", LoRAType.LORA_A),
            ("embed_tokens", LoRAType.LORA_B),
            ("added_tokens", LoRAType.LORA_A),
        ):
            with self.subTest(module=module, lora_type=lora_type):
                self._assert_zero(self._slot(module, lora_type))
        torch.testing.assert_close(self._slot("lm_head"), weights[HEAD_A])
        torch.testing.assert_close(
            self._slot("lm_head", LoRAType.LORA_B)[:VOCAB], weights[HEAD_B]
        )

    def test_embedding_only_evicts_head(self):
        self._load("head", _head_weights())
        self._load("embed", _embedding_weights(0, 1.0))

        for lora_type in LoRAType:
            with self.subTest(lora_type=lora_type):
                self._assert_zero(self._slot("lm_head", lora_type))

    def test_reuse_with_fewer_added_tokens(self):
        for added_tokens in (0, 1):
            with self.subTest(added_tokens=added_tokens):
                self._load("large", _embedding_weights(MAX_ADDED, 1.0), MAX_ADDED)
                weights = _embedding_weights(added_tokens, 2.0)
                self._load("small", weights, added_tokens, value=2.0)

                vocab = VOCAB + added_tokens
                embedding_a = self._slot("embed_tokens")
                added = self._slot("added_tokens")
                torch.testing.assert_close(embedding_a[:, :vocab], weights[EMBED_A])
                torch.testing.assert_close(
                    added[:added_tokens], torch.full((added_tokens, HIDDEN), 2.0)
                )
                with self.subTest(buffer="embedding A"):
                    self._assert_zero(embedding_a[:, vocab:])
                with self.subTest(buffer="added tokens"):
                    self._assert_zero(added[added_tokens:])

    def test_added_embeddings_load_without_embedding_lora(self):
        for max_rank_spaced_layout in (False, True):
            with self.subTest(max_rank_spaced_layout=max_rank_spaced_layout):
                self.pool.max_rank_spaced_layout = max_rank_spaced_layout
                self._load("embed", _embedding_weights(MAX_ADDED, 1.0), MAX_ADDED)
                self._load("added-only", {}, added_tokens=1, value=2.0)

                expected = torch.zeros(MAX_ADDED, HIDDEN)
                expected[0].fill_(2.0)
                torch.testing.assert_close(self._slot("added_tokens"), expected)
                for lora_type in LoRAType:
                    self._assert_zero(self._slot("embed_tokens", lora_type))

                self._load("empty", {})
                self._assert_zero(self._slot("added_tokens"))


if __name__ == "__main__":
    unittest.main()
