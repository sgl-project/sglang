"""CPU coverage of byte-row embeddings, weight ownership and TP row layout."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import vocab_parallel_embedding as vocab_module
from sglang.srt.mem_cache.engram import embedding as embedding_module
from sglang.srt.mem_cache.engram.embedding import HiCacheVocabParallelEmbedding
from sglang.srt.mem_cache.hicache_storage import HiCacheEngramStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _CpuRows(HiCacheEngramStorage):
    """In-memory replacement for the external storage, not the embedding logic."""

    def __init__(self):
        self.rows = None
        self.initialize_count = 0

    def initialize(self, rows):
        if self.rows is not None:
            raise RuntimeError("table already published")
        self.rows = rows.clone()
        self.initialize_count += 1

    def lookup(self, ids):
        out = torch.zeros((*ids.shape, self.rows.shape[1]), dtype=torch.uint8)
        valid = ids >= 0
        out[valid] = self.rows[ids[valid].long()]
        return out

    def close(self):
        self.rows = None


class TestHiCacheEngramEmbedding(CustomTestCase):
    def setUp(self):
        self.storage = _CpuRows()
        self.addCleanup(self.storage.close)
        self.enterContext(
            mock.patch.object(
                embedding_module, "create_storage", return_value=self.storage
            )
        )
        self.enterContext(
            mock.patch.object(
                embedding_module,
                "get_memory",
                return_value=SimpleNamespace(hi_engram_config=None),
            )
        )
        parallel = SimpleNamespace(
            tp_group=None, tp_size=2, tp_rank=1, attn_tp_size=2, attn_tp_rank=1
        )
        self.enterContext(
            mock.patch.object(embedding_module, "get_parallel", return_value=parallel)
        )
        self.enterContext(
            mock.patch.object(vocab_module, "get_parallel", return_value=parallel)
        )
        self.enterContext(
            mock.patch.object(vocab_module, "_resolve_linear_group", return_value=None)
        )
        # NCCL allocation and collectives are external to the byte-row adapter.
        self.enterContext(
            mock.patch.object(
                embedding_module, "use_symmetric_memory", return_value=nullcontext()
            )
        )
        self.enterContext(
            mock.patch.object(
                embedding_module, "is_allocation_symmetric", return_value=False
            )
        )
        # Exercise the real mask arithmetic eagerly on CPU, without compiling it.
        self.enterContext(torch._dynamo.config.patch(disable=True))

    def _embedding(self, *, enable_tp=False, **kwargs):
        return HiCacheVocabParallelEmbedding(
            num_embeddings=10,
            org_num_embeddings=6,
            embedding_dim=3,
            params_dtype=torch.bfloat16,
            padding_size=4,
            enable_tp=enable_tp,
            **kwargs,
        )

    def test_weight_loader_publication_and_reuse_preserve_values(self):
        embedding = self._embedding()
        rows = torch.arange(18, dtype=torch.bfloat16).reshape(6, 3) / 2
        embedding.weight.weight_loader(embedding.weight, rows)
        self.assertEqual(embedding.weight.device.type, "cpu")
        self.assertEqual(list(embedding.state_dict()), ["weight"])
        self.assertIsNone(embedding.quant_method)
        for ids in (torch.tensor([[5, 0], [2, 5]]), torch.tensor([1, 3])):
            actual = embedding.gather(ids)
            torch.testing.assert_close(actual, rows[ids], rtol=0, atol=0)
        self.assertEqual(self.storage.initialize_count, 1)
        self.assertEqual(embedding.weight.numel(), 0)
        self.assertEqual(embedding.weight.dtype, torch.bfloat16)

    def test_tp_masks_other_shard_and_accounts_for_padding_before_added_vocab(self):
        embedding = self._embedding(enable_tp=True)
        # Rank 1 owns original IDs [4, 6) and added IDs [8, 10).
        # Original IDs occupy four local slots; added IDs start at slot 4.
        local_rows = torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
                [90, 90, 90],
                [91, 91, 91],
                [30, 31, 32],
                [40, 41, 42],
            ],
            dtype=torch.bfloat16,
        )
        embedding.weight.data.copy_(local_rows)
        ids = torch.tensor([[-1, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.int32)
        expected = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [10, 11, 12], [20, 21, 22]],
                [[0, 0, 0], [0, 0, 0], [30, 31, 32], [40, 41, 42]],
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(embedding.gather(ids), expected, rtol=0, atol=0)

    def test_gather_converts_dtype_and_reuses_supplied_output(self):
        embedding = self._embedding(output_dtype=torch.float32)
        rows = torch.arange(18, dtype=torch.bfloat16).reshape(6, 3) / 2
        embedding.weight_loader(embedding.weight, rows)
        ids = torch.tensor([[5, 1], [0, 4]], dtype=torch.int32)
        output = embedding.allocate_output((*ids.shape, 3), ids.device)
        output.fill_(float("nan"))
        result = embedding.gather(ids, out=output)
        self.assertIs(result, output)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, rows[ids.long()].float(), rtol=0, atol=0)

    def test_invalid_output_is_rejected_before_publishing_weights(self):
        embedding = self._embedding()
        ids = torch.tensor([1, 2])
        for output in (
            torch.empty((2, 4), dtype=torch.bfloat16),
            torch.empty((2, 3), dtype=torch.float32),
            torch.empty((2, 3), dtype=torch.bfloat16, device="meta"),
        ):
            with self.subTest(
                shape=output.shape, dtype=output.dtype, device=output.device
            ):
                with self.assertRaisesRegex(ValueError, "shape, dtype or device"):
                    embedding.gather(ids, out=output)
        self.assertEqual(self.storage.initialize_count, 0)
        self.assertGreater(embedding.weight.numel(), 0)

    def test_failed_publication_retains_loaded_weights_for_retry(self):
        embedding = self._embedding()
        rows = torch.arange(18, dtype=torch.bfloat16).reshape(6, 3)
        embedding.weight_loader(embedding.weight, rows)
        with mock.patch.object(
            self.storage, "initialize", side_effect=OSError("publish failed")
        ):
            with self.assertRaisesRegex(OSError, "publish failed"):
                embedding.gather(torch.tensor([4]))
        torch.testing.assert_close(embedding.weight[:6], rows)
        torch.testing.assert_close(embedding.gather(torch.tensor([4])), rows[4:5])
        self.assertEqual(embedding.weight.numel(), 0)

    def test_reduce_selects_tp_group_and_skips_already_scattered_or_empty_rows(self):
        embedding = self._embedding(enable_tp=True)
        values = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        with (
            mock.patch.object(
                embedding_module,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            mock.patch.object(
                embedding_module,
                "tensor_model_parallel_all_reduce",
                side_effect=lambda x: x + 10,
            ),
            mock.patch.object(
                embedding_module, "attn_tp_all_reduce", side_effect=lambda x: x + 20
            ),
        ):
            torch.testing.assert_close(embedding.reduce(values), values + 10)
            embedding.use_attn_tp_group = True
            torch.testing.assert_close(embedding.reduce(values), values + 20)
        with (
            mock.patch.object(
                embedding_module,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=True),
            ),
            mock.patch.object(
                embedding_module,
                "attn_tp_all_reduce",
                side_effect=AssertionError("unexpected collective"),
            ),
        ):
            self.assertIs(embedding.reduce(values), values)
        with mock.patch.object(
            embedding_module,
            "get_attn_tp_context",
            side_effect=AssertionError("empty batch needs no context"),
        ):
            empty = values[:0]
            self.assertIs(embedding.reduce(empty), empty)
        embedding.tp_size = 1
        self.assertIs(embedding.reduce(values), values)


if __name__ == "__main__":
    unittest.main()
