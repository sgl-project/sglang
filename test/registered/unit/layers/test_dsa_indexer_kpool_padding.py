"""CPU regression coverage for DSA KPool target-verify padding."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool, kpool_fp8_index
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSAIndexerKPoolTargetVerifyPadding(CustomTestCase):
    @staticmethod
    def _indexer(num_tokens: int):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool_compress_ape = torch.empty(4, 128)
        indexer.scale_fmt = None
        indexer._get_q_k_bf16 = MagicMock(
            return_value=(
                torch.randn(num_tokens, 2, 128),
                torch.randn(num_tokens, 128),
                torch.randn(num_tokens, 128),
            )
        )
        indexer._compute_gate_score_if_missing = MagicMock(
            return_value=torch.randn(num_tokens, 128)
        )
        return indexer

    @staticmethod
    def _pool():
        pool = MagicMock()
        pool.get_compress_tail_buffers.return_value = (
            torch.empty(1, 12, 128),
            torch.empty(1, 12, 128),
        )
        pool.get_index_k_with_scale_buffer.return_value = torch.empty(1)
        return pool

    def test_cache_write_drops_mlp_sync_padding_tokens(self):
        indexer = self._indexer(num_tokens=8)
        pool = self._pool()
        plan = SimpleNamespace(
            num_draft_tokens=6,
            req=torch.tensor([17, 99]),
            write_start=torch.tensor([23, 0], dtype=torch.int32),
            tail_logical_start=torch.tensor([19, 0], dtype=torch.int32),
            write_loc=torch.tensor([[29], [-1]]),
            effective_n_per_batch=torch.tensor([6, 0], dtype=torch.int32),
        )
        metadata = SimpleNamespace(attn_metadata=SimpleNamespace(kpool_write_plan=plan))
        forward_batch = SimpleNamespace(
            num_token_non_padded_cpu=6,
            spec_info=SimpleNamespace(ragged_verify_layout=None),
            out_cache_loc=torch.arange(8) + 100,
        )

        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            patch.object(
                kpool_fp8_index, "kpool_write_tail_and_maybe_compress"
            ) as write,
        ):
            result = indexer._forward_cuda_target_verify(
                x=torch.randn(8, 64),
                q_lora=torch.randn(8, 32),
                positions=torch.arange(8),
                forward_batch=forward_batch,
                layer_id=7,
                act_quant=MagicMock(),
                metadata=metadata,
                enable_dual_stream=False,
                return_indices=False,
            )

        self.assertIsNone(result)
        kwargs = write.call_args.kwargs
        self.assertEqual(kwargs["key"].shape, (6, 128))
        self.assertEqual(kwargs["score"].shape, (6, 128))
        torch.testing.assert_close(kwargs["out_cache_loc"], torch.arange(6) + 100)
        torch.testing.assert_close(kwargs["req_pool_indices"], torch.tensor([17]))
        torch.testing.assert_close(
            kwargs["write_start"], torch.tensor([23], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kwargs["tail_logical_start"], torch.tensor([19], dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["write_loc"], torch.tensor([[29]]))
        torch.testing.assert_close(
            kwargs["effective_n_per_batch"],
            torch.tensor([6], dtype=torch.int32),
        )

    def test_cache_write_rejects_ragged_groups(self):
        indexer = self._indexer(num_tokens=4)
        pool = self._pool()
        plan = SimpleNamespace(num_draft_tokens=3)
        metadata = SimpleNamespace(attn_metadata=SimpleNamespace(kpool_write_plan=plan))
        forward_batch = SimpleNamespace(
            num_token_non_padded_cpu=4,
            out_cache_loc=torch.arange(4),
            spec_info=SimpleNamespace(ragged_verify_layout=object()),
        )

        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            self.assertRaisesRegex(AssertionError, "fixed-width write plan"),
        ):
            indexer._forward_cuda_target_verify(
                x=torch.randn(4, 64),
                q_lora=torch.randn(4, 32),
                positions=torch.arange(4),
                forward_batch=forward_batch,
                layer_id=7,
                act_quant=MagicMock(),
                metadata=metadata,
                enable_dual_stream=False,
                return_indices=False,
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
