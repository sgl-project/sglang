"""Sequence-order mixers and KPool metadata under interleave CP."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp import interleave
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestInterleaveMixer(CustomTestCase):
    def test_multimodal_mixer_input_is_rejected_only_during_cp_extend(self):
        metadata = InterleaveContextParallelMetadata(
            total_seq_lens=4, per_rank_actual_token=[1] * 4
        )
        for extend in (False, True):
            for cp_metadata in (None, metadata):
                for multimodal in (False, True):
                    with self.subTest(extend=extend, cp=cp_metadata, mm=multimodal):
                        batch = SimpleNamespace(
                            forward_mode=SimpleNamespace(
                                is_context_parallel_extend=lambda: extend
                            ),
                            attn_cp_metadata=cp_metadata,
                            contains_mm_inputs=lambda: multimodal,
                        )
                        if extend and cp_metadata is not None and multimodal:
                            with self.assertRaisesRegex(ValueError, "text-only"):
                                interleave.validate_mixer_batch(batch)
                        else:
                            interleave.validate_mixer_batch(batch)

    def test_pooling_gathers_keys_and_scores_but_keeps_queries_local(self):
        from sglang.srt.layers.cp import base, interleave, interleave_kpool

        keys = torch.arange(33).reshape(11, 3).bfloat16()
        scores = keys + 100
        positions = torch.arange(11)
        packed = torch.full((4, 4, 6), -999, dtype=torch.bfloat16)
        for token in range(11):
            packed[token % 4, token // 4] = torch.cat((keys[token], scores[token]))
        metadata = InterleaveContextParallelMetadata(
            total_seq_lens=11,
            per_rank_actual_token=[4] * 4,
            per_rank_logical_token=[3, 3, 3, 2],
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
            attn_cp_metadata=metadata,
            positions=positions,
        )
        for rank in range(4):
            with self.subTest(rank=rank):

                def gather(output, local):
                    torch.testing.assert_close(
                        local[: len(keys[rank::4])], packed[rank, : len(keys[rank::4])]
                    )
                    output.copy_(packed.flatten(0, 1))

                parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_group=None)
                with (
                    patch.object(base, "get_parallel", return_value=parallel),
                    patch.object(interleave, "get_parallel", return_value=parallel),
                    patch.object(interleave, "attn_cp_all_gather_into_tensor", gather),
                    patch.object(
                        interleave, "is_allocation_symmetric", return_value=False
                    ),
                    patch.object(
                        interleave_kpool,
                        "get_cp_strategy",
                        return_value=interleave.InterleaveCPStrategy(4),
                    ),
                ):
                    full_key, full_score, full_positions = (
                        interleave_kpool.materialize_compression_inputs(
                            packed[rank, :, :3],
                            packed[rank, :, 3:],
                            positions[rank::4],
                            batch,
                        )
                    )
                torch.testing.assert_close(full_key, keys)
                torch.testing.assert_close(full_score, scores)
                self.assertIs(full_positions, positions)

    def test_sequence_order_and_padding_round_trip(self):
        for tokens, shard_rows in ((8, 2), (11, 3), (11, 8), (1, 1), (0, 0)):
            with self.subTest(tokens=tokens, shard_rows=shard_rows):
                sequence = torch.arange(tokens * 2).reshape(tokens, 2).float()
                shards = torch.zeros(4, shard_rows, 2)
                for token in range(tokens):
                    shards[token % 4, token // 4] = sequence[token]
                packed = shards.flatten(0, 1)
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(
                        is_context_parallel_extend=lambda: True
                    ),
                    attn_cp_metadata=InterleaveContextParallelMetadata(
                        total_seq_lens=tokens,
                        per_rank_actual_token=[shard_rows] * 4,
                    ),
                )
                with patch.object(
                    interleave,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_cp_size=4),
                ):
                    restored = interleave.mixer_to_sequence_order(packed, batch)
                    torch.testing.assert_close(restored, sequence)
                    # A causal mixer must see the original order, not rank-major rows.
                    mixed = restored.cumsum(0)
                    output = interleave.mixer_to_rank_order(
                        mixed, batch, packed.shape[0]
                    )
                    torch.testing.assert_close(
                        interleave.mixer_to_rank_order(
                            sequence, batch, packed.shape[0]
                        ),
                        packed,
                    )
                for token in range(tokens):
                    torch.testing.assert_close(
                        output.reshape(4, shard_rows, 2)[token % 4, token // 4],
                        sequence[: token + 1].sum(0),
                    )

    def test_decode_does_not_reorder_rows(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: False),
            attn_cp_metadata=None,
        )
        hidden = torch.arange(6).reshape(3, 2)
        self.assertIs(interleave.mixer_to_sequence_order(hidden, batch), hidden)
        self.assertIs(interleave.mixer_to_rank_order(hidden, batch, 3), hidden)

    def test_kpool_local_queries_keep_full_compression_request_order(self):
        from sglang.srt.layers.cp import interleave_kpool

        batch = SimpleNamespace(
            extend_seq_lens_cpu=[1, 3, 5],
            req_pool_indices=torch.tensor([9, 2, 7]),
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
            attn_cp_metadata=InterleaveContextParallelMetadata(total_seq_lens=9),
        )
        metadata = SimpleNamespace(
            real_page_table=torch.tensor([[20], [70]]),
            dsa_seqlens_expanded=torch.tensor([6, 11]),
            indexer_seq_lens_cpu=torch.tensor([8, 14]),
            dsa_extend_seq_lens_list=[1, 1],
        )
        with patch.object(
            interleave_kpool,
            "get_parallel",
            return_value=SimpleNamespace(attn_cp_size=4, attn_cp_rank=2),
        ):
            local = interleave_kpool.local_query_inputs(batch, metadata)
        self.assertEqual(local["local_extend_seq_lens_cpu"], [1, 1])
        self.assertEqual(local["local_seq_lens_cpu"], [8, 14])
        self.assertEqual(local["local_req_pool_indices"].tolist(), [2, 7])
        self.assertIs(local["local_real_page_table"], metadata.real_page_table)
        self.assertIs(local["local_seqlens_expanded"], metadata.dsa_seqlens_expanded)


if __name__ == "__main__":
    unittest.main()
