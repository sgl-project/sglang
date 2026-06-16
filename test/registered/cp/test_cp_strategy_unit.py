import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.base import (
    ContextParallelStrategyKind,
    get_cp_strategy,
    get_cp_strategy_kind,
    init_cp_strategy,
    is_cp_enabled,
    is_interleave,
    is_zigzag,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ExtendMode:
    def is_context_parallel_extend(self):
        return True


class _FakeCPGroup:
    def __init__(self, all_rank_tensors):
        self.all_rank_tensors = all_rank_tensors

    def cp_all_gather_into_tensor_async(self, output, input_tensor, stream):
        del input_tensor, stream
        torch.cat(self.all_rank_tensors, dim=0, out=output)


class TestCPStrategyUnit(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))

    def _init_zigzag(self, cp_size=4):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=cp_size,
                attention_backend="fa3",
            )
        )

    def _metadata_for_rank(self, rank, cp_size=4, num_tokens=19):
        strategy = ZigzagCPStrategy(cp_size=cp_size)
        with patch(
            "sglang.srt.layers.dp_attention.get_attention_cp_rank",
            return_value=rank,
        ):
            return strategy.build_metadata(
                num_tokens=num_tokens,
                seqs_len=[11, 13],
                extend_seqs_len=[9, 10],
            )

    def _forward_batch(self, metadata):
        return SimpleNamespace(
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[9, 10],
            attn_cp_metadata=metadata,
        )

    def _padded_rank_tensors(self, x, cp_size=4):
        per_rank = []
        metas = []
        for rank in range(cp_size):
            metadata = self._metadata_for_rank(rank, cp_size=cp_size)
            metas.append(metadata)
            fb = self._forward_batch(metadata)
            with patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_rank",
                return_value=rank,
            ):
                local = ZigzagCPStrategy(cp_size=cp_size).shard_hidden_states(x, fb)
            pad = metadata.max_rank_len[0] - local.shape[0]
            if pad:
                local = torch.nn.functional.pad(
                    local,
                    [0, 0] * (local.ndim - 1) + [0, pad],
                )
            per_rank.append(local)
        return metas, per_rank

    def test_strategy_kind_maps_cli_values(self):
        self.assertEqual(ContextParallelStrategyKind.NONE.value, 0)
        self.assertEqual(
            ContextParallelStrategyKind.from_string("zigzag"),
            ContextParallelStrategyKind.ZIGZAG,
        )
        self.assertEqual(
            ContextParallelStrategyKind.from_string("interleave"),
            ContextParallelStrategyKind.INTERLEAVE,
        )
        self.assertEqual(ContextParallelStrategyKind.ZIGZAG.cli_value, "zigzag")
        self.assertEqual(ContextParallelStrategyKind.INTERLEAVE.cli_value, "interleave")

    def test_cp_v2_strategy_is_only_active_when_env_enabled(self):
        self._init_zigzag()

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=False
        ):
            self.assertIsNone(get_cp_strategy())
            self.assertFalse(is_cp_enabled())

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ):
            self.assertIsNotNone(get_cp_strategy())
            self.assertTrue(is_cp_enabled())
            self.assertTrue(is_zigzag())
            self.assertFalse(is_interleave())
            self.assertEqual(get_cp_strategy_kind(), ContextParallelStrategyKind.ZIGZAG)

    def test_zigzag_metadata_for_batched_sequences(self):
        metadata = self._metadata_for_rank(rank=2)

        self.assertEqual(metadata.bs, 2)
        self.assertEqual(metadata.total_seq_lens, 19)
        self.assertEqual(
            metadata.split_list,
            [2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(metadata.zigzag_index, [2, 10, 5, 13])
        self.assertEqual(metadata.per_rank_actual_token, [6, 5, 4, 4])
        self.assertEqual(metadata.max_rank_len, [6, 6, 6, 6])
        self.assertEqual(
            metadata.reverse_split_len,
            [2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            metadata.cp_reverse_index,
            [0, 4, 8, 12, 14, 10, 6, 2, 1, 5, 9, 13, 15, 11, 7, 3],
        )
        self.assertEqual(metadata.kv_len_prev_list, [6, 8])
        self.assertEqual(metadata.kv_len_next_list, [9, 11])
        self.assertEqual(metadata.actual_seq_q_prev_list, [1, 1])
        self.assertEqual(metadata.actual_seq_q_next_list, [1, 1])
        self.assertEqual(metadata.total_q_prev_tokens, 2)
        self.assertEqual(metadata.total_q_next_tokens, 2)
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_q_prev_tensor.cpu(), torch.tensor([0, 1, 2])
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_q_next_tensor.cpu(), torch.tensor([0, 1, 2])
            )
        )

    def test_zigzag_shards_hidden_states_and_position_ids(self):
        metadata = self._metadata_for_rank(rank=2)
        fb = self._forward_batch(metadata)
        x = torch.arange(19 * 2).view(19, 2)
        positions = torch.arange(19)

        with patch(
            "sglang.srt.layers.dp_attention.get_attention_cp_rank",
            return_value=2,
        ):
            strategy = ZigzagCPStrategy(cp_size=4)
            local_x = strategy.shard_hidden_states(x, fb)
            local_positions = strategy.shard_position_ids(positions, fb)

        expected_indices = [3, 13, 6, 16]
        self.assertTrue(torch.equal(local_x, x[expected_indices]))
        self.assertTrue(torch.equal(local_positions, positions[expected_indices]))

    def test_zigzag_gathers_hidden_states_to_original_order(self):
        cp_size = 4
        x = torch.arange(19 * 2).view(19, 2)
        metas, padded_rank_tensors = self._padded_rank_tensors(x, cp_size=cp_size)
        rank = 2
        local_x = padded_rank_tensors[rank][: metas[rank].per_rank_actual_token[rank]]
        fb = self._forward_batch(metas[rank])

        with (
            patch(
                "sglang.srt.layers.cp.zigzag.get_attention_cp_group",
                return_value=_FakeCPGroup(padded_rank_tensors),
            ),
            patch(
                "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
                return_value=torch.no_grad(),
            ),
        ):
            gathered = ZigzagCPStrategy(cp_size=cp_size).gather_hidden_states(
                local_x, fb, stream=None
            )

        self.assertTrue(torch.equal(gathered, x))

    def test_zigzag_gathers_kv_cache_to_original_order(self):
        cp_size = 4
        kv = torch.arange(19 * 2 * 3).view(19, 2, 3)
        metas, padded_rank_tensors = self._padded_rank_tensors(kv, cp_size=cp_size)
        rank = 1
        local_kv = padded_rank_tensors[rank][: metas[rank].per_rank_actual_token[rank]]
        fb = self._forward_batch(metas[rank])

        with (
            patch(
                "sglang.srt.layers.cp.zigzag.get_attention_cp_group",
                return_value=_FakeCPGroup(padded_rank_tensors),
            ),
            patch(
                "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
                return_value=torch.no_grad(),
            ),
        ):
            gathered = ZigzagCPStrategy(cp_size=cp_size).gather_kv_cache(
                local_kv, fb, stream=None
            )

        self.assertTrue(torch.equal(gathered, kv))

    def test_zigzag_attention_dispatch_runs_prev_then_next(self):
        with patch(
            "sglang.srt.layers.dp_attention.get_attention_cp_rank",
            return_value=0,
        ):
            metadata = ZigzagCPStrategy(cp_size=2).build_metadata(
                num_tokens=8,
                seqs_len=[8],
                extend_seqs_len=[8],
            )
        fb = SimpleNamespace(attn_cp_metadata=metadata)
        q = torch.arange(4 * 2).view(4, 2)
        calls = []

        def attn_fn(q_chunk, cu_seqlens_q, cache_seqlens, max_seqlen_q):
            calls.append(
                (
                    q_chunk.clone(),
                    cu_seqlens_q.clone(),
                    cache_seqlens.clone(),
                    max_seqlen_q,
                )
            )
            return q_chunk + 100

        out = ZigzagCPStrategy(cp_size=2).run_attention(
            q, fb, device=torch.device("cpu"), attn_fn=attn_fn
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][0], q[:2]))
        self.assertTrue(torch.equal(calls[1][0], q[2:]))
        self.assertTrue(torch.equal(out, q + 100))


if __name__ == "__main__":
    unittest.main()
