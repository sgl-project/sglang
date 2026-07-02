import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.communicator import _get_moe_cp_padded_tokens_per_rank
from sglang.srt.layers.cp.base import (
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    get_cp_strategy,
    get_cp_strategy_kind,
    init_cp_strategy,
    is_cp_enabled,
    is_interleave,
    is_zigzag,
)
from sglang.srt.layers.cp.utils import (
    CP_V2_DEFAULT_MODEL_CLASSES,
    cp_split_before_forward,
    enable_cp_v2,
    is_cp_v2_active,
    prepare_cp_forward,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ExtendMode:
    def is_context_parallel_extend(self):
        return True


class _FakeCPGroup:
    def __init__(self, all_rank_tensors):
        self.all_rank_tensors = all_rank_tensors
        self.all_gather_calls = 0
        self.async_all_gather_calls = 0
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1

    def all_gather_into_tensor(self, output, input_tensor):
        del input_tensor
        self.all_gather_calls += 1
        torch.cat(self.all_rank_tensors, dim=0, out=output)

    def cp_all_gather_into_tensor_async(self, output, input_tensor, stream):
        del input_tensor, stream
        self.async_all_gather_calls += 1
        torch.cat(self.all_rank_tensors, dim=0, out=output)


class TestCPStrategyUnit(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))

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

    def test_flash_attention_cp_backend_accepts_fa4(self):
        self.assertEqual(
            CPAttentionBackendKind.from_string("fa4"),
            CPAttentionBackendKind.FLASH_ATTENTION,
        )

    def test_init_cp_strategy_binds_zigzag_strategy(self):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=4,
            )
        )

        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_zigzag())
        self.assertFalse(is_interleave())
        self.assertEqual(get_cp_strategy_kind(), ContextParallelStrategyKind.ZIGZAG)

    def test_get_cp_strategy_is_initialized_under_cp_v1_and_cp_v2(self):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="interleave",
                attn_cp_size=4,
            )
        )

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=False
        ):
            self.assertIsNotNone(get_cp_strategy())
            self.assertTrue(is_cp_enabled())
            self.assertTrue(is_interleave())

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ):
            self.assertIsNotNone(get_cp_strategy())


class TestCPZigzagStrategy(CustomTestCase):
    def setUp(self):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=4,
                attention_backend="fa3",
            )
        )

    def tearDown(self):
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))

    def _metadata_for_rank(self, rank, *, cp_size, seq_lens, extend_seq_lens):
        strategy = ZigzagCPStrategy(cp_size=cp_size)
        with get_parallel().override(attn_cp_rank=rank):
            return strategy.build_metadata(
                num_tokens=sum(extend_seq_lens),
                seqs_len=seq_lens,
                extend_seqs_len=extend_seq_lens,
            )

    def _forward_batch(self, metadata, extend_seq_lens):
        return SimpleNamespace(
            input_ids=torch.arange(sum(extend_seq_lens)),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=extend_seq_lens,
            attn_cp_metadata=metadata,
        )

    def test_enable_cp_v2_and_is_cp_v2_active(self):
        active_batch = SimpleNamespace(
            input_ids=torch.arange(128),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[128],
        )
        inactive_batch = SimpleNamespace(
            input_ids=torch.arange(127),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[127],
        )
        uneven_active_batch = SimpleNamespace(
            input_ids=torch.arange(129),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[129],
        )
        padded_inactive_batch = SimpleNamespace(
            input_ids=torch.arange(128),
            num_token_non_padded_cpu=7,
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[7],
        )

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=False
        ):
            self.assertFalse(enable_cp_v2())
            self.assertFalse(is_cp_v2_active(active_batch))

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ):
            self.assertTrue(enable_cp_v2())
            self.assertTrue(is_cp_v2_active(active_batch))
            self.assertFalse(is_cp_v2_active(inactive_batch))
            self.assertTrue(is_cp_v2_active(uneven_active_batch))
            self.assertFalse(is_cp_v2_active(padded_inactive_batch))

    def test_cp_v2_defaults_include_mimo_v2_architectures(self):
        self.assertIn("MiMoV2ForCausalLM", CP_V2_DEFAULT_MODEL_CLASSES)
        self.assertIn("MiMoV2FlashForCausalLM", CP_V2_DEFAULT_MODEL_CLASSES)

    def test_moe_cp_padding_uses_only_largest_local_shard(self):
        self.assertEqual(_get_moe_cp_padded_tokens_per_rank([160] * 4, 4), 160)
        self.assertEqual(_get_moe_cp_padded_tokens_per_rank([65, 64, 64, 64], 4), 65)

    def test_prepare_cp_forward_uses_real_token_count_before_padding(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(640),
            num_token_non_padded_cpu=515,
            forward_mode=_ExtendMode(),
            seq_lens_cpu=[515],
            extend_seq_lens_cpu=[515],
            attn_cp_metadata=None,
        )

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ), get_parallel().override(attn_cp_rank=0):
            prepare_cp_forward(forward_batch)

        metadata = forward_batch.attn_cp_metadata
        self.assertEqual(metadata.total_seq_lens, 515)
        self.assertEqual(metadata.per_rank_actual_token, [129, 129, 129, 128])

    def _expected_metadata(self, *, rank, cp_size, seq_lens, extend_seq_lens):
        bs = len(extend_seq_lens)
        cp_segment_num = cp_size * 2
        prefix_offsets = [
            max(int(seq_lens[i]) - int(extend_seq_lens[i]), 0) for i in range(bs)
        ]

        per_seq_block_sizes = []
        split_list = []
        for length in extend_seq_lens:
            base = length // cp_segment_num
            rem = length % cp_segment_num
            block_sizes = [
                base + 1 if block_id < rem else base
                for block_id in range(cp_segment_num)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        per_rank_actual_token = [
            sum(
                block_sizes[rank_id] + block_sizes[cp_segment_num - 1 - rank_id]
                for block_sizes in per_seq_block_sizes
            )
            for rank_id in range(cp_size)
        ]
        max_rank_len = [max(per_rank_actual_token)] * cp_size

        zigzag_index = list(range(rank, rank + bs * cp_segment_num, cp_segment_num))
        zigzag_index += list(
            range(cp_segment_num - rank - 1, bs * cp_segment_num, cp_segment_num)
        )

        cp_reverse_index = []
        for batch_id in range(bs):
            cp_reverse_index.extend(
                list(range(batch_id, cp_segment_num * bs, 2 * bs))
                + list(range((cp_segment_num - 1) * bs + batch_id, 0, -2 * bs))
            )

        reverse_split_len = []
        for rank_id in range(cp_size):
            for batch_id in range(bs):
                reverse_split_len.append(per_seq_block_sizes[batch_id][rank_id])
            for batch_id in range(bs):
                reverse_split_len.append(
                    per_seq_block_sizes[batch_id][cp_segment_num - 1 - rank_id]
                )

        kv_len_prev_list = []
        kv_len_next_list = []
        actual_seq_q_prev_list = []
        actual_seq_q_next_list = []
        for batch_id, block_sizes in enumerate(per_seq_block_sizes):
            kv_len_prev_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: rank + 1])
            )
            kv_len_next_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_segment_num - rank])
            )
            actual_seq_q_prev_list.append(block_sizes[rank])
            actual_seq_q_next_list.append(block_sizes[cp_segment_num - rank - 1])

        return {
            "bs": bs,
            "total_seq_lens": sum(extend_seq_lens),
            "split_list": split_list,
            "zigzag_index": zigzag_index,
            "per_rank_actual_token": per_rank_actual_token,
            "max_rank_len": max_rank_len,
            "reverse_split_len": reverse_split_len,
            "cp_reverse_index": cp_reverse_index,
            "kv_len_prev_list": kv_len_prev_list,
            "kv_len_next_list": kv_len_next_list,
            "actual_seq_q_prev_list": actual_seq_q_prev_list,
            "actual_seq_q_next_list": actual_seq_q_next_list,
        }

    def _assert_metadata_matches(self, metadata, expected):
        self.assertEqual(metadata.bs, expected["bs"])
        self.assertEqual(metadata.total_seq_lens, expected["total_seq_lens"])
        self.assertEqual(metadata.split_list, expected["split_list"])
        self.assertEqual(metadata.zigzag_index, expected["zigzag_index"])
        self.assertEqual(
            metadata.per_rank_actual_token, expected["per_rank_actual_token"]
        )
        self.assertEqual(metadata.max_rank_len, expected["max_rank_len"])
        self.assertEqual(metadata.reverse_split_len, expected["reverse_split_len"])
        self.assertEqual(metadata.cp_reverse_index, expected["cp_reverse_index"])
        self.assertEqual(metadata.kv_len_prev_list, expected["kv_len_prev_list"])
        self.assertEqual(metadata.kv_len_next_list, expected["kv_len_next_list"])
        self.assertEqual(
            metadata.actual_seq_q_prev_list, expected["actual_seq_q_prev_list"]
        )
        self.assertEqual(
            metadata.actual_seq_q_next_list, expected["actual_seq_q_next_list"]
        )
        self.assertEqual(
            metadata.cu_seqlens_q_prev_tensor.cpu().tolist(),
            [0]
            + list(
                torch.tensor(expected["actual_seq_q_prev_list"]).cumsum(dim=0).tolist()
            ),
        )
        self.assertEqual(
            metadata.cu_seqlens_q_next_tensor.cpu().tolist(),
            [0]
            + list(
                torch.tensor(expected["actual_seq_q_next_list"]).cumsum(dim=0).tolist()
            ),
        )

    def _padded_rank_tensors(self, x, *, cp_size, seq_lens, extend_seq_lens):
        per_rank = []
        metas = []
        for rank in range(cp_size):
            metadata = self._metadata_for_rank(
                rank,
                cp_size=cp_size,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
            )
            metas.append(metadata)
            fb = self._forward_batch(metadata, extend_seq_lens)
            local = ZigzagCPStrategy(cp_size=cp_size).shard_hidden_states(x, fb)
            pad = metadata.max_rank_len[0] - local.shape[0]
            if pad:
                local = torch.nn.functional.pad(
                    local,
                    [0, 0] * (local.ndim - 1) + [0, pad],
                )
            per_rank.append(local)
        return metas, per_rank

    def test_zigzag_metadata_for_batched_sequences(self):
        cases = [
            (4, [11, 13], [9, 10]),
            (2, [8], [8]),
            (4, [100000, 200000, 80], [100000, 200000, 64]),
            (4, [100005, 200011, 25], [100000, 200000, 16]),
        ]

        for cp_size, seq_lens, extend_seq_lens in cases:
            for rank in range(cp_size):
                with self.subTest(
                    cp_size=cp_size,
                    rank=rank,
                    seq_lens=seq_lens,
                    extend_seq_lens=extend_seq_lens,
                ):
                    metadata = self._metadata_for_rank(
                        rank,
                        cp_size=cp_size,
                        seq_lens=seq_lens,
                        extend_seq_lens=extend_seq_lens,
                    )
                    expected = self._expected_metadata(
                        rank=rank,
                        cp_size=cp_size,
                        seq_lens=seq_lens,
                        extend_seq_lens=extend_seq_lens,
                    )
                    self._assert_metadata_matches(metadata, expected)

    def test_zigzag_shards_hidden_states_and_position_ids(self):
        cp_size = 4
        seq_lens = [130, 142]
        extend_seq_lens = [128, 136]
        x = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        positions = torch.arange(sum(extend_seq_lens))

        for rank in range(cp_size):
            metadata = self._metadata_for_rank(
                rank,
                cp_size=cp_size,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
            )
            fb = self._forward_batch(metadata, extend_seq_lens)
            strategy = ZigzagCPStrategy(cp_size=cp_size)
            chunks = torch.split(x, metadata.split_list, dim=0)
            position_chunks = torch.split(positions, metadata.split_list, dim=-1)
            expected_x = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=0)
            expected_positions = torch.cat(
                [position_chunks[i] for i in metadata.zigzag_index], dim=-1
            )

            local_x = strategy.shard_hidden_states(x, fb)
            local_positions = strategy.shard_position_ids(positions, fb)
            with patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
            ):
                helper_x, helper_positions = cp_split_before_forward(
                    x,
                    positions,
                    fb,
                )

            self.assertTrue(torch.equal(local_x, expected_x))
            self.assertTrue(torch.equal(local_positions, expected_positions))
            self.assertTrue(torch.equal(helper_x, expected_x))
            self.assertTrue(torch.equal(helper_positions, expected_positions))

    def test_zigzag_shards_uneven_rank_local_inputs(self):
        cp_size = 4
        seq_lens = [515]
        extend_seq_lens = [515]
        rank = 3
        metadata = self._metadata_for_rank(
            rank,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )
        fb = self._forward_batch(metadata, extend_seq_lens)
        x = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        positions = torch.arange(sum(extend_seq_lens))
        chunks = torch.split(x, metadata.split_list, dim=0)
        position_chunks = torch.split(positions, metadata.split_list, dim=-1)
        expected_x = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=0)
        expected_positions = torch.cat(
            [position_chunks[i] for i in metadata.zigzag_index], dim=-1
        )

        with get_parallel().override(attn_cp_rank=rank):
            local_x = ZigzagCPStrategy(cp_size=cp_size).shard_hidden_states(x, fb)
            local_positions = ZigzagCPStrategy(cp_size=cp_size).shard_position_ids(
                positions, fb
            )

        self.assertEqual(local_x.shape[0], metadata.per_rank_actual_token[rank])
        self.assertEqual(local_positions.shape[0], metadata.per_rank_actual_token[rank])
        self.assertTrue(torch.equal(local_x, expected_x))
        self.assertTrue(torch.equal(local_positions, expected_positions))

    def test_zigzag_gathers_hidden_states_to_original_order(self):
        cp_size = 4
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
        x = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        metas, padded_rank_tensors = self._padded_rank_tensors(
            x,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )

        for rank in range(cp_size):
            local_x = padded_rank_tensors[rank][
                : metas[rank].per_rank_actual_token[rank]
            ]
            fb = self._forward_batch(metas[rank], extend_seq_lens)

            with (
                get_parallel().override(
                    attn_cp_group=_FakeCPGroup(padded_rank_tensors)
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
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
        kv = torch.arange(sum(extend_seq_lens) * 2 * 3).view(sum(extend_seq_lens), 2, 3)
        metas, padded_rank_tensors = self._padded_rank_tensors(
            kv,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )

        for rank in range(cp_size):
            local_kv = padded_rank_tensors[rank][
                : metas[rank].per_rank_actual_token[rank]
            ]
            fb = self._forward_batch(metas[rank], extend_seq_lens)

            with (
                get_parallel().override(
                    attn_cp_group=_FakeCPGroup(padded_rank_tensors)
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

    def test_zigzag_gather_barriers_before_uneven_cp_collectives(self):
        cp_size = 4
        rank = 3
        seq_lens = [515]
        extend_seq_lens = [515]
        kv = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        metas, padded_rank_tensors = self._padded_rank_tensors(
            kv,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )
        local_kv = padded_rank_tensors[rank][: metas[rank].per_rank_actual_token[rank]]
        fb = self._forward_batch(metas[rank], extend_seq_lens)
        fake_group = _FakeCPGroup(padded_rank_tensors)

        with (
            get_parallel().override(attn_cp_group=fake_group),
            patch(
                "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
                return_value=torch.no_grad(),
            ),
        ):
            gathered = ZigzagCPStrategy(cp_size=cp_size).gather_kv_cache(
                local_kv, fb, stream=None
            )

        self.assertTrue(torch.equal(gathered, kv))
        self.assertEqual(fake_group.barrier_calls, 1)

    def test_zigzag_gather_skips_barrier_for_even_cp_collectives(self):
        cp_size = 4
        rank = 0
        seq_lens = [512]
        extend_seq_lens = [512]
        kv = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        metas, padded_rank_tensors = self._padded_rank_tensors(
            kv,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )
        local_kv = padded_rank_tensors[rank][: metas[rank].per_rank_actual_token[rank]]
        fb = self._forward_batch(metas[rank], extend_seq_lens)
        fake_group = _FakeCPGroup(padded_rank_tensors)

        with (
            get_parallel().override(attn_cp_group=fake_group),
            patch(
                "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
                return_value=torch.no_grad(),
            ),
        ):
            gathered = ZigzagCPStrategy(cp_size=cp_size).gather_kv_cache(
                local_kv, fb, stream=None
            )

        self.assertTrue(torch.equal(gathered, kv))
        self.assertEqual(fake_group.barrier_calls, 0)

    def test_zigzag_attention_dispatch_runs_prev_then_next(self):
        cp_size = 2
        seq_lens = [8]
        extend_seq_lens = [8]
        metadata = self._metadata_for_rank(
            0,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
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

        out = ZigzagCPStrategy(cp_size=cp_size).run_attention(
            q, fb, device=torch.device("cpu"), attn_fn=attn_fn
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][0], q[:2]))
        self.assertTrue(torch.equal(calls[1][0], q[2:]))
        self.assertTrue(torch.equal(out, q + 100))

    def test_zigzag_attention_handles_uneven_rank_lengths(self):
        cp_size = 4
        rank = 3
        seq_lens = [515]
        extend_seq_lens = [515]
        metadata = self._metadata_for_rank(
            rank,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )
        fb = SimpleNamespace(attn_cp_metadata=metadata)
        q = torch.arange(metadata.per_rank_actual_token[rank] * 2).view(
            metadata.per_rank_actual_token[rank], 2
        )
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

        with get_parallel().override(attn_cp_rank=rank):
            out = ZigzagCPStrategy(cp_size=cp_size).run_attention(
                q, fb, device=torch.device("cpu"), attn_fn=attn_fn
            )

        actual_len = metadata.per_rank_actual_token[rank]
        self.assertEqual(len(calls), 2)
        self.assertEqual(out.shape[0], actual_len)
        self.assertTrue(torch.equal(out, q + 100))
        self.assertEqual(calls[0][0].shape[0], metadata.total_q_prev_tokens)
        self.assertEqual(calls[1][0].shape[0], metadata.total_q_next_tokens)

    def test_zigzag_materialize_full_kv_preserves_swa_loc(self):
        strategy = ZigzagCPStrategy(cp_size=2)
        cache_loc = torch.tensor([4, 5])
        swa_loc = torch.tensor([1, 2])
        k = torch.arange(4).view(2, 2)
        v = k + 10
        layer = SimpleNamespace(
            is_cross_attention=False,
            k_scale=0.5,
            v_scale=0.25,
        )
        forward_batch = SimpleNamespace(out_cache_loc=cache_loc)
        calls = []

        class Pool:
            def set_kv_buffer(
                self,
                layer_arg,
                loc_info,
                key_cache,
                value_cache,
                k_scale,
                v_scale,
            ):
                calls.append(
                    (layer_arg, loc_info, key_cache, value_cache, k_scale, v_scale)
                )

        with (
            patch(
                "sglang.srt.layers.cp.zigzag.get_token_to_kv_pool",
                return_value=Pool(),
            ),
            patch.object(strategy, "gather_kv_cache", side_effect=lambda x, *_: x),
            patch(
                "sglang.srt.layers.cp.zigzag.torch.cuda.current_stream",
                return_value=None,
            ),
        ):
            strategy.materialize_full_kv(
                forward_batch,
                layer,
                k,
                v,
                swa_loc=swa_loc,
            )

        self.assertEqual(len(calls), 1)
        layer_arg, loc_info, key_cache, value_cache, k_scale, v_scale = calls[0]
        self.assertIs(layer_arg, layer)
        self.assertIs(loc_info.loc, cache_loc)
        self.assertIs(loc_info.swa_loc, swa_loc)
        self.assertTrue(torch.equal(key_cache, k))
        self.assertTrue(torch.equal(value_cache, v))
        self.assertEqual(k_scale, layer.k_scale)
        self.assertEqual(v_scale, layer.v_scale)


if __name__ == "__main__":
    unittest.main()
