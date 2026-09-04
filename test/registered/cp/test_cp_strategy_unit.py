import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.cp.base import (
    ContextParallelStrategyKind,
    get_cp_strategy,
    get_cp_strategy_kind,
    init_cp_strategy,
    is_cp_enabled,
    is_interleave,
    is_zigzag,
)
from sglang.srt.layers.cp.bcg import PrefillCPBCGInput
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.padding import (
    get_cp_padding_align_size,
    pad_local_rows,
    pad_logical_token_to_physical,
)
from sglang.srt.layers.cp.utils import (
    cp_split_before_forward,
    enable_cp_v2,
    is_cp_v2_active,
    prepare_cp_forward,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
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

    def all_gather_into_tensor(self, output, input_tensor):
        del input_tensor
        torch.cat(self.all_rank_tensors, dim=0, out=output)


class TestCPStrategyUnit(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

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

    def test_init_cp_strategy_binds_zigzag_strategy(self):
        init_cp_strategy(
            enable_prefill_cp=True,
            cp_size=4,
            cp_strategy="zigzag",
        )

        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_zigzag())
        self.assertFalse(is_interleave())
        self.assertEqual(get_cp_strategy_kind(), ContextParallelStrategyKind.ZIGZAG)

    def test_get_cp_strategy_is_initialized_under_cp_v2(self):
        init_cp_strategy(
            enable_prefill_cp=True,
            cp_size=4,
            cp_strategy="interleave",
        )

        self.assertIsNotNone(get_cp_strategy())
        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_interleave())

    def test_hip_dsa_cp_uses_protected_legacy_runtime_flag(self):
        parallel = SimpleNamespace(
            enable_dsa_prefill_context_parallel=False,
            attn_cp_size=2,
        )
        model_config = SimpleNamespace(hf_config=SimpleNamespace())

        with (
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.process_model_config",
                return_value=model_config,
            ),
            patch("sglang.srt.layers.attention.dsa.utils.is_hip", return_value=True),
            patch(
                "sglang.srt.configs.model_config.is_deepseek_dsa",
                return_value=True,
            ),
        ):
            self.assertFalse(is_dsa_enable_prefill_cp())

    @patch("sglang.srt.utils.is_npu", return_value=False)
    @patch("sglang.srt.utils.is_hip", return_value=True)
    def test_hip_keeps_strategy_cp_disabled(self, _mock_is_hip, _mock_is_npu):
        self.assertFalse(enable_cp_v2())

    @patch("sglang.srt.utils.is_npu", return_value=True)
    @patch("sglang.srt.utils.is_hip", return_value=False)
    def test_npu_keeps_strategy_cp_disabled(self, _mock_is_hip, _mock_is_npu):
        self.assertFalse(enable_cp_v2())


class TestPrefillCPBCGReplay(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [2048, 2304]
        runner.max_num_tokens = 2304
        runner.enable_cp_v2_bcg_capture = True
        return runner

    def _make_forward_batch(self):
        return SimpleNamespace(
            batch_size=3,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(2048)),
            seq_lens_cpu=[1534, 161, 353],
            extend_seq_lens_cpu=[1534, 161, 353],
            extend_prefix_lens_cpu=[0, 0, 0],
        )

    def _enable_zigzag(self):
        init_cp_strategy(
            enable_prefill_cp=True,
            cp_size=4,
            cp_strategy="zigzag",
        )

    def test_local_capacity_overflow_uses_next_capture_bucket(self):
        runner = self._make_runner()
        runner.capture_num_tokens.append(2560)
        runner.max_num_tokens = 2560
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 576, 2560: 640},
        )
        forward_batch = self._make_forward_batch()
        self._enable_zigzag()

        with (
            patch(
                "sglang.srt.layers.cp.bcg.get_cp_padding_align_size",
                return_value=8,
            ),
        ):
            selected_buckets = []
            for cp_rank in range(4):
                with get_parallel().override(attn_cp_rank=cp_rank, attn_cp_size=4):
                    selected_buckets.append(
                        runner.prefill_cp_bcg_input.select_replay_bucket_for_batch(
                            num_tokens=2048,
                            extend_seq_lens=[1534, 161, 353],
                            capture_num_tokens=runner.capture_num_tokens,
                            max_padding_factor=2,
                        )
                    )

            self.assertEqual(selected_buckets, [2304, 2304, 2304, 2304])
            with get_parallel().override(attn_cp_rank=0, attn_cp_size=4):
                self.assertTrue(runner.can_run_graph(forward_batch))

    def test_bucket_search_preserves_two_x_padding_limit(self):
        runner = self._make_runner()
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 576},
        )

        self.assertIsNone(
            runner.prefill_cp_bcg_input.select_replay_bucket(
                num_tokens=1024,
                required_local_tokens=520,
                capture_num_tokens=runner.capture_num_tokens,
                max_padding_factor=2,
            )
        )

    def test_bucket_search_falls_back_when_no_capture_has_capacity(self):
        runner = self._make_runner()
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 516},
        )
        forward_batch = self._make_forward_batch()
        self._enable_zigzag()

        with (
            get_parallel().override(attn_cp_rank=0, attn_cp_size=4),
            patch(
                "sglang.srt.layers.cp.bcg.get_cp_padding_align_size",
                return_value=8,
            ),
        ):
            self.assertFalse(runner.can_run_graph(forward_batch))

    def test_load_batch_uses_selected_larger_bucket(self):
        class StopAfterRecordingFill(Exception):
            pass

        class RecordingRegistry:
            padded_num_tokens = None

            def fill_from(self, _source, **kwargs):
                self.padded_num_tokens = kwargs["padded_num_tokens"]
                raise StopAfterRecordingFill

        runner = self._make_runner()
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 576},
        )
        runner.buffer_registry = RecordingRegistry()
        forward_batch = self._make_forward_batch()
        self._enable_zigzag()

        with (
            get_parallel().override(attn_cp_rank=0, attn_cp_size=4),
            patch(
                "sglang.srt.layers.cp.bcg.get_cp_padding_align_size",
                return_value=8,
            ),
            self.assertRaises(StopAfterRecordingFill),
        ):
            runner.load_batch(forward_batch)

        self.assertEqual(runner.buffer_registry.padded_num_tokens, 2304)


class TestCPZigzagStrategy(CustomTestCase):
    def setUp(self):
        init_cp_strategy(
            enable_prefill_cp=True,
            cp_size=4,
            cp_strategy="zigzag",
        )

    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

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
            input_ids=torch.arange(8),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[8],
        )
        inactive_batch = SimpleNamespace(
            input_ids=torch.arange(7),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[7],
        )

        with patch.dict("os.environ", {"SGLANG_ENABLE_CP_V2": "0"}):
            self.assertTrue(enable_cp_v2())
            self.assertTrue(is_cp_v2_active(active_batch))
            self.assertFalse(is_cp_v2_active(inactive_batch))

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
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
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
            helper_x, helper_positions = cp_split_before_forward(
                x,
                positions,
                fb,
            )

            self.assertTrue(torch.equal(local_x, expected_x))
            self.assertTrue(torch.equal(local_positions, expected_positions))
            self.assertTrue(torch.equal(helper_x, expected_x))
            self.assertTrue(torch.equal(helper_positions, expected_positions))

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

    def test_zigzag_padding_aligns_local_tensors(self):
        cp_size = 2
        metadata = SimpleNamespace(
            per_rank_actual_token=[7, 6],
            per_rank_logical_token=None,
            max_rank_len=[7, 7],
        )

        with (
            get_parallel().override(attn_cp_size=cp_size),
            patch(
                "sglang.srt.layers.utils.cp_utils.is_prefill_cp_in_seq_split",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_in_seq_split",
                return_value=False,
            ),
        ):
            align_size = get_cp_padding_align_size()
            pad_logical_token_to_physical(metadata)

        self.assertEqual(align_size, 2 * cp_size)
        self.assertEqual(metadata.per_rank_logical_token, [7, 6])
        self.assertEqual(metadata.per_rank_actual_token, [8, 8])
        self.assertEqual(metadata.max_rank_len, [8, 8])
        for logical_len in metadata.per_rank_logical_token:
            local_hidden = torch.arange(logical_len * 2).view(logical_len, 2)
            padded_hidden = pad_local_rows(local_hidden, metadata, dim=0)
            physical_len = metadata.per_rank_actual_token[0]
            self.assertEqual(padded_hidden.shape, (physical_len, 2))
            self.assertTrue(torch.equal(padded_hidden[:logical_len], local_hidden))
            self.assertTrue(
                torch.equal(
                    padded_hidden[logical_len:],
                    local_hidden.new_zeros(physical_len - logical_len, 2),
                )
            )

    def test_zigzag_materialize_full_kv_gathers_once_and_preserves_swa_location(self):
        key = torch.arange(6).view(3, 2)
        value = torch.arange(9).view(3, 3) + 10
        local_kv = torch.cat([key, value], dim=-1)
        cache_loc = torch.arange(3)
        swa_loc = torch.arange(5) + 16
        forward_batch = SimpleNamespace(
            out_cache_loc=cache_loc,
            encoder_out_cache_loc=torch.arange(3) + 32,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            k_scale="key-scale",
            v_scale="value-scale",
        )
        writes = []
        pool = SimpleNamespace(set_kv_buffer=lambda *args: writes.append(args))
        strategy = ZigzagCPStrategy(cp_size=2)

        with (
            patch.object(strategy, "gather_kv_cache", return_value=local_kv) as gather,
            patch(
                "sglang.srt.layers.cp.zigzag.get_token_to_kv_pool",
                return_value=pool,
            ),
        ):
            strategy.materialize_full_kv(forward_batch, layer, key, value, swa_loc)

        gather.assert_called_once()
        gathered_kv, gathered_forward_batch = gather.call_args.args
        self.assertTrue(gathered_kv.is_contiguous())
        self.assertTrue(torch.equal(gathered_kv, local_kv))
        self.assertIs(gathered_forward_batch, forward_batch)
        self.assertEqual(len(writes), 1)
        written_layer, write_loc, written_key, written_value, k_scale, v_scale = writes[
            0
        ]
        self.assertIs(written_layer, layer)
        self.assertIsInstance(write_loc, KVWriteLoc)
        self.assertIs(write_loc.loc, cache_loc)
        self.assertTrue(torch.equal(write_loc.swa_loc, swa_loc[:3]))
        self.assertTrue(written_key.is_contiguous())
        self.assertTrue(written_value.is_contiguous())
        self.assertTrue(torch.equal(written_key, key))
        self.assertTrue(torch.equal(written_value, value))
        self.assertEqual((k_scale, v_scale), ("key-scale", "value-scale"))

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

    def test_zigzag_combined_attention_matches_two_half_reference(self):
        def reference_attention(
            q,
            cu_seqlens_q,
            cache_seqlens,
            cu_seqlens_kv,
            *,
            sequence_offset,
        ):
            outputs = []
            for seq_id in range(cache_seqlens.numel()):
                q_start = int(cu_seqlens_q[seq_id])
                q_end = int(cu_seqlens_q[seq_id + 1])
                q_seq = q[q_start:q_end]
                q_len = q_end - q_start
                kv_len = int(cache_seqlens[seq_id])
                self.assertEqual(
                    int(cu_seqlens_kv[seq_id + 1] - cu_seqlens_kv[seq_id]),
                    kv_len,
                )

                absolute_seq_id = sequence_offset + seq_id + 1
                positions = torch.arange(kv_len, dtype=q.dtype)
                k_seq = torch.stack(
                    (
                        positions / (kv_len + 1),
                        torch.sin(positions + absolute_seq_id),
                        torch.full_like(positions, absolute_seq_id / 10),
                    ),
                    dim=1,
                )
                v_seq = torch.stack(
                    (
                        torch.cos(positions + absolute_seq_id),
                        positions / (absolute_seq_id + 1),
                        torch.full_like(positions, absolute_seq_id),
                    ),
                    dim=1,
                )
                q_positions = kv_len - q_len + torch.arange(q_len)
                allowed = torch.arange(kv_len)[None, :] <= q_positions[:, None]
                scores = q_seq @ k_seq.T / q.shape[-1] ** 0.5
                outputs.append(
                    torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
                    @ v_seq
                )
            return torch.cat(outputs, dim=0)

        cp_size = 4
        seq_lens = [19, 27]
        extend_seq_lens = [11, 13]
        for rank in range(cp_size):
            with self.subTest(rank=rank):
                metadata = self._metadata_for_rank(
                    rank,
                    cp_size=cp_size,
                    seq_lens=seq_lens,
                    extend_seq_lens=extend_seq_lens,
                )
                logical_tokens = (
                    metadata.total_q_prev_tokens + metadata.total_q_next_tokens
                )
                q = torch.linspace(
                    -0.75,
                    0.75,
                    steps=logical_tokens * 3,
                    dtype=torch.float32,
                ).view(logical_tokens, 3)
                q_prev = q[: metadata.total_q_prev_tokens]
                q_next = q[metadata.total_q_prev_tokens :]

                two_half_out = torch.cat(
                    (
                        reference_attention(
                            q_prev,
                            metadata.cu_seqlens_q_prev_tensor,
                            metadata.kv_len_prev_tensor,
                            metadata.cu_seqlens_kv_prev_tensor,
                            sequence_offset=0,
                        ),
                        reference_attention(
                            q_next,
                            metadata.cu_seqlens_q_next_tensor,
                            metadata.kv_len_next_tensor,
                            metadata.cu_seqlens_kv_next_tensor,
                            sequence_offset=metadata.bs,
                        ),
                    ),
                    dim=0,
                )
                combined_out = reference_attention(
                    q,
                    metadata.cu_seqlens_q_combined_tensor,
                    metadata.kv_len_combined_tensor,
                    metadata.cu_seqlens_kv_combined_tensor,
                    sequence_offset=0,
                )

                torch.testing.assert_close(
                    combined_out, two_half_out, atol=1e-5, rtol=1e-5
                )


class TestCPInterleaveStrategy(CustomTestCase):
    def setUp(self):
        init_cp_strategy(
            enable_prefill_cp=True,
            cp_size=4,
            cp_strategy="interleave",
        )

    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def _metadata_for_rank(self, rank, *, cp_size, seq_lens, extend_seq_lens):
        strategy = InterleaveCPStrategy(cp_size=cp_size)
        with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
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

    def _rank_tensors(self, x, *, cp_size, seq_lens, extend_seq_lens):
        per_rank = []
        metas = []
        with self._patch_legacy_round_robin_mode():
            for rank in range(cp_size):
                metadata = self._metadata_for_rank(
                    rank,
                    cp_size=cp_size,
                    seq_lens=seq_lens,
                    extend_seq_lens=extend_seq_lens,
                )
                metas.append(metadata)
                fb = self._forward_batch(metadata, extend_seq_lens)
                strategy = InterleaveCPStrategy(cp_size=cp_size)
                with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
                    per_rank.append(strategy.shard_hidden_states(x, fb))
        return metas, per_rank

    @contextmanager
    def _patch_legacy_round_robin_mode(self):
        with patch(
            "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
            return_value=True,
        ):
            yield

    @contextmanager
    def _patch_interleave_all_gather(self, rank_tensors):
        def all_gather(output, input_tensor):
            del input_tensor
            torch.cat(rank_tensors, dim=0, out=output)

        patchers = (
            patch(
                "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
                side_effect=all_gather,
            ),
            patch(
                "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.cp.interleave.use_symmetric_memory",
                return_value=torch.no_grad(),
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
                return_value=True,
            ),
        )
        with (
            patchers[0],
            patchers[1],
            patchers[2],
            patchers[3],
            get_parallel().override(attn_cp_group=object()),
        ):
            yield

    def test_interleave_metadata_supports_shared_padding(self):
        metadata = InterleaveCPStrategy(cp_size=4).build_metadata(
            num_tokens=10,
            seqs_len=[10],
            extend_seqs_len=[10],
        )

        self.assertEqual(metadata.per_rank_actual_token, [3, 3, 2, 2])
        with patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ):
            pad_logical_token_to_physical(metadata)

        self.assertEqual(metadata.per_rank_logical_token, [3, 3, 2, 2])
        self.assertEqual(metadata.per_rank_actual_token, [4, 4, 4, 4])
        self.assertEqual(metadata.max_rank_len, [4, 4, 4, 4])

    def test_prepare_cp_forward_sizes_gather_buffer_for_all_cp_ranks(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(10),
            positions=torch.arange(10),
            forward_mode=_ExtendMode(),
            seq_lens_cpu=[10],
            extend_seq_lens_cpu=[10],
            attn_cp_metadata=None,
            global_num_tokens_cpu=[10],
            out_cache_loc=None,
        )

        with (
            get_parallel().override(attn_cp_rank=2, attn_cp_size=4),
            patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.dp_attention.set_local_dp_buffer_len"
            ) as set_buffer_len,
        ):
            prepare_cp_forward(forward_batch)

        self.assertEqual(
            forward_batch.attn_cp_metadata.per_rank_actual_token,
            [4, 4, 4, 4],
        )
        set_buffer_len.assert_called_once_with(16)

    def test_interleave_shards_hidden_states_and_position_ids(self):
        cp_size = 4
        seq_lens = [8]
        extend_seq_lens = [8]
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
            strategy = InterleaveCPStrategy(cp_size=cp_size)
            expected_x = x[rank::cp_size]
            expected_positions = positions[rank::cp_size]

            with (
                get_parallel().override(
                    attn_cp_rank=rank,
                    attn_cp_size=cp_size,
                ),
                self._patch_legacy_round_robin_mode(),
            ):
                local_x = strategy.shard_hidden_states(x, fb)
                local_positions = strategy.shard_position_ids(positions, fb)

                helper_x, helper_positions = cp_split_before_forward(
                    x,
                    positions,
                    fb,
                )

            self.assertTrue(torch.equal(local_x, expected_x))
            self.assertTrue(torch.equal(local_positions, expected_positions))
            self.assertTrue(torch.equal(helper_x, expected_x))
            self.assertTrue(torch.equal(helper_positions, expected_positions))

    def test_interleave_padding_preserves_shard_and_gather(self):
        cp_size = 4
        total_tokens = 10
        x = torch.arange(total_tokens * 2).view(total_tokens, 2)
        rank_tensors = []
        metas = []

        for rank in range(cp_size):
            metadata = self._metadata_for_rank(
                rank,
                cp_size=cp_size,
                seq_lens=[total_tokens],
                extend_seq_lens=[total_tokens],
            )
            with patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=4,
            ):
                pad_logical_token_to_physical(metadata)
            fb = self._forward_batch(metadata, [total_tokens])
            with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
                local_x = InterleaveCPStrategy(cp_size=cp_size).shard_hidden_states(
                    x, fb
                )

            logical_len = metadata.per_rank_logical_token[rank]
            self.assertEqual(local_x.shape[0], 4)
            self.assertTrue(torch.equal(local_x[:logical_len], x[rank::cp_size]))
            self.assertTrue(torch.count_nonzero(local_x[logical_len:]) == 0)
            metas.append(metadata)
            rank_tensors.append(local_x)

        for rank in range(cp_size):
            fb = self._forward_batch(metas[rank], [total_tokens])
            with (
                get_parallel().override(
                    attn_cp_rank=rank,
                    attn_cp_size=cp_size,
                ),
                self._patch_interleave_all_gather(rank_tensors),
            ):
                gathered = InterleaveCPStrategy(cp_size=cp_size).gather_hidden_states(
                    rank_tensors[rank], fb, stream=None
                )

            self.assertTrue(torch.equal(gathered, x))

    def test_interleave_gathers_hidden_states_to_original_order(self):
        cp_size = 4
        seq_lens = [10]
        extend_seq_lens = [10]
        x = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        metas, rank_tensors = self._rank_tensors(
            x,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )
        max_rank_len = max(t.shape[0] for t in rank_tensors)
        padded_rank_tensors = []
        for tensor in rank_tensors:
            if tensor.shape[0] < max_rank_len:
                padded = tensor.new_zeros((max_rank_len, *tensor.shape[1:]))
                padded[: tensor.shape[0]] = tensor
                padded_rank_tensors.append(padded)
            else:
                padded_rank_tensors.append(tensor)

        for rank in range(cp_size):
            fb = self._forward_batch(metas[rank], extend_seq_lens)
            with (
                get_parallel().override(
                    attn_cp_rank=rank,
                    attn_cp_size=cp_size,
                ),
                self._patch_interleave_all_gather(padded_rank_tensors),
            ):
                gathered = InterleaveCPStrategy(cp_size=cp_size).gather_hidden_states(
                    rank_tensors[rank], fb, stream=None
                )

            self.assertTrue(torch.equal(gathered, x))

    def test_interleave_gathers_kv_cache_to_original_order(self):
        cp_size = 4
        seq_lens = [8]
        extend_seq_lens = [8]
        kv = torch.arange(sum(extend_seq_lens) * 2 * 3).view(sum(extend_seq_lens), 2, 3)
        metas, rank_tensors = self._rank_tensors(
            kv,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )

        for rank in range(cp_size):
            fb = self._forward_batch(metas[rank], extend_seq_lens)
            with (
                get_parallel().override(
                    attn_cp_rank=rank,
                    attn_cp_size=cp_size,
                ),
                self._patch_interleave_all_gather(rank_tensors),
            ):
                gathered = InterleaveCPStrategy(cp_size=cp_size).gather_kv_cache(
                    rank_tensors[rank], fb, stream=None
                )

            self.assertTrue(torch.equal(gathered, kv))

    def test_interleave_materializes_full_mla_kv(self):
        strategy = InterleaveCPStrategy(cp_size=2)
        forward_batch = SimpleNamespace()
        layer = object()
        k_nope = torch.arange(6).view(2, 1, 3)
        k_rope = torch.arange(4).view(2, 1, 2) + 10
        full_latent = torch.arange(20).view(4, 5)

        with (
            patch.object(
                strategy,
                "gather_kv_cache",
                return_value=full_latent,
            ) as gather,
            patch(
                "sglang.srt.layers.cp.interleave.torch.cuda.current_stream",
                return_value=None,
            ),
        ):
            full_k_nope, full_k_rope = strategy.materialize_full_mla_kv(
                forward_batch,
                layer,
                k_nope,
                k_rope,
            )

        gather.assert_called_once()
        packed_kv, gathered_forward_batch, stream = gather.call_args.args
        self.assertTrue(
            torch.equal(packed_kv, torch.cat([k_nope, k_rope], dim=-1).squeeze(1))
        )
        self.assertIs(gathered_forward_batch, forward_batch)
        self.assertIsNone(stream)
        self.assertTrue(torch.equal(full_k_nope, full_latent[:, :3].unsqueeze(1)))
        self.assertTrue(torch.equal(full_k_rope, full_latent[:, 3:].unsqueeze(1)))


if __name__ == "__main__":
    unittest.main()
