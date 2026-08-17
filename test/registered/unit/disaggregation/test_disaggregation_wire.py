import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.staging_handler import (
    handle_staging_req,
)
from sglang.srt.disaggregation.common.utils import (
    build_dcp_staging_transfer_blocks,
    dcp_staging_required_bytes,
    group_concurrent_contiguous,
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.srt.disaggregation.utils import (
    MetadataBuffers,
    get_dsv4_c128_state_indices,
    is_mla_or_hybrid_mla_backend,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import should_use_dsa_fused_topk
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_mooncake_registration_staging_fields(self):
        msg = [
            b"room",
            b"127.0.0.1",
            b"1234",
            b"session",
            struct.pack("Q", 0x1000),
            struct.pack("Q", 0x2000),
            b"",
            b"0",
            b"1",
            b"128",
            b"",
            b"",
            b"",
            b"",
            struct.pack("Q", 0x3000),
            b"4096",
            b"4",
            b"2",
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.staging_base_ptr, 0x3000)
        self.assertEqual(info.staging_total_size, 4096)
        self.assertEqual(info.dst_dcp_size, 4)
        self.assertEqual(info.dst_dcp_rank, 2)

    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_prebuilt_skips_unused_prompt_tensor(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            prefix_indices=[0, 1],
            extend_range=SimpleNamespace(length=3),
            origin_input_ids=[0, 1, 2, 3, 4],
            output_ids=[],
            retracted_stain=True,
            is_retracted=True,
            multimodal_inputs=None,
            get_fill_ids=Mock(side_effect=AssertionError("prompt should not be read")),
        )
        batch = SimpleNamespace(
            reqs=[req],
            device="cpu",
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(5, dtype=torch.int64).reshape(1, 5)
            ),
            return_logprob=False,
            model_config=SimpleNamespace(vocab_size=32),
        )

        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "SamplingBatchInfo.from_schedule_batch",
            return_value=Mock(),
        ):
            ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt(batch)

        self.assertIsNone(batch.input_ids)
        self.assertEqual(batch.extend_num_tokens, 3)
        self.assertTrue(torch.equal(batch.out_cache_loc, torch.tensor([2, 3, 4])))
        req.get_fill_ids.assert_not_called()

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestGroupConcurrentContiguous(unittest.TestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


class TestDCPStagingLayout(unittest.TestCase):
    def test_required_bytes_uses_layer_major_layout(self):
        self.assertEqual(dcp_staging_required_bytes(64, [576, 320]), 64 * 896)

    def test_contiguous_destination_page_becomes_one_block_per_layer(self):
        blocks = build_dcp_staging_transfer_blocks(
            staging_ptr=0x1000,
            dst_data_ptrs=[0x100000, 0x200000],
            dst_token_indices=np.arange(640, 704, dtype=np.int64),
            token_item_lens=[576, 320],
        )

        self.assertEqual(
            blocks,
            [
                (0x1000, 0x100000 + 640 * 576, 64 * 576),
                (0x1000 + 64 * 576, 0x200000 + 640 * 320, 64 * 320),
            ],
        )

    def test_destination_page_gap_splits_blocks_but_source_stays_packed(self):
        blocks = build_dcp_staging_transfer_blocks(
            staging_ptr=0x8000,
            dst_data_ptrs=[0x300000],
            dst_token_indices=np.array([10, 11, 12, 64, 65], dtype=np.int64),
            token_item_lens=[576],
        )

        self.assertEqual(
            blocks,
            [
                (0x8000, 0x300000 + 10 * 576, 3 * 576),
                (0x8000 + 3 * 576, 0x300000 + 64 * 576, 2 * 576),
            ],
        )

    def test_invalid_layer_geometry_raises(self):
        with self.assertRaises(ValueError):
            build_dcp_staging_transfer_blocks(
                staging_ptr=0,
                dst_data_ptrs=[1, 2],
                dst_token_indices=np.array([0], dtype=np.int64),
                token_item_lens=[576],
            )


class TestMLACacheDetection(unittest.TestCase):
    def test_direct_mla_pool_is_supported(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        pool = object.__new__(MLATokenToKVPool)
        self.assertTrue(is_mla_or_hybrid_mla_backend(pool))

    def test_hybrid_pool_with_mla_full_cache_is_supported(self):
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MLATokenToKVPool,
        )

        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = object.__new__(MLATokenToKVPool)
        self.assertTrue(is_mla_or_hybrid_mla_backend(pool))

    def test_hybrid_pool_without_mla_full_cache_is_rejected(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = object()
        self.assertFalse(is_mla_or_hybrid_mla_backend(pool))


class TestMooncakeDCPStaging(unittest.TestCase):
    @staticmethod
    def _manager():
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=64,
            kv_layer_ids=[0, 1],
            kv_data_ptrs=[0x100000, 0x200000],
            gpu_id=0,
        )
        manager._transfer_data = Mock(return_value=0)
        return manager

    @staticmethod
    def _staging(num_tokens):
        staging_size = num_tokens * (576 + 320)
        return SimpleNamespace(
            get_size=lambda: staging_size,
            get_ptr=lambda: 0x9000,
            fits=lambda required: required <= staging_size,
        )

    @patch(
        "sglang.srt.disaggregation.common.staging_buffer."
        "gather_dcp_tokens_to_staging"
    )
    def test_strided_source_is_gathered_into_page_sized_transfers(self, gather):
        manager = self._manager()
        staging = self._staging(64)

        ret = manager.send_kvcache_dcp_staged(
            "peer",
            np.arange(100, 108, dtype=np.int32),
            [0x300000, 0x400000],
            np.array([200], dtype=np.int32),
            dcp_token_item_lens=[576, 320],
            dst_dcp_size=8,
            dst_dcp_rank=3,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=512,
            executor=Mock(),
            dst_layer_ids=[0, 1],
            staging_buffer=staging,
        )

        self.assertEqual(ret, 0)
        gather.assert_called_once()
        gathered_indices = gather.call_args.args[1]
        np.testing.assert_array_equal(
            gathered_indices[:4], np.array([6403, 6411, 6419, 6427])
        )
        manager._transfer_data.assert_called_once_with(
            "peer",
            [
                (0x9000, 0x300000 + 200 * 64 * 576, 64 * 576),
                (
                    0x9000 + 64 * 576,
                    0x400000 + 200 * 64 * 320,
                    64 * 320,
                ),
            ],
        )

    @patch(
        "sglang.srt.disaggregation.common.staging_buffer."
        "gather_dcp_tokens_to_staging"
    )
    def test_large_relayout_is_split_at_staging_capacity(self, gather):
        manager = self._manager()

        ret = manager.send_kvcache_dcp_staged(
            "peer",
            np.arange(100, 116, dtype=np.int32),
            [0x300000, 0x400000],
            np.array([200, 400], dtype=np.int32),
            dcp_token_item_lens=[576, 320],
            dst_dcp_size=8,
            dst_dcp_rank=3,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=1024,
            executor=Mock(),
            dst_layer_ids=[0, 1],
            staging_buffer=self._staging(64),
        )

        self.assertEqual(ret, 0)
        self.assertEqual(gather.call_count, 2)
        self.assertEqual(manager._transfer_data.call_count, 2)
        self.assertEqual(
            [len(call.args[1]) for call in manager._transfer_data.call_args_list],
            [2, 2],
        )

    @patch(
        "sglang.srt.disaggregation.common.staging_buffer."
        "gather_dcp_tokens_to_staging",
        side_effect=RuntimeError("Triton compilation failed"),
    )
    def test_gather_failure_disables_staging_and_falls_back(self, gather):
        manager = self._manager()
        manager.send_kvcache_dcp = Mock(return_value=0)
        args = (
            "peer",
            np.arange(100, 108, dtype=np.int32),
            [0x300000, 0x400000],
            np.array([200], dtype=np.int32),
        )
        kwargs = dict(
            dcp_token_item_lens=[576, 320],
            dst_dcp_size=8,
            dst_dcp_rank=3,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=512,
            executor=Mock(),
            dst_layer_ids=[0, 1],
            staging_buffer=self._staging(64),
        )

        self.assertEqual(manager.send_kvcache_dcp_staged(*args, **kwargs), 0)
        self.assertTrue(manager._dcp_staging_disabled)
        self.assertEqual(manager.send_kvcache_dcp_staged(*args, **kwargs), 0)

        gather.assert_called_once()
        self.assertEqual(manager.send_kvcache_dcp.call_count, 2)
        manager._transfer_data.assert_not_called()

    @patch(
        "sglang.srt.disaggregation.common.staging_buffer."
        "gather_dcp_tokens_to_staging"
    )
    def test_transfer_failure_does_not_disable_staging(self, gather):
        manager = self._manager()
        manager.send_kvcache_dcp = Mock(return_value=0)
        manager._transfer_data.side_effect = RuntimeError("RDMA transfer failed")

        with self.assertRaisesRegex(RuntimeError, "RDMA transfer failed"):
            manager.send_kvcache_dcp_staged(
                "peer",
                np.arange(100, 108, dtype=np.int32),
                [0x300000, 0x400000],
                np.array([200], dtype=np.int32),
                dcp_token_item_lens=[576, 320],
                dst_dcp_size=8,
                dst_dcp_rank=3,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=512,
                executor=Mock(),
                dst_layer_ids=[0, 1],
                staging_buffer=self._staging(64),
            )

        gather.assert_called_once()
        self.assertFalse(getattr(manager, "_dcp_staging_disabled", False))
        manager.send_kvcache_dcp.assert_not_called()


class TestMooncakePPStaging(unittest.TestCase):
    def test_staging_response_targets_requesting_pp_rank(self):
        sock = Mock()
        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            _connect_to_bootstrap_server=Mock(return_value=(sock, threading.Lock())),
        )
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )
        kv_args = SimpleNamespace(
            page_size=64,
            kv_item_lens=[4096, 4096],
            total_kv_head_num=4,
            engine_rank=0,
        )
        target = {"pp_rank": 3}

        handle_staging_req(
            [b"STAGING_REQ", b"7", b"0", b"1", b"peer", b"3"],
            allocator,
            kv_args,
            attn_tp_size=16,
            prefill_attn_tp_size=1,
            kv_buffer_tensors=None,
            room_receivers={7: receiver},
            room_bootstrap={7: [{"pp_rank": 2}, target]},
        )

        receiver._connect_to_bootstrap_server.assert_called_once_with(target)
        sock.send_multipart.assert_called_once()

    @patch(
        "sglang.srt.disaggregation.common.staging_buffer.gather_all_layers_to_staging"
    )
    def test_pp_stage_writes_its_global_layer_slots(self, gather):
        manager = object.__new__(MooncakeKVManager)
        tensor = SimpleNamespace(shape=(1, 1, 8), element_size=lambda: 2)
        manager.kv_buffer_tensors = {
            "k_buffers": [tensor],
            "v_buffers": [tensor],
            "page_size": 2,
        }
        manager.attn_tp_size = 1
        manager.pp_size = 16
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            gpu_id=0,
            total_kv_head_num=4,
            kv_head_num=4,
            kv_layer_ids=[7, 7],
        )
        manager._transfer_data = Mock(return_value=0)
        staging = SimpleNamespace(fits=lambda size: True, get_ptr=lambda: 0x9000)

        ret = manager.send_kvcache_staged(
            "peer",
            np.array([1, 2], dtype=np.int32),
            dst_staging_ptr=0x100000,
            dst_staging_size=1 << 20,
            dst_tp_rank=0,
            dst_attn_tp_size=16,
            dst_kv_item_len=128,
            dst_layer_ids=[3, 7, 11, 3, 7, 11],
            staging_buffer=staging,
        )

        self.assertEqual(ret, 0)
        gather.assert_called_once()
        manager._transfer_data.assert_called_once_with(
            "peer",
            [
                (0x9000, 0x100000 + 64, 64),
                (0x9000 + 64, 0x100000 + 4 * 64, 64),
            ],
        )


class TestEagleDsaSeedTransfer(unittest.TestCase):
    @staticmethod
    def _make_req(seed, metadata_buffer_index=0):
        return SimpleNamespace(
            metadata_buffer_index=metadata_buffer_index,
            output_ids=[101],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            multimodal_inputs=None,
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=torch.tensor([1.0, 2.0]),
            output_topk_p=torch.tensor([1.0]),
            output_topk_index=torch.tensor([7]),
            output_dsa_topk_indices=seed,
            bootstrap_room=9,
        )

    def test_metadata_buffer_copies_seed_and_uses_invalid_sentinel(self):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=2,
            hidden_states_dtype=torch.float32,
            output_dsa_topk_indices_dim=3,
        )
        seed = torch.tensor([4, 5, 6], dtype=torch.int32)
        buffers.set_buf(self._make_req(seed))
        buffers.set_buf(self._make_req(None, metadata_buffer_index=1))

        self.assertTrue(torch.equal(buffers.output_dsa_topk_indices[0], seed))
        self.assertEqual(buffers.output_dsa_topk_indices[1].tolist(), [-1, -1, -1])
        ptrs, data_lens, item_lens = buffers.get_buf_infos()
        self.assertEqual(ptrs[-2], buffers.output_dsa_topk_indices.data_ptr())
        self.assertEqual(data_lens[-2], buffers.output_dsa_topk_indices.nbytes)
        self.assertEqual(item_lens[-2], buffers.output_dsa_topk_indices[0].nbytes)

    def test_decode_input_requires_valid_seed_for_every_request(self):
        seeds = (
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5, 6], dtype=torch.int32),
        )
        batch = SimpleNamespace(
            reqs=[self._make_req(seed) for seed in seeds],
            device="cpu",
            enable_overlap=False,
        )
        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            enable_multi_layer_eagle=False,
            disaggregation_mode="null",
        )
        last_tokens = torch.tensor([11, 12], dtype=torch.int64)

        draft_input = build_eagle_disagg_draft_input(
            batch, server_args, last_tokens, None
        )
        self.assertTrue(torch.equal(draft_input.dsa_topk_indices, torch.stack(seeds)))

        for invalid_seed in (
            None,
            torch.full((3,), -1, dtype=torch.int32),
        ):
            batch.reqs[1].output_dsa_topk_indices = invalid_seed
            draft_input = build_eagle_disagg_draft_input(
                batch, server_args, last_tokens, None
            )
            self.assertIsNone(draft_input.dsa_topk_indices)

    def test_pd_decode_fused_topk_remaps_wire_positions_to_local_slots(self):
        wire_positions = (
            torch.tensor([2, 0, -1], dtype=torch.int32),
            torch.tensor([1, 3, -1], dtype=torch.int32),
        )
        req_to_token = torch.tensor(
            [
                [0, 0, 0, 0],
                [700, 801, 902, 990],
                [410, 420, 430, 440],
                [101, 205, 309, 450],
            ],
            dtype=torch.int32,
        )
        batch = SimpleNamespace(
            reqs=[self._make_req(seed) for seed in wire_positions],
            device="cpu",
            enable_overlap=False,
            req_pool_indices=torch.tensor([3, 1], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        )
        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            enable_multi_layer_eagle=False,
            disaggregation_mode="decode",
            enable_hisparse=False,
        )

        with envs.SGLANG_DSA_FUSE_TOPK.override(True), patch(
            "sglang.srt.layers.attention.dsa.utils.is_cuda", return_value=True
        ):
            self.assertTrue(
                should_use_dsa_fused_topk(
                    server_args, seed_dsa_topk_from_draft_extend=True
                )
            )
            draft_input = build_eagle_disagg_draft_input(
                batch, server_args, torch.tensor([11, 12], dtype=torch.int64), None
            )

        self.assertEqual(
            draft_input.dsa_topk_indices.tolist(),
            [[309, 101, -1], [801, 990, -1]],
        )

    def test_future_map_initializes_seed_buffer_after_seedless_payload(self):
        future_map = object.__new__(FutureMap)
        future_map.dsa_topk_indices_buf = None
        future_map.req_pool_size = 4
        future_map.device = "cpu"
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(bonus_tokens=torch.zeros((2,), dtype=torch.int64))
        )
        self.assertIsNone(future_map.dsa_topk_indices_buf)

        seeds = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(
                bonus_tokens=torch.zeros((2,), dtype=torch.int64),
                dsa_topk_indices=seeds,
            )
        )
        self.assertEqual(future_map.dsa_topk_indices_buf.shape, (4, 3))
        self.assertEqual(future_map.dsa_topk_indices_buf.dtype, torch.int32)


class TestDSV4C128StateIndices(unittest.TestCase):
    def test_online_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=True, ring_size=1),
            np.empty((0,), dtype=np.int32),
        )

    def test_online_partial_boundary_uses_request_slot(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 257, online=True, ring_size=1),
            np.array([7], dtype=np.int32),
        )

    def test_offline_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=False, ring_size=128),
            np.empty((0,), dtype=np.int32),
        )

    def test_offline_partial_boundary_uses_request_local_page(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 129, online=False, ring_size=256),
            np.array([15], dtype=np.int32),
        )


def _buf_infos(*ptrs):
    return list(ptrs), [ptr + 100 for ptr in ptrs], [ptr + 200 for ptr in ptrs]


def _make_dsv4_target(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.get_state_buf_infos = lambda: _buf_infos(11)
    pool.get_unified_swa_ring_buf_infos = lambda: (
        _buf_infos(12) if unified else ([], [], [])
    )
    pool.get_c128_state_buf_infos = lambda: ([], [], [])
    return pool


def _make_dsv4_draft(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.compression_ratios = [0]
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.compress_state_pools = [None]
    pool.indexer_compress_state_pools = [None]
    if unified:
        pool.unified_kv_pool = SimpleNamespace(
            swa_pages=524,
            kv_buffer=[torch.empty((524, 16), dtype=torch.uint8)],
        )
    else:
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty((2, 16), dtype=torch.uint8)]
        )
    return pool


class TestDSV4DraftStateRegistration(unittest.TestCase):
    def test_draft_state_is_a_separate_component(self):
        mapping = torch.arange(16)
        cases = [
            (
                "paged",
                _make_dsv4_target(unified=False, mapping=mapping),
                _make_dsv4_draft(unified=False, mapping=mapping),
                [StateType.SWA, StateType.SWA],
                [[11]],
            ),
            (
                "unified",
                _make_dsv4_target(unified=True),
                _make_dsv4_draft(unified=True),
                [StateType.SWA, StateType.SWA_RING, StateType.SWA_RING],
                [[11], [12]],
            ),
        ]

        for name, target, draft, expected_types, target_ptrs in cases:
            with self.subTest(name=name):
                if draft._unified_kv:
                    expected_infos = draft.get_unified_swa_ring_buf_infos()
                else:
                    expected_infos = draft.get_state_buf_infos()
                kv_args = KVArgs()

                setup_state_kv_args(kv_args, target, draft)

                self.assertEqual(kv_args.state_types, expected_types)
                self.assertEqual(kv_args.state_data_ptrs[:-1], target_ptrs)
                self.assertEqual(kv_args.state_data_ptrs[-1], expected_infos[0])
                self.assertEqual(kv_args.state_data_lens[-1], expected_infos[1])
                self.assertEqual(kv_args.state_item_lens[-1], expected_infos[2])


if __name__ == "__main__":
    unittest.main()
