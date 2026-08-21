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
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import should_use_dsa_fused_topk
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.runtime_context import get_context
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
        # The draft-input shape comes from the spec bag.
        override = get_context().override_server_args(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            enable_multi_layer_eagle=False,
        )
        override.install()
        self.addCleanup(override.restore)
        last_tokens = torch.tensor([11, 12], dtype=torch.int64)

        draft_input = build_eagle_disagg_draft_input(batch, last_tokens, None)
        self.assertTrue(torch.equal(draft_input.dsa_topk_indices, torch.stack(seeds)))

        for invalid_seed in (
            None,
            torch.full((3,), -1, dtype=torch.int32),
        ):
            batch.reqs[1].output_dsa_topk_indices = invalid_seed
            draft_input = build_eagle_disagg_draft_input(batch, last_tokens, None)
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
        override = get_context().override_server_args(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            enable_multi_layer_eagle=False,
            disaggregation_mode="decode",
            enable_hisparse=False,
        )
        override.install()
        self.addCleanup(override.restore)

        with envs.SGLANG_DSA_FUSE_TOPK.override(True), patch(
            "sglang.srt.layers.attention.dsa.utils.is_cuda", return_value=True
        ):
            self.assertTrue(
                should_use_dsa_fused_topk(seed_dsa_topk_from_draft_extend=True)
            )
            draft_input = build_eagle_disagg_draft_input(
                batch, torch.tensor([11, 12], dtype=torch.int64), None
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
