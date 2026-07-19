import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_list,
    pack_int_lists,
    pack_list_of_buffers,
    pack_nested_transfer_layout,
    pack_transfer_layout,
    unpack_int_list,
    unpack_int_lists,
    unpack_list_of_buffers,
    unpack_nested_transfer_layout,
    unpack_transfer_layout,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import (
    MetadataBuffers,
    append_draft_kv_data,
    get_dsv4_c128_state_indices,
    setup_state_kv_args,
    should_transfer_draft_cache,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
    CpCacheLayerSplitPoolBase,
)
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_TRANSFER_C128_STATE,
    DSV4_TRANSFER_SWA_KV,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
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

    def test_flat_int_list_roundtrip(self):
        self.assertEqual(unpack_int_list(pack_int_list([7, 8, 9], "I"), "I"), [7, 8, 9])
        self.assertEqual(pack_int_list([], "I"), b"")
        self.assertEqual(unpack_int_list(b"", "I"), [])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestCpCacheLayerSplitTransferLayoutWire(CustomTestCase):
    def test_layer_split_transfer_layout_roundtrip_preserves_none_slots(self):
        layout = [("dsv4_c4_kv", 1), None, ("dsv4_c128_kv", 3)]

        self.assertEqual(unpack_transfer_layout(pack_transfer_layout(layout)), layout)
        self.assertEqual(pack_transfer_layout([]), b"")
        self.assertEqual(unpack_transfer_layout(b""), [])

    def test_layer_split_state_layout_roundtrip_preserves_component_boundaries(self):
        layouts = [
            [("dsv4_swa_kv", 0), None],
            [("dsv4_attention_state", 1), ("dsv4_indexer_state", 1)],
        ]

        self.assertEqual(
            unpack_nested_transfer_layout(pack_nested_transfer_layout(layouts)),
            layouts,
        )
        self.assertEqual(pack_nested_transfer_layout([]), b"")
        self.assertEqual(unpack_nested_transfer_layout(b""), [])


class TestCpCacheLayerSplitDescriptorMatching(CustomTestCase):
    def _build_params(self, **kwargs):
        params = dict(
            src_data_ptrs=[100],
            dst_data_ptrs=[200],
            item_lens=[16],
            src_data_layout=[("dsv4_c4_kv", 1)],
            dst_data_layout=[("dsv4_c4_kv", 1)],
            dst_item_lens=[16],
        )
        params.update(kwargs)
        return CommonKVManager.build_descriptor_matched_transfer_params(**params)

    def test_descriptor_matching_checks_destination_item_size(self):
        with self.assertRaisesRegex(RuntimeError, "item size mismatch"):
            self._build_params(dst_item_lens=[32])

    def test_required_descriptor_matching_rejects_missing_layouts(self):
        with self.assertRaisesRegex(RuntimeError, "descriptors on both"):
            self._build_params(
                src_data_layout=[],
                dst_data_layout=[],
            )

    def test_descriptor_matching_returns_pointer_item_len_tuples(self):
        self.assertEqual(
            self._build_params(),
            [(100, 200, 16)],
        )


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
    pool.get_state_transfer_layout = lambda: []
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
    pool.get_state_transfer_layout = lambda: (
        [] if unified else [(DSV4_TRANSFER_SWA_KV, 0)]
    )
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


class TestDSV4DraftStateRegistration(CustomTestCase):
    def test_draft_state_is_a_separate_component(self):
        mapping = torch.arange(16)
        cases = [
            (
                "paged",
                _make_dsv4_target(unified=False, mapping=mapping),
                _make_dsv4_draft(unified=False, mapping=mapping),
                [StateType.SWA, StateType.SWA],
                [[11]],
                [(DSV4_TRANSFER_SWA_KV, 0)],
            ),
            (
                "unified",
                _make_dsv4_target(unified=True),
                _make_dsv4_draft(unified=True),
                [StateType.SWA, StateType.SWA_RING, StateType.SWA_RING],
                [[11], [12]],
                [],
            ),
        ]

        for name, target, draft, expected_types, target_ptrs, draft_layout in cases:
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
                self.assertEqual(kv_args.state_data_layouts[-1], draft_layout)


class TestDSV4DraftLayerSplitTransfer(CustomTestCase):
    def test_empty_draft_kv_data_preserves_target_descriptors(self):
        ptrs, lens, item_lens = [1], [2], [3]
        layout = [("dsv4_c4_kv", 4)]
        draft = SimpleNamespace(get_contiguous_buf_infos=lambda: ([], [], []))

        added = append_draft_kv_data(ptrs, lens, item_lens, layout, draft)

        self.assertEqual(added, 0)
        self.assertEqual(layout, [("dsv4_c4_kv", 4)])

    def test_only_last_layer_split_rank_transfers_replicated_draft(self):
        pool = object.__new__(CpCacheLayerSplitPoolBase)
        pool.cp_size = 4

        pool.cp_rank = 0
        self.assertFalse(should_transfer_draft_cache(pool))
        pool.cp_rank = 3
        self.assertTrue(should_transfer_draft_cache(pool))
        self.assertTrue(should_transfer_draft_cache(object()))

    def test_empty_layer_split_c128_component_is_not_transferred(self):
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            state_types=[StateType.C128_STATE],
            state_data_ptrs=[[]],
            state_item_lens=[[]],
            require_descriptor_matched_transfer=True,
        )
        req = SimpleNamespace(dst_state_indices=[[0]])

        self.assertEqual(
            manager.maybe_send_extra(req, [[0]], executor=None),
            0,
        )


class TestDSV4C128StateRegistration(CustomTestCase):
    def test_c128_state_layout_is_registered_as_separate_component(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.get_state_buf_infos = lambda: _buf_infos(11)
        pool.get_state_transfer_layout = lambda: [(DSV4_TRANSFER_SWA_KV, 0)]
        pool.get_c128_state_buf_infos = lambda: _buf_infos(12)
        pool.get_c128_state_transfer_layout = lambda: [(DSV4_TRANSFER_C128_STATE, 5)]
        kv_args = KVArgs()

        setup_state_kv_args(kv_args, pool)

        self.assertEqual(kv_args.state_types, [StateType.SWA, StateType.C128_STATE])
        self.assertEqual(
            kv_args.state_data_layouts,
            [
                [(DSV4_TRANSFER_SWA_KV, 0)],
                [(DSV4_TRANSFER_C128_STATE, 5)],
            ],
        )

    def test_layer_split_keeps_empty_c128_slot_before_draft_state(self):
        mapping = torch.arange(16)
        target = object.__new__(CpCacheLayerSplitDeepSeekV4TokenToKVPool)
        target._unified_kv = False
        target.compression_ratios = [0, 128]
        target.page_size = 256
        target.sliding_window = 128
        target.full_to_swa_index_mapping = mapping
        target.get_state_buf_infos = lambda: _buf_infos(11)
        target.get_state_transfer_layout = lambda: [(DSV4_TRANSFER_SWA_KV, 0)]
        target.get_c128_state_buf_infos = lambda: ([], [], [])
        target.get_c128_state_transfer_layout = lambda: []
        draft = _make_dsv4_draft(unified=False, mapping=mapping)
        kv_args = KVArgs()

        setup_state_kv_args(kv_args, target, draft)

        self.assertEqual(
            kv_args.state_types,
            [StateType.SWA, StateType.C128_STATE, StateType.SWA],
        )
        self.assertEqual(kv_args.state_data_ptrs[1], [])
        self.assertEqual(
            kv_args.state_data_layouts,
            [
                [(DSV4_TRANSFER_SWA_KV, 0)],
                [],
                [(DSV4_TRANSFER_SWA_KV, 0)],
            ],
        )


if __name__ == "__main__":
    unittest.main()
