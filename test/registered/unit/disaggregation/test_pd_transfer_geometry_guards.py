"""Slot-count guards and MHA V-half strides of the NIXL PD transfer, driven on
``object.__new__`` managers with fake NIXL agents (no GPU, no RDMA)."""

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.nixl.conn import KVArgsRegisterInfo, NixlKVManager
from sglang.srt.disaggregation.utils import (
    kv_region_slot_counts,
    log_kv_registration_summary,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SRC_BASE, DST_BASE, REGION_STRIDE = 0x100000, 0x900000, 0x10000
ITEM_LEN = 64
RAGGED_SLOT_COUNTS = [4, 4, 2, 2, 2]
RAGGED_DATA_LENS = [n * ITEM_LEN for n in RAGGED_SLOT_COUNTS]
RAGGED_ITEM_LENS = [ITEM_LEN] * len(RAGGED_SLOT_COUNTS)


class ExplodingAgent:
    def prep_xfer_dlist(self, *args, **kwargs):
        raise AssertionError("prep_xfer_dlist must not be called on this path")


class PrepRecordingAgent:
    def __init__(self):
        self.prep_calls = []

    def prep_xfer_dlist(self, peer_name, arr, mem_kind):
        self.prep_calls.append((peer_name, arr, mem_kind))
        return f"prep_{len(self.prep_calls)}"


class XferRecordingAgent(ExplodingAgent):
    def __init__(self):
        self.get_xfer_descs_calls = []

    def get_xfer_descs(self, reqs, mem_kind):
        self.get_xfer_descs_calls.append((reqs, mem_kind))
        return f"descs_{len(self.get_xfer_descs_calls)}"

    def initialize_xfer(self, *args):
        return "handle"

    def transfer(self, handle):
        return "DONE"


def _make_kv_args(kv_data_lens, kv_item_lens):
    n = len(kv_data_lens)
    return SimpleNamespace(
        kv_data_ptrs=[SRC_BASE + i * REGION_STRIDE for i in range(n)],
        kv_data_lens=list(kv_data_lens),
        kv_item_lens=list(kv_item_lens),
        kv_data_mem_kinds=["VRAM"] * n,
        kv_layer_ids=[],
        state_types=[],
        aux_data_ptrs=[],
        aux_item_lens=[],
        gpu_id=0,
        page_size=1,
        prefill_start_layer=0,
    )


def _make_prefill_manager(kv_args, agent, is_mla_backend=True, attn_tp_size=1):
    mgr = object.__new__(NixlKVManager)
    mgr.kv_args = kv_args
    mgr.agent = agent
    mgr.src_mem_kind = "VRAM"
    mgr.is_mla_backend = is_mla_backend
    mgr.is_hybrid_mla_backend = False
    mgr.attn_tp_size = attn_tp_size
    mgr.pp_size = 1
    mgr.prep_handles = {}
    mgr.prep_handle_slice_src = None
    mgr.prep_handles_slice_dst = {}
    mgr.prep_handles_segment_src = {}
    mgr._num_slots_src = kv_args.kv_data_lens[0] // kv_args.kv_item_lens[0]
    return mgr


def _make_peer_info(
    n_regions, dst_kv_item_lens, dst_num_slots, dst_kv_mem_kinds=None, decode_tp_size=1
):
    return KVArgsRegisterInfo(
        room="None",
        endpoint="127.0.0.1",
        dst_port=1000,
        agent_name="decode_agent",
        agent_metadata=b"",
        dst_kv_ptrs=[DST_BASE + i * REGION_STRIDE for i in range(n_regions)],
        dst_kv_mem_kinds=(
            list(dst_kv_mem_kinds)
            if dst_kv_mem_kinds is not None
            else ["VRAM"] * n_regions
        ),
        dst_aux_ptrs=[],
        dst_state_data_ptrs=[],
        gpu_id=1,
        decode_tp_size=decode_tp_size,
        decode_tp_rank=0,
        dst_kv_item_len=dst_kv_item_lens[0],
        dst_kv_item_lens=list(dst_kv_item_lens),
        dst_num_slots=dst_num_slots,
    )


class TestSlotCountGuards(CustomTestCase):
    def test_homogeneous_prepped_falls_back_on_ragged_source(self):
        kv_args = _make_kv_args(RAGGED_DATA_LENS, RAGGED_ITEM_LENS)
        agent = XferRecordingAgent()
        mgr = _make_prefill_manager(kv_args, agent)
        peer_info = _make_peer_info(5, RAGGED_ITEM_LENS, dst_num_slots=4)

        mgr._prepare_payload_xfer(peer_info)

        self.assertEqual(mgr.prep_handles, {})
        self.assertIsNone(peer_info.kv_xfer_segments)
        self.assertEqual(peer_info.dst_homogeneous_mem_kind, "VRAM")

        mgr.send_kvcache(
            "decode_agent",
            np.array([1], dtype=np.int32),
            peer_info.dst_kv_ptrs,
            np.array([3], dtype=np.int32),
            peer_info.gpu_id,
            "room_kv_0_1_0",
        )
        (src_reqs, _), (dst_reqs, _) = agent.get_xfer_descs_calls
        regions = np.arange(5, dtype=np.uint64) * REGION_STRIDE
        np.testing.assert_array_equal(src_reqs[:, 0], SRC_BASE + regions + 1 * ITEM_LEN)
        np.testing.assert_array_equal(dst_reqs[:, 0], DST_BASE + regions + 3 * ITEM_LEN)
        np.testing.assert_array_equal(
            src_reqs[:, 1], np.full(5, ITEM_LEN, dtype=np.uint64)
        )

    def test_homogeneous_prepped_builds_when_uniform(self):
        kv_args = _make_kv_args([4 * ITEM_LEN] * 5, RAGGED_ITEM_LENS)
        agent = PrepRecordingAgent()
        mgr = _make_prefill_manager(kv_args, agent)
        peer_info = _make_peer_info(5, RAGGED_ITEM_LENS, dst_num_slots=4)

        mgr._prepare_payload_xfer(peer_info)

        self.assertIn("", mgr.prep_handles)
        self.assertIn("decode_agent", mgr.prep_handles)
        self.assertEqual(len(agent.prep_calls), 2)

    def test_decode_extra_regions_keep_prepped_path(self):
        # build_transfer_entry_pairs drops decode's two extra (draft) regions,
        # so only the source's uniform geometry gates the prep handles.
        kv_args = _make_kv_args([4 * ITEM_LEN] * 2, [ITEM_LEN] * 2)
        agent = PrepRecordingAgent()
        mgr = _make_prefill_manager(kv_args, agent)
        peer_info = _make_peer_info(4, [ITEM_LEN] * 4, dst_num_slots=4)

        mgr._prepare_payload_xfer(peer_info)

        self.assertIn("", mgr.prep_handles)
        _, arr, _ = agent.prep_calls[1]
        self.assertEqual(arr.shape, (8, 3))
        np.testing.assert_array_equal(
            arr[[0, 4], 0], [DST_BASE, DST_BASE + REGION_STRIDE]
        )

    def test_prepped_only_paths_raise_on_ragged_source(self):
        cases = [
            (
                "mixed_memory",
                {},
                {"dst_kv_mem_kinds": ["VRAM", "DRAM"]},
                r"mixed-memory prepped transfer.*\[2, 4\]",
            ),
            (
                "hetero_tp",
                {"is_mla_backend": False, "attn_tp_size": 2},
                {"decode_tp_size": 4},
                r"heterogeneous-TP prepped transfer.*\[2, 4\]",
            ),
        ]
        for name, mgr_kwargs, peer_kwargs, regex in cases:
            with self.subTest(name):
                kv_args = _make_kv_args([4 * ITEM_LEN, 2 * ITEM_LEN], [ITEM_LEN] * 2)
                mgr = _make_prefill_manager(kv_args, ExplodingAgent(), **mgr_kwargs)
                peer_info = _make_peer_info(
                    2, [ITEM_LEN] * 2, dst_num_slots=4, **peer_kwargs
                )

                with self.assertRaisesRegex(RuntimeError, regex):
                    mgr._prepare_payload_xfer(peer_info)

    def test_mixed_memory_prepped_builds_when_uniform(self):
        kv_args = _make_kv_args([4 * ITEM_LEN] * 2, [ITEM_LEN] * 2)
        agent = PrepRecordingAgent()
        mgr = _make_prefill_manager(kv_args, agent)
        peer_info = _make_peer_info(
            2, [ITEM_LEN] * 2, dst_num_slots=4, dst_kv_mem_kinds=["VRAM", "DRAM"]
        )

        mgr._prepare_payload_xfer(peer_info)

        self.assertEqual(len(peer_info.kv_xfer_segments), 2)
        self.assertEqual(len(agent.prep_calls), 4)
        self.assertEqual(mgr.prep_handles, {})

    def test_kv_region_slot_counts(self):
        self.assertEqual(
            kv_region_slot_counts(_make_kv_args(RAGGED_DATA_LENS, RAGGED_ITEM_LENS)),
            {4, 2},
        )
        self.assertEqual(
            kv_region_slot_counts(_make_kv_args([4 * ITEM_LEN] * 3, [ITEM_LEN] * 3)),
            {4},
        )

    def test_registration_summary_names_the_geometry(self):
        with self.assertLogs("sglang.srt.disaggregation.utils", level="INFO") as logs:
            log_kv_registration_summary(
                _make_kv_args(RAGGED_DATA_LENS, RAGGED_ITEM_LENS), "prefill"
            )
        self.assertEqual(len(logs.output), 1)
        self.assertIn("kv_entries=5", logs.output[0])
        self.assertIn("slot_counts=[2, 4]", logs.output[0])


class TestMhaVHalfStrides(CustomTestCase):
    K_PTRS, V_PTRS = [0x1000, 0x2000], [0x3000, 0x4000]
    DST_K_PTRS, DST_V_PTRS = [0x11000, 0x12000], [0x13000, 0x14000]

    def _run_send(self, item_lens):
        mgr = object.__new__(NixlKVManager)
        agent = XferRecordingAgent()
        mgr.agent = agent
        mgr.is_mla_backend = False
        mgr.prep_handles = {}
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=[], gpu_id=3, prefill_start_layer=0)
        mgr._send_kvcache_generic(
            peer_name="decode_agent",
            src_data_ptrs=self.K_PTRS + self.V_PTRS,
            dst_data_ptrs=self.DST_K_PTRS + self.DST_V_PTRS,
            item_lens=item_lens,
            prefill_data_indices=np.array([2, 3], dtype=np.int32),
            dst_data_indices=np.array([5, 6], dtype=np.int32),
            dst_gpu_id=7,
            notif="room_kv_0_1_0",
        )
        (src_reqs, src_kind), (dst_reqs, dst_kind) = agent.get_xfer_descs_calls
        self.assertEqual((src_kind, dst_kind), ("VRAM", "VRAM"))
        return src_reqs, dst_reqs

    def test_v_half_uses_entry_aligned_item_lens(self):
        src_reqs, dst_reqs = self._run_send(item_lens=[64, 64, 24, 24])
        np.testing.assert_array_equal(
            src_reqs[:, 0],
            np.array(
                [0x1000 + 2 * 64, 0x2000 + 2 * 64, 0x3000 + 2 * 24, 0x4000 + 2 * 24],
                dtype=np.uint64,
            ),
        )
        np.testing.assert_array_equal(
            dst_reqs[:, 0],
            np.array(
                [
                    0x11000 + 5 * 64,
                    0x12000 + 5 * 64,
                    0x13000 + 5 * 24,
                    0x14000 + 5 * 24,
                ],
                dtype=np.uint64,
            ),
        )
        np.testing.assert_array_equal(
            src_reqs[:, 1], np.array([2 * 64, 2 * 64, 2 * 24, 2 * 24], dtype=np.uint64)
        )
        np.testing.assert_array_equal(src_reqs[:, 2], np.full(4, 3, dtype=np.uint64))
        np.testing.assert_array_equal(dst_reqs[:, 2], np.full(4, 7, dtype=np.uint64))

    def test_uniform_item_lens_unchanged(self):
        src_reqs, _ = self._run_send(item_lens=[64, 64, 64, 64])
        np.testing.assert_array_equal(
            src_reqs[:, 0],
            np.array(
                [0x1000 + 128, 0x2000 + 128, 0x3000 + 128, 0x4000 + 128],
                dtype=np.uint64,
            ),
        )
        np.testing.assert_array_equal(src_reqs[:, 1], np.full(4, 128, dtype=np.uint64))


if __name__ == "__main__":
    unittest.main()
