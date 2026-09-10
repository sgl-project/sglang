import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.deepseek_v4_hook import (  # noqa: E402
    validate_deepseek_v41_features,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager  # noqa: E402
from sglang.srt.disaggregation.nixl.conn import NixlKVManager  # noqa: E402
from sglang.srt.disaggregation.utils import (  # noqa: E402
    get_dsv4_request_state_indices,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (  # noqa: E402
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.cuda_graph_config import Backend  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PairPool:
    kv_pools = {1: None, 2: None}

    def __init__(self, ring_size):
        self.ring_size = ring_size

    def get_ring_size(self, ratio):
        assert ratio == 2
        return self.ring_size


class TestDSV41PairTransfer(CustomTestCase):
    def test_pending_row_uses_local_ring_and_request_slot(self):
        for ring in (2, 8, 16):
            for req_slot in (0, 3, 7):
                for length in (1, 3, 7, 9, 101, 257, 4097):
                    with self.subTest(ring=ring, slot=req_slot, length=length):
                        indices = get_dsv4_request_state_indices(
                            _PairPool(ring), req_slot, length
                        )
                        np.testing.assert_array_equal(
                            indices, [req_slot * ring + (length - 1) % ring]
                        )
                        self.assertEqual(indices.dtype, np.int32)

    def test_even_boundary_has_no_payload(self):
        for length in (0, 2, 8, 256, 4096):
            self.assertEqual(
                get_dsv4_request_state_indices(_PairPool(8), 3, length).size, 0
            )

    def test_c128_page_indices_are_unchanged(self):
        pool = SimpleNamespace(kv_pools={128: None}, get_ring_size=lambda _: 256)
        with patch(
            "sglang.srt.disaggregation.utils.is_dsv4_c128_online_enabled",
            return_value=False,
        ):
            np.testing.assert_array_equal(
                get_dsv4_request_state_indices(pool, 3, 129), [7]
            )
            self.assertEqual(get_dsv4_request_state_indices(pool, 3, 256).size, 0)

    def test_pair_state_registers_one_row_per_transfer_item(self):
        states = [torch.empty((64, 1024), dtype=torch.float32) for _ in range(3)]
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.compress_state_pools = [
            SimpleNamespace(
                ratio=2,
                ring_size=8,
                kv_score_buffer=SimpleNamespace(kv_score=state),
            )
            for state in states
        ]

        ptrs, lens, item_lens = pool.get_c128_state_buf_infos()

        self.assertEqual(ptrs, [state.data_ptr() for state in states])
        self.assertEqual(lens, [state.nbytes for state in states])
        self.assertEqual(item_lens, [state[0].nbytes for state in states])
        self.assertEqual(sum(item_lens), 12 * 1024)

    def test_prefill_uses_minimal_pair_ring_while_decode_uses_spec_ring(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        spec = SimpleNamespace(
            speculative_algorithm="DSPARK", speculative_num_draft_tokens=5
        )

        with (
            patch(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
            patch(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_spec",
                return_value=spec,
            ),
        ):
            self.assertEqual(pool.get_ring_size(2), 2)

        with (
            patch(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            ),
            patch(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_spec",
                return_value=spec,
            ),
        ):
            self.assertEqual(pool.get_ring_size(2), 8)


class TestDSV41PDDSparkValidation(CustomTestCase):
    def _config(self, algorithm):
        return SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            speculative_algorithm=algorithm,
            disaggregation_mode="decode",
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            pp_size=1,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=None)
            ),
            enable_decoder_swa_bounded_replay=False,
        )

    def _validate(self, algorithm):
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
                return_value=self._config(algorithm),
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41")
                ),
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            validate_deepseek_v41_features(SimpleNamespace())

    def test_pd_dspark_is_allowed(self):
        self._validate("DSPARK")

    def test_pd_other_speculative_algorithm_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "speculative decoding other than DSpark"
        ):
            self._validate("EAGLE")


class _RecordingMooncakeManager:
    def __init__(self):
        self.is_mla_backend = True
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.blocks = []

    def get_mla_kv_ptrs_with_pp(self, src, dst, state_type):
        return src, dst, len(src)

    def _transfer_data(self, session_id, blocks):
        self.blocks.extend(blocks)
        return 0


class _RecordingNixlAgent:
    def __init__(self):
        self.requests = []

    def get_xfer_descs(self, requests, mem_kind):
        self.requests.append(requests)
        return requests

    def initialize_xfer(self, *args):
        return "handle"

    def transfer(self, handle):
        return "DONE"


class TestDSV41PairTransferAddressing(CustomTestCase):
    def test_mooncake_maps_source_and_destination_ring_rows(self):
        manager = _RecordingMooncakeManager()

        MooncakeKVManager._send_kvcache_generic(
            manager,
            mooncake_session_id="session",
            src_data_ptrs=[1000],
            dst_data_ptrs=[2000],
            item_lens=[4096],
            prefill_data_indices=np.array([6], dtype=np.int32),
            dst_data_indices=np.array([28], dtype=np.int32),
            executor=None,
        )

        self.assertEqual(manager.blocks, [(1000 + 6 * 4096, 2000 + 28 * 4096, 4096)])

    def test_nixl_maps_source_and_destination_ring_rows(self):
        manager = object.__new__(NixlKVManager)
        manager.is_mla_backend = True
        manager.prep_handles = {}
        manager.kv_args = SimpleNamespace(gpu_id=0, kv_data_ptrs=[])
        manager.agent = _RecordingNixlAgent()

        handle = NixlKVManager._send_kvcache_generic(
            manager,
            peer_name="decode",
            src_data_ptrs=[1000],
            dst_data_ptrs=[2000],
            item_lens=[4096],
            prefill_data_indices=np.array([6], dtype=np.int32),
            dst_data_indices=np.array([28], dtype=np.int32),
            dst_gpu_id=1,
            notif="state",
        )

        self.assertEqual(handle, "handle")
        src, dst = manager.agent.requests
        np.testing.assert_array_equal(src[:, :2], [[1000 + 6 * 4096, 4096]])
        np.testing.assert_array_equal(dst[:, :2], [[2000 + 28 * 4096, 4096]])


if __name__ == "__main__":
    unittest.main()
