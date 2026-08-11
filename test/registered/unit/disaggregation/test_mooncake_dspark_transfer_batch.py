import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeDSparkTransferBatch(unittest.TestCase):
    def _manager(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.enable_custom_mem_pool = False
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100],
            kv_item_lens=[4],
            kv_layer_ids=[40],
            state_types=[StateType.SWA, StateType.C128_STATE, StateType.SWA],
            state_data_ptrs=[[200], [300], [400]],
            state_item_lens=[[8], [16], [8]],
        )
        manager._validate_envelope_kv_layout = MagicMock()
        manager._transfer_data = MagicMock(return_value=0)
        return manager

    def test_only_repeated_swa_has_a_draft_component(self):
        manager = self._manager()
        self.assertEqual(
            manager._find_draft_swa_component_index([[1], [2], [3]]), 2
        )

        manager.kv_args.state_types = [StateType.SWA, StateType.C128_STATE]
        self.assertIsNone(manager._find_draft_swa_component_index([[1], [2]]))

    def test_target_kv_and_draft_swa_share_one_submit(self):
        manager = self._manager()
        req = SimpleNamespace(
            mooncake_session_id="session",
            dst_state_indices=[[11], [12], [7]],
        )
        registration = SimpleNamespace(
            dst_kv_item_len=4,
            dst_attn_tp_size=1,
            dst_kv_layer_ids=[40],
            dst_state_types=[
                StateType.SWA,
                StateType.C128_STATE,
                StateType.SWA,
            ],
            dst_state_data_ptrs=[[600], [700], [800]],
        )

        result = manager.send_kvcache_with_draft_swa(
            req=req,
            prefill_kv_indices=np.asarray([1, 2], dtype=np.int32),
            dst_kv_ptrs=[500],
            dst_kv_indices=np.asarray([3, 4], dtype=np.int32),
            prefill_state_indices=[[9], [10], [5]],
            target_rank_registration_info=registration,
        )

        self.assertEqual(result, (0, 2))
        manager._validate_envelope_kv_layout.assert_called_once_with([500], 4, 1)
        manager._transfer_data.assert_called_once_with(
            "session",
            [
                (104, 512, 8),
                (440, 856, 8),
            ],
        )


if __name__ == "__main__":
    unittest.main()
