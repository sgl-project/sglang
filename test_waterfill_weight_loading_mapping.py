import unittest


class TestWaterfillWeightLoadingMapping(unittest.TestCase):
    def setUp(self):
        # Lazily import to avoid side effects at module import time
        from types import SimpleNamespace

        import sglang.srt.layers.moe.utils as moe_utils
        import sglang.srt.server_args as server_args_mod

        self.moe_utils = moe_utils
        self.server_args_mod = server_args_mod

        # Save and override globals
        self._old_backend = moe_utils.MOE_A2A_BACKEND
        self._old_global_server_args = getattr(
            server_args_mod, "_global_server_args", None
        )

        moe_utils.MOE_A2A_BACKEND = moe_utils.MoeA2ABackend.DEEPEP
        server_args_mod.set_global_server_args_for_scheduler(
            SimpleNamespace(enable_deepep_waterfill=True)
        )
        self.server_args = server_args_mod.get_global_server_args()

    def tearDown(self):
        # Restore globals
        self.moe_utils.MOE_A2A_BACKEND = self._old_backend
        self.server_args_mod._global_server_args = self._old_global_server_args

    def _make_fusedmoe_stub(self, ep_rank: int, ep_size: int):
        # We only need the fields accessed by FusedMoE._map_global_expert_id_to_local_expert_id.
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        m = object.__new__(FusedMoE)
        m.num_fused_shared_experts = 0
        m.num_experts = 264  # DeepSeekV3 routed(256) + ep_size(8)
        m.num_local_experts = 33  # (264 / 8)
        m.moe_ep_rank = ep_rank
        m.moe_ep_size = ep_size
        return m

    def test_maps_checkpoint_expert_ids_with_old_experts_per_rank(self):
        # Waterfill expands expert layout to 33 per rank at runtime, but checkpoint expert IDs are 0..255
        # laid out as 32 per rank. This test asserts we map checkpoint IDs using old_epr=32.
        ep_size = 8

        # Rank1 should own experts [32..63]
        m1 = self._make_fusedmoe_stub(ep_rank=1, ep_size=ep_size)
        self.assertEqual(m1._map_global_expert_id_to_local_expert_id(63), 31)
        self.assertEqual(m1._map_global_expert_id_to_local_expert_id(64), -1)

        # Rank2 should own experts [64..95]
        m2 = self._make_fusedmoe_stub(ep_rank=2, ep_size=ep_size)
        self.assertEqual(m2._map_global_expert_id_to_local_expert_id(64), 0)
        self.assertEqual(m2._map_global_expert_id_to_local_expert_id(95), 31)
        self.assertEqual(m2._map_global_expert_id_to_local_expert_id(96), -1)

    def test_mapping_is_not_applied_when_waterfill_disabled(self):
        # When Waterfill is disabled, the mapping should fall back to the standard layout
        # (num_local_routed_experts = num_local_experts).
        self.server_args.enable_deepep_waterfill = False

        ep_size = 8
        m1 = self._make_fusedmoe_stub(ep_rank=1, ep_size=ep_size)

        # With the expanded 33-per-rank layout, expert 64 would be considered owned by rank1
        # (start=33,end=66) and map to local 31. This is intentionally different from the Waterfill mapping.
        self.assertEqual(m1._map_global_expert_id_to_local_expert_id(64), 31)


if __name__ == "__main__":
    unittest.main()
