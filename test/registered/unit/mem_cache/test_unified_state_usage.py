"""The state sub-pool's usage sums like every other sub-pool's: held plus
available, even when free state ids have no bytes to land on."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, StateGeometry, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _build():
    setup_allocator_context()
    kv = MHAGeometry(
        head_num=1, head_dim=128, v_head_dim=128, kv_cache_dtype=torch.bfloat16
    )
    plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=kv,
            state_layer_ids=(1,),
            state_geometry=StateGeometry(
                conv_state_shapes=((3, 3072),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(16, 128, 128),
                temporal_dtype=torch.float32,
            ),
            checkpoint_interval=64,
        ),
        page_size=1,
    )
    sized = size_plan(
        plan,
        KVSizingInputs(
            budget_bytes=64 << 20,
            max_total_tokens=4096,
            max_running_requests=2,
            context_len=4096,
            alignment=1,
        ),
    )
    return build_kv_storage(
        plan,
        sized,
        UnifiedBuildContext(
            device="cpu",
            start_layer=0,
            end_layer=2,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=True,
            max_num_reqs=2,
            model_context_len=4096,
            state_cache_params=SimpleNamespace(layers=(1,)),
        ),
    )


class TestStateUsage(unittest.TestCase):
    def _state_usage(self, storage):
        (usage,) = [u for u in storage.usage() if u.metrics == "mamba"]
        return usage

    def test_held_plus_available_when_state_ids_outnumber_their_bytes(self):
        storage = _build()
        allocator = storage.allocator
        state = allocator.state_slots
        held = state.alloc(2)
        self.assertIsNotNone(held)
        # Fill the token sub-pool until it holds bytes a free state id maps to:
        # then that id is free but no take can place it.
        while allocator.state_slots_available() >= state.available_size():
            if allocator.alloc(allocator.page_size) is None:
                break
        self.assertLess(allocator.state_slots_available(), state.available_size())

        usage = self._state_usage(storage)
        self.assertEqual(usage.available, allocator.state_slots_available())
        # Every slot is held or available: none appears used by nobody.
        self.assertEqual(usage.capacity - usage.available, 2)

    def test_no_state_capacity_reports_no_usage(self):
        """With the token sub-pool holding every byte, the state capacity is zero
        and its reported usage is zero, not a division by it."""
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )

        storage = _build()
        allocator = storage.allocator
        while allocator.state_slots_available():
            if allocator.alloc(allocator.page_size) is None:
                break
        usage = self._state_usage(storage)
        self.assertEqual((usage.capacity, usage.available), (0, 0))

        (full,) = [u for u in storage.usage() if u.metrics == "full"]
        observer = SchedulerPoolStatsObserver(
            tree_cache=SimpleNamespace(
                supports_mamba=lambda: False, supports_prefix_sharing=lambda: True
            ),
            storage=storage,
            token_to_kv_pool_allocator=SimpleNamespace(
                state=SimpleNamespace(ckpt_pool=None)
            ),
            session_controller=None,
            hisparse_coordinator=None,
            enable_hisparse=False,
        )
        stats = observer._get_mamba_token_info(full, usage)
        self.assertEqual((stats.mamba_capacity, stats.mamba_usage), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
