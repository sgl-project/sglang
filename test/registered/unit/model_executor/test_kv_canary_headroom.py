import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.runner.canary_manager import CanaryManager
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.model_runner_components import kv_pool_runtime
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

_GIB = 1 << 30


def _manager(slots, groups):
    manager = CanaryManager.__new__(CanaryManager)
    manager._launch_capacities = CanaryLaunchCapacities(
        per_forward_verify_capacity=3 * slots,
        per_forward_write_req_capacity=128,
        per_forward_write_entry_capacity=4096,
    )
    manager._buffer_groups = (None,) * groups
    return manager


class TestCanaryHeadroom(CustomTestCase):
    def _resize(self, target, drafts=(), *, graph_borrow=False, eager_gap=False):
        config = MemoryPoolConfig(max_total_num_tokens=1024)
        pool = Mock(post_capture_backed_bytes=2 * _GIB, dtype="bfloat16")
        runner = SimpleNamespace(
            token_to_kv_pool=pool,
            device="cuda",
            gpu_id=0,
            pre_model_load_memory=32,
            mem_fraction_static=0.875,
            max_running_requests=16 if eager_gap else None,
            model_config=SimpleNamespace(is_multimodal=False),
            sampling_prewarm_result=SimpleNamespace(sampling_headroom_bytes=6 * _GIB),
            canary_manager=target,
            kv_cache_configurator=Mock(),
            max_total_num_tokens=1_000_000,
            token_to_kv_pool_allocator=Mock(),
            req_to_token_pool=Mock(),
        )
        runner.kv_cache_configurator.config_from_budget.return_value = config
        runner.kv_cache_configurator.resolve_max_num_reqs.return_value = 16
        with (
            get_context().override_server_args(
                disaggregation_mode="null",
                cuda_graph_config=CudaGraphConfig(
                    decode=PhaseConfig(backend=Backend.FULL, max_bs=8)
                ),
            ),
            patch.object(kv_pool_runtime.torch.cuda, "synchronize"),
            patch.object(
                kv_pool_runtime,
                "get_world_group",
                return_value=SimpleNamespace(world_size=1, cpu_group=None),
            ),
            patch.object(kv_pool_runtime, "get_available_gpu_memory", return_value=20),
            patch.object(kv_pool_runtime, "mambaish_config", return_value=None),
            patch.object(
                kv_pool_runtime, "get_device_memory_capacity", return_value=32
            ),
            patch.object(
                kv_pool_runtime,
                "pre_capture_activation_reserve_mb",
                return_value=8 * 1024,
            ),
            patch.object(
                kv_pool_runtime, "graph_pool_borrow_enabled", return_value=graph_borrow
            ),
            patch.object(kv_pool_runtime, "mm_runtime_reservation_gb", return_value=1),
        ):
            resize = kv_pool_runtime.compute_post_capture_kv_resize(
                runner,
                draft_runners=tuple(SimpleNamespace(canary_manager=m) for m in drafts),
            )
        pool.finalize_backing.assert_called_once_with(config)
        runner.token_to_kv_pool_allocator.resize.assert_called_once_with(config)
        runner.req_to_token_pool.reset_aux_cache_allocator.assert_called_once_with()
        self.assertEqual(resize.max_total_num_tokens, config.max_total_num_tokens)
        args, kwargs = runner.kv_cache_configurator.config_from_budget.call_args
        self.assertEqual(kwargs, {"cap_tokens": 1_000_000})
        return args[0]

    def test_canary_off_preserves_every_budget_byte(self):
        for graph_borrow, eager_gap, headroom in (
            (False, False, 6),
            (True, False, 4),
            (True, True, 8),
        ):
            with self.subTest(graph_borrow=graph_borrow, eager_gap=eager_gap):
                self.assertEqual(
                    self._resize(
                        None, (None,), graph_borrow=graph_borrow, eager_gap=eager_gap
                    ),
                    (20 - headroom - 1 + 2) * _GIB,
                )

    def test_workspace_is_added_to_other_headroom_at_installed_capacity(self):
        manager = _manager(1_000_000, 3)
        workspace = manager.per_forward_workspace_bytes()
        for graph_borrow, eager_gap in ((False, False), (True, False), (True, True)):
            with self.subTest(graph_borrow=graph_borrow, eager_gap=eager_gap):
                baseline = self._resize(
                    None, graph_borrow=graph_borrow, eager_gap=eager_gap
                )
                self.assertEqual(
                    self._resize(
                        manager, graph_borrow=graph_borrow, eager_gap=eager_gap
                    ),
                    baseline - workspace,
                )

    def test_sequential_target_and_drafts_reserve_largest_workspace(self):
        small, large = _manager(1024, 1), _manager(4096, 4)
        for target, drafts in (
            (small, (large, None)),
            (large, (small,)),
            (None, (small, large)),
        ):
            with self.subTest(target=target, drafts=drafts):
                self.assertEqual(
                    self._resize(target, drafts),
                    self._resize(None) - large.per_forward_workspace_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
