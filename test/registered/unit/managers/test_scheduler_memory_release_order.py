import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSchedulerMemoryReleaseOrder(CustomTestCase):
    def test_kv_cache_is_flushed_before_its_memory_is_paused(self):
        events = []
        memory_saver = MagicMock()
        memory_saver.pause.side_effect = lambda tag: events.append(("pause", tag))

        def flush_cache():
            self.assertNotIn(GPU_MEMORY_TYPE_KV_CACHE, manager.offload_tags)
            events.append(("flush", None))
            return True

        manager = SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=memory_saver,
            flush_cache=flush_cache,
            is_fully_idle=MagicMock(return_value=True),
        )

        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.torch.get_device_module"
        ) as get_device_module:
            get_device_module.return_value.synchronize.side_effect = (
                lambda: events.append(("sync", None))
            )
            manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )

        self.assertEqual(
            events,
            [
                ("flush", None),
                ("sync", None),
                ("pause", GPU_MEMORY_TYPE_KV_CACHE),
            ],
        )
        self.assertIn(GPU_MEMORY_TYPE_KV_CACHE, manager.offload_tags)
        get_device_module.return_value.synchronize.assert_called_once_with()

    def test_static_state_copy_finishes_before_weight_memory_is_paused(self):
        events = []
        model = object()
        memory_saver = MagicMock()
        memory_saver.pause.side_effect = lambda tag: events.append(("pause", tag))
        manager = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    model=model,
                    server_args=SimpleNamespace(weight_cache_mode="off"),
                )
            ),
            draft_worker=None,
            tp_cpu_group=object(),
            memory_saver_adapter=memory_saver,
            flush_cache=MagicMock(return_value=True),
            is_fully_idle=MagicMock(return_value=True),
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater._export_static_state",
                side_effect=lambda actual_model: events.append(("copy", actual_model))
                or {"buffers": []},
            ),
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.barrier",
                side_effect=lambda group: events.append(("barrier", group)),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater.torch.get_device_module"
            ) as get_device_module,
        ):
            get_device_module.return_value.synchronize.side_effect = (
                lambda: events.append(("sync", None))
            )
            manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            )

        self.assertEqual(
            [event[0] for event in events],
            ["copy", "barrier", "sync", "pause"],
        )
        self.assertIs(events[0][1], model)
        self.assertEqual(events[-1][1], GPU_MEMORY_TYPE_WEIGHTS)

    def test_all_regions_preserve_low_memory_release_order(self):
        events = []
        model = object()
        memory_saver = MagicMock()

        def pause(tag):
            self.assertNotIn(tag, manager.offload_tags)
            events.append(("pause", tag))

        memory_saver.pause.side_effect = pause

        def flush_cache():
            self.assertNotIn(GPU_MEMORY_TYPE_KV_CACHE, manager.offload_tags)
            events.append(("flush", None))
            return True

        manager = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    model=model,
                    server_args=SimpleNamespace(weight_cache_mode="off"),
                )
            ),
            draft_worker=None,
            tp_cpu_group=object(),
            memory_saver_adapter=memory_saver,
            flush_cache=flush_cache,
            is_fully_idle=MagicMock(return_value=True),
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater._export_static_state",
                side_effect=lambda actual_model: events.append(("copy", actual_model))
                or {"buffers": []},
            ),
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.barrier",
                side_effect=lambda group: events.append(("barrier", group)),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater.torch.get_device_module"
            ) as get_device_module,
        ):
            get_device_module.return_value.synchronize.side_effect = (
                lambda: events.append(("sync", None))
            )
            manager.release_memory_occupation(ReleaseMemoryOccupationReqInput())

        self.assertEqual(
            [event[0] for event in events],
            [
                "flush",
                "sync",
                "pause",
                "copy",
                "barrier",
                "sync",
                "pause",
                "pause",
            ],
        )
        self.assertEqual(
            [event[1] for event in events if event[0] == "pause"],
            [
                GPU_MEMORY_TYPE_KV_CACHE,
                GPU_MEMORY_TYPE_WEIGHTS,
                GPU_MEMORY_TYPE_CUDA_GRAPH,
            ],
        )
        self.assertEqual(
            manager.offload_tags,
            {
                GPU_MEMORY_TYPE_KV_CACHE,
                GPU_MEMORY_TYPE_WEIGHTS,
                GPU_MEMORY_TYPE_CUDA_GRAPH,
            },
        )
        self.assertEqual(get_device_module.return_value.synchronize.call_count, 2)

    def test_cuda_graph_only_synchronizes_before_pause(self):
        events = []
        memory_saver = MagicMock()
        memory_saver.pause.side_effect = lambda tag: events.append(("pause", tag))
        manager = SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=memory_saver,
            flush_cache=MagicMock(return_value=True),
            is_fully_idle=MagicMock(return_value=True),
        )

        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.torch.get_device_module"
        ) as get_device_module:
            get_device_module.return_value.synchronize.side_effect = (
                lambda: events.append(("sync", None))
            )
            manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
            )

        self.assertEqual(
            events,
            [
                ("sync", None),
                ("pause", GPU_MEMORY_TYPE_CUDA_GRAPH),
            ],
        )
        self.assertIn(GPU_MEMORY_TYPE_CUDA_GRAPH, manager.offload_tags)

    def test_flush_cache_is_a_noop_while_kv_memory_is_released(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.weight_updater = SimpleNamespace(
            offload_tags={GPU_MEMORY_TYPE_KV_CACHE}
        )
        scheduler._engine_paused = True
        scheduler.is_fully_idle = MagicMock(return_value=True)
        scheduler.cur_batch_for_debug = object()
        scheduler.last_batch = object()
        scheduler.tree_cache = MagicMock()
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.grammar_manager = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.draft_worker = None

        self.assertTrue(scheduler.flush_cache(empty_cache=False))
        scheduler.is_fully_idle.assert_not_called()
        scheduler.tree_cache.reset.assert_not_called()
        scheduler.req_to_token_pool.clear.assert_not_called()
        scheduler.token_to_kv_pool_allocator.clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
