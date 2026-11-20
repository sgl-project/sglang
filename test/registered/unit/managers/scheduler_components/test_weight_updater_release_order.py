"""``release_memory_occupation`` must drain the device before each memory-saver
pause: pause() unmaps without waiting, so in-flight kernels (kv_cache) or the
async D2H clones from ``_export_static_state`` (weights) would fault or corrupt
once their pages are released. The cases assert the recorded call order."""

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
from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ALL_TAGS = [
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
]


class TestReleaseMemoryOccupationSynchronize(CustomTestCase):
    def _release(self, tags):
        events = []
        memory_saver = MagicMock()
        memory_saver.pause.side_effect = lambda tag: events.append(("pause", tag))
        with patch.object(
            weight_updater_module,
            "get_parallel",
            lambda: SimpleNamespace(tp_group=SimpleNamespace(cpu_group=object())),
        ):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=object())),
                draft_worker=None,
                memory_saver_adapter=memory_saver,
                flush_cache=lambda **kwargs: events.append(("flush", None)) or True,
                is_fully_idle=lambda **kwargs: True,
            )
        with (
            patch.object(
                weight_updater_module,
                "_export_static_state",
                lambda model: events.append(("export", None)),
            ),
            patch.object(
                weight_updater_module,
                "get_model",
                lambda: SimpleNamespace(weight_cache_mode="off"),
            ),
            patch("torch.distributed.all_reduce", lambda *args, **kwargs: None),
            patch(
                "torch.distributed.barrier",
                lambda *args, **kwargs: events.append(("barrier", None)),
            ),
            patch.object(
                weight_updater_module.torch,
                "get_device_module",
            ) as get_device_module,
        ):
            get_device_module.return_value.synchronize.side_effect = lambda: (
                events.append(("sync", None))
            )
            out = manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=tags)
            )
        return out, events

    def test_full_release_order(self):
        out, events = self._release(ALL_TAGS)

        self.assertTrue(out.success)
        self.assertEqual(
            events,
            [
                ("sync", None),
                ("pause", GPU_MEMORY_TYPE_KV_CACHE),
                ("flush", None),
                ("export", None),
                ("sync", None),
                ("barrier", None),
                ("pause", GPU_MEMORY_TYPE_WEIGHTS),
                ("pause", GPU_MEMORY_TYPE_CUDA_GRAPH),
                ("sync", None),
                ("barrier", None),
            ],
        )

    def test_every_pause_is_preceded_by_a_sync(self):
        _, events = self._release(ALL_TAGS)

        last_sync = -1
        for index, (kind, payload) in enumerate(events):
            if kind == "sync":
                last_sync = index
            elif kind == "pause":
                self.assertGreater(
                    last_sync, -1, f"pause({payload}) happened before any sync"
                )
