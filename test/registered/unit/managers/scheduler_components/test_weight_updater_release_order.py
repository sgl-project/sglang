"""Check that each memory-saver unmap follows a device drain.

The weights export enqueues on-device buffer clones that must finish before
their source pages are unmapped.
"""

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
from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.utils import torch_memory_saver_adapter as adapter_module

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ALL_TAGS = [
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
]


class TestReleaseMemoryOccupationSynchronize(CustomTestCase):
    def _release(self, tags):
        events = []
        memory_saver = MagicMock(enabled=True)
        memory_saver.pause.side_effect = lambda tag: events.append(("pause", tag))
        with patch.object(
            weight_updater_module,
            "get_parallel",
            lambda: SimpleNamespace(tp_group=SimpleNamespace(cpu_group=object())),
        ):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=object())),
                draft_worker=None,
                memory_saver_adapter=adapter_module._TorchMemorySaverAdapterReal(),
                flush_cache=lambda **kwargs: events.append(("flush", None)) or True,
                is_fully_idle=lambda **kwargs: True,
                scheduler=SimpleNamespace(
                    pause_state=SimpleNamespace(paused=False),
                    request_lists=dict,
                    disaggregation_mode=weight_updater_module.DisaggregationMode.DECODE,
                    disagg_decode_prealloc_queue=SimpleNamespace(
                        release_memory_occupation=lambda: events.append(
                            ("release_queue", None)
                        ),
                    ),
                ),
            )
        with (
            patch.object(adapter_module, "_memory_saver", memory_saver, create=True),
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

    def test_release_order(self):
        sync = ("sync", None)
        barrier = ("barrier", None)
        kv = [
            ("release_queue", None),
            sync,
            ("pause", GPU_MEMORY_TYPE_KV_CACHE),
            ("flush", None),
        ]
        weights = [("export", None), barrier, sync, ("pause", GPU_MEMORY_TYPE_WEIGHTS)]
        graph = [sync, ("pause", GPU_MEMORY_TYPE_CUDA_GRAPH)]
        for tags, expected in (
            (ALL_TAGS, kv + weights + graph),
            ([GPU_MEMORY_TYPE_KV_CACHE], kv),
            ([GPU_MEMORY_TYPE_WEIGHTS], weights),
            ([GPU_MEMORY_TYPE_CUDA_GRAPH], graph),
        ):
            with self.subTest(tags=tags):
                out, events = self._release(tags)
                self.assertTrue(out.success)
                self.assertEqual(events, expected + [sync, barrier])

    def test_disabled_adapter_does_not_synchronize(self):
        adapter = adapter_module.TorchMemorySaverAdapter.create(enable=False)
        with patch.object(adapter_module.torch, "get_device_module") as get_device:
            adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)
        get_device.assert_not_called()

    def test_disabled_backend_preserves_pause_delegation(self):
        adapter = adapter_module._TorchMemorySaverAdapterReal()
        memory_saver = MagicMock(enabled=False)
        with (
            patch.object(adapter_module, "_memory_saver", memory_saver, create=True),
            patch.object(adapter_module.torch, "get_device_module") as get_device,
        ):
            adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)
        get_device.assert_not_called()
        memory_saver.pause.assert_called_once_with(tag=GPU_MEMORY_TYPE_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
