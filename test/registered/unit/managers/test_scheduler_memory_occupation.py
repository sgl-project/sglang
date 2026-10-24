import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MemorySaver:
    def __init__(self):
        self.resident = set(GPU_MEMORY_ALL_TYPES)

    def pause(self, tag):
        self.resident.remove(tag)

    def resume(self, tag):
        self.resident.add(tag)


class TestWeightCacheMemoryOccupation(CustomTestCase):
    def _manager(self):
        cache = ["cached KV tokens"]
        adapter = _MemorySaver()
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=None,
                draft_worker=None,
                memory_saver_adapter=adapter,
                flush_cache=lambda **kwargs: cache.clear() or True,
                is_fully_idle=lambda **kwargs: True,
            )
        return manager, adapter, cache

    def test_rejected_release_preserves_memory_and_cache(self):
        """An unsupported weight release must not first unmap KV or mark tags paused."""
        for mode in ("client", "daemon"):
            for tags in (
                None,
                [],
                [GPU_MEMORY_TYPE_WEIGHTS],
                [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS],
            ):
                with self.subTest(mode=mode, tags=tags):
                    manager, adapter, cache = self._manager()
                    with patch(
                        "sglang.srt.managers.scheduler_components.weight_updater.get_model",
                        return_value=SimpleNamespace(weight_cache_mode=mode),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "weight cache is active"
                        ):
                            manager.release_memory_occupation(
                                ReleaseMemoryOccupationReqInput(tags=tags)
                            )

                    self.assertEqual(manager.offload_tags, set())
                    self.assertEqual(adapter.resident, set(GPU_MEMORY_ALL_TYPES))
                    self.assertEqual(cache, ["cached KV tokens"])

    def test_rejected_resume_preserves_paused_regions(self):
        """A rejected resume must retain the state needed for a subsequent valid resume."""
        for mode in ("client", "daemon"):
            for tags in (
                None,
                [],
                [GPU_MEMORY_TYPE_WEIGHTS],
                [GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_WEIGHTS],
            ):
                with self.subTest(mode=mode, tags=tags):
                    manager, adapter, _ = self._manager()
                    paused = {GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH}
                    manager.offload_tags.update(paused)
                    adapter.resident.difference_update(paused)
                    with patch(
                        "sglang.srt.managers.scheduler_components.weight_updater.get_model",
                        return_value=SimpleNamespace(weight_cache_mode=mode),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "weight cache is active"
                        ):
                            manager.resume_memory_occupation(
                                ResumeMemoryOccupationReqInput(tags=tags)
                            )

                    self.assertEqual(manager.offload_tags, paused)
                    self.assertEqual(adapter.resident, {GPU_MEMORY_TYPE_WEIGHTS})
                    manager.resume_memory_occupation(
                        ResumeMemoryOccupationReqInput(tags=sorted(paused))
                    )
                    self.assertEqual(manager.offload_tags, set())
                    self.assertEqual(adapter.resident, set(GPU_MEMORY_ALL_TYPES))

    def test_weight_release_without_weight_cache_restores_static_state(self):
        """Preflight must still allow private weights and restore their saved buffers."""
        manager, adapter, _ = self._manager()
        model = torch.nn.Module()
        model.register_buffer("state", torch.tensor([7.0]))
        manager.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        with (
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            patch("torch.distributed.barrier"),
            patch("torch.get_device_module"),
        ):
            manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            )
            self.assertNotIn(GPU_MEMORY_TYPE_WEIGHTS, adapter.resident)
            self.assertEqual(manager.offload_tags, {GPU_MEMORY_TYPE_WEIGHTS})
            model.state.zero_()

            manager.resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            )

        self.assertEqual(model.state.tolist(), [7.0])
        self.assertEqual(manager.offload_tags, set())
        self.assertEqual(adapter.resident, set(GPU_MEMORY_ALL_TYPES))

    def test_kv_only_release_and_resume_remain_supported(self):
        """Shared weights must not prevent releasing and restoring private KV memory."""
        manager, adapter, cache = self._manager()
        with (
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="client"),
            ),
            patch("torch.get_device_module"),
        ):
            manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )
            self.assertEqual(cache, [])
            self.assertNotIn(GPU_MEMORY_TYPE_KV_CACHE, adapter.resident)
            self.assertEqual(manager.offload_tags, {GPU_MEMORY_TYPE_KV_CACHE})

            manager.resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )

        self.assertEqual(manager.offload_tags, set())
        self.assertEqual(adapter.resident, set(GPU_MEMORY_ALL_TYPES))


if __name__ == "__main__":
    unittest.main()
