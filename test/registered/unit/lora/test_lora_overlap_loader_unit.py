# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import torch
from torch.cuda import Event as CudaEvent
from torch.cuda import Stream as CudaStream

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader, LoRAOverlapLoadStatus
from sglang.srt.lora.lora_pipeline_sync import LoRAPipelineFlag
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


class TestLoRAOverlapLoaderUnitTests(CustomTestCase):

    mock_lora_manager: MagicMock
    mock_stream: MagicMock
    mock_stream_context: MagicMock
    mock_device_module: MagicMock
    mock_torch: MagicMock

    def setUp(self):
        self.torch_patcher = patch("sglang.srt.lora.lora_overlap_loader.torch")
        self.mock_torch = self.torch_patcher.start()

        self.mock_device_module = MagicMock()
        self.mock_stream = MagicMock(spec=CudaStream)
        self.mock_stream_context = MagicMock()
        self.mock_event = MagicMock(spec=CudaEvent)

        self.mock_device_module.Stream.return_value = self.mock_stream
        self.mock_device_module.stream.return_value = self.mock_stream_context
        self.mock_device_module.Event.return_value = self.mock_event
        self.mock_torch.get_device_module.return_value = self.mock_device_module
        self.mock_torch.cuda.current_stream.return_value = MagicMock(spec=CudaStream)

        self.mock_lora_manager = MagicMock(spec=LoRAManager)
        self.mock_lora_manager.device = "cuda:0"
        self.mock_lora_manager.memory_pool = MagicMock()
        self.mock_lora_manager.memory_pool.uid_to_buffer_id = {}
        self.mock_lora_manager.validate_lora_batch.return_value = True
        self.mock_lora_manager.fetch_new_loras.side_effect = self._mark_loras_loaded
        self.mock_lora_manager.pending_lora_load_events = {}

    def tearDown(self):
        self.torch_patcher.stop()

    def _create_loader(self) -> LoRAOverlapLoader:
        return LoRAOverlapLoader(cast(LoRAManager, self.mock_lora_manager))

    def _mark_loras_loaded(self, new_loras, _loras_to_be_loaded, **kwargs):
        for lora_id in new_loras:
            self.mock_lora_manager.memory_pool.uid_to_buffer_id[lora_id] = len(
                self.mock_lora_manager.memory_pool.uid_to_buffer_id
            )

    def _create_mock_event(self, query_return: bool = False) -> MagicMock:
        event = MagicMock(spec=CudaEvent)
        event.query.return_value = query_return
        return event

    def test_completed_stale_loads_are_reaped_before_capacity_check(self):
        loader = self._create_loader()
        events = [
            self._create_mock_event(query_return=True),
            self._create_mock_event(query_return=False),
        ]
        self.mock_device_module.Event.side_effect = events
        self.mock_lora_manager.validate_lora_batch.side_effect = (
            lambda lora_ids: len(lora_ids) <= 1
        )

        self.assertTrue(
            loader._try_start_overlap_load("stale_lora", running_loras=set())
        )
        self.assertIn("stale_lora", loader.lora_to_overlap_load_event)

        self.mock_lora_manager.fetch_new_loras.reset_mock()
        result = loader.try_overlap_load_lora("new_lora", running_loras=set())

        # With pipelining, returns True as soon as load starts
        self.assertTrue(result)
        self.assertNotIn("stale_lora", loader.lora_to_overlap_load_event)
        self.assertIn("new_lora", loader.lora_to_overlap_load_event)
        self.assertIn("new_lora", loader.pipelined_loading_loras)
        self.mock_lora_manager.fetch_new_loras.assert_called_once_with(
            {"new_lora"}, set(), loading_stream=self.mock_stream
        )

    def test_loaded_lora_reused_after_stale_event_drain(self):
        loader = self._create_loader()
        self.mock_lora_manager.memory_pool = MagicMock()
        self.mock_lora_manager.memory_pool.uid_to_buffer_id = {}
        events = [
            self._create_mock_event(query_return=True),
            self._create_mock_event(query_return=False),
        ]
        self.mock_device_module.Event.side_effect = events
        self.mock_lora_manager.validate_lora_batch.side_effect = (
            lambda lora_ids: len(lora_ids) <= 2
        )

        self.assertTrue(loader._try_start_overlap_load("lora_A", running_loras=set()))
        self.assertIn("lora_A", loader.lora_to_overlap_load_event)

        self.mock_lora_manager.fetch_new_loras.reset_mock()
        # With pipelining, returns True as soon as load starts
        self.assertTrue(loader.try_overlap_load_lora("lora_B", running_loras=set()))
        self.assertNotIn("lora_A", loader.lora_to_overlap_load_event)
        self.assertIn("lora_B", loader.lora_to_overlap_load_event)
        self.assertIn("lora_B", loader.pipelined_loading_loras)
        self.mock_lora_manager.fetch_new_loras.assert_called_once_with(
            {"lora_B"}, set(), loading_stream=self.mock_stream
        )

        self.mock_lora_manager.fetch_new_loras.reset_mock()
        self.assertTrue(loader.try_overlap_load_lora("lora_A", running_loras=set()))
        self.assertIn("lora_B", loader.lora_to_overlap_load_event)
        self.mock_lora_manager.fetch_new_loras.assert_not_called()

    def test_pending_lora_load_must_complete_even_if_memory_pool_has_slot(self):
        loader = self._create_loader()
        self.mock_lora_manager.memory_pool = MagicMock()
        self.mock_lora_manager.memory_pool.uid_to_buffer_id = {"lora_A": 0}

        loader.lora_to_overlap_load_event["lora_A"] = self._create_mock_event(False)

        result = loader.try_overlap_load_lora("lora_A", running_loras=set())

        self.assertFalse(result)
        self.mock_lora_manager.fetch_new_loras.assert_not_called()
        self.assertIn("lora_A", loader.lora_to_overlap_load_event)

    def test_loader_uses_manager_pending_event_store(self):
        loader = self._create_loader()

        self.assertIs(
            loader.lora_to_overlap_load_event,
            self.mock_lora_manager.pending_lora_load_events,
        )

    def test_pending_load_is_synchronized_before_unload(self):
        manager = LoRAManager.__new__(LoRAManager)
        manager.device = torch.device("cuda:0")
        manager.pending_lora_load_events = {}
        manager.memory_pool = MagicMock()
        manager.configs = {"lora_A": object()}
        manager.loras = {"lora_A": object()}
        lora_ref = MagicMock()
        lora_ref.lora_id = "lora_A"
        lora_ref.lora_name = "lora_A"
        lora_ref.lora_path = "/tmp/lora_A"
        lora_ref.pinned = False
        manager.lora_refs = {"lora_A": lora_ref}
        manager.num_pinned_loras = 0
        manager.lora_modules = []

        order = []
        event = self._create_mock_event(False)
        event.synchronize.side_effect = lambda: order.append("synchronize")
        manager.memory_pool.remove_lora.side_effect = lambda _uid: (
            order.append("remove") or 0
        )
        loader = LoRAOverlapLoader(manager)
        loader.lora_to_overlap_load_event["lora_A"] = event

        result = manager.unload_lora_adapter(lora_ref)

        self.assertTrue(result.success)
        self.assertEqual(order, ["synchronize", "remove"])
        event.synchronize.assert_called_once_with()
        self.assertNotIn("lora_A", manager.pending_lora_load_events)

    def test_full_lifecycle_single_lora_load(self):
        loader = self._create_loader()

        # Initially not loaded
        status = loader._check_overlap_load_status("lora_A")
        self.assertEqual(status, LoRAOverlapLoadStatus.NOT_LOADED)

        # First call starts async load, returns True immediately (pipelined)
        result = loader.try_overlap_load_lora("lora_A", running_loras=set())
        self.assertTrue(result)
        self.assertIn("lora_A", loader.lora_to_overlap_load_event)
        self.assertIn("lora_A", loader.pipelined_loading_loras)
        self.mock_lora_manager.fetch_new_loras.assert_called_once_with(
            {"lora_A"}, set(), loading_stream=self.mock_stream
        )

        # While pipelined and still in memory pool, returns True (forward gates on per-layer flags)
        result = loader.try_overlap_load_lora("lora_A", running_loras=set())
        self.assertTrue(result)

        # Even after event completes, pipelined path keeps returning True
        # as long as adapter remains in memory pool
        loader.lora_to_overlap_load_event["lora_A"].query.return_value = True
        result = loader.try_overlap_load_lora("lora_A", running_loras=set())
        self.assertTrue(result)

    def test_capacity_constraints_block_new_loads(self):
        loader = self._create_loader()

        events = [self._create_mock_event() for _ in range(4)]
        self.mock_device_module.Event.side_effect = events

        # Load 3 loras successfully
        for i in range(3):
            self.assertTrue(
                loader._try_start_overlap_load(f"lora_{i}", running_loras=set())
            )
        self.assertEqual(len(loader.lora_to_overlap_load_event), 3)

        # Capacity full - new load blocked
        self.mock_lora_manager.validate_lora_batch.return_value = False
        self.mock_lora_manager.fetch_new_loras.reset_mock()
        result = loader.try_overlap_load_lora("lora_3", running_loras=set())
        self.assertFalse(result)
        self.mock_lora_manager.fetch_new_loras.assert_not_called()
        self.assertNotIn("lora_3", loader.lora_to_overlap_load_event)

        # First lora completes, freeing capacity
        loader.lora_to_overlap_load_event["lora_0"].query.return_value = True

        loader._drain_completed_overlap_loads()
        self.assertEqual(
            loader._check_overlap_load_status("lora_0"), LoRAOverlapLoadStatus.LOADED
        )

        # Now new load succeeds
        self.mock_lora_manager.validate_lora_batch.return_value = True
        self.assertTrue(loader._try_start_overlap_load("lora_3", running_loras=set()))

    def test_validation_includes_pending_and_running_loras(self):
        loader = self._create_loader()

        events = [self._create_mock_event() for _ in range(5)]
        self.mock_device_module.Event.side_effect = events

        # Start pending loads
        loader._try_start_overlap_load("pending_1", running_loras=set())
        loader._try_start_overlap_load("pending_2", running_loras=set())

        # Load new lora with running_loras
        self.mock_lora_manager.validate_lora_batch.reset_mock()
        running = {"running_1", "running_2"}
        loader.try_overlap_load_lora("new_lora", running_loras=running)

        # Validation should include: pending + running + new
        call_args = self.mock_lora_manager.validate_lora_batch.call_args[0][0]
        expected = {"pending_1", "pending_2", "running_1", "running_2", "new_lora"}
        self.assertEqual(call_args, expected)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestLoRAPipelineSync(CustomTestCase):
    """Tests for per-layer pipelined LoRA loading synchronization primitives."""

    def test_flag_initial_state(self):
        """Flag starts ready (no load in progress)."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        flag.wait_until_ready(torch.cuda.current_stream())

    def test_mark_loading_and_ready(self):
        """Test basic mark_loading -> mark_ready cycle."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        stream = torch.cuda.Stream()

        flag.mark_loading()
        with torch.cuda.stream(stream):
            t = torch.zeros(256, 256, device="cuda:0")
            t.fill_(1.0)
        flag.mark_ready(stream)

        flag.wait_until_ready(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        self.assertEqual(t[0, 0].item(), 1.0)

    def test_cross_stream_synchronization(self):
        """Compute stream waits for loading stream to finish."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        load_stream = torch.cuda.Stream()

        flag.mark_loading()
        with torch.cuda.stream(load_stream):
            large_tensor = torch.zeros(1024, 1024, device="cuda:0")
            large_tensor.fill_(42.0)
        flag.mark_ready(load_stream)

        flag.wait_until_ready(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        self.assertEqual(large_tensor[0, 0].item(), 42.0)

    def test_no_wait_when_not_loading(self):
        """wait_until_ready is a no-op when no load is in progress."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        flag.wait_until_ready(torch.cuda.current_stream())
        flag.wait_until_ready(torch.cuda.current_stream())

    def test_multiple_flags_independent(self):
        """Multiple flags operate independently."""
        flag1 = LoRAPipelineFlag(torch.device("cuda:0"))
        flag2 = LoRAPipelineFlag(torch.device("cuda:0"))
        stream = torch.cuda.Stream()

        flag1.mark_loading()
        flag2.wait_until_ready(torch.cuda.current_stream())

        flag1.mark_ready(stream)
        stream.synchronize()

    def test_mark_ready_does_not_clear_pending_wait(self):
        """Regression guard: recording the completion event must NOT satisfy the
        pending wait."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        load_stream = torch.cuda.Stream()

        # Fresh flag owes no wait.
        self.assertFalse(flag.needs_wait)

        # A load starts -> a wait is owed.
        flag.mark_loading()
        self.assertTrue(flag.needs_wait)

        # Recording completion must NOT clear the owed wait.
        flag.mark_ready(load_stream)
        self.assertTrue(flag.needs_wait)

        # Only the compute stream consuming the wait clears it, and it is a no-op
        # thereafter (e.g. a subsequent decode step).
        flag.wait_until_ready(torch.cuda.current_stream())
        self.assertFalse(flag.needs_wait)
        flag.wait_until_ready(torch.cuda.current_stream())
        self.assertFalse(flag.needs_wait)

    def test_pending_wait_re_arms_on_each_load(self):
        """A per-layer flag is reused across successive adapter loads; each new
        load must re-arm its own pending wait.
        """
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        load_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.current_stream()

        # First load cycle.
        flag.mark_loading()
        flag.mark_ready(load_stream)
        flag.wait_until_ready(compute_stream)
        self.assertFalse(flag.needs_wait)

        # Second load on the same flag must arm a fresh wait.
        flag.mark_loading()
        self.assertTrue(flag.needs_wait)
        flag.mark_ready(load_stream)
        self.assertTrue(flag.needs_wait)
        flag.wait_until_ready(compute_stream)
        self.assertFalse(flag.needs_wait)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
