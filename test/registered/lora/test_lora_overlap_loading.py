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

import multiprocessing as mp
import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from torch.cuda import Event as CudaEvent
from torch.cuda import Stream as CudaStream

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader, LoRAOverlapLoadStatus
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=75, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=75, suite="stage-b-test-small-1-gpu-amd")


class TestLoRAOverlapLoading(CustomTestCase):
    def test_ci_lora_models_batch_splitting(self):
        run_lora_batch_splitting_equivalence_test(
            CI_MULTI_LORA_MODELS, enable_lora_overlap_loading=True
        )


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
        self.mock_lora_manager.validate_lora_batch.return_value = True

    def tearDown(self):
        self.torch_patcher.stop()

    def _create_loader(self) -> LoRAOverlapLoader:
        return LoRAOverlapLoader(cast(LoRAManager, self.mock_lora_manager))

    def _create_mock_event(self, query_return: bool = False) -> MagicMock:
        event = MagicMock(spec=CudaEvent)
        event.query.return_value = query_return
        return event

    def test_full_lifecycle_single_lora_load(self):
        loader = self._create_loader()

        # Initially not loaded
        status = loader._check_overlap_load_status("lora_A")
        self.assertEqual(status, LoRAOverlapLoadStatus.NOT_LOADED)

        # First call starts async load, returns False
        result = loader.try_overlap_load_lora("lora_A", running_loras=set())
        self.assertFalse(result)
        self.assertIn("lora_A", loader.lora_to_overlap_load_event)
        self.mock_lora_manager.fetch_new_loras.assert_called_once_with(
            {"lora_A"}, set()
        )

        # Simulate load still in progress - returns False, event persists
        loader.lora_to_overlap_load_event["lora_A"].query.return_value = False
        result = loader.try_overlap_load_lora("lora_A", running_loras=set())
        self.assertFalse(result)
        self.assertEqual(
            loader._check_overlap_load_status("lora_A"), LoRAOverlapLoadStatus.LOADING
        )

        # Simulate load complete - returns True, event removed
        loader.lora_to_overlap_load_event["lora_A"].query.return_value = True
        result = loader.try_overlap_load_lora("lora_A", running_loras=set())
        self.assertTrue(result)
        self.assertNotIn("lora_A", loader.lora_to_overlap_load_event)

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


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
