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

import torch
from torch.cuda import Event as CudaEvent
from torch.cuda import Stream as CudaStream

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader, LoRAOverlapLoadStatus
from sglang.srt.lora.lora_pipeline_sync import LoRAPipelineFlag
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    TORCH_DTYPES,
    LoRAAdaptor,
    LoRAModelCase,
    run_lora_test_one_by_one,
)
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")

# Two adapters on a freely available base model
LORA_A = "algoprog/fact-generation-llama-3.1-8b-instruct-lora"
LORA_B = "nvidia/llama-3.1-nemoguard-8b-topic-control"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_1 = "AI is a field of computer science focused on"
PROMPT_2 = "The capital of France is"


class TestLoRAOverlapLoadingSingleRequest(CustomTestCase):
    """1. Single request, single LoRA loading with max 1 LoRA in GPU."""

    def test_single_request_single_lora(self):
        model_case = LoRAModelCase(
            base=BASE_MODEL,
            adaptors=[LoRAAdaptor(name=LORA_A)],
            max_loras_per_batch=1,
            max_loaded_loras=1,
        )
        for torch_dtype in TORCH_DTYPES:
            run_lora_test_one_by_one(
                [PROMPT_1],
                model_case,
                torch_dtype,
                max_new_tokens=32,
                enable_lora_overlap_loading=True,
                disable_cuda_graph=True,
                disable_radix_cache=True,
                test_tag="overlap_single_request_single_lora",
            )


class TestLoRAOverlapLoadingBatchReplace(CustomTestCase):
    """2. Two new LoRAs replace two in GPU, batch runs with correct output."""

    def test_two_loras_batch_replace(self):
        with SRTRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=[LORA_A, LORA_B],
            enable_lora_overlap_loading=True,
            max_loras_per_batch=2,
            max_loaded_loras=2,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            sleep_on_idle=True,
        ) as srt_runner:
            # Batch with both LoRAs
            outputs = srt_runner.batch_forward(
                [PROMPT_1, PROMPT_2],
                max_new_tokens=32,
                lora_paths=[LORA_A, LORA_B],
            )
            # Different adapters should produce different outputs
            self.assertNotEqual(
                outputs.output_strs[0].strip(),
                outputs.output_strs[1].strip(),
                "Two different LoRA adapters produced identical output",
            )
            # Both should produce non-empty output
            self.assertTrue(len(outputs.output_strs[0].strip()) > 0)
            self.assertTrue(len(outputs.output_strs[1].strip()) > 0)


class TestLoRAOverlapLoadingEviction(CustomTestCase):
    """3. Two requests: one LoRA already in GPU, one needs loading (eviction).
    Max 2 LoRAs means one must be replaced. Verify correct output."""

    def test_eviction_on_new_lora(self):
        with SRTRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=[LORA_A, LORA_B],
            enable_lora_overlap_loading=True,
            max_loras_per_batch=2,
            max_loaded_loras=2,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            sleep_on_idle=True,
        ) as srt_runner:
            # First batch: load LORA_A into GPU
            out1 = srt_runner.batch_forward(
                [PROMPT_1],
                max_new_tokens=32,
                lora_paths=[LORA_A],
            )
            self.assertTrue(len(out1.output_strs[0].strip()) > 0)

            # Second batch: LORA_A already loaded, LORA_B needs loading
            out2 = srt_runner.batch_forward(
                [PROMPT_1, PROMPT_2],
                max_new_tokens=32,
                lora_paths=[LORA_A, LORA_B],
            )
            # Both should produce non-empty, different outputs
            self.assertTrue(len(out2.output_strs[0].strip()) > 0)
            self.assertTrue(len(out2.output_strs[1].strip()) > 0)
            self.assertNotEqual(
                out2.output_strs[0].strip(),
                out2.output_strs[1].strip(),
                "Different adapters should produce different outputs",
            )

            # Verify LORA_A output is consistent across batches
            self.assertEqual(
                out1.output_strs[0].strip(),
                out2.output_strs[0].strip(),
                "Same adapter + prompt should produce consistent output",
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
        self.mock_lora_manager.memory_pool = MagicMock()
        self.mock_lora_manager.memory_pool.uid_to_buffer_id = {}
        self.mock_lora_manager.validate_lora_batch.return_value = True
        self.mock_lora_manager.fetch_new_loras.side_effect = self._mark_loras_loaded

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


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
