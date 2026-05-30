# Copyright 2024 SGLang Team
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
"""Integration tests for spec worker implementations and scheduler compatibility."""

import unittest

from sglang.srt.managers.tp_worker import BaseTpWorker
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker


class TestSpecWorkerImplementations(unittest.TestCase):
    """Verify all spec worker implementations properly inherit from BaseSpecWorker."""

    def get_spec_worker_classes(self):
        """Collect all BaseSpecWorker subclasses."""
        # Import known spec worker classes
        try:
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            spec_workers = [EAGLEWorkerV2]
        except ImportError:
            spec_workers = []

        try:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            spec_workers.append(MultiLayerEagleWorkerV2)
        except ImportError:
            pass

        try:
            from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2

            spec_workers.append(StandaloneWorkerV2)
        except ImportError:
            pass

        return spec_workers

    def test_spec_workers_inherit_from_base_spec_worker(self):
        """Verify spec workers inherit from BaseSpecWorker."""
        spec_workers = self.get_spec_worker_classes()
        for worker_class in spec_workers:
            self.assertTrue(
                issubclass(worker_class, BaseSpecWorker),
                f"{worker_class.__name__} does not inherit from BaseSpecWorker",
            )

    def test_spec_workers_inherit_from_base_tp_worker(self):
        """Verify spec workers (via BaseSpecWorker) inherit from BaseTpWorker."""
        spec_workers = self.get_spec_worker_classes()
        for worker_class in spec_workers:
            self.assertTrue(
                issubclass(worker_class, BaseTpWorker),
                f"{worker_class.__name__} does not inherit from BaseTpWorker via BaseSpecWorker",
            )

    def test_spec_workers_implement_required_methods(self):
        """Verify spec workers implement or properly inherit required methods."""
        spec_workers = self.get_spec_worker_classes()

        required_methods = {
            "forward_batch_generation",
            "model_runner",
            "target_worker",
            "draft_worker",
            "spec_v2_attn_backends",
            "clear_cache_pool",
            "check_weights",
            "save_remote_model",
            "save_sharded_model",
        }

        for worker_class in spec_workers:
            for method_name in required_methods:
                self.assertTrue(
                    hasattr(worker_class, method_name),
                    f"{worker_class.__name__} missing {method_name}",
                )


class TestSchedulerOverlapAssumptions(unittest.TestCase):
    """Test that scheduler overlap logic works with unified API."""

    def test_scheduler_can_use_unified_attn_backends_check(self):
        """Verify scheduler's hasattr check works reliably with unified API."""
        # Simulate the scheduler's unified check pattern
        from unittest.mock import MagicMock

        from sglang.srt.speculative.base_spec_worker import (
            BaseSpecWorker as SpecWorkerType,
        )

        mock_target = MagicMock()
        mock_target.model_runner = MagicMock()
        mock_target.model_runner.attn_backend = "flashinfer"

        # Create a minimal spec worker implementation
        class TestWorker(SpecWorkerType):
            def __init__(self):
                self._target = mock_target
                self._draft = MagicMock()

            @property
            def target_worker(self):
                return self._target

            @property
            def draft_worker(self):
                return self._draft

            def forward_batch_generation(self, batch):
                pass

            def clear_cache_pool(self):
                pass

        worker = TestWorker()

        # Scheduler's new unified check pattern (from init_overlap)
        if hasattr(worker, "spec_v2_attn_backends"):
            attn_backends = worker.spec_v2_attn_backends
        else:
            attn_backends = (mock_target.model_runner.attn_backend,)

        # Verify we get the expected backends
        self.assertEqual(len(attn_backends), 1)
        self.assertEqual(attn_backends[0], "flashinfer")

    def test_scheduler_model_worker_abstraction(self):
        """Verify scheduler can treat model_worker uniformly (TpModelWorker or spec worker)."""
        from unittest.mock import MagicMock

        from sglang.srt.speculative.base_spec_worker import (
            BaseSpecWorker as SpecWorkerType,
        )

        # Test with a mock spec worker
        mock_target = MagicMock()
        mock_target.model_runner = MagicMock()
        mock_target.model_runner.attn_backend = "triton"
        mock_target.device = "cuda:0"
        mock_target.model_config = MagicMock()
        mock_target.tokenizer = MagicMock()
        mock_target.processor = MagicMock()
        mock_target.get_pad_input_ids_func.return_value = lambda x: x

        class UnifiedWorker(SpecWorkerType):
            def __init__(self):
                self._target = mock_target
                self._draft = MagicMock()

            @property
            def target_worker(self):
                return self._target

            @property
            def draft_worker(self):
                return self._draft

            def forward_batch_generation(self, batch):
                return MagicMock()

            def clear_cache_pool(self):
                pass

        worker = UnifiedWorker()

        # Simulate scheduler operations on model_worker
        _ = worker.model_runner
        _ = worker.model_config
        _ = worker.tokenizer
        _ = worker.processor
        _ = worker.device
        _ = worker.get_pad_input_ids_func()
        _ = worker.check_weights("summary")

        # Should not raise any errors
        self.assertIsNotNone(worker)


if __name__ == "__main__":
    unittest.main()
