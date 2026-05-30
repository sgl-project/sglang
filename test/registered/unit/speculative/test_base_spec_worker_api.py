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
"""Tests for BaseSpecWorker API compatibility with TpModelWorker."""

import inspect
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.tp_worker import BaseTpWorker
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker


class TestBaseSpecWorkerInheritance(unittest.TestCase):
    """Test that BaseSpecWorker properly inherits from BaseTpWorker."""

    def test_base_spec_worker_inherits_from_base_tp_worker(self):
        """Verify BaseSpecWorker is a subclass of BaseTpWorker."""
        self.assertTrue(issubclass(BaseSpecWorker, BaseTpWorker))

    def test_base_spec_worker_implements_abstract_methods(self):
        """Verify BaseSpecWorker implements or remains abstract for BaseTpWorker methods."""
        abstract_methods = {
            name
            for name, method in inspect.getmembers(BaseTpWorker)
            if getattr(method, "__isabstractmethod__", False)
        }
        # forward_batch_generation and model_runner are abstract in BaseTpWorker
        # BaseSpecWorker must implement or remain abstract for these
        self.assertIn("forward_batch_generation", abstract_methods)
        self.assertIn("model_runner", abstract_methods)

    def test_base_spec_worker_has_spec_specific_methods(self):
        """Verify BaseSpecWorker retains spec-specific hooks."""
        methods = {
            "spec_v2_attn_backends",
            "clear_cache_pool",
            "on_verify_complete_cpu",
            "target_worker",
            "draft_worker",
        }
        for method in methods:
            self.assertTrue(
                hasattr(BaseSpecWorker, method),
                f"BaseSpecWorker missing spec-specific method: {method}",
            )


class MockSpecWorker(BaseSpecWorker):
    """Mock spec worker for testing."""

    def __init__(self, target_worker, draft_worker):
        self._target_worker = target_worker
        self._draft_worker = draft_worker

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def forward_batch_generation(self, batch):
        # Delegate to target for this test
        return self.target_worker.forward_batch_generation(batch)

    def clear_cache_pool(self):
        pass


class MockTpWorker(BaseTpWorker):
    """Mock TP worker for testing BaseTpWorker defaults."""

    def __init__(self, model_runner, target_worker=None):
        self._model_runner = model_runner
        if target_worker is not None:
            self.target_worker = target_worker

    @property
    def model_runner(self):
        return self._model_runner

    def forward_batch_generation(self, batch):
        pass


class TestBaseTpWorkerDefaults(unittest.TestCase):
    """Test default worker APIs shared by TP and legacy spec workers."""

    def test_check_and_save_default_to_model_runner(self):
        model_runner = MagicMock()
        worker = MockTpWorker(model_runner)
        params = {
            "url": "target://model",
            "path": "/tmp/model",
            "pattern": "*",
            "max_size": "5GB",
        }

        worker.check_weights("summary")
        model_runner.check_weights.assert_called_once_with(action="summary")

        worker.save_remote_model(params)
        model_runner.save_remote_model.assert_called_once_with("target://model")

        worker.save_sharded_model(params)
        model_runner.save_sharded_model.assert_called_once_with(
            path="/tmp/model",
            pattern="*",
            max_size="5GB",
        )

    def test_check_and_save_preserve_legacy_spec_target_semantics(self):
        target_worker = MagicMock()
        draft_runner = MagicMock()
        worker = MockTpWorker(draft_runner, target_worker=target_worker)
        params = {"url": "target://model", "draft_url": "draft://model"}

        worker.check_weights("summary")
        target_worker.check_weights.assert_called_once_with("summary")

        worker.save_remote_model(params)
        target_worker.save_remote_model.assert_called_once_with(params)
        draft_runner.save_remote_model.assert_called_once_with("draft://model")


class TestBaseSpecWorkerDelegation(unittest.TestCase):
    """Test that BaseSpecWorker properly delegates to target_worker."""

    def setUp(self):
        """Set up mocks for target and draft workers."""
        self.mock_target = MagicMock()
        self.mock_target.model_runner = MagicMock()
        self.mock_target.model_runner.attn_backend = "flashinfer"
        self.mock_draft = MagicMock()

        self.spec_worker = MockSpecWorker(self.mock_target, self.mock_draft)

    def test_model_runner_delegation(self):
        """Verify model_runner delegates to target."""
        result = self.spec_worker.model_runner
        self.assertEqual(result, self.mock_target.model_runner)

    def test_spec_v2_attn_backends_delegation(self):
        """Verify spec_v2_attn_backends returns target backend by default."""
        backends = self.spec_worker.spec_v2_attn_backends
        self.assertEqual(backends, (self.mock_target.model_runner.attn_backend,))

    def test_device_delegation(self):
        """Verify device property delegates to target."""
        self.mock_target.device = "cuda:0"
        self.assertEqual(self.spec_worker.device, "cuda:0")

    def test_tp_rank_delegation(self):
        """Verify tp_rank property delegates to target."""
        self.mock_target.tp_rank = 1
        self.assertEqual(self.spec_worker.tp_rank, 1)

    def test_pp_rank_delegation(self):
        """Verify pp_rank property delegates to target."""
        self.mock_target.pp_rank = 0
        self.assertEqual(self.spec_worker.pp_rank, 0)

    def test_dp_rank_delegation(self):
        """Verify dp_rank property delegates to target."""
        self.mock_target.dp_rank = None
        self.assertEqual(self.spec_worker.dp_rank, None)

    def test_gpu_id_delegation(self):
        """Verify gpu_id property delegates to target."""
        self.mock_target.gpu_id = 2
        self.assertEqual(self.spec_worker.gpu_id, 2)

    def test_capacity_properties_delegation(self):
        """Verify capacity properties delegate to target."""
        self.mock_target.max_total_num_tokens = 16384
        self.mock_target.max_prefill_tokens = 8192
        self.mock_target.max_running_requests = 32
        self.mock_target.max_queued_requests = 128
        self.mock_target.max_req_len = 4096
        self.mock_target.max_req_input_len = 4091
        self.mock_target.random_seed = 42

        self.assertEqual(self.spec_worker.max_total_num_tokens, 16384)
        self.assertEqual(self.spec_worker.max_prefill_tokens, 8192)
        self.assertEqual(self.spec_worker.max_running_requests, 32)
        self.assertEqual(self.spec_worker.max_queued_requests, 128)
        self.assertEqual(self.spec_worker.max_req_len, 4096)
        self.assertEqual(self.spec_worker.max_req_input_len, 4091)
        self.assertEqual(self.spec_worker.random_seed, 42)

    def test_sliding_window_delegation(self):
        """Verify sliding_window_size delegates to target."""
        self.mock_target.sliding_window_size = 1024
        self.assertEqual(self.spec_worker.sliding_window_size, 1024)

    def test_is_hybrid_swa_delegation(self):
        """Verify is_hybrid_swa delegates to target."""
        self.mock_target.is_hybrid_swa = True
        self.assertTrue(self.spec_worker.is_hybrid_swa)

    def test_get_worker_info_delegation(self):
        """Verify get_worker_info delegates to target."""
        expected_info = (
            16384,
            8192,
            32,
            128,
            4096,
            4091,
            42,
            "cuda:0",
            None,
            512,
            1024,
            512,
        )
        self.mock_target.get_worker_info.return_value = expected_info
        result = self.spec_worker.get_worker_info()
        self.assertEqual(result, expected_info)
        self.mock_target.get_worker_info.assert_called_once()

    def test_get_pad_input_ids_func_delegation(self):
        """Verify get_pad_input_ids_func delegates to target."""
        mock_func = MagicMock()
        self.mock_target.get_pad_input_ids_func.return_value = mock_func
        result = self.spec_worker.get_pad_input_ids_func()
        self.assertEqual(result, mock_func)
        self.mock_target.get_pad_input_ids_func.assert_called_once()

    def test_get_memory_pool_delegation(self):
        """Verify get_memory_pool delegates to target."""
        mock_pool = (MagicMock(), MagicMock())
        self.mock_target.get_memory_pool.return_value = mock_pool
        result = self.spec_worker.get_memory_pool()
        self.assertEqual(result, mock_pool)
        self.mock_target.get_memory_pool.assert_called_once()

    def test_forward_batch_embedding_delegation(self):
        """Verify forward_batch_embedding delegates to target."""
        mock_batch = MagicMock()
        mock_result = MagicMock()
        self.mock_target.forward_batch_embedding.return_value = mock_result
        result = self.spec_worker.forward_batch_embedding(mock_batch)
        self.assertEqual(result, mock_result)
        self.mock_target.forward_batch_embedding.assert_called_once_with(mock_batch)

    def test_get_tokens_per_layer_info_delegation(self):
        """Verify get_tokens_per_layer_info delegates to target."""
        expected_info = (16384, 8192)
        self.mock_target.get_tokens_per_layer_info.return_value = expected_info
        result = self.spec_worker.get_tokens_per_layer_info()
        self.assertEqual(result, expected_info)
        self.mock_target.get_tokens_per_layer_info.assert_called_once()

    def test_weight_update_delegation(self):
        """Verify weight update methods delegate to target."""
        mock_req = MagicMock()
        self.mock_target.update_weights_from_disk.return_value = (True, "OK")

        result = self.spec_worker.update_weights_from_disk(mock_req)
        self.assertEqual(result, (True, "OK"))
        self.mock_target.update_weights_from_disk.assert_called_once_with(mock_req)

    def test_lora_adapter_delegation(self):
        """Verify LoRA adapter methods delegate to target."""
        mock_req = MagicMock()
        self.mock_target.load_lora_adapter.return_value = (True, "Loaded")
        self.mock_target.load_lora_adapter_from_tensors.return_value = (True, "Loaded")
        self.mock_target.unload_lora_adapter.return_value = (True, "Unloaded")

        result = self.spec_worker.load_lora_adapter(mock_req)
        self.assertEqual(result, (True, "Loaded"))
        self.mock_target.load_lora_adapter.assert_called_once_with(mock_req)

        result = self.spec_worker.load_lora_adapter_from_tensors(mock_req)
        self.assertEqual(result, (True, "Loaded"))
        self.mock_target.load_lora_adapter_from_tensors.assert_called_once_with(
            mock_req
        )

        result = self.spec_worker.unload_lora_adapter(mock_req)
        self.assertEqual(result, (True, "Unloaded"))
        self.mock_target.unload_lora_adapter.assert_called_once_with(mock_req)

    def test_scheduler_field_delegation(self):
        """Verify scheduler-facing fields delegate through the active worker API."""
        self.mock_target.model_config = MagicMock()
        self.mock_target.tokenizer = MagicMock()
        self.mock_target.processor = MagicMock()

        self.assertEqual(self.spec_worker.model_config, self.mock_target.model_config)
        self.assertEqual(self.spec_worker.tokenizer, self.mock_target.tokenizer)
        self.assertEqual(self.spec_worker.processor, self.mock_target.processor)

    def test_check_and_save_api_delegation(self):
        """Verify check/save APIs are available through BaseSpecWorker."""
        params = {
            "url": "target://model",
            "draft_url": "draft://model",
            "path": "/tmp/model",
            "pattern": "*.safetensors",
            "max_size": "5GB",
        }
        draft_runner = MagicMock()
        self.mock_draft.draft_runner = draft_runner
        self.mock_target.check_weights.return_value = {"ok": True}

        self.assertEqual(self.spec_worker.check_weights("summary"), {"ok": True})
        self.mock_target.check_weights.assert_called_once_with("summary")

        self.spec_worker.save_remote_model(params)
        self.mock_target.save_remote_model.assert_called_once_with(params)
        draft_runner.save_remote_model.assert_called_once_with("draft://model")

        self.spec_worker.save_sharded_model(params)
        self.mock_target.save_sharded_model.assert_called_once_with(params)

    def test_save_remote_model_supports_draft_runner_list(self):
        """Verify multi-layer draft runners are saved through the unified API."""
        runner_a = MagicMock()
        runner_b = MagicMock()
        self.mock_draft.draft_runner_list = [runner_a, runner_b]

        self.spec_worker.save_remote_model(
            {"url": "target://model", "draft_url": "draft://model"}
        )

        runner_a.save_remote_model.assert_called_once_with("draft://model")
        runner_b.save_remote_model.assert_called_once_with("draft://model")


class TestSpecWorkerUnifiedAPI(unittest.TestCase):
    """Test that BaseSpecWorker provides unified API compatible with TpModelWorker."""

    def setUp(self):
        """Set up mock spec worker."""
        self.mock_target = MagicMock()
        self.mock_target.model_runner = MagicMock()
        self.mock_target.model_runner.attn_backend = "flashinfer"
        self.mock_draft = MagicMock()

        self.spec_worker = MockSpecWorker(self.mock_target, self.mock_draft)

    def test_spec_worker_has_required_base_tp_worker_methods(self):
        """Verify all BaseTpWorker methods are accessible on spec worker."""
        required_methods = [
            "forward_batch_generation",
            "forward_batch_embedding",
            "model_runner",
            "get_worker_info",
            "get_pad_input_ids_func",
            "get_memory_pool",
            "get_tokens_per_layer_info",
            "update_weights_from_disk",
            "load_lora_adapter",
            "unload_lora_adapter",
            "load_lora_adapter_from_tensors",
            "check_weights",
            "save_remote_model",
            "save_sharded_model",
        ]

        for method_name in required_methods:
            self.assertTrue(
                hasattr(self.spec_worker, method_name),
                f"Spec worker missing {method_name} from BaseTpWorker interface",
            )

    def test_spec_worker_has_required_rank_properties(self):
        """Verify spec worker exposes rank/device properties."""
        self.mock_target.device = "cuda:0"
        self.mock_target.tp_rank = 1
        self.mock_target.pp_rank = 0
        self.mock_target.dp_rank = None

        # Should be accessible without error
        _ = self.spec_worker.device
        _ = self.spec_worker.tp_rank
        _ = self.spec_worker.pp_rank
        _ = self.spec_worker.dp_rank

    def test_spec_worker_can_be_used_where_tp_worker_expected(self):
        """Verify spec worker can be used in place of TpModelWorker."""
        # This is a duck-typing test: if all methods exist and work, the API is compatible
        self.mock_target.get_worker_info.return_value = (
            16384,
            8192,
            32,
            128,
            4096,
            4091,
            42,
            "cuda:0",
            None,
            512,
            1024,
            512,
        )

        # Try accessing as if it were a TpModelWorker
        info = self.spec_worker.get_worker_info()
        self.assertIsNotNone(info)
        self.assertEqual(len(info), 12)

        self.mock_target.model_config = MagicMock()
        self.mock_target.tokenizer = MagicMock()
        self.mock_target.processor = MagicMock()
        _ = self.spec_worker.model_config
        _ = self.spec_worker.tokenizer
        _ = self.spec_worker.processor


class TestSpecWorkerOverlapScheduling(unittest.TestCase):
    """Test spec_v2_attn_backends for overlap scheduling."""

    def setUp(self):
        """Set up mock workers."""
        self.mock_target = MagicMock()
        self.mock_target.model_runner = MagicMock()
        self.mock_target.model_runner.attn_backend = "flashinfer"
        self.mock_draft = MagicMock()

        self.spec_worker = MockSpecWorker(self.mock_target, self.mock_draft)

    def test_default_spec_v2_attn_backends(self):
        """Verify default spec_v2_attn_backends returns target backend."""
        backends = self.spec_worker.spec_v2_attn_backends
        self.assertEqual(len(backends), 1)
        self.assertEqual(backends[0], "flashinfer")

    def test_spec_worker_always_has_spec_v2_attn_backends(self):
        """Verify spec_v2_attn_backends is always available (no getattr fallback needed)."""
        # This is the key test: scheduler can now use hasattr() safely
        self.assertTrue(hasattr(self.spec_worker, "spec_v2_attn_backends"))
        backends = self.spec_worker.spec_v2_attn_backends
        self.assertIsNotNone(backends)
        self.assertIsInstance(backends, tuple)


if __name__ == "__main__":
    unittest.main()
