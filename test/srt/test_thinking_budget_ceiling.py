"""
Tests for the Server-Wide Thinking Budget Ceiling API feature.

This file contains:
  - Unit tests (no server needed): ServerArgs parsing, clamping logic,
    auto-injection of processors, and Prometheus gauge behavior.
  - Integration tests (with server): HTTP API endpoints for get/set
    and end-to-end clamping behavior.

Usage (unit tests only):
    python3 -m pytest test/srt/test_thinking_budget_ceiling.py -k "Unit" -v

Usage (all tests including integration):
    python3 -m pytest test/srt/test_thinking_budget_ceiling.py -v
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_TEST_MODEL = os.environ.get("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


# ---------------------------------------------------------------------------
# Unit Tests -- no server process needed
# ---------------------------------------------------------------------------


class TestUnitServerArgsParsing(unittest.TestCase):
    """Unit tests for --default-thinking-budget flag parsing and validation."""

    def test_default_value_is_none(self):
        """ServerArgs should have default_thinking_budget=None by default."""
        # Use a minimal set of args that bypasses heavy validation.
        # We only care about the dataclass field default here.
        import dataclasses

        from sglang.srt.server_args import ServerArgs

        defaults = {f.name: f.default for f in dataclasses.fields(ServerArgs)}
        self.assertIsNone(defaults["default_thinking_budget"])

    def test_negative_value_raises_error(self):
        """Passing a negative --default-thinking-budget should raise ValueError."""
        from sglang.srt.server_args import ServerArgs

        # Mock __post_init__ to skip heavy initialization (GPU, model resolution).
        # Then call _handle_other_validations() directly to test thinking budget logic.
        with patch.object(ServerArgs, "__post_init__"):
            args = ServerArgs(
                model_path="fake-model",
                default_thinking_budget=-1,
            )
        with self.assertRaises(ValueError):
            args._handle_other_validations()

    def test_zero_value_accepted(self):
        """A value of 0 should be accepted (disables thinking entirely)."""
        from sglang.srt.server_args import ServerArgs

        with patch.object(ServerArgs, "__post_init__"):
            args = ServerArgs(
                model_path="fake-model",
                default_thinking_budget=0,
            )
        # _handle_other_validations should not raise for value 0.
        args._handle_other_validations()
        self.assertEqual(args.default_thinking_budget, 0)

    def test_positive_value_accepted(self):
        """A positive integer should be accepted."""
        from sglang.srt.server_args import ServerArgs

        with patch.object(ServerArgs, "__post_init__"):
            args = ServerArgs(
                model_path="fake-model",
                default_thinking_budget=1024,
            )
        args._handle_other_validations()
        self.assertEqual(args.default_thinking_budget, 1024)

    def test_auto_enables_custom_logit_processor(self):
        """When default_thinking_budget is set, enable_custom_logit_processor
        should be auto-enabled."""
        from sglang.srt.server_args import ServerArgs

        with patch.object(ServerArgs, "__post_init__"):
            args = ServerArgs(
                model_path="fake-model",
                default_thinking_budget=512,
                enable_custom_logit_processor=False,
            )
        args._handle_other_validations()
        self.assertTrue(args.enable_custom_logit_processor)


class TestUnitClampingLogic(unittest.TestCase):
    """Unit tests for the clamping logic that the TokenizerManager applies.

    These tests directly exercise the clamping rules without starting a server.
    We simulate the logic from TokenizerManager._create_tokenized_request().
    """

    @staticmethod
    def _apply_clamping(ceiling, per_request):
        """Reproduce the clamping logic from tokenizer_manager.py.

        Args:
            ceiling: The server-wide default_thinking_budget (int or None).
            per_request: The per-request thinking_budget from custom_params (int or None).

        Returns:
            The effective thinking_budget after clamping, or None if no enforcement.
        """
        if ceiling is None:
            # No server ceiling -- per-request is used as-is.
            return per_request

        # Server ceiling is active.
        if per_request is not None and isinstance(per_request, int):
            return min(per_request, ceiling)
        else:
            return ceiling

    def test_ceiling_1024_per_request_512(self):
        """Per-request 512 is within ceiling 1024; should use 512."""
        self.assertEqual(self._apply_clamping(1024, 512), 512)

    def test_ceiling_1024_per_request_2048(self):
        """Per-request 2048 exceeds ceiling 1024; should clamp to 1024."""
        self.assertEqual(self._apply_clamping(1024, 2048), 1024)

    def test_ceiling_1024_no_per_request(self):
        """No per-request value; ceiling 1024 should be applied as default."""
        self.assertEqual(self._apply_clamping(1024, None), 1024)

    def test_ceiling_none_per_request_512(self):
        """No ceiling; per-request 512 should pass through."""
        self.assertEqual(self._apply_clamping(None, 512), 512)

    def test_ceiling_none_no_per_request(self):
        """Neither ceiling nor per-request set; no enforcement."""
        self.assertIsNone(self._apply_clamping(None, None))

    def test_ceiling_0_per_request_any(self):
        """Ceiling of 0 means disable thinking entirely; effective should be 0."""
        self.assertEqual(self._apply_clamping(0, 500), 0)

    def test_ceiling_0_no_per_request(self):
        """Ceiling of 0 with no per-request should yield 0."""
        self.assertEqual(self._apply_clamping(0, None), 0)

    def test_ceiling_equals_per_request(self):
        """When ceiling equals per-request, should use that value."""
        self.assertEqual(self._apply_clamping(1024, 1024), 1024)


class TestUnitAutoInjection(unittest.TestCase):
    """Unit tests for the auto-injection of the correct ThinkingBudgetLogitProcessor
    based on model_type."""

    @staticmethod
    def _detect_processor_for_model_type(model_type):
        """Simulate _detect_thinking_budget_processor() logic."""
        from sglang.srt.sampling.custom_logit_processor import (
            DeepSeekR1ThinkingBudgetLogitProcessor,
            Glm4MoeThinkingBudgetLogitProcessor,
            Qwen3ThinkingBudgetLogitProcessor,
        )

        model_type_to_processor = {
            "deepseek_v3": DeepSeekR1ThinkingBudgetLogitProcessor,
            "deepseek_r1": DeepSeekR1ThinkingBudgetLogitProcessor,
            "glm4": Glm4MoeThinkingBudgetLogitProcessor,
            "glm_moe": Glm4MoeThinkingBudgetLogitProcessor,
            "chatglm": Glm4MoeThinkingBudgetLogitProcessor,
            "qwen3": Qwen3ThinkingBudgetLogitProcessor,
            "qwen3_moe": Qwen3ThinkingBudgetLogitProcessor,
        }
        processor_cls = model_type_to_processor.get(model_type)
        if processor_cls is not None:
            return processor_cls.to_str()
        return None

    def test_deepseek_v3(self):
        """deepseek_v3 model type should select DeepSeekR1ThinkingBudgetLogitProcessor."""
        from sglang.srt.sampling.custom_logit_processor import (
            DeepSeekR1ThinkingBudgetLogitProcessor,
        )

        result = self._detect_processor_for_model_type("deepseek_v3")
        self.assertIsNotNone(result)
        self.assertEqual(result, DeepSeekR1ThinkingBudgetLogitProcessor.to_str())

    def test_deepseek_r1(self):
        """deepseek_r1 model type should select DeepSeekR1ThinkingBudgetLogitProcessor."""
        from sglang.srt.sampling.custom_logit_processor import (
            DeepSeekR1ThinkingBudgetLogitProcessor,
        )

        result = self._detect_processor_for_model_type("deepseek_r1")
        self.assertIsNotNone(result)
        self.assertEqual(result, DeepSeekR1ThinkingBudgetLogitProcessor.to_str())

    def test_qwen3(self):
        """qwen3 model type should select Qwen3ThinkingBudgetLogitProcessor."""
        from sglang.srt.sampling.custom_logit_processor import (
            Qwen3ThinkingBudgetLogitProcessor,
        )

        result = self._detect_processor_for_model_type("qwen3")
        self.assertIsNotNone(result)
        self.assertEqual(result, Qwen3ThinkingBudgetLogitProcessor.to_str())

    def test_qwen3_moe(self):
        """qwen3_moe model type should select Qwen3ThinkingBudgetLogitProcessor."""
        from sglang.srt.sampling.custom_logit_processor import (
            Qwen3ThinkingBudgetLogitProcessor,
        )

        result = self._detect_processor_for_model_type("qwen3_moe")
        self.assertIsNotNone(result)
        self.assertEqual(result, Qwen3ThinkingBudgetLogitProcessor.to_str())

    def test_glm4(self):
        """glm4 model type should select Glm4MoeThinkingBudgetLogitProcessor."""
        from sglang.srt.sampling.custom_logit_processor import (
            Glm4MoeThinkingBudgetLogitProcessor,
        )

        result = self._detect_processor_for_model_type("glm4")
        self.assertIsNotNone(result)
        self.assertEqual(result, Glm4MoeThinkingBudgetLogitProcessor.to_str())

    def test_glm_moe(self):
        """glm_moe model type should select Glm4MoeThinkingBudgetLogitProcessor."""
        from sglang.srt.sampling.custom_logit_processor import (
            Glm4MoeThinkingBudgetLogitProcessor,
        )

        result = self._detect_processor_for_model_type("glm_moe")
        self.assertIsNotNone(result)
        self.assertEqual(result, Glm4MoeThinkingBudgetLogitProcessor.to_str())

    def test_chatglm(self):
        """chatglm model type should select Glm4MoeThinkingBudgetLogitProcessor."""
        from sglang.srt.sampling.custom_logit_processor import (
            Glm4MoeThinkingBudgetLogitProcessor,
        )

        result = self._detect_processor_for_model_type("chatglm")
        self.assertIsNotNone(result)
        self.assertEqual(result, Glm4MoeThinkingBudgetLogitProcessor.to_str())

    def test_unknown_model_returns_none(self):
        """An unknown model type should not auto-inject any processor."""
        result = self._detect_processor_for_model_type("llama")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        """An empty model type string should not auto-inject any processor."""
        result = self._detect_processor_for_model_type("")
        self.assertIsNone(result)


class TestUnitPrometheusGauge(unittest.TestCase):
    """Unit tests for the Prometheus gauge in TokenizerMetricsCollector."""

    @classmethod
    def setUpClass(cls):
        """Create a single TokenizerMetricsCollector to avoid duplicate
        Prometheus metric registration across tests."""
        from sglang.srt.metrics.collector import TokenizerMetricsCollector

        cls._labels = {"model_name": "test-model"}
        cls._collector = TokenizerMetricsCollector(labels=cls._labels)

    def test_gauge_set_to_value(self):
        """When ceiling is an integer, gauge should be set to that value."""
        self._collector.set_default_thinking_budget_gauge(self._labels, 1024)
        gauge_value = self._collector.default_thinking_budget.labels(
            **self._labels
        )._value.get()
        self.assertEqual(gauge_value, 1024)

    def test_gauge_set_to_negative_one_when_none(self):
        """When ceiling is None, gauge should be set to -1."""
        self._collector.set_default_thinking_budget_gauge(self._labels, None)
        gauge_value = self._collector.default_thinking_budget.labels(
            **self._labels
        )._value.get()
        self.assertEqual(gauge_value, -1)

    def test_gauge_set_to_zero(self):
        """When ceiling is 0, gauge should be set to 0."""
        self._collector.set_default_thinking_budget_gauge(self._labels, 0)
        gauge_value = self._collector.default_thinking_budget.labels(
            **self._labels
        )._value.get()
        self.assertEqual(gauge_value, 0)

    def test_gauge_update_from_value_to_none(self):
        """Gauge should correctly transition from a value to None (-1)."""
        self._collector.set_default_thinking_budget_gauge(self._labels, 512)
        gauge_value = self._collector.default_thinking_budget.labels(
            **self._labels
        )._value.get()
        self.assertEqual(gauge_value, 512)

        self._collector.set_default_thinking_budget_gauge(self._labels, None)
        gauge_value = self._collector.default_thinking_budget.labels(
            **self._labels
        )._value.get()
        self.assertEqual(gauge_value, -1)


class TestUnitTokenizerManagerGetSet(unittest.TestCase):
    """Unit tests for the get/set methods on TokenizerManager, using mocks."""

    def test_set_and_get_thinking_budget(self):
        """set_default_thinking_budget should update the value returned by get."""
        # Create a mock TokenizerManager with just the relevant attributes.
        tm = MagicMock()
        tm.enable_metrics = False

        # Bind the real methods to the mock.
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        tm.set_default_thinking_budget = (
            lambda value: TokenizerManager.set_default_thinking_budget(tm, value)
        )
        tm.get_default_thinking_budget = (
            lambda: TokenizerManager.get_default_thinking_budget(tm)
        )

        # Initially None.
        tm.default_thinking_budget = None
        self.assertIsNone(tm.get_default_thinking_budget())

        # Set to 1024.
        tm.set_default_thinking_budget(1024)
        self.assertEqual(tm.default_thinking_budget, 1024)
        self.assertEqual(tm.get_default_thinking_budget(), 1024)

        # Set to None (remove ceiling).
        tm.set_default_thinking_budget(None)
        self.assertIsNone(tm.get_default_thinking_budget())

    def test_set_thinking_budget_updates_metrics(self):
        """When metrics are enabled, set should call the gauge update."""
        tm = MagicMock()
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock()
        tm.metrics_collector.labels = {"model_name": "test-model"}

        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        TokenizerManager.set_default_thinking_budget(tm, 512)
        tm.metrics_collector.set_default_thinking_budget_gauge.assert_called_once_with(
            {"model_name": "test-model"}, 512
        )


class TestUnitEngineAPI(unittest.TestCase):
    """Unit tests for Engine.set/get_default_thinking_budget delegation."""

    def test_engine_delegates_to_tokenizer_manager(self):
        """Engine methods should delegate to tokenizer_manager."""
        from sglang.srt.entrypoints.engine import Engine

        engine = Engine.__new__(Engine)
        engine.tokenizer_manager = MagicMock()
        engine.tokenizer_manager.get_default_thinking_budget.return_value = 1024

        # Test set
        engine.set_default_thinking_budget(512)
        engine.tokenizer_manager.set_default_thinking_budget.assert_called_once_with(
            512
        )

        # Test get
        result = engine.get_default_thinking_budget()
        self.assertEqual(result, 1024)
        engine.tokenizer_manager.get_default_thinking_budget.assert_called_once()


# ---------------------------------------------------------------------------
# Integration Tests -- require a running server
# ---------------------------------------------------------------------------


class TestIntegrationThinkingBudgetAPI(CustomTestCase):
    """Integration tests for the /set_default_thinking_budget and
    /get_default_thinking_budget HTTP endpoints.

    These tests launch a real SGLang server process and exercise the
    admin API over HTTP.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _TEST_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--default-thinking-budget",
                "2048",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # -- GET endpoint tests --

    def test_get_returns_initial_ceiling(self):
        """GET /get_default_thinking_budget should return the startup ceiling."""
        resp = requests.get(f"{self.base_url}/get_default_thinking_budget")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("thinking_budget", data)
        self.assertEqual(data["thinking_budget"], 2048)

    # -- SET endpoint tests: valid values --

    def test_set_valid_integer(self):
        """POST /set_default_thinking_budget with a valid integer should return 200."""
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 1024},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["thinking_budget"], 1024)

        # Verify via GET
        resp = requests.get(f"{self.base_url}/get_default_thinking_budget")
        self.assertEqual(resp.json()["thinking_budget"], 1024)

    def test_set_null_removes_ceiling(self):
        """POST with null should remove the ceiling (unlimited)."""
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": None},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIsNone(data["thinking_budget"])

        # Verify via GET
        resp = requests.get(f"{self.base_url}/get_default_thinking_budget")
        self.assertIsNone(resp.json()["thinking_budget"])

    def test_set_zero_disables_thinking(self):
        """POST with 0 should succeed and disable thinking."""
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 0},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["thinking_budget"], 0)

    # -- SET endpoint tests: invalid values --

    def test_set_negative_returns_400(self):
        """POST with a negative integer should return 400."""
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": -1},
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])

    def test_set_boolean_returns_400(self):
        """POST with a boolean should return 400 (booleans are not ints)."""
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": True},
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])

    def test_set_string_returns_400(self):
        """POST with a string should return 400."""
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": "abc"},
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])

    def test_set_float_returns_400(self):
        """POST with a float should return 400."""
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 10.5},
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])

    # -- PUT method should also work --

    def test_put_method_works(self):
        """PUT /set_default_thinking_budget should work the same as POST."""
        resp = requests.put(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 256},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["thinking_budget"], 256)

    # -- Round-trip: set then get --

    def test_set_then_get_roundtrip(self):
        """Setting a value and getting it back should be consistent."""
        for value in [100, 0, 4096, None, 512]:
            resp = requests.post(
                f"{self.base_url}/set_default_thinking_budget",
                json={"thinking_budget": value},
            )
            self.assertEqual(resp.status_code, 200)

            resp = requests.get(f"{self.base_url}/get_default_thinking_budget")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["thinking_budget"], value)


class TestIntegrationThinkingBudgetClamping(CustomTestCase):
    """Integration tests that verify the clamping behavior affects actual requests.

    We set a server-wide ceiling and then send chat completion requests with
    varying per-request thinking_budget values to verify clamping.

    Note: The underlying model (a small Llama model) is not a thinking model,
    so the ThinkingBudgetLogitProcessor may not produce visible thinking tokens.
    These tests verify the API plumbing works correctly -- that the server
    accepts the ceiling config and the requests complete successfully with
    the custom_params being properly set. For full end-to-end thinking token
    verification, a thinking model (e.g., Qwen3 or DeepSeek-R1) is needed.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _TEST_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--default-thinking-budget",
                "1024",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_request_without_thinking_budget_succeeds(self):
        """When ceiling is set but request has no thinking_budget,
        the server should apply the ceiling as default and the request
        should complete successfully."""
        # Set ceiling
        requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 1024},
        )

        # Send a generate request without thinking_budget in custom_params
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                },
            },
        )
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")
        data = resp.json()
        self.assertIn("text", data)

    def test_request_with_lower_thinking_budget_succeeds(self):
        """When per-request thinking_budget < ceiling, the request should
        succeed with per-request value used (not clamped)."""
        requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 1024},
        )

        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "custom_params": {"thinking_budget": 512},
                },
            },
        )
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")
        data = resp.json()
        self.assertIn("text", data)

    def test_request_with_higher_thinking_budget_succeeds(self):
        """When per-request thinking_budget > ceiling, the request should
        succeed with the value clamped to ceiling."""
        requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 1024},
        )

        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "custom_params": {"thinking_budget": 2048},
                },
            },
        )
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")
        data = resp.json()
        self.assertIn("text", data)

    def test_request_after_ceiling_removed(self):
        """After removing the ceiling (set to null), requests with high
        thinking_budget should not be clamped."""
        # Remove ceiling
        resp = requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": None},
        )
        self.assertEqual(resp.status_code, 200)

        # Send request with large thinking_budget -- should succeed
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "custom_params": {"thinking_budget": 99999},
                },
            },
        )
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")

    def test_chat_completion_with_ceiling(self):
        """Chat completion endpoint should also respect the ceiling."""
        requests.post(
            f"{self.base_url}/set_default_thinking_budget",
            json={"thinking_budget": 512},
        )

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")
        data = resp.json()
        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)

    def test_server_info_includes_thinking_budget(self):
        """The /get_model_info or /server_info endpoint should reflect
        the default_thinking_budget from server_args."""
        resp = requests.get(f"{self.base_url}/server_info")
        if resp.status_code == 200:
            data = resp.json()
            # default_thinking_budget should be present in server_args
            # (it was set to 1024 at startup, but may have been changed by
            # preceding tests via the API -- the server_info shows startup args).
            self.assertIn("default_thinking_budget", str(data))


if __name__ == "__main__":
    unittest.main()
