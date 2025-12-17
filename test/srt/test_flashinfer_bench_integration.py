"""
Unit tests for FlashInfer-Bench integration module.

These tests verify the integration layer between SGLang and FlashInfer-Bench.
They test both the case where flashinfer-bench is installed and where it isn't.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestFlashInferBenchIntegrationWithoutPackage(unittest.TestCase):
    """Tests for when flashinfer-bench is NOT installed."""

    def test_initialize_returns_false_without_package(self):
        """Test that initialize returns False when flashinfer-bench is not installed."""
        # Mock the import to fail
        with patch.dict("sys.modules", {"flashinfer_bench": None}):
            # Need to reload the module to pick up the mocked import
            import importlib

            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            # Force reload to re-evaluate the try/except import
            importlib.reload(fib_integration)

            # When package is not available, HAS_FLASHINFER_BENCH should be False
            # and initialize should return False
            result = fib_integration.initialize_flashinfer_bench(
                tracing=True, apply=False, dataset_path="/tmp/test"
            )
            self.assertFalse(result)

    def test_shutdown_safe_without_package(self):
        """Test that shutdown doesn't raise when flashinfer-bench is not installed."""
        with patch.dict("sys.modules", {"flashinfer_bench": None}):
            import importlib

            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            importlib.reload(fib_integration)

            # Should not raise
            fib_integration.shutdown_flashinfer_bench()

    def test_status_functions_return_false_without_init(self):
        """Test that status functions return False when not initialized."""
        import importlib

        import sglang.srt.layers.flashinfer_bench_integration as fib_integration

        importlib.reload(fib_integration)

        self.assertFalse(fib_integration.is_flashinfer_bench_enabled())
        self.assertFalse(fib_integration.is_tracing_enabled())
        self.assertFalse(fib_integration.is_apply_enabled())


class TestFlashInferBenchIntegrationWithMockedPackage(unittest.TestCase):
    """Tests for when flashinfer-bench IS installed (mocked)."""

    def setUp(self):
        """Set up mocked flashinfer_bench module."""
        self.mock_enable_tracing = MagicMock()
        self.mock_enable_apply = MagicMock()
        self.mock_disable_tracing = MagicMock()
        self.mock_disable_apply = MagicMock()

        self.mock_fib = MagicMock()
        self.mock_fib.enable_tracing = self.mock_enable_tracing
        self.mock_fib.enable_apply = self.mock_enable_apply
        self.mock_fib.disable_tracing = self.mock_disable_tracing
        self.mock_fib.disable_apply = self.mock_disable_apply

    def test_initialize_with_tracing_only(self):
        """Test initialization with tracing enabled."""
        with patch.dict("sys.modules", {"flashinfer_bench": self.mock_fib}):

            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            # Patch the imported functions
            fib_integration.HAS_FLASHINFER_BENCH = True
            fib_integration.enable_tracing = self.mock_enable_tracing
            fib_integration.enable_apply = self.mock_enable_apply
            fib_integration.disable_tracing = self.mock_disable_tracing
            fib_integration.disable_apply = self.mock_disable_apply

            # Reset state
            fib_integration._initialized = False
            fib_integration._tracing_enabled = False
            fib_integration._apply_enabled = False

            result = fib_integration.initialize_flashinfer_bench(
                tracing=True, apply=False, dataset_path="/tmp/test"
            )

            self.assertTrue(result)
            self.mock_enable_tracing.assert_called_once_with(
                dataset_path="/tmp/test", tracing_configs=None
            )
            self.mock_enable_apply.assert_not_called()
            self.assertTrue(fib_integration.is_tracing_enabled())
            self.assertFalse(fib_integration.is_apply_enabled())

    def test_initialize_with_apply_only(self):
        """Test initialization with apply enabled."""
        with patch.dict("sys.modules", {"flashinfer_bench": self.mock_fib}):
            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            fib_integration.HAS_FLASHINFER_BENCH = True
            fib_integration.enable_tracing = self.mock_enable_tracing
            fib_integration.enable_apply = self.mock_enable_apply
            fib_integration.disable_tracing = self.mock_disable_tracing
            fib_integration.disable_apply = self.mock_disable_apply

            # Reset state
            fib_integration._initialized = False
            fib_integration._tracing_enabled = False
            fib_integration._apply_enabled = False

            result = fib_integration.initialize_flashinfer_bench(
                tracing=False, apply=True, dataset_path="/tmp/test"
            )

            self.assertTrue(result)
            self.mock_enable_tracing.assert_not_called()
            self.mock_enable_apply.assert_called_once_with(dataset_path="/tmp/test")
            self.assertFalse(fib_integration.is_tracing_enabled())
            self.assertTrue(fib_integration.is_apply_enabled())

    def test_initialize_with_both(self):
        """Test initialization with both tracing and apply enabled."""
        with patch.dict("sys.modules", {"flashinfer_bench": self.mock_fib}):
            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            fib_integration.HAS_FLASHINFER_BENCH = True
            fib_integration.enable_tracing = self.mock_enable_tracing
            fib_integration.enable_apply = self.mock_enable_apply
            fib_integration.disable_tracing = self.mock_disable_tracing
            fib_integration.disable_apply = self.mock_disable_apply

            # Reset state
            fib_integration._initialized = False
            fib_integration._tracing_enabled = False
            fib_integration._apply_enabled = False

            result = fib_integration.initialize_flashinfer_bench(
                tracing=True, apply=True, dataset_path="/tmp/test"
            )

            self.assertTrue(result)
            self.mock_enable_tracing.assert_called_once()
            self.mock_enable_apply.assert_called_once()
            self.assertTrue(fib_integration.is_flashinfer_bench_enabled())

    def test_initialize_with_neither(self):
        """Test that initialization returns False when neither tracing nor apply."""
        with patch.dict("sys.modules", {"flashinfer_bench": self.mock_fib}):
            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            fib_integration.HAS_FLASHINFER_BENCH = True

            # Reset state
            fib_integration._initialized = False
            fib_integration._tracing_enabled = False
            fib_integration._apply_enabled = False

            result = fib_integration.initialize_flashinfer_bench(
                tracing=False, apply=False, dataset_path="/tmp/test"
            )

            self.assertFalse(result)
            self.mock_enable_tracing.assert_not_called()
            self.mock_enable_apply.assert_not_called()

    def test_shutdown(self):
        """Test shutdown disables tracing and apply."""
        with patch.dict("sys.modules", {"flashinfer_bench": self.mock_fib}):
            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            fib_integration.HAS_FLASHINFER_BENCH = True
            fib_integration.enable_tracing = self.mock_enable_tracing
            fib_integration.enable_apply = self.mock_enable_apply
            fib_integration.disable_tracing = self.mock_disable_tracing
            fib_integration.disable_apply = self.mock_disable_apply

            # Simulate initialized state
            fib_integration._initialized = True
            fib_integration._tracing_enabled = True
            fib_integration._apply_enabled = True

            fib_integration.shutdown_flashinfer_bench()

            self.mock_disable_tracing.assert_called_once()
            self.mock_disable_apply.assert_called_once()
            self.assertFalse(fib_integration._initialized)
            self.assertFalse(fib_integration._tracing_enabled)
            self.assertFalse(fib_integration._apply_enabled)


class TestEnvironmentVariableDefaults(unittest.TestCase):
    """Test that environment variables are used as defaults."""

    def test_uses_env_vars_when_args_none(self):
        """Test that env vars are used when function args are None."""
        mock_enable_tracing = MagicMock()
        mock_enable_apply = MagicMock()

        mock_fib = MagicMock()
        mock_fib.enable_tracing = mock_enable_tracing
        mock_fib.enable_apply = mock_enable_apply

        with patch.dict("sys.modules", {"flashinfer_bench": mock_fib}):
            import sglang.srt.layers.flashinfer_bench_integration as fib_integration

            fib_integration.HAS_FLASHINFER_BENCH = True
            fib_integration.enable_tracing = mock_enable_tracing
            fib_integration.enable_apply = mock_enable_apply

            # Reset state
            fib_integration._initialized = False
            fib_integration._tracing_enabled = False
            fib_integration._apply_enabled = False

            # Mock the environment variable getters
            with patch.object(
                fib_integration.envs.FIB_ENABLE_TRACING, "get", return_value=True
            ):
                with patch.object(
                    fib_integration.envs.FIB_ENABLE_APPLY, "get", return_value=False
                ):
                    with patch.object(
                        fib_integration.envs.FIB_DATASET_PATH,
                        "get",
                        return_value="/env/path",
                    ):
                        result = fib_integration.initialize_flashinfer_bench()

                        self.assertTrue(result)
                        mock_enable_tracing.assert_called_once_with(
                            dataset_path="/env/path", tracing_configs=None
                        )


if __name__ == "__main__":
    unittest.main()
