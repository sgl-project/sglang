import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from sglang.srt.entrypoints.grpc_bridge import _VALID_PAUSE_MODES, RuntimeHandle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPauseGenerationModeValidation(CustomTestCase):
    def _run_payload(self, mode: str) -> dict:
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = MagicMock()
        handle.tokenizer_manager.pause_generation = AsyncMock()

        captured = {}

        def fake_submit(name, payload_fn, chunk_callback, **kwargs):
            captured["result"] = asyncio.run(payload_fn())

        handle._submit_json_unary = fake_submit
        handle.pause_generation(mode, chunk_callback=MagicMock())
        return captured["result"]

    def test_invalid_mode_rejected_at_bridge(self):
        """Invalid pause mode must raise ValueError before constructing the scheduler request."""
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = MagicMock()
        handle.tokenizer_manager.pause_generation = AsyncMock()

        errors = {}

        def fake_submit(name, payload_fn, chunk_callback, **kwargs):
            try:
                asyncio.run(payload_fn())
            except ValueError as e:
                errors["error"] = e

        handle._submit_json_unary = fake_submit
        handle.pause_generation("bogus", chunk_callback=MagicMock())

        self.assertIn("error", errors)
        self.assertIn("bogus", str(errors["error"]))
        handle.tokenizer_manager.pause_generation.assert_not_awaited()

    def test_empty_mode_rejected_at_bridge(self):
        """Proto-default empty mode string must be rejected at the bridge."""
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = MagicMock()
        handle.tokenizer_manager.pause_generation = AsyncMock()

        errors = {}

        def fake_submit(name, payload_fn, chunk_callback, **kwargs):
            try:
                asyncio.run(payload_fn())
            except ValueError as e:
                errors["error"] = e

        handle._submit_json_unary = fake_submit
        handle.pause_generation("", chunk_callback=MagicMock())

        self.assertIn("error", errors)
        handle.tokenizer_manager.pause_generation.assert_not_awaited()

    def test_valid_modes_pass_validation(self):
        """All modes in _VALID_PAUSE_MODES must pass bridge validation and dispatch."""
        for mode in _VALID_PAUSE_MODES:
            result = self._run_payload(mode)
            self.assertEqual(
                result, {"message": f"Generation paused (mode={mode})."}
            )


if __name__ == "__main__":
    unittest.main()
