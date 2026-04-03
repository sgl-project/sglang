import unittest
from types import SimpleNamespace
from typing import Optional

try:
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
    from sglang.srt.managers.tokenizer_communicator_mixin import (
        TokenizerCommunicatorMixin,
    )
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Missing test dependency: {exc}")


class DummyTokenizerManager:
    def __init__(self, weight_version: str, payload_digest: Optional[str]):
        self.server_args = SimpleNamespace(weight_version=weight_version)
        self.weight_update_payload_digest = payload_digest


class TestWeightUpdateVersioning(unittest.TestCase):
    def test_rejects_base_version_mismatch(self):
        manager = DummyTokenizerManager(weight_version="v2", payload_digest=None)
        request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[],
            base_weight_version="v1",
        )

        success, message, is_noop = (
            TokenizerCommunicatorMixin._validate_weight_update_request(manager, request)
        )

        self.assertFalse(success)
        self.assertFalse(is_noop)
        self.assertIn("Base weight version mismatch", message)

    def test_duplicate_payload_becomes_noop(self):
        manager = DummyTokenizerManager(weight_version="v2", payload_digest="abc123")
        request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[],
            weight_version="v2",
            payload_digest="abc123",
        )

        success, _, is_noop = TokenizerCommunicatorMixin._validate_weight_update_request(
            manager, request
        )

        self.assertTrue(success)
        self.assertTrue(is_noop)

    def test_duplicate_payload_with_stale_base_version_becomes_noop(self):
        manager = DummyTokenizerManager(weight_version="v2", payload_digest="abc123")
        request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[],
            base_weight_version="v1",
            weight_version="v2",
            payload_digest="abc123",
        )

        success, _, is_noop = TokenizerCommunicatorMixin._validate_weight_update_request(
            manager, request
        )

        self.assertTrue(success)
        self.assertTrue(is_noop)

    def test_conflicting_payload_is_rejected(self):
        manager = DummyTokenizerManager(weight_version="v2", payload_digest="abc123")
        request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[],
            weight_version="v2",
            payload_digest="different",
        )

        success, message, is_noop = (
            TokenizerCommunicatorMixin._validate_weight_update_request(manager, request)
        )

        self.assertFalse(success)
        self.assertFalse(is_noop)
        self.assertIn("different payload digest", message)


if __name__ == "__main__":
    unittest.main()
