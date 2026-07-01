import asyncio
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_stub_self():
    # update_lora_adapter_communicator raises so the test fails if the guard
    # is bypassed and the backend call is reached.
    def _fail(*args, **kwargs):
        raise AssertionError(
            "update_lora_adapter_communicator should not be reached when "
            "lora_name/lora_path validation fails"
        )

    stub = types.SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        server_args=types.SimpleNamespace(enable_lora=True, dp_size=1),
        update_lora_adapter_communicator=_fail,
    )
    return stub


class TestLoadLoRAAdapterValidation(unittest.TestCase):
    def test_empty_lora_name_rejected(self):
        stub = _make_stub_self()
        obj = LoadLoRAAdapterReqInput(lora_name="", lora_path="/some/path")

        result = asyncio.run(TokenizerControlMixin.load_lora_adapter(stub, obj))

        self.assertFalse(result.success)
        self.assertEqual(
            result.error_message,
            "Both 'lora_name' and 'lora_path' must be provided.",
        )

    def test_empty_lora_path_rejected(self):
        stub = _make_stub_self()
        obj = LoadLoRAAdapterReqInput(lora_name="adapter", lora_path="")

        result = asyncio.run(TokenizerControlMixin.load_lora_adapter(stub, obj))

        self.assertFalse(result.success)
        self.assertEqual(
            result.error_message,
            "Both 'lora_name' and 'lora_path' must be provided.",
        )

    def test_whitespace_only_lora_name_rejected(self):
        stub = _make_stub_self()
        obj = LoadLoRAAdapterReqInput(lora_name="   ", lora_path="/some/path")

        result = asyncio.run(TokenizerControlMixin.load_lora_adapter(stub, obj))

        self.assertFalse(result.success)
        self.assertEqual(
            result.error_message,
            "Both 'lora_name' and 'lora_path' must be provided.",
        )

    def test_whitespace_only_lora_path_rejected(self):
        stub = _make_stub_self()
        obj = LoadLoRAAdapterReqInput(lora_name="adapter", lora_path="   ")

        result = asyncio.run(TokenizerControlMixin.load_lora_adapter(stub, obj))

        self.assertFalse(result.success)
        self.assertEqual(
            result.error_message,
            "Both 'lora_name' and 'lora_path' must be provided.",
        )

    def test_empty_lora_name_from_tensors_rejected(self):
        stub = _make_stub_self()
        obj = LoadLoRAAdapterFromTensorsReqInput(
            lora_name="",
            config_dict={},
            serialized_tensors="",
        )

        result = asyncio.run(
            TokenizerControlMixin.load_lora_adapter_from_tensors(stub, obj)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "'lora_name' must be provided.")

    def test_whitespace_only_lora_name_from_tensors_rejected(self):
        stub = _make_stub_self()
        obj = LoadLoRAAdapterFromTensorsReqInput(
            lora_name="   ",
            config_dict={},
            serialized_tensors="",
        )

        result = asyncio.run(
            TokenizerControlMixin.load_lora_adapter_from_tensors(stub, obj)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "'lora_name' must be provided.")


if __name__ == "__main__":
    unittest.main()
