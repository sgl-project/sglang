import asyncio
import unittest
from types import SimpleNamespace

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
    def _fail(*args, **kwargs):
        raise AssertionError("LoRA backend should not be reached")

    return SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        server_args=SimpleNamespace(enable_lora=True, dp_size=1),
        lora_update_lock=asyncio.Lock(),
        update_lora_adapter_communicator=_fail,
    )


class TestLoadLoRAAdapterValidation(unittest.TestCase):
    def test_file_adapter_rejects_blank_name_or_path(self):
        for lora_name, lora_path in (
            ("", "/some/path"),
            ("   ", "/some/path"),
            ("adapter", ""),
            ("adapter", "   "),
        ):
            with self.subTest(lora_name=lora_name, lora_path=lora_path):
                result = asyncio.run(
                    TokenizerControlMixin.load_lora_adapter(
                        _make_stub_self(),
                        LoadLoRAAdapterReqInput(
                            lora_name=lora_name,
                            lora_path=lora_path,
                        ),
                    )
                )

                self.assertFalse(result.success)
                self.assertEqual(
                    result.error_message,
                    "Both 'lora_name' and 'lora_path' must be provided.",
                )

    def test_tensor_adapter_rejects_blank_name(self):
        for lora_name in ("", "   "):
            with self.subTest(lora_name=lora_name):
                result = asyncio.run(
                    TokenizerControlMixin.load_lora_adapter_from_tensors(
                        _make_stub_self(),
                        LoadLoRAAdapterFromTensorsReqInput(
                            lora_name=lora_name,
                            config_dict={},
                            serialized_named_tensors=[],
                        ),
                    )
                )

                self.assertFalse(result.success)
                self.assertEqual(result.error_message, "'lora_name' must be provided.")


if __name__ == "__main__":
    unittest.main()
