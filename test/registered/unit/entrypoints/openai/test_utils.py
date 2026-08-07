"""Unit tests for OpenAI entrypoint utility helpers."""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_TRITON_MODULES_BEFORE = {
    name: module
    for name, module in sys.modules.items()
    if name == "triton" or name.startswith("triton.")
}
_SYS_META_PATH_BEFORE = tuple(sys.meta_path)
_INSTALLED_TRITON_STUB = False

try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    stub_path = (
        Path(__file__).resolve().parents[5] / "python" / "sglang" / "_triton_stub.py"
    )
    spec = importlib.util.spec_from_file_location("_sglang_triton_stub", stub_path)
    triton_stub = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(triton_stub)
    triton_stub.install()
    _INSTALLED_TRITON_STUB = True

import unittest

import torch

_ORIGINAL_TORCH_COMPILE = torch.compile


def _identity_compile(fn=None, **kwargs):
    if fn is None:
        return lambda inner_fn: inner_fn
    return fn


torch.compile = _identity_compile

try:
    from sglang.srt.entrypoints.openai.protocol import StreamOptions
    from sglang.srt.entrypoints.openai.utils import (
        cached_tokens_details_from_dict,
        convert_embeds_to_tensors,
        should_include_usage,
    )
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
finally:
    torch.compile = _ORIGINAL_TORCH_COMPILE

    if _INSTALLED_TRITON_STUB:
        sys.meta_path[:] = _SYS_META_PATH_BEFORE
        for name in list(sys.modules):
            if (name == "triton" or name.startswith("triton.")) and (
                name not in _TRITON_MODULES_BEFORE
            ):
                del sys.modules[name]
        sys.modules.update(_TRITON_MODULES_BEFORE)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestShouldIncludeUsage(CustomTestCase):
    def test_no_stream_options_uses_false_default_without_continuous_stats(self):
        include_usage, continuous_usage_stats = should_include_usage(None, False)

        self.assertFalse(include_usage)
        self.assertFalse(continuous_usage_stats)

    def test_no_stream_options_uses_true_default_without_continuous_stats(self):
        include_usage, continuous_usage_stats = should_include_usage(None, True)

        self.assertTrue(include_usage)
        self.assertFalse(continuous_usage_stats)

    def test_request_include_usage_enables_final_usage_when_default_false(self):
        include_usage, continuous_usage_stats = should_include_usage(
            StreamOptions(include_usage=True),
            False,
        )

        self.assertTrue(include_usage)
        self.assertFalse(continuous_usage_stats)

    def test_continuous_usage_stats_does_not_imply_final_usage(self):
        include_usage, continuous_usage_stats = should_include_usage(
            StreamOptions(continuous_usage_stats=True),
            False,
        )

        self.assertFalse(include_usage)
        self.assertTrue(continuous_usage_stats)

    def test_stream_options_preserve_true_server_default(self):
        include_usage, continuous_usage_stats = should_include_usage(
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            True,
        )

        self.assertTrue(include_usage)
        self.assertFalse(continuous_usage_stats)


class TestCachedTokensDetailsFromDict(CustomTestCase):
    def test_empty_details_default_numeric_counts_and_omit_storage_fields(self):
        details = cached_tokens_details_from_dict({})

        self.assertEqual(details.device, 0)
        self.assertEqual(details.host, 0)
        self.assertIsNone(details.storage)
        self.assertIsNone(details.storage_backend)
        dumped = details.model_dump()
        self.assertEqual(dumped["device"], 0)
        self.assertEqual(dumped["host"], 0)
        self.assertNotIn("storage", dumped)
        self.assertNotIn("storage_backend", dumped)

    def test_device_and_host_counts_without_storage_fields(self):
        details = cached_tokens_details_from_dict({"device": 7, "host": 3})

        self.assertEqual(details.device, 7)
        self.assertEqual(details.host, 3)
        self.assertIsNone(details.storage)
        self.assertIsNone(details.storage_backend)
        dumped = details.model_dump()
        self.assertEqual(dumped["device"], 7)
        self.assertEqual(dumped["host"], 3)
        self.assertNotIn("storage", dumped)
        self.assertNotIn("storage_backend", dumped)

    def test_storage_details_include_storage_fields(self):
        details = cached_tokens_details_from_dict(
            {
                "device": 4,
                "host": 2,
                "storage": 6,
                "storage_backend": "file",
            }
        )

        self.assertEqual(details.device, 4)
        self.assertEqual(details.host, 2)
        self.assertEqual(details.storage, 6)
        self.assertEqual(details.storage_backend, "file")
        dumped = details.model_dump()
        self.assertEqual(dumped["device"], 4)
        self.assertEqual(dumped["host"], 2)
        self.assertEqual(dumped["storage"], 6)
        self.assertEqual(dumped["storage_backend"], "file")

    def test_zero_storage_key_still_uses_storage_aware_details(self):
        details = cached_tokens_details_from_dict({"storage": 0})

        self.assertEqual(details.device, 0)
        self.assertEqual(details.host, 0)
        self.assertEqual(details.storage, 0)
        self.assertIsNone(details.storage_backend)
        dumped = details.model_dump()
        self.assertEqual(dumped["device"], 0)
        self.assertEqual(dumped["host"], 0)
        self.assertEqual(dumped["storage"], 0)
        self.assertNotIn("storage_backend", dumped)


class TestConvertEmbedsToTensors(CustomTestCase):
    def test_none_passthrough(self):
        self.assertIsNone(convert_embeds_to_tensors(None))

    def test_empty_list_passthrough(self):
        self.assertEqual(convert_embeds_to_tensors([]), [])

    def test_all_none_batch_preserves_batch_length(self):
        result = convert_embeds_to_tensors([None, None])

        self.assertEqual(result, [None, None])

    def test_single_input_embeddings_convert_to_float32_tensors(self):
        result = convert_embeds_to_tensors([[1.0, 2.5], [3.0, 4.5]])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        self.assertIsInstance(result[0][0], torch.Tensor)
        self.assertEqual(result[0][0].dtype, torch.float32)
        self.assertEqual(tuple(result[0][0].shape), (2,))
        self.assertEqual(result[0][0].tolist(), [1.0, 2.5])
        self.assertIsInstance(result[0][1], torch.Tensor)
        self.assertEqual(result[0][1].dtype, torch.float32)
        self.assertEqual(tuple(result[0][1].shape), (2,))
        self.assertEqual(result[0][1].tolist(), [3.0, 4.5])

    def test_batch_embeddings_with_leading_none_preserve_none_and_convert_later_data(
        self,
    ):
        result = convert_embeds_to_tensors(
            [
                None,
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            ]
        )

        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0])
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(len(result[2]), 2)

        self.assertIsInstance(result[1][0], torch.Tensor)
        self.assertEqual(result[1][0].dtype, torch.float32)
        self.assertEqual(tuple(result[1][0].shape), (3,))
        self.assertEqual(result[1][0].tolist(), [1.0, 2.0, 3.0])

        self.assertIsInstance(result[2][0], torch.Tensor)
        self.assertEqual(result[2][0].dtype, torch.float32)
        self.assertEqual(tuple(result[2][0].shape), (3,))
        self.assertEqual(result[2][0].tolist(), [4.0, 5.0, 6.0])

        self.assertIsInstance(result[2][1], torch.Tensor)
        self.assertEqual(result[2][1].dtype, torch.float32)
        self.assertEqual(tuple(result[2][1].shape), (3,))
        self.assertEqual(result[2][1].tolist(), [7.0, 8.0, 9.0])


if __name__ == "__main__":
    unittest.main()
