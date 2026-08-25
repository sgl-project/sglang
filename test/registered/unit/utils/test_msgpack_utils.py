"""CPU-only regression tests for MessagePack transport helpers."""

import unittest
from array import array

import msgspec
import numpy as np
import torch

from sglang.srt.utils.msgpack_utils import (
    _MSGPACK_EXT_ARRAY,
    _MSGPACK_EXT_NP_ARRAY,
    _MSGPACK_EXT_TORCH_TENSOR,
    _from_msgpack_state,
    _to_msgpack_state,
    enc_hook,
    ext_hook,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _round_trip(value: object) -> object:
    encoded = msgspec.msgpack.encode(value, enc_hook=enc_hook)
    return msgspec.msgpack.decode(encoded, ext_hook=ext_hook)


class TestMsgpackBufferExtensions(CustomTestCase):
    def test_round_trip_preserves_array_and_numpy_metadata(self):
        value = {
            "array": array("i", [3, -1, 7]),
            "numpy": np.array([[1.5, -2.0]], dtype=np.float32),
        }

        restored = _round_trip(value)

        self.assertEqual(restored["array"], value["array"])
        self.assertEqual(restored["numpy"].dtype, np.float32)
        np.testing.assert_array_equal(restored["numpy"], value["numpy"])

    def test_round_trip_preserves_non_contiguous_cpu_tensor(self):
        tensor = torch.arange(12, dtype=torch.int64).reshape(3, 4).transpose(0, 1)
        self.assertFalse(tensor.is_contiguous())

        restored = _round_trip(tensor)

        self.assertIsInstance(restored, torch.Tensor)
        self.assertEqual(restored.device.type, "cpu")
        self.assertEqual(restored.dtype, tensor.dtype)
        self.assertEqual(tuple(restored.shape), tuple(tensor.shape))
        self.assertTrue(torch.equal(restored, tensor))

    def test_round_trip_preserves_empty_tensor_shape_and_dtype(self):
        tensor = torch.empty((0, 3), dtype=torch.float16)

        restored = _round_trip(tensor)

        self.assertEqual(tuple(restored.shape), (0, 3))
        self.assertEqual(restored.dtype, torch.float16)
        self.assertEqual(restored.device.type, "cpu")

    def test_unknown_extension_is_preserved_for_forward_compatibility(self):
        value = msgspec.msgpack.Ext(99, b"future-wire-format")

        restored = _round_trip(value)

        self.assertIsInstance(restored, msgspec.msgpack.Ext)
        self.assertEqual(restored.code, 99)
        self.assertEqual(restored.data, b"future-wire-format")

    def test_rejects_truncated_and_inconsistent_buffer_metadata(self):
        with self.assertRaisesRegex(msgspec.DecodeError, "missing metadata"):
            ext_hook(_MSGPACK_EXT_ARRAY, memoryview(b"\x00\x00"))

        declared_size_without_payload = (8).to_bytes(4, "big") + b"\x81"
        with self.assertRaisesRegex(msgspec.DecodeError, "invalid metadata"):
            ext_hook(_MSGPACK_EXT_NP_ARRAY, memoryview(declared_size_without_payload))

    def test_known_extension_codes_remain_distinct(self):
        self.assertNotEqual(_MSGPACK_EXT_ARRAY, _MSGPACK_EXT_TORCH_TENSOR)
        self.assertNotEqual(_MSGPACK_EXT_ARRAY, _MSGPACK_EXT_NP_ARRAY)
        self.assertNotEqual(_MSGPACK_EXT_TORCH_TENSOR, _MSGPACK_EXT_NP_ARRAY)


class TestMsgpackStateConversion(CustomTestCase):
    def test_nested_transport_state_round_trips_special_types(self):
        state = {
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "shape": torch.Size([2, 3]),
            "numpy_dtype": np.dtype("<i4"),
            "nested": (torch.float16, [torch.Size([1])]),
        }

        restored = _from_msgpack_state(_to_msgpack_state(state))

        self.assertEqual(restored["dtype"], torch.float32)
        self.assertEqual(restored["device"], torch.device("cpu"))
        self.assertEqual(restored["shape"], torch.Size([2, 3]))
        self.assertEqual(restored["numpy_dtype"], np.dtype("<i4"))
        self.assertEqual(restored["nested"], (torch.float16, [torch.Size([1])]))


if __name__ == "__main__":
    unittest.main()
