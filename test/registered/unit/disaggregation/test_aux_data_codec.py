import ctypes
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.utils import AuxDataCodec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAuxDataCodec(CustomTestCase):
    def _kv_args(self, buffers):
        return SimpleNamespace(
            aux_data_ptrs=[ctypes.addressof(b) for b, _ in buffers],
            aux_item_lens=[item_len for _, item_len in buffers],
        )

    def test_round_trip_into_the_indexed_slot(self):
        item_len, num_items = 64, 4
        dst = (ctypes.c_byte * (item_len * num_items))()
        kv_args = self._kv_args([(dst, item_len)])
        payload = bytes(range(item_len))

        AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 2, payload)

        raw = bytes(dst)
        self.assertEqual(raw[2 * item_len : 3 * item_len], payload)
        self.assertEqual(raw[: 2 * item_len], bytes(2 * item_len))
        self.assertEqual(raw[3 * item_len :], bytes(item_len))
        src_addr = ctypes.addressof(dst) + 2 * item_len
        self.assertEqual(
            AuxDataCodec.serialize_data_from_buffer(src_addr, item_len), payload
        )

    def test_partial_item_and_buffer_types(self):
        item_len = 32
        other = (ctypes.c_byte * item_len)()
        dst = (ctypes.c_byte * (item_len * 3))()
        kv_args = self._kv_args([(other, item_len), (dst, item_len)])
        payload = bytes([255, 0, 128, 7, 1])
        for i, data in enumerate((payload, bytearray(payload), memoryview(payload))):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 1, i, data)
            start = i * item_len
            self.assertEqual(bytes(dst)[start : start + len(payload)], payload)
            self.assertEqual(
                bytes(dst)[start + len(payload) : start + item_len],
                bytes(item_len - len(payload)),
            )
        self.assertEqual(bytes(other), bytes(item_len))

    def test_empty_payload_is_a_no_op(self):
        dst = (ctypes.c_byte * 8)(*range(8))
        AuxDataCodec.deserialize_data_to_buffer(self._kv_args([(dst, 8)]), 0, 0, b"")
        self.assertEqual(bytes(dst), bytes(range(8)))


if __name__ == "__main__":
    unittest.main()
