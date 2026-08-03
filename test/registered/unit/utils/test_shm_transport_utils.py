"""Unit tests for the shm rung of the TensorRef ladder."""

import unittest
from multiprocessing import shared_memory

import numpy as np
import torch

from sglang.srt.managers.tokenizer_manager import _input_embeds_len
from sglang.srt.utils.shm_transport_utils import (
    is_shm_ref,
    package_hidden_states,
    read_shm_tensor,
    write_shm_tensor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _unlink(ref):
    segment = shared_memory.SharedMemory(name=ref["name"])
    segment.close()
    segment.unlink()


class TestShmTensorTransport(CustomTestCase):
    def test_write_read_round_trip(self):
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        ref = write_shm_tensor(tensor, kind="test")
        try:
            self.assertTrue(is_shm_ref(ref))
            self.assertEqual(ref["shape"], [3, 4])
            np.testing.assert_array_equal(read_shm_tensor(ref), tensor)
        finally:
            _unlink(ref)

    def test_package_hidden_states_concatenates_chunks(self):
        chunks = [torch.ones(2, 4), torch.zeros(4)]
        ref = package_hidden_states(chunks, kind="test")
        try:
            rows = read_shm_tensor(ref)
            self.assertEqual(rows.shape, (3, 4))
            self.assertEqual(rows.dtype, np.float32)
            np.testing.assert_array_equal(rows[-1], np.zeros(4))
        finally:
            _unlink(ref)

    def test_is_shm_ref(self):
        self.assertFalse(is_shm_ref([[0.0]]))
        self.assertFalse(is_shm_ref({"transport": "rdma"}))
        self.assertTrue(is_shm_ref({"transport": "shm", "name": "x"}))


class TestInputEmbedsLen(CustomTestCase):
    def test_inline_rows(self):
        self.assertEqual(_input_embeds_len([[0.0], [0.0]]), 2)

    def test_valid_ref(self):
        ref = {"transport": "shm", "name": "x", "dtype": "float32", "shape": [5, 8]}
        self.assertEqual(_input_embeds_len(ref), 5)

    def test_malformed_refs_raise(self):
        for shape in (None, [5], [0, 8], [5, "8"]):
            with self.assertRaises(ValueError):
                _input_embeds_len(
                    {"transport": "shm", "name": "x", "dtype": "f4", "shape": shape}
                )
        with self.assertRaises(ValueError):
            _input_embeds_len({"transport": "shm", "shape": [5, 8]})


if __name__ == "__main__":
    unittest.main()
