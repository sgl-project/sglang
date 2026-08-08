"""Unit tests for the shm rung of the TensorRef ladder."""

import os
import unittest
import uuid
from contextlib import contextmanager
from multiprocessing import shared_memory
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.managers.tokenizer_manager import _input_embeds_shape
from sglang.srt.utils.shm_transport_utils import (
    is_shm_ref,
    package_hidden_states,
    read_shm_tensor,
    validate_shm_tensor_buffer,
    validate_shm_tensor_ref,
    write_shm_tensor_buffer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _owned_buffer(shape, dtype=np.float32):
    dtype = np.dtype(dtype)
    name = f"sgl_shm_test_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    size = int(np.prod(shape)) * dtype.itemsize
    segment = shared_memory.SharedMemory(name=name, create=True, size=size)
    ref = {
        "transport": "shm",
        "name": name,
        "dtype": dtype.name,
        "shape": list(shape),
    }
    try:
        yield segment, ref
    finally:
        segment.close()
        segment.unlink()


class TestShmTensorTransport(CustomTestCase):
    def test_read_peer_owned_buffer(self):
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        with _owned_buffer(tensor.shape) as (segment, ref):
            np.ndarray(tensor.shape, dtype=tensor.dtype, buffer=segment.buf)[...] = (
                tensor
            )
            with patch("sglang.srt.utils.shm_transport_utils._untrack"):
                np.testing.assert_array_equal(read_shm_tensor(ref), tensor)

    def test_package_hidden_states_writes_caller_owned_buffer(self):
        chunks = [torch.ones(2, 4), torch.zeros(4)]
        with _owned_buffer((3, 4)) as (segment, ref):
            with patch("sglang.srt.utils.shm_transport_utils._untrack"):
                self.assertIs(
                    package_hidden_states(chunks, output_buffer=ref),
                    ref,
                )
            rows = np.ndarray((3, 4), dtype=np.float32, buffer=segment.buf).copy()
            np.testing.assert_array_equal(rows[:2], np.ones((2, 4)))
            np.testing.assert_array_equal(rows[-1], np.zeros(4))

            probe = shared_memory.SharedMemory(name=ref["name"])
            probe.close()

    def test_srt_only_attaches_closes_and_never_unlinks(self):
        tensor = np.arange(6, dtype=np.float32).reshape(2, 3)
        ref = {
            "transport": "shm",
            "name": "sgl_shm_test_123_deadbeef",
            "dtype": "float32",
            "shape": [2, 3],
        }
        segment = MagicMock(
            _name=f"/{ref['name']}",
            size=tensor.nbytes,
            buf=bytearray(tensor.nbytes),
        )
        with (
            patch(
                "sglang.srt.utils.shm_transport_utils.shared_memory.SharedMemory",
                return_value=segment,
            ) as open_segment,
            patch("sglang.srt.utils.shm_transport_utils._untrack"),
        ):
            validate_shm_tensor_buffer(
                ref,
                shape=(2, 3),
                dtype="float32",
            )
            write_shm_tensor_buffer(ref, tensor)

        self.assertEqual(open_segment.call_count, 2)
        for call in open_segment.call_args_list:
            self.assertEqual(call.kwargs, {"name": ref["name"]})
        self.assertEqual(segment.close.call_count, 2)
        segment.unlink.assert_not_called()
        np.testing.assert_array_equal(
            np.ndarray(tensor.shape, dtype=tensor.dtype, buffer=segment.buf), tensor
        )

    def test_write_failure_closes_but_does_not_unlink(self):
        tensor = np.ones((2, 3), dtype=np.float32)
        ref = {
            "transport": "shm",
            "name": "sgl_shm_test_123_deadbeef",
            "dtype": "float32",
            "shape": [2, 3],
        }
        segment = MagicMock(
            _name=f"/{ref['name']}",
            size=tensor.nbytes - 1,
            buf=bytearray(tensor.nbytes - 1),
        )
        with (
            patch(
                "sglang.srt.utils.shm_transport_utils.shared_memory.SharedMemory",
                return_value=segment,
            ),
            patch("sglang.srt.utils.shm_transport_utils._untrack"),
            self.assertRaisesRegex(ValueError, "segment has"),
        ):
            write_shm_tensor_buffer(ref, tensor)

        segment.close.assert_called_once_with()
        segment.unlink.assert_not_called()

    def test_buffer_validation_requires_exact_layout_and_capacity(self):
        with (
            _owned_buffer((2, 3)) as (_, ref),
            patch("sglang.srt.utils.shm_transport_utils._untrack"),
        ):
            validate_shm_tensor_buffer(
                ref,
                shape=(2, 3),
                dtype="float32",
            )
            with self.assertRaisesRegex(ValueError, "expected shape"):
                validate_shm_tensor_buffer(
                    ref,
                    shape=(3, 2),
                    dtype="float32",
                )
            with self.assertRaisesRegex(ValueError, "expected dtype"):
                validate_shm_tensor_buffer(
                    ref,
                    shape=(2, 3),
                    dtype="float16",
                )

        for physical_shape in ((1, 1), (3, 3)):
            with _owned_buffer(physical_shape) as (_, mismatched_ref):
                ref = {**mismatched_ref, "shape": [2, 3]}
                with (
                    self.subTest(physical_shape=physical_shape),
                    patch("sglang.srt.utils.shm_transport_utils._untrack"),
                    self.assertRaisesRegex(ValueError, "needs exactly"),
                ):
                    validate_shm_tensor_buffer(
                        ref,
                        shape=(2, 3),
                        dtype="float32",
                    )

    def test_rejects_unsafe_dtypes_and_names_before_opening(self):
        for dtype in (None, "object", "str32", "complex64"):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                read_shm_tensor(
                    {
                        "transport": "shm",
                        "name": "sgl_shm_test_123_deadbeef",
                        "dtype": dtype,
                        "shape": [1],
                    }
                )

        ref = {"transport": "shm", "dtype": "float32", "shape": [1]}
        for name in (
            "x",
            "psm_deadbeef",
            "sgl_shm_test_0_deadbeef",
            "sgl_shm_test_123_not-hex!",
            f"sgl_shm_{'x' * 33}_123_deadbeef",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_shm_tensor_ref({**ref, "name": name})
        self.assertEqual(
            validate_shm_tensor_ref(
                {**ref, "name": "sgl_shm_input_embeds_123_deadbeef"}
            )[0],
            (1,),
        )

    def test_is_shm_ref(self):
        self.assertFalse(is_shm_ref([[0.0]]))
        self.assertFalse(is_shm_ref({"transport": "rdma"}))
        self.assertTrue(is_shm_ref({"transport": "shm", "name": "x"}))


class TestInputEmbedsLen(CustomTestCase):
    def test_inline_rows(self):
        self.assertEqual(_input_embeds_shape([[0.0], [0.0]])[0], 2)

    def test_valid_ref(self):
        ref = {
            "transport": "shm",
            "name": "sgl_shm_test_123_deadbeef",
            "dtype": "float32",
            "shape": [5, 8],
        }
        self.assertEqual(_input_embeds_shape(ref)[0], 5)

    def test_malformed_refs_raise(self):
        for shape in (None, [5], [0, 8], [5, "8"]):
            with self.assertRaises(ValueError):
                _input_embeds_shape(
                    {
                        "transport": "shm",
                        "name": "sgl_shm_test_123_deadbeef",
                        "dtype": "f4",
                        "shape": shape,
                    }
                )
        with self.assertRaises(ValueError):
            _input_embeds_shape({"transport": "shm", "shape": [5, 8]})
        with self.assertRaises(ValueError):
            _input_embeds_shape(
                {
                    "transport": "shm",
                    "name": "sgl_shm_test_123_deadbeef",
                    "dtype": "object",
                    "shape": [5, 8],
                }
            )


if __name__ == "__main__":
    unittest.main()
