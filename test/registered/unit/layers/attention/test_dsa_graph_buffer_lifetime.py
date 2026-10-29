import os
import unittest
from unittest import mock

from sglang.srt.layers.attention.dsa import graph_buffer_lifetime as lifetime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeStorage:
    def __init__(self, start: int, nbytes: int):
        self._start = start
        self._nbytes = nbytes

    def data_ptr(self):
        return self._start

    def nbytes(self):
        return self._nbytes


class _FakeTensor:
    is_cuda = True
    device = "cuda:0"
    dtype = "torch.float32"

    def __init__(
        self,
        data_start: int,
        *,
        shape=(2, 4),
        stride=(8, 1),
        element_size=4,
        storage_start=None,
        storage_nbytes=64,
    ):
        self._data_start = data_start
        self.shape = shape
        self._stride = stride
        self._element_size = element_size
        self._storage = _FakeStorage(
            data_start if storage_start is None else storage_start, storage_nbytes
        )

    def stride(self):
        return self._stride

    def data_ptr(self):
        return self._data_start

    def element_size(self):
        return self._element_size

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def untyped_storage(self):
        return self._storage


class TestGraphBufferLifetime(unittest.TestCase):
    def setUp(self):
        lifetime._reset_for_test()

    def tearDown(self):
        lifetime._reset_for_test()

    def test_probe_records_capture_without_retaining_owner(self):
        tensor = _FakeTensor(1024)
        with (
            mock.patch.dict(os.environ, {"SGLANG_DSA_GRAPH_BUFFER_LIFETIME": "probe"}),
            mock.patch.object(
                lifetime.torch.cuda, "is_current_stream_capturing", return_value=True
            ),
        ):
            lifetime.register_graph_buffer(tensor, "logits")

        records, owners = lifetime._state_for_test()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].label, "logits")
        self.assertEqual(owners, [])

    def test_retain_keeps_captured_tensor_owner(self):
        tensor = _FakeTensor(2048)
        with (
            mock.patch.dict(os.environ, {"SGLANG_DSA_GRAPH_BUFFER_LIFETIME": "retain"}),
            mock.patch.object(
                lifetime.torch.cuda, "is_current_stream_capturing", return_value=True
            ),
        ):
            lifetime.register_graph_buffer(tensor, "topk")

        _, owners = lifetime._state_for_test()
        self.assertEqual(owners, [tensor])

    def test_reports_logical_overlap_before_nearest_buffer(self):
        overlapping = _FakeTensor(4096, shape=(1, 8), stride=(8, 1))
        nearest = _FakeTensor(8192, shape=(1, 8), stride=(8, 1))
        with (
            mock.patch.dict(os.environ, {"SGLANG_DSA_GRAPH_BUFFER_LIFETIME": "probe"}),
            mock.patch.object(
                lifetime.torch.cuda, "is_current_stream_capturing", return_value=True
            ),
        ):
            lifetime.register_graph_buffer(nearest, "nearest")
            lifetime.register_graph_buffer(overlapping, "overlap")

        target = _FakeTensor(4100, shape=(1, 2), stride=(2, 1))
        description = lifetime.describe_graph_buffer_overlap(target)
        self.assertTrue(description.startswith("logical-overlap:overlap:"))

    def test_warmup_does_not_record(self):
        tensor = _FakeTensor(12288)
        with (
            mock.patch.dict(os.environ, {"SGLANG_DSA_GRAPH_BUFFER_LIFETIME": "retain"}),
            mock.patch.object(
                lifetime.torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            lifetime.register_graph_buffer(tensor, "warmup")

        self.assertEqual(lifetime._state_for_test(), ([], []))


if __name__ == "__main__":
    unittest.main()
