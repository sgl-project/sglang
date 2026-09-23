"""Single-NPU PD checksum checks against Python's independent zlib oracle."""

import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.disaggregation.checksum import (
    KvChecksumComputer,
    page_indices_for_request,
)
from sglang.srt.hardware_backend.npu.checksum import adler32_strided_checksum
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="full-1-npu-a3", nightly=True)


def reference(tensors, strides, indices):
    value = 1
    for tensor, stride, idx in zip(tensors, strides, indices):
        raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        for row in idx.cpu().tolist():
            value = zlib.adler32(raw[row * stride : (row + 1) * stride], value)
    return value


class TestNPUKvChecksum(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        torch.manual_seed(42)

    def check_descriptors(self, tensors, strides, indices):
        actual = adler32_strided_checksum(
            [t.data_ptr() for t in tensors], strides, indices
        )
        self.assertEqual(actual, reference(tensors, strides, indices))
        return actual

    def test_empty(self):
        self.assertEqual(adler32_strided_checksum([], [], []), 1)
        empty = torch.empty(0, dtype=torch.int64, device=self.device)
        self.assertEqual(adler32_strided_checksum([0], [64], [empty]), 1)
        idx = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.assertEqual(adler32_strided_checksum([0], [0], [idx]), 1)

    def test_byte_tails_order_and_duplicates(self):
        for stride in (1, 31, 1023, 1024, 1025, 65521, 70001):
            with self.subTest(stride=stride):
                data = torch.randint(0, 256, (8, stride), dtype=torch.uint8).to(
                    self.device
                )
                idx = torch.tensor(
                    [7, 0, 3, 3, 1], dtype=torch.int64, device=self.device
                )
                self.check_descriptors([data], [stride], [idx])

    def test_multiple_dtypes_and_empty_component(self):
        tensors = [
            torch.randn(9, 35, dtype=dtype).to(self.device)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ]
        indices = [
            torch.tensor(rows, dtype=torch.int32, device=self.device)
            for rows in ([8, 1, 4], [], [2, 0])
        ]
        self.check_descriptors(tensors, [t[0].nbytes for t in tensors], indices)

    def test_modular_overflow(self):
        data = torch.full((257, 4097), 255, dtype=torch.uint8, device=self.device)
        idx = torch.arange(256, -1, -1, dtype=torch.int64, device=self.device)
        self.check_descriptors([data], [4097], [idx])

    def test_selection_and_corruption(self):
        data = torch.arange(256, dtype=torch.int32).view(8, 32).to(self.device)
        idx = torch.tensor([5, 1], dtype=torch.int64, device=self.device)
        first = self.check_descriptors([data], [128], [idx])
        data[0, 0] += 1
        self.assertEqual(first, self.check_descriptors([data], [128], [idx]))
        data[5, 0] += 1
        self.assertNotEqual(first, self.check_descriptors([data], [128], [idx]))
        self.assertNotEqual(
            self.check_descriptors([data], [128], [idx]),
            self.check_descriptors([data], [128], [idx.flip(0)]),
        )

    def test_paged_and_fia_views(self):
        page_size = 4
        storage = torch.randn(2, 8, page_size, 3, 16, dtype=torch.bfloat16).to(
            self.device
        )
        # One full page and one partial page, in request order, at nonadjacent
        # physical locations. Like the CUDA path, checksum includes full pages.
        locations = torch.tensor(
            [[20, 21, 22, 23, 8, 9]], dtype=torch.int64, device=self.device
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
            req_to_token_pool=SimpleNamespace(req_to_token=locations),
        )
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))
        idx = page_indices_for_request(scheduler, req, 6)
        self.assertEqual(idx.cpu().tolist(), [5, 2])
        for fia in (False, True):
            with self.subTest(fia=fia):
                tensors = [
                    layer.view(-1, 1, 3, 16) if fia else layer for layer in storage
                ]
                strides = [
                    t[0].nbytes * page_size if fia else t[0].nbytes for t in tensors
                ]
                computer = KvChecksumComputer(
                    self.device, [t.data_ptr() for t in tensors], strides
                )
                self.assertEqual(
                    computer.compute(idx), reference(tensors, strides, [idx, idx])
                )

    def test_nested_state_and_npu_dispatch(self):
        kv = torch.randn(8, 35, dtype=torch.bfloat16).to(self.device)
        states = [torch.randn(4, width).to(self.device) for width in (13, 27)]
        kv_idx = torch.tensor([6, 1, 0], dtype=torch.int64, device=self.device)
        state_idx = torch.tensor([2], dtype=torch.int64, device=self.device)
        computer = KvChecksumComputer(
            self.device,
            [kv.data_ptr()],
            [kv[0].nbytes],
            [[states[0].data_ptr()], [states[1].data_ptr()]],
            [[states[0][0].nbytes], [states[1][0].nbytes]],
            state_types=["mamba", "mamba"],
        )
        expected = reference(
            [kv] + states,
            [t[0].nbytes for t in [kv] + states],
            [kv_idx, state_idx, state_idx],
        )
        with patch(
            "sglang.srt.hardware_backend.npu.checksum.adler32_strided_checksum",
            wraps=adler32_strided_checksum,
        ) as npu_checksum:
            self.assertEqual(computer.compute(kv_idx, state_idx), expected)
            npu_checksum.assert_called_once()
        states[1][2, 0] += 1
        self.assertNotEqual(computer.compute(kv_idx, state_idx), expected)

    def test_heterogeneous_state_rejected(self):
        for state_types in (["swa", "dsv4_c128"], ["mamba", "dsa"]):
            with self.subTest(state_types=state_types):
                with self.assertRaisesRegex(NotImplementedError, "state components"):
                    KvChecksumComputer(
                        self.device, [1], [1], state_types=state_types
                    )

    def test_invalid_descriptors(self):
        idx = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        data = torch.zeros((2, 64), dtype=torch.uint8, device=self.device)
        with self.assertRaises(ValueError):
            adler32_strided_checksum([data.data_ptr()], [], [idx])
        for invalid in (idx.view(1, 2), idx.to(torch.float32), idx.cpu()):
            with self.assertRaises(ValueError):
                adler32_strided_checksum([data.data_ptr()], [64], [invalid])


if __name__ == "__main__":
    unittest.main()
