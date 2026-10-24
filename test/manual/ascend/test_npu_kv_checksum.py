"""NPU PD checksum integration checks for page mapping and kernel dispatch."""

import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mem_cache.checksum import adler32_strided_checksum

from sglang.srt.disaggregation.checksum import (
    KvChecksumComputer,
    page_indices_for_request,
)


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
        )
        expected = reference(
            [kv] + states,
            [t[0].nbytes for t in [kv] + states],
            [kv_idx, state_idx, state_idx],
        )
        with patch(
            "sgl_kernel_npu.mem_cache.checksum.adler32_strided_checksum",
            wraps=adler32_strided_checksum,
        ) as npu_checksum:
            self.assertEqual(computer.compute(kv_idx, state_idx), expected)
            npu_checksum.assert_called_once()
        states[1][2, 0] += 1
        self.assertNotEqual(computer.compute(kv_idx, state_idx), expected)


if __name__ == "__main__":
    unittest.main()
