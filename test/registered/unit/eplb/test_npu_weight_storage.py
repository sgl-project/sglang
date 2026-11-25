"""Regress EPLB rejecting packed NZ views or corrupting shared expert slots."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.eplb import _ExpertWeightAdapter
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import torch_npu
except ImportError:
    torch_npu = None

register_npu_ci(est_time=15, suite="nightly-test-1-npu-a3", nightly=True)


@unittest.skipUnless(torch_npu is not None and torch.npu.is_available(), "NPU required")
class TestNpuExpertWeightStorage(CustomTestCase):
    def setUp(self):
        self.previous_internal_format = (
            torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
        )
        torch.npu.config.allow_internal_format = True

    def tearDown(self):
        torch.npu.synchronize()
        torch.npu.config.allow_internal_format = self.previous_internal_format

    @staticmethod
    def _physical_copy(weight):
        out = torch.empty(
            (weight.shape[0], weight.untyped_storage().nbytes() // weight.shape[0]),
            dtype=torch.uint8,
            device=weight.device,
        )
        torch.ops.npu.copy_memory_(out, weight.view(torch.uint8), True)
        return out

    def test_packed_nz_migration_preserves_storage_and_graph(self):
        # Match ModelSlim: convert INT8 to NZ, then export an INT32 dtype view.
        # The last expert is shared, and the unaligned dimensions exercise NZ padding.
        data = (
            torch.arange(4 * 129 * 68, dtype=torch.int32)
            .remainder(251)
            .to(torch.int8)
            .reshape(4, 129, 68)
            .npu()
        )
        full_weight = torch.ops.npu.npu_format_cast(data, 29).view(torch.int32)
        weight = full_weight[:3].detach()
        descriptor = (weight.data_ptr(), weight.shape, weight.stride())
        expected_bytes = self._physical_copy(full_weight).cpu()
        adapter = _ExpertWeightAdapter({0: [weight]}, [0], rank=0)
        view = adapter.make_weight_views([weight])[0]
        temp = adapter.create_temp_buffers()[0]
        self.assertEqual(temp.shape, expected_bytes[:3].shape)
        observed = torch.empty_like(expected_bytes, device=weight.device)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch.ops.npu.copy_memory_(observed, full_weight.view(torch.uint8), True)

        # Stage all sources before writing aliased destinations.
        expected_bytes[:3] = expected_bytes[:3].roll(1, dims=0)
        temp.copy_(expected_bytes[:3].to(weight.device))
        view.copy_(temp)
        graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.equal(observed.cpu(), expected_bytes))
        self.assertEqual((weight.data_ptr(), weight.shape, weight.stride()), descriptor)
        self.assertEqual(torch.ops.npu.get_npu_format(weight), 29)
        unpacked = torch.ops.npu.npu_format_cast(full_weight.view(torch.int8), 2)
        expected_data = data.cpu()
        expected_data[:3] = expected_data[:3].roll(1, dims=0)
        self.assertTrue(torch.equal(unpacked.cpu(), expected_data))


if __name__ == "__main__":
    unittest.main()
