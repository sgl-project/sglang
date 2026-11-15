"""Declared request windows retain packed bytes and reset row ownership."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.dsv41_request_window import RequestWindowSpec
from sglang.srt.mem_cache.kv.memory.request_window import RequestWindowMemory
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRequestWindowMemory(CustomTestCase):
    def test_packed_extents_and_fresh_request_row_reset(self):
        for layout in KVLayout:
            with self.subTest(layout=layout):
                spec = RequestWindowSpec.create(
                    num_slots=4,
                    layers=2,
                    page_size=64,
                    capacity=129,
                    workspace_rows=513,
                    kv_layout=layout,
                )
                memory = RequestWindowMemory(spec, dtype=torch.uint8, device="cpu")
                window = memory.view
                self.assertEqual(spec.capacity, 192)
                self.assertEqual(
                    memory.nbytes,
                    2 * 14 * layout.page_bytes(64)
                    + 2 * 832 * 8
                    + 10 * layout.page_bytes(64),
                )
                window.tags.fill_(19)
                memory.state.tensors["pages"].fill_(7)
                table = ReqToTokenPool(3, 16, "cpu", enable_memory_saver=False)
                table.register_on_alloc_rows(memory.reset_rows)
                (slot,) = table.alloc_rows(1)
                self.assertTrue(
                    torch.all(window.tags[:, slot * 192 : (slot + 1) * 192] == -1)
                )
                self.assertTrue(torch.all(window.tags[:, :192] == 19))
                # Tags invalidate the packed bytes; acquiring a row need not
                # clear or move its large payload, nor another request's data.
                self.assertTrue(torch.all(memory.state.tensors["pages"] == 7))
                window.tags[:, slot * 192 : (slot + 1) * 192].fill_(29)
                table.free_rows([slot])
                self.assertEqual(table.alloc_rows(1), [slot])
                self.assertTrue(
                    torch.all(window.tags[:, slot * 192 : (slot + 1) * 192] == -1)
                )
                expected = torch.tensor(
                    [slot * 3, slot * 3 + 1, slot * 3 + 2], dtype=torch.int32
                ).numpy()
                self.assertTrue((window.page_indices(slot, 0, 192) == expected).all())

    def test_attention_view_preserves_packed_token_dimension(self):
        """Request-owned packed pages must retain the HIP attention view contract."""
        for layout in KVLayout:
            with self.subTest(layout=layout):
                spec = RequestWindowSpec.create(
                    num_slots=2,
                    layers=1,
                    page_size=64,
                    capacity=128,
                    workspace_rows=256,
                    kv_layout=layout,
                )
                memory = RequestWindowMemory(spec, dtype=torch.uint8, device="cpu")
                for view in (memory.view.state, memory.view.workspace):
                    packed = view.kv_buffer[0]
                    attention = packed[:, : 64 * view.kv_cache_total_dim].view(
                        packed.shape[0], 64, 1, view.kv_cache_total_dim
                    )
                    self.assertEqual(attention.shape[-1], layout.bytes_per_token)
                    self.assertEqual(attention.data_ptr(), packed.data_ptr())

    def test_activation_cannot_resize_captured_backing(self):
        spec = RequestWindowSpec.create(
            num_slots=2,
            layers=1,
            page_size=64,
            capacity=128,
            workspace_rows=256,
            kv_layout=KVLayout.V41,
        )
        memory = RequestWindowMemory(spec, dtype=torch.uint8, device="cpu")
        view = memory.view
        pointer = view.workspace.kv_buffer[0].data_ptr()
        view.activate(SimpleNamespace(size=256))
        with self.assertRaisesRegex(RuntimeError, "workspace too small"):
            view.activate(SimpleNamespace(size=257))
        self.assertEqual(view.workspace.kv_buffer[0].data_ptr(), pointer)


if __name__ == "__main__":
    unittest.main()
