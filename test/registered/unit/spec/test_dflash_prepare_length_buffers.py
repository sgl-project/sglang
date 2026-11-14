"""Length-buffer reuse must follow PyTorch's CUDA device resolution."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import dflash_info_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-small")


class TestPrepareLengthBuffers(unittest.TestCase):
    def setUp(self):
        self.draft = DFlashDraftInputV2.create_idle_input("cpu")
        self.pin_memory = patch.object(
            dflash_info_v2, "is_pin_memory_available", return_value=False
        )
        self.pin_memory.start()
        self.addCleanup(self.pin_memory.stop)

    @contextmanager
    def cuda_allocations(self):
        """Keep real CPU storage and simulate only CUDA allocation metadata."""
        empty = torch.empty
        allocations = []
        with patch.object(torch.cuda, "current_device", return_value=0) as current:

            def allocate(shape, *, device, **kwargs):
                target = torch.device(device)
                if target.type != "cuda":
                    return empty(shape, device=device, **kwargs)
                index = (
                    target.index if target.index is not None else current.return_value
                )
                buffer = SimpleNamespace(
                    device=torch.device("cuda", index), numel=lambda: shape[0]
                )
                allocations.append(buffer)
                return buffer

            with patch.object(torch, "empty", side_effect=allocate):
                yield current, allocations

    def test_unindexed_cuda_reuses_both_buffers(self):
        with self.cuda_allocations() as (_, allocations):
            self.draft._ensure_prepare_length_buffers(4, "cuda")
            cur = self.draft._prepare_cur_kv_lens_gpu_buf
            nxt = self.draft._prepare_nxt_kv_lens_gpu_buf
            for device in ("cuda", torch.device("cuda")) * 3:
                self.draft._ensure_prepare_length_buffers(4, device)
                self.assertIs(self.draft._prepare_cur_kv_lens_gpu_buf, cur)
                self.assertIs(self.draft._prepare_nxt_kv_lens_gpu_buf, nxt)
                self.assertEqual(cur.numel(), 32)
            self.assertEqual(len(allocations), 2)

    def test_explicit_cuda_index_does_not_use_current_device(self):
        with self.cuda_allocations() as (current, allocations):
            for device in ("cuda:2", torch.device("cuda:2")):
                self.draft._ensure_prepare_length_buffers(4, device)
            self.assertEqual(len(allocations), 2)
            self.assertEqual(allocations[0].device, torch.device("cuda:2"))
            current.assert_not_called()

    def test_capacity_grows_only_when_the_batch_outgrows_it(self):
        with self.cuda_allocations() as (_, allocations):
            for bs, capacity in ((4, 32), (32, 32), (33, 64), (40, 64)):
                self.draft._ensure_prepare_length_buffers(bs, "cuda")
                self.assertEqual(
                    self.draft._prepare_cur_kv_lens_gpu_buf.numel(), capacity
                )
                self.assertEqual(
                    self.draft._prepare_nxt_kv_lens_gpu_buf.numel(), capacity
                )
                self.assertEqual(
                    self.draft._prepare_batch_seq_lens_cpu_buf.numel(), capacity
                )
            self.assertEqual(len(allocations), 4)

    def test_unindexed_cuda_follows_a_changed_current_device(self):
        with self.cuda_allocations() as (current, allocations):
            self.draft._ensure_prepare_length_buffers(4, "cuda")
            old = self.draft._prepare_cur_kv_lens_gpu_buf
            cpu = self.draft._prepare_batch_seq_lens_cpu_buf
            current.return_value = 1
            self.draft._ensure_prepare_length_buffers(4, torch.device("cuda"))
            self.assertIsNot(self.draft._prepare_cur_kv_lens_gpu_buf, old)
            self.assertEqual(allocations[-1].device, torch.device("cuda:1"))
            self.assertIs(self.draft._prepare_batch_seq_lens_cpu_buf, cpu)
            self.draft._ensure_prepare_length_buffers(4, "cuda")
            self.assertEqual(len(allocations), 4)

    def test_explicit_cuda_device_change_reallocates_both_buffers(self):
        with self.cuda_allocations() as (current, allocations):
            self.draft._ensure_prepare_length_buffers(4, "cuda:0")
            old = self.draft._prepare_cur_kv_lens_gpu_buf
            self.draft._ensure_prepare_length_buffers(4, "cuda:1")
            self.assertIsNot(self.draft._prepare_cur_kv_lens_gpu_buf, old)
            self.assertEqual(len(allocations), 4)
            for buffer in allocations[-2:]:
                self.assertEqual(buffer.device, torch.device("cuda:1"))
            current.assert_not_called()

    def test_cpu_buffers_keep_storage_and_dtypes(self):
        with patch.object(torch.cuda, "current_device") as current:
            self.draft._ensure_prepare_length_buffers(4, "cpu")
            names = (
                "_prepare_batch_seq_lens_cpu_buf",
                "_prepare_cur_kv_lens_cpu_buf",
                "_prepare_nxt_kv_lens_cpu_buf",
                "_prepare_cur_kv_lens_gpu_buf",
                "_prepare_nxt_kv_lens_gpu_buf",
            )
            buffers = [getattr(self.draft, name) for name in names]
            self.draft._ensure_prepare_length_buffers(32, torch.device("cpu"))
            for name, buffer in zip(names, buffers):
                self.assertIs(getattr(self.draft, name), buffer)
                self.assertEqual(buffer.device, torch.device("cpu"))
                self.assertEqual(buffer.numel(), 32)
            self.assertEqual(buffers[0].dtype, torch.int64)
            self.assertTrue(all(buffer.dtype == torch.int32 for buffer in buffers[1:]))
            current.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPrepareLengthBuffersCUDA(unittest.TestCase):
    def test_real_cuda_storage_is_reused_until_capacity_is_exceeded(self):
        draft = DFlashDraftInputV2.create_idle_input("cpu")
        with torch.cuda.device(torch.cuda.current_device()):
            draft._ensure_prepare_length_buffers(4, "cuda")
            cur, nxt = (
                draft._prepare_cur_kv_lens_gpu_buf,
                draft._prepare_nxt_kv_lens_gpu_buf,
            )
            cur.fill_(7)
            for device in ("cuda", torch.device("cuda"), cur.device):
                draft._ensure_prepare_length_buffers(32, device)
                self.assertEqual(
                    draft._prepare_cur_kv_lens_gpu_buf.data_ptr(), cur.data_ptr()
                )
                self.assertEqual(
                    draft._prepare_nxt_kv_lens_gpu_buf.data_ptr(), nxt.data_ptr()
                )
                torch.testing.assert_close(cur, torch.full_like(cur, 7))
            draft._ensure_prepare_length_buffers(33, "cuda")
            self.assertEqual(draft._prepare_cur_kv_lens_gpu_buf.numel(), 64)
            self.assertNotEqual(
                draft._prepare_cur_kv_lens_gpu_buf.data_ptr(), cur.data_ptr()
            )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_real_current_device_change_moves_both_buffers(self):
        draft = DFlashDraftInputV2.create_idle_input("cpu")
        for index in (0, 1):
            with torch.cuda.device(index):
                draft._ensure_prepare_length_buffers(4, "cuda")
                self.assertEqual(
                    draft._prepare_cur_kv_lens_gpu_buf.device,
                    torch.device("cuda", index),
                )
                self.assertEqual(
                    draft._prepare_nxt_kv_lens_gpu_buf.device,
                    torch.device("cuda", index),
                )


if __name__ == "__main__":
    unittest.main()
