import unittest

import torch

from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
from sglang.srt.lora.torch_ops.lora_ops import (
    sgmv_expand,
    sgmv_expand_slice,
    sgmv_shrink,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase


class TestTorchNativeLoRABackend(CustomTestCase):

    device = "cpu"
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        req_pool_indices=None,
        seq_lens=None,
        out_cache_loc=None,
        seq_lens_sum=6,
        extend_seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        extend_seq_lens_cpu=[1, 1],
    )
    weight_indices = [0, 1]
    lora_ranks = [1, 1]
    scalings = [1.0, 0.5]
    use_cuda_graph = False

    @classmethod
    def setUpClass(cls):
        cls.backend = TorchNativeLoRABackend(max_loras_per_batch=2, device=cls.device)
        cls.backend.prepare_lora_batch(
            forward_batch=cls.forward_batch,
            weight_indices=cls.weight_indices,
            lora_ranks=cls.lora_ranks,
            scalings=cls.scalings,
            use_cuda_graph=cls.use_cuda_graph,
        )

    def test_run_lora_a_sgemm(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        num_loras = 3
        dtype = torch.float32

        x = torch.randn(batch_size, input_dim, dtype=dtype)
        weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)

        total_seq_len, _ = x.shape
        _, weight_output_dim, _ = weights.shape
        output_tensor = torch.zeros(
            (total_seq_len, weight_output_dim), dtype=dtype, device=self.device
        )
        sgmv_shrink(
            x,
            weights,
            output_tensor,
            self.backend.batch_info.seg_lens,
            self.backend.batch_info.weight_indices,
            1.0,
        )
        scaling = torch.repeat_interleave(
            self.backend.batch_info.scalings[self.backend.batch_info.weight_indices],
            self.backend.batch_info.seg_lens,
            output_size=total_seq_len,
        ).unsqueeze(-1)
        expect_output = output_tensor * scaling

        actual_output = self.backend.run_lora_a_sgemm(x, weights)

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_run_lora_b_sgemm(self):
        batch_size = 2
        input_dim = 6
        output_dim = 4
        num_loras = 3
        dtype = torch.float32

        x = torch.randn(batch_size, input_dim, dtype=dtype)
        weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)

        total_seq_len, _ = x.shape
        _, weight_output_dim, _ = weights.shape
        output_tensor = torch.zeros(
            (total_seq_len, weight_output_dim), dtype=dtype, device=self.device
        )
        sgmv_expand(
            x,
            weights,
            output_tensor,
            self.backend.batch_info.seg_lens,
            self.backend.batch_info.weight_indices,
            True,
        )
        expect_output = output_tensor

        actual_output = self.backend.run_lora_b_sgemm(x, weights)

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_run_qkv_lora(self):
        batch_size = 2
        input_dim = 6
        output_dim = 4
        num_loras = 3
        dtype = torch.float32

        x = torch.randn(batch_size, input_dim, dtype=dtype)
        qkv_lora_a = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        qkv_lora_b = torch.randn(num_loras, input_dim, output_dim, dtype=dtype)
        output_offset_cpu = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32)

        num_slices = 3
        total_seq_len, _ = x.shape
        _, weight_intermediate_dim, _ = qkv_lora_a.shape
        _, weight_out_dim, _ = qkv_lora_b.shape
        max_rank = weight_intermediate_dim // num_slices
        output_tensor = torch.zeros(
            (total_seq_len, weight_out_dim), device=x.device, dtype=x.dtype
        )
        lora_a_output = torch.zeros(
            total_seq_len, weight_intermediate_dim, dtype=x.dtype, device=x.device
        )
        sgmv_shrink(
            x,
            qkv_lora_a,
            lora_a_output,
            self.backend.batch_info.seg_lens,
            self.backend.batch_info.weight_indices,
            1.0,
        )
        scaling = torch.repeat_interleave(
            self.backend.batch_info.scalings[self.backend.batch_info.weight_indices],
            self.backend.batch_info.seg_lens,
            output_size=total_seq_len,
        ).unsqueeze(-1)
        lora_a_output = lora_a_output * scaling
        for slice_id in range(num_slices):
            slice_offset = output_offset_cpu[slice_id]
            slice_offset_next = output_offset_cpu[slice_id + 1]
            slice_size = slice_offset_next - slice_offset
            sgmv_expand_slice(
                lora_a_output[:, (max_rank * slice_id) : (max_rank * (slice_id + 1))],
                qkv_lora_b[:, slice_offset:slice_offset_next],
                output_tensor,
                self.backend.batch_info.seg_lens,
                self.backend.batch_info.weight_indices,
                slice_offset,
                slice_size,
                True,
            )
        expect_output = output_tensor
        actual_output = self.backend.run_qkv_lora(
            x, qkv_lora_a, qkv_lora_b, None, output_offset_cpu, 0
        )
        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_run_gate_up_lora(self):
        batch_size = 2
        input_dim = 6
        output_dim = 4
        num_loras = 3
        dtype = torch.float32

        num_slices = 2

        x = torch.randn(batch_size, input_dim, dtype=dtype)
        gate_up_lora_a = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        gate_up_lora_b = torch.randn(
            num_loras, output_dim, output_dim // num_slices, dtype=dtype
        )

        total_seq_len, _ = x.shape
        _, weight_intermediate_dim, _ = gate_up_lora_a.shape
        _, weight_out_dim, _ = gate_up_lora_b.shape
        slice_size = weight_out_dim // num_slices
        max_rank = weight_intermediate_dim // num_slices
        output_tensor = torch.zeros(
            (total_seq_len, weight_out_dim), device=x.device, dtype=x.dtype
        )
        lora_a_output = torch.zeros(
            total_seq_len, weight_intermediate_dim, dtype=x.dtype, device=x.device
        )
        sgmv_shrink(
            x,
            gate_up_lora_a,
            lora_a_output,
            self.backend.batch_info.seg_lens,
            self.backend.batch_info.weight_indices,
            1.0,
        )
        scaling = torch.repeat_interleave(
            self.backend.batch_info.scalings[self.backend.batch_info.weight_indices],
            self.backend.batch_info.seg_lens,
            output_size=total_seq_len,
        ).unsqueeze(-1)
        lora_a_output = lora_a_output * scaling
        slice_offset = 0
        for slice_id in range(num_slices):
            sgmv_expand_slice(
                lora_a_output[:, (max_rank * slice_id) : (max_rank * (slice_id + 1))],
                gate_up_lora_b[:, slice_offset : slice_offset + slice_size],
                output_tensor,
                self.backend.batch_info.seg_lens,
                self.backend.batch_info.weight_indices,
                slice_offset,
                slice_size,
                True,
            )
            slice_offset += slice_size
        expect_output = output_tensor
        actual_output = self.backend.run_gate_up_lora(x, gate_up_lora_a, gate_up_lora_b)
        self.assertTrue(torch.allclose(actual_output, expect_output))


if __name__ == "__main__":
    unittest.main()
