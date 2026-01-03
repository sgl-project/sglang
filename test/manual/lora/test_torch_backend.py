import unittest

import torch

from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.lora_utils import reference_sgmv_expand, reference_sgmv_shrink
from sglang.test.test_utils import CustomTestCase


class TestTorchNativeLoRABackend(CustomTestCase):

    device = "cpu"
    weight_indices = [0, 1]
    lora_ranks = [1, 1]
    scalings = [1.0, 0.5]
    seq_lens = [1, 1]
    use_cuda_graph = False

    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        req_pool_indices=None,
        seq_lens=None,
        out_cache_loc=None,
        seq_lens_sum=6,
        extend_seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        extend_seq_lens_cpu=seq_lens,
    )

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

        weight_indices_tensor = torch.tensor(
            self.weight_indices, dtype=torch.int32, device=self.device
        )

        seg_len_tensor = torch.tensor(
            self.seq_lens, dtype=torch.int32, device=self.device
        )

        lora_ranks_tensor = torch.tensor(
            self.lora_ranks, dtype=torch.int32, device=self.device
        )

        scalings_tensor = torch.tensor(
            self.scalings, dtype=torch.float, device=self.device
        )

        expect_output = reference_sgmv_shrink(
            x,
            weights,
            weight_indices_tensor,
            seg_len_tensor,
            lora_ranks_tensor,
            scalings_tensor,
        )

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
        _, weight_out_dim, _ = weights.shape

        weight_indices_tensor = torch.tensor(
            self.weight_indices, dtype=torch.int32, device=self.device
        )

        seg_len_tensor = torch.tensor(
            self.seq_lens, dtype=torch.int32, device=self.device
        )

        lora_ranks_tensor = torch.tensor(
            self.lora_ranks, dtype=torch.int32, device=self.device
        )

        expect_output = reference_sgmv_expand(
            x,
            weights,
            weight_indices_tensor,
            seg_len_tensor,
            lora_ranks_tensor,
            slice_offsets=torch.tensor(
                [0, weight_out_dim], dtype=torch.int32, device="cpu"
            ),
        )

        actual_output = self.backend.run_lora_b_sgemm(x, weights)

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_run_qkv_lora(self):
        batch_size = 2
        num_loras = 3
        input_dim = 6
        output_offset = [0, 3, 6, 9, 12]
        output_dim = output_offset[-1]
        num_slices = len(output_offset) - 1
        max_lora_rank = max(self.lora_ranks)
        dtype = torch.float32

        x = torch.randn(batch_size, input_dim, dtype=dtype)
        output_offset_cpu = torch.tensor(output_offset, dtype=torch.int32)
        qkv_lora_a = torch.randn(
            num_loras, max_lora_rank * num_slices, input_dim, dtype=dtype
        )
        qkv_lora_b = torch.randn(
            num_loras, output_dim, max_lora_rank * num_slices, dtype=dtype
        )

        weight_indices_tensor = torch.tensor(
            self.weight_indices, dtype=torch.int32, device=self.device
        )

        seg_len_tensor = torch.tensor(
            self.seq_lens, dtype=torch.int32, device=self.device
        )

        lora_ranks_tensor = torch.tensor(
            self.lora_ranks, dtype=torch.int32, device=self.device
        )

        scalings_tensor = torch.tensor(
            self.scalings, dtype=torch.float, device=self.device
        )

        expect_lora_a_output = reference_sgmv_shrink(
            x,
            qkv_lora_a,
            weight_indices_tensor,
            seg_len_tensor,
            lora_ranks_tensor,
            scalings_tensor,
            num_slices,
        )

        expect_output = reference_sgmv_expand(
            expect_lora_a_output,
            qkv_lora_b,
            weight_indices_tensor,
            seg_len_tensor,
            lora_ranks_tensor,
            output_offset_cpu,
        )

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

        max_lora_rank = max(self.lora_ranks)

        num_slices = 2

        x = torch.randn(batch_size, input_dim, dtype=dtype)
        gate_up_lora_a = torch.randn(
            num_loras, max_lora_rank * num_slices, input_dim, dtype=dtype
        )
        gate_up_lora_b = torch.randn(
            num_loras, output_dim, max_lora_rank * num_slices, dtype=dtype
        )

        _, weight_out_dim, _ = gate_up_lora_b.shape
        slice_size = weight_out_dim // num_slices
        output_offset = torch.tensor(
            [0, slice_size, weight_out_dim], dtype=torch.int32, device="cpu"
        )

        weight_indices_tensor = torch.tensor(
            self.weight_indices, dtype=torch.int32, device=self.device
        )

        seg_len_tensor = torch.tensor(
            self.seq_lens, dtype=torch.int32, device=self.device
        )

        lora_ranks_tensor = torch.tensor(
            self.lora_ranks, dtype=torch.int32, device=self.device
        )

        scalings_tensor = torch.tensor(
            self.scalings, dtype=torch.float, device=self.device
        )

        expect_lora_a_output = reference_sgmv_shrink(
            x,
            gate_up_lora_a,
            weight_indices_tensor,
            seg_len_tensor,
            lora_ranks_tensor,
            scalings_tensor,
            num_slices,
        )

        expect_output = reference_sgmv_expand(
            expect_lora_a_output,
            gate_up_lora_b,
            weight_indices_tensor,
            seg_len_tensor,
            lora_ranks_tensor,
            slice_offsets=output_offset,
        )

        actual_output = self.backend.run_gate_up_lora(x, gate_up_lora_a, gate_up_lora_b)
        self.assertTrue(torch.allclose(actual_output, expect_output))


if __name__ == "__main__":
    unittest.main()
