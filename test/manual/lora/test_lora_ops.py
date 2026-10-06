import random
import unittest

import torch

from sglang.srt.lora.torch_ops.graph_lora_ops import (
    sgemm_lora_a_embedding_graph_fwd,
    sgemm_lora_a_graph_fwd,
    sgemm_lora_b_graph_fwd,
)
from sglang.srt.lora.torch_ops.lora_ops import (
    sgemm_lora_a_embedding_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.test.lora_utils import (
    reference_embedding_lora_a_shrink,
    reference_sgmv_expand,
    reference_sgmv_shrink,
)
from sglang.test.test_utils import CustomTestCase


class TestLoraOps(CustomTestCase):
    def test_sgemm_lora_a_embedding_fwd(self):
        batch_size = 64
        input_dim = 1024
        num_loras = 3
        dtype = torch.float32
        vocab_size = 32000

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        inputs = torch.randint(vocab_size, (batch_size,), dtype=torch.int32)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, vocab_size, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float16, device="cpu"
        )

        expect_output = reference_embedding_lora_a_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
            vocab_size,
        )

        actual_output = sgemm_lora_a_embedding_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
            vocab_size,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgemm_lora_a_fwd(self):
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float16, device="cpu"
        )

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        actual_output = sgemm_lora_a_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgemm_lora_b_fwd(self):
        batch_size = 2
        output_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        inputs = torch.randn(batch_size, max_lora_rank, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, max_lora_rank, dtype=dtype)
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cpu")

        expect_output = reference_sgmv_expand(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        actual_output = sgemm_lora_b_fwd(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgemm_lora_a_embedding_fwd_expand(self):
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float32
        vocab_size = 32000

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        seq_len_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )

        seq_len = sum(seq_len_tensor)

        inputs = torch.randint(vocab_size, (seq_len,), dtype=torch.int32)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, vocab_size, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float16, device="cpu"
        )

        expect_output = reference_embedding_lora_a_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
            vocab_size,
        )

        actual_output = sgemm_lora_a_embedding_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
            vocab_size,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgemm_lora_a_fwd_expand(self):
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        seq_len_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )

        seq_len = sum(seq_len_tensor)

        inputs = torch.randn(seq_len, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float16, device="cpu"
        )

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        actual_output = sgemm_lora_a_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgemm_lora_b_fwd_expand(self):
        batch_size = 2
        output_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        seq_len_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )

        seq_len = sum(seq_len_tensor)

        inputs = torch.randn(seq_len, max_lora_rank, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, max_lora_rank, dtype=dtype)
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cpu")

        expect_output = reference_sgmv_expand(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        actual_output = sgemm_lora_b_fwd(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgemm_lora_a_embedding_graph_fwd(self):
        batch_size = 4
        input_dim = 1024
        num_loras = 3
        dtype = torch.float16
        vocab_size = 32000

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        inputs = torch.randint(vocab_size, (batch_size,), dtype=torch.int32)
        lora_a_weights = torch.zeros(num_loras, max_lora_rank, vocab_size, dtype=dtype)
        for idx, rank in enumerate(lora_ranks):
            lora_a_weights[idx, :rank] = torch.randn(rank, vocab_size, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float16, device="cpu"
        )

        expect_output = reference_embedding_lora_a_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
            vocab_size,
        )

        actual_output = sgemm_lora_a_embedding_graph_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_scaling_tensor,
            vocab_size,
        )

        self.assertTrue(
            torch.allclose(actual_output, expect_output, rtol=1e-3, atol=1e-5)
        )

    def test_sgemm_lora_a_graph_fwd(self):
        batch_size = 4
        input_dim = 1024
        num_loras = 3
        dtype = torch.float16

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.zeros(num_loras, max_lora_rank, input_dim, dtype=dtype)
        for idx, rank in enumerate(lora_ranks):
            lora_a_weights[idx, :rank] = torch.randn(rank, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float16, device="cpu"
        )

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        actual_output = sgemm_lora_a_graph_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_scaling_tensor,
        )

        self.assertTrue(
            torch.allclose(actual_output, expect_output, rtol=1e-3, atol=1e-5)
        )

    def test_sgemm_lora_b_graph_fwd(self):
        batch_size = 4
        output_dim = 1024
        num_loras = 3
        dtype = torch.float16

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        inputs = torch.randn(batch_size, max_lora_rank, dtype=dtype)
        lora_b_weights = torch.zeros(num_loras, output_dim, max_lora_rank, dtype=dtype)
        for idx, rank in enumerate(lora_ranks):
            lora_b_weights[idx, ..., :rank] = torch.randn(output_dim, rank, dtype=dtype)
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cpu")

        expect_output = reference_sgmv_expand(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        actual_output = sgemm_lora_b_graph_fwd(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            slice_offsets,
        )

        self.assertTrue(
            torch.allclose(actual_output, expect_output, rtol=1e-3, atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
