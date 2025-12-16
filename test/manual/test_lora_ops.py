import unittest

import torch

from sglang.srt.lora.torch_ops.lora_ops import (
    bgmv_expand,
    bgmv_expand_slice,
    bgmv_shrink,
    sgmv_expand,
    sgmv_expand_slice,
    sgmv_shrink,
)
from sglang.test.test_utils import CustomTestCase


class TestLoraOps(CustomTestCase):
    def test_sgmv_expand(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
        add_inputs = True

        total_seq_len, _ = inputs.shape
        exploded_indices = torch.repeat_interleave(
            lora_indices_tensor, seq_len_tensor, output_size=total_seq_len
        )
        expect_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        bgmv_expand(inputs, lora_b_weights, expect_output, exploded_indices, add_inputs)

        actual_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        sgmv_expand(
            inputs,
            lora_b_weights,
            actual_output,
            seq_len_tensor,
            lora_indices_tensor,
            add_inputs,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_bgmv_expand(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))

        selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
        selected_loras = selected_loras.squeeze(dim=1)
        inputs = inputs.to(dtype=dtype)
        outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
        limit = batch_size
        common_len = min(outputs.shape[1], output_dim)
        expect_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        expect_output[:, :common_len] = outputs[:limit, :common_len]

        actual_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        bgmv_expand(
            inputs,
            lora_b_weights,
            actual_output,
            lora_indices_tensor,
            add_inputs=False,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_bgmv_expand_add_residual(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))

        selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
        selected_loras = selected_loras.squeeze(dim=1)
        inputs = inputs.to(dtype=dtype)
        outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
        limit = batch_size
        common_len = min(outputs.shape[1], output_dim)
        expect_output = torch.randn(batch_size, output_dim, dtype=dtype)
        actual_output = expect_output.clone()

        expect_output[:, :common_len] += outputs[:limit, :common_len]

        bgmv_expand(
            inputs,
            lora_b_weights,
            actual_output,
            lora_indices_tensor,
            add_inputs=True,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgmv_shrink(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
        scaling = 0.9

        total_seq_len, _ = inputs.shape
        exploded_indices = torch.repeat_interleave(
            lora_indices_tensor, seq_len_tensor, output_size=total_seq_len
        )
        expect_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        bgmv_shrink(inputs, lora_a_weights, expect_output, exploded_indices, scaling)

        actual_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        sgmv_shrink(
            inputs,
            lora_a_weights,
            actual_output,
            seq_len_tensor,
            lora_indices_tensor,
            scaling,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_bgmv_shrink(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
        scaling = 0.9

        selected_loras = lora_a_weights[lora_indices_tensor].to(dtype=dtype)
        inputs = inputs.to(dtype=dtype)
        outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

        expect_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        expect_output[:, : outputs.shape[1]] = scaling * outputs[:]

        actual_output = torch.zeros(batch_size, output_dim, dtype=dtype)
        bgmv_shrink(
            inputs,
            lora_a_weights,
            actual_output,
            lora_indices_tensor,
            scaling=scaling,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_sgmv_expand_slice(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        output_dim_slice = 12
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
        slice_offset = 2
        slice_size = 6
        add_inputs = False

        total_seq_len, _ = inputs.shape
        exploded_indices = torch.repeat_interleave(
            lora_indices_tensor, seq_len_tensor, output_size=total_seq_len
        )
        expect_output = torch.randn(batch_size, output_dim_slice, dtype=dtype)
        actual_output = expect_output.clone()
        bgmv_expand_slice(
            inputs,
            lora_b_weights,
            expect_output,
            exploded_indices,
            slice_offset,
            slice_size,
            add_inputs,
        )

        sgmv_expand_slice(
            inputs,
            lora_b_weights,
            actual_output,
            seq_len_tensor,
            lora_indices_tensor,
            slice_offset,
            slice_size,
            add_inputs,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_bgmv_expand_slice(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        output_dim_slice = 12
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
        slice_offset = 2
        slice_size = 6

        selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
        inputs = inputs.to(dtype=dtype)
        outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
        expect_output = torch.zeros(batch_size, output_dim_slice, dtype=dtype)
        expect_output[:, slice_offset : slice_offset + slice_size] = outputs[:]

        actual_output = torch.zeros(batch_size, output_dim_slice, dtype=dtype)
        bgmv_expand_slice(
            inputs,
            lora_b_weights,
            actual_output,
            lora_indices_tensor,
            slice_offset,
            slice_size,
            add_inputs=False,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    def test_bgmv_expand_slice_add_residual(self):
        batch_size = 2
        input_dim = 4
        output_dim = 6
        output_dim_slice = 12
        num_loras = 3
        dtype = torch.float32

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
        slice_offset = 2
        slice_size = 6

        selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
        inputs = inputs.to(dtype=dtype)
        outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
        expect_output = torch.randn(batch_size, output_dim_slice, dtype=dtype)
        actual_output = expect_output.clone()
        expect_output[:, slice_offset : slice_offset + slice_size] += outputs[:]

        bgmv_expand_slice(
            inputs,
            lora_b_weights,
            actual_output,
            lora_indices_tensor,
            slice_offset,
            slice_size,
            add_inputs=True,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))


if __name__ == "__main__":
    unittest.main()
