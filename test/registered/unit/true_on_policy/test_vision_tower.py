"""TOP vision support: the tower's biased projection around its GEMM, and the positions of image requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.true_on_policy import vision as top_vision
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestVisionLinear(unittest.TestCase):
    def _linear(self, x, weight, bias):
        """``linear`` with an exact FP64 product standing in for the GPU GEMM, recording the GEMM's operands."""
        calls = []

        def gemm(a, b):
            calls.append((a, b))
            return a @ b

        with (
            patch.object(top_vision, "_require_top"),
            patch("sglang.srt.batch_invariant_ops.matmul_persistent", gemm),
        ):
            return top_vision.linear(x, weight, bias), calls

    def test_k_is_zero_padded_to_the_gemm_multiple(self):
        """The tower MLP's K is 4304, which DeepGEMM does not take; the pad must not change any output."""
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(5, 4304, generator=generator, dtype=torch.float64)
        weight = torch.randn(16, 4304, generator=generator, dtype=torch.float64)
        bias = torch.randn(16, generator=generator, dtype=torch.float64)
        out, calls = self._linear(x, weight, bias)
        ((a, b),) = calls
        self.assertEqual((a.shape, b.shape), ((5, 4352), (4352, 16)))
        self.assertTrue(a.is_contiguous() and b.t().is_contiguous())
        self.assertEqual(float(a[:, 4304:].abs().sum() + b[4304:].abs().sum()), 0.0)
        torch.testing.assert_close(out, x @ weight.t() + bias, rtol=0, atol=1e-9)

    def test_leading_dims_and_aligned_k_pass_through(self):
        x = torch.arange(2 * 3 * 128, dtype=torch.float64).reshape(2, 3, 128)
        weight = torch.ones(32, 128, dtype=torch.float64)
        out, calls = self._linear(x, weight, None)
        self.assertEqual(out.shape, (2, 3, 32))
        self.assertEqual(calls[0][0].shape, (6, 128))
        self.assertTrue(torch.equal(out[..., 0], x.sum(-1)))

    def test_the_program_refuses_a_process_without_top(self):
        with self.assertRaises((RuntimeError, ValueError)):
            top_vision.linear(torch.zeros(1, 128), torch.zeros(16, 128), None)


class TestImagePositions(unittest.TestCase):
    def test_image_requests_keep_their_mrope_positions_under_the_contract(self):
        """A contract engine rotates image tokens by the model's mRoPE positions, not their text positions, so an
        image-conditioned log-prob equals the trainer's."""
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.true_on_policy import TRUE_ON_POLICY_V1

        # Two text tokens, a 2x2 image at temporal position 2, one text token after it.
        prompt_positions = torch.tensor(
            [[0, 1, 2, 2, 2, 2, 4], [0, 1, 2, 2, 3, 3, 4], [0, 1, 2, 3, 2, 3, 4]]
        )
        mm_input = SimpleNamespace(
            mrope_positions=prompt_positions,
            mrope_position_delta=torch.tensor([[-2]]),
            mrope_position_delta_repeated_cache=None,
        )
        batch = SimpleNamespace(
            multimodal_inputs=[mm_input],
            extend_lens=[7],
            prefix_lens=[0],
            reqs=[SimpleNamespace(session=None)],
        )
        runner = SimpleNamespace(device="cpu")
        contract = SimpleNamespace(
            rl_on_policy_target=None,
            true_on_policy_contract=TRUE_ON_POLICY_V1,
            tp_size=1,
        )
        deterministic = SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        )
        with (
            patch(
                "sglang.srt.true_on_policy.config._get_global_server_args",
                return_value=contract,
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.get_exec",
                return_value=deterministic,
            ),
        ):
            prefill = SimpleNamespace(
                seq_lens_cpu=torch.tensor([7]), mrope_positions=None
            )
            ForwardBatch._compute_mrope_positions_extend(prefill, runner, batch)
            decode = SimpleNamespace(
                seq_lens_cpu=torch.tensor([8]),
                seq_lens=torch.tensor([8]),
                mrope_positions=None,
            )
            ForwardBatch._compute_mrope_positions_decode(decode, runner, batch)
        self.assertTrue(torch.equal(prefill.mrope_positions, prompt_positions))
        # The first generated token continues after the image's largest position, not at its token index 7.
        self.assertEqual(decode.mrope_positions.tolist(), [[5], [5], [5]])


if __name__ == "__main__":
    unittest.main()
