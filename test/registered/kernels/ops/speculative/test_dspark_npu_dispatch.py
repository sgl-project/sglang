"""Numerical coverage for DSpark's SGLang-to-NPU dispatch boundary."""

import unittest
from unittest import mock

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

from sglang.kernels.ops.sampling import argmax as sampling_argmax
from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    SoftmaxTemp,
)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import MarkovGreedyStep
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildOutTokens,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=90, suite="base-b-test-1-npu-a3")


class TestDSparkNpuDispatch(CustomTestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_build_out_tokens_matches_full_torch_buffer(self):
        from sgl_kernel_npu.operators import sample

        for bs, gamma in ((0, 7), (1, 0), (33, 7)):
            for len_dtype in (torch.int32, torch.int64):
                for step in (1, 2):
                    with self.subTest(bs=bs, gamma=gamma, dtype=len_dtype, step=step):
                        draft = torch.arange(bs * gamma * step).reshape(
                            bs, gamma * step
                        )
                        draft = (draft + 2**40).npu()[:, ::step]
                        lengths = (
                            (torch.arange(bs * step) // step % (gamma + 1))
                            .to(len_dtype)
                            .npu()[::step]
                        )
                        bonus = (torch.arange(bs * step) - 2**40).npu()[::step]
                        inputs = {
                            "draft_tokens": draft,
                            "correct_len": lengths,
                            "bonus": bonus,
                            "verify_num_draft_tokens": gamma + 1,
                            "gamma": gamma,
                        }
                        reference = dict(
                            inputs,
                            draft_tokens=draft.cpu(),
                            correct_len=lengths.cpu(),
                            bonus=bonus.cpu(),
                        )
                        expected = BuildOutTokens.torch(**reference)
                        with mock.patch.object(
                            sample,
                            "build_out_tokens_triton",
                            wraps=sample.build_out_tokens_triton,
                        ) as kernel:
                            actual = BuildOutTokens.execute(**inputs)
                        kernel.assert_called_once_with(**inputs)
                        self.assertEqual(actual.dtype, torch.int64)
                        self.assertEqual(actual.device, draft.device)
                        self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_placement_uses_first_tensor(self):
        cpu = torch.zeros(1)
        npu = cpu.npu()
        self.assertFalse(inputs_on_cuda(None, cpu, npu))
        self.assertTrue(inputs_on_cuda(None, npu, cpu))
        with self.assertRaises(AssertionError):
            inputs_on_cuda(1, flag=True)

    def test_markov_greedy_matches_native_rounding(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base = torch.randn(3, 4097, device="npu", dtype=dtype)
            embeds = torch.randn(3, 33, device="npu", dtype=dtype)
            weight = torch.randn(4097, 33, device="npu", dtype=dtype)
            expected = (base + torch.nn.functional.linear(embeds, weight)).argmax(-1)
            actual = MarkovGreedyStep.execute(
                base_logits=base, prev_embeds=embeds, w2_weight=weight
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_argmax_and_accept_match_torch(self):
        rows, block, vocab = 9, 7, 4097
        logits = torch.randn(rows * block, vocab, device="npu")
        logits[0, 5] = float("nan")
        expected_top1 = logits.cpu().argmax(-1)
        self.assertTrue(torch.equal(sampling_argmax(logits).cpu(), expected_top1))
        predict = expected_top1.reshape(rows, block)
        candidates = predict.clone()
        candidates[:, 1:] = predict[:, :-1]
        for row in range(rows):
            if row % block < block - 1:
                col = row % block
                candidates[row, col + 1] = (predict[row, col] + 1) % vocab
        lengths = (candidates[:, 1:] == predict[:, :-1]).int().cumprod(1).sum(1)
        bonus = predict[torch.arange(rows), lengths]
        npu_candidates = candidates.npu()
        integrated = AcceptGreedy.execute(
            candidates=npu_candidates,
            target_logits=logits,
            verify_num_draft_tokens=block,
        )
        self.assertEqual(integrated[0].dtype, torch.int32)
        self.assertEqual(integrated[1].dtype, torch.int64)
        self.assertTrue(torch.equal(integrated[0].cpu(), lengths.int()))
        self.assertTrue(torch.equal(integrated[1].cpu(), bonus))

    def test_softmax_temperature_mapping(self):
        logits = torch.randn(9, 4097, device="npu")
        temperatures = torch.tensor([0.3, 0.7, 1.2], device="npu")
        actual = SoftmaxTemp.execute(
            logits=logits, temperatures=temperatures, rows_per_request=3
        )
        expected = torch.softmax(
            logits.cpu() / temperatures.cpu().repeat_interleave(3)[:, None], -1
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=1e-7)

    def test_cpu_reference_stays_available(self):
        logits = torch.randn(6, 257)
        temperatures = torch.tensor([0.7, 1.3])
        expected = torch.softmax(
            logits / temperatures.repeat_interleave(3)[:, None], -1
        )
        actual = SoftmaxTemp.execute(
            logits=logits, temperatures=temperatures, rows_per_request=3
        )
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
