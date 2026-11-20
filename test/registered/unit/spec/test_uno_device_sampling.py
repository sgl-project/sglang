"""Device-portable sampling contracts for linear UNO."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.uno_cuda_graph_runner import UnoDecodeCudaGraphRunner
from sglang.srt.speculative.uno_utils import (
    _build_sparse_target_support_tensors,
    _dense_rejection_torch,
    _sparse_rejection_from_support,
    _top_k,
    run_uno_sampling,
    sample_uno_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnoDeviceSampling(CustomTestCase):
    def test_sparse_topk_stays_outside_compiled_tensor_math(self):
        module = "sglang.srt.speculative.uno_utils"
        logits, values, indices, temperatures, top_ks, top_ps = (
            object() for _ in range(6)
        )
        expected = object()

        def topk(tensor, k):
            self.assertFalse(torch.compiler.is_compiling())
            self.assertIs(tensor, logits)
            self.assertEqual(k, 3)
            return values, indices

        with (
            patch(module + "._top_k", side_effect=topk) as select,
            patch(
                module + "._sparse_target_support_from_topk", return_value=expected
            ) as normalize,
        ):
            actual = _build_sparse_target_support_tensors(
                logits, temperatures, top_ks, top_ps, 2, 4, 3
            )
        self.assertIs(actual, expected)
        select.assert_called_once()
        normalize.assert_called_once_with(
            topk_logits=values,
            topk_ids=indices,
            next_token_logits=logits,
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            batch_size=2,
            forward_width=4,
            max_top_k=3,
        )

    def test_npu_capture_resets_prefix_for_each_variant_and_bucket(self):
        runner = UnoDecodeCudaGraphRunner.__new__(UnoDecodeCudaGraphRunner)
        runner.model_runner = SimpleNamespace(device="npu")
        runner.seq_len_fill_value = 0
        runner.buffers = SimpleNamespace(
            seq_lens=torch.full((64,), 8, dtype=torch.int32),
            seq_lens_cpu=torch.full((64,), 8, dtype=torch.int32),
        )

        def prepare(size, num_tokens=None):
            self.assertEqual(torch.count_nonzero(runner.buffers.seq_lens).item(), 0)
            self.assertEqual(torch.count_nonzero(runner.buffers.seq_lens_cpu).item(), 0)
            runner.buffers.seq_lens[:size].add_(8)
            return SimpleNamespace(lora_ids=["synthetic"]), None, None

        with patch.object(
            UnoDecodeCudaGraphRunner.__bases__[0],
            "capture_prepare",
            side_effect=prepare,
        ):
            for bs in (64, 64, 1, 1):
                batch, _, _ = runner.capture_prepare(bs, num_tokens=bs * 8)
                self.assertIsNone(batch.lora_ids)

    def test_torch_topk_fallback_matches_reference(self):
        logits = torch.tensor([[0.1, 0.9, -0.2, 0.4]], dtype=torch.float32)
        expected_values, expected_indices = torch.topk(logits, k=3, dim=-1)

        with patch("sglang.srt.speculative.uno_utils._flashinfer_top_k", None):
            values, indices = _top_k(logits, 3)

        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(indices, expected_indices)

    def test_npu_sampling_pipeline_avoids_shared_triton_sampler(self):
        module = "sglang.srt.speculative.uno_utils"
        # Use deterministic one-hot support so each output is independent of
        # RNG implementation, for both compact and dense draft distributions.
        for top_k in (2, 129):
            for width in (1, 4):
                with self.subTest(top_k=top_k, width=width):
                    logits = torch.full((2, width, 130), float("-inf"))
                    logits[:, :, 7] = 0
                    sampling = SimpleNamespace(
                        temperatures=torch.ones(2, 1),
                        top_ks=torch.full((2,), top_k, dtype=torch.int32),
                        top_ps=torch.ones(2),
                        need_top_k_sampling=True,
                        need_top_p_sampling=False,
                    )
                    frontiers = torch.tensor([124, 252], dtype=torch.int32)
                    with (
                        patch(module + "._is_npu", True),
                        patch(
                            module + ".fast_sample",
                            side_effect=AssertionError("shared Triton sampler reached"),
                        ),
                    ):
                        candidates, draft = sample_uno_candidates(
                            draft_logits=logits,
                            sampling_info=sampling,
                            max_top_k=top_k,
                            uniform_top_k_value=top_k,
                        )
                        result = run_uno_sampling(
                            candidates=candidates,
                            next_token_logits=logits.flatten(0, 1),
                            sampling_info=sampling,
                            committed_frontiers=frontiers,
                            draft_distribution=draft,
                            max_top_k=top_k,
                            uniform_top_k_value=top_k,
                        )
                    self.assertEqual(candidates.tolist(), [[7] * width] * 2)
                    self.assertEqual(
                        result.output_ids.tolist(), [[7] * (width + 1)] * 2
                    )
                    self.assertEqual(result.accept_lens.tolist(), [width + 1] * 2)
                    torch.testing.assert_close(
                        result.new_seq_lens, frontiers + width + 1
                    )

    def test_dense_rejection_recovers_target_distribution(self):
        # q=(3/4,1/4), p=(1/4,3/4). Each of these four equally weighted
        # rows represents 1/4 of proposal/acceptance probability mass.
        candidates = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 1]])
        target = torch.tensor([[[0.25, 0.75], [0.25, 0.75]]]).expand(4, -1, -1)
        draft = torch.tensor([[[0.75, 0.25]]]).expand(4, -1, -1)
        accepted, bonus = _dense_rejection_torch(
            candidates=candidates,
            target_probs=target,
            draft_probs=draft,
            accept_uniforms=torch.tensor([[1 / 6, 0], [0.5, 0], [5 / 6, 0], [0.5, 0]]),
            final_uniforms=torch.full((4,), 0.5),
        )
        emitted = torch.where(accepted > 0, candidates[:, 1], bonus)
        self.assertEqual(accepted.tolist(), [1, 0, 0, 1])
        self.assertEqual(torch.bincount(emitted, minlength=2).tolist(), [1, 3])

    def test_sparse_rejection_handles_partially_disjoint_supports(self):
        # q puts 3/4 mass on token 1 (absent from p); p=(1/2,0,1/2).
        # Three rejected token-1 proposals and one accepted token-2 proposal
        # each represent 1/4 probability mass. Correction must recover p.
        candidates = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 2]])
        target_ids = torch.tensor([[[0, 2], [0, 2]]]).expand(4, -1, -1)
        draft_ids = torch.tensor([[[1, 2]]]).expand(4, -1, -1)
        accepted, bonus = _sparse_rejection_from_support(
            candidates=candidates,
            target_ids=target_ids,
            target_probs=torch.full((4, 2, 2), 0.5),
            draft_ids=draft_ids,
            draft_probs=torch.tensor([[[0.75, 0.25]]]).expand(4, -1, -1),
            accept_uniforms=torch.full((4, 2), 0.5),
            final_uniforms=torch.tensor([1 / 6, 0.5, 5 / 6, 0.5]),
        )
        emitted = torch.where(accepted > 0, candidates[:, 1], bonus)
        self.assertEqual(accepted.tolist(), [0, 0, 0, 1])
        self.assertEqual(torch.bincount(emitted, minlength=3).tolist(), [2, 0, 2])

    def test_dense_rejection_all_accepted_and_single_token_width(self):
        for width in (1, 4):
            with self.subTest(width=width):
                target = torch.tensor([[[0.25, 0.75]]]).expand(1, width, 2)
                accepted, bonus = _dense_rejection_torch(
                    candidates=torch.ones((1, width), dtype=torch.int64),
                    target_probs=target,
                    draft_probs=target[:, : width - 1],
                    accept_uniforms=torch.full((1, width), 0.5),
                    final_uniforms=torch.tensor([0.25]),
                )
                self.assertEqual(accepted.tolist(), [width - 1])
                self.assertEqual(bonus.tolist(), [1])

    def test_dense_rejection_fallback_accepts_prefix_and_samples_residual(self):
        candidates = torch.tensor([[7, 1, 2]], dtype=torch.int64)
        target_probs = torch.tensor(
            [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        )
        draft_probs = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

        accepted, bonus = _dense_rejection_torch(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            accept_uniforms=torch.tensor([[0.5, 0.5, 0.5]]),
            final_uniforms=torch.tensor([0.0]),
        )

        self.assertEqual(accepted.tolist(), [1])
        self.assertEqual(bonus.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
