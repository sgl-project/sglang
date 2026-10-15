import unittest

import torch

from sglang.kernels.ops.speculative.ragged_verify_kernels import (
    build_ragged_dense_row_maps,
    gather_rows_masked,
)
from sglang.srt.speculative.acceptance_estimator import OnlineAcceptanceEstimator
from sglang.srt.speculative.eagle_adaptive_verify import EagleAdaptiveVerifier
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _simulate_verify(
    *,
    features: torch.Tensor,
    true_slope: float,
    true_intercepts: torch.Tensor,
    verify_lens: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Chain acceptance: drafts are accepted in order until the first rejection."""
    probs = torch.sigmoid(true_slope * features + true_intercepts)
    coins = torch.rand(probs.shape, device=probs.device, generator=generator)
    accepted = (coins < probs).to(torch.int32)
    steps = torch.arange(features.shape[1], device=features.device)
    accepted = accepted * (steps < (verify_lens - 1).unsqueeze(1)).to(torch.int32)
    return torch.cumprod(accepted, dim=1).sum(dim=1).to(torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestOnlineAcceptanceEstimator(CustomTestCase):
    def test_fit_recovers_chain_acceptance_model(self):
        """The arrowhead Newton solve must converge to the generating coefficients
        using only labels up to each chain's first rejection."""
        device = torch.device("cuda")
        num_steps, num_reqs = 4, 256
        true_slope = 0.8
        true_intercepts = torch.tensor([1.2, 0.4, -0.3, -0.9], device=device)
        estimator = OnlineAcceptanceEstimator(num_steps=num_steps, device=device)
        g = torch.Generator(device=device).manual_seed(0)
        for _ in range(40 * estimator.REFIT_INTERVAL):
            features = torch.randn((num_reqs, num_steps), device=device, generator=g)
            features = features * 3.0
            q = torch.sigmoid(features)
            features, predictions = estimator.predict(q)
            verify_lens = torch.full(
                (num_reqs,), num_steps + 1, dtype=torch.int32, device=device
            )
            num_correct = _simulate_verify(
                features=features,
                true_slope=true_slope,
                true_intercepts=true_intercepts,
                verify_lens=verify_lens,
                generator=g,
            )
            estimator.observe(
                features=features,
                predictions=predictions,
                num_correct_drafts=num_correct,
                verify_lens=verify_lens,
            )
        self.assertAlmostEqual(estimator.slope.item(), true_slope, delta=0.05)
        torch.testing.assert_close(
            estimator.intercepts, true_intercepts, atol=0.1, rtol=0
        )

    def test_saturated_drafts_keep_fit_bounded(self):
        """Drafts whose top-1 probability rounds to 1 carry ~0 Newton weight; their
        rejections must not blow an intercept up so far that every later draft is
        trimmed (and so never labeled again)."""
        device = torch.device("cuda")
        estimator = OnlineAcceptanceEstimator(num_steps=3, device=device)
        intercepts = estimator.intercepts.clone()
        q = torch.ones((64, 3), device=device)
        num_correct = torch.zeros(64, dtype=torch.int32, device=device)
        num_correct[::2] = 3
        verify_lens = torch.full((64,), 4, dtype=torch.int32, device=device)
        for _ in range(estimator.REFIT_INTERVAL):
            features, predictions = estimator.predict(q)
            estimator.observe(
                features=features,
                predictions=predictions,
                num_correct_drafts=num_correct,
                verify_lens=verify_lens,
            )
        self.assertEqual(estimator.num_refits, 1)
        self.assertLessEqual(
            (estimator.intercepts - intercepts).abs().max().item(), 1.0 + 1e-6
        )
        self.assertTrue(0.0 <= estimator.slope.item() <= 4.0)

    def test_trimmed_drafts_are_not_labeled(self):
        """Drafts the target never verified carry no label, so a round of fully
        trimmed chains must leave the fit untouched."""
        device = torch.device("cuda")
        estimator = OnlineAcceptanceEstimator(num_steps=3, device=device)
        slope = estimator.slope.clone()
        intercepts = estimator.intercepts.clone()
        q = torch.full((8, 3), 0.9, device=device)
        for _ in range(estimator.REFIT_INTERVAL):
            features, predictions = estimator.predict(q)
            estimator.observe(
                features=features,
                predictions=predictions,
                num_correct_drafts=torch.zeros(8, dtype=torch.int32, device=device),
                verify_lens=torch.ones(8, dtype=torch.int32, device=device),
            )
        self.assertEqual(estimator.num_refits, 1)
        torch.testing.assert_close(estimator.slope, slope, rtol=0, atol=0)
        torch.testing.assert_close(estimator.intercepts, intercepts, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestAdaptiveVerifyPacking(CustomTestCase):
    def test_pack_then_unpack_restores_verified_slots(self):
        """Packed rows must hold each request's verified prefix in order, and the
        strided view must read every verified slot back from its own packed row and
        every trimmed slot from the request's last verified row."""
        device = torch.device("cuda")
        num_draft_tokens = 4
        verify_lens = torch.tensor([1, 4, 2, 3], dtype=torch.int32, device=device)
        bs = verify_lens.numel()
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens, graph_num_tokens=16
        )
        verifier = EagleAdaptiveVerifier.__new__(EagleAdaptiveVerifier)
        verifier.num_draft_tokens = num_draft_tokens
        verifier.device = device

        strided_tokens = torch.arange(bs * num_draft_tokens, device=device) + 100

        class _Model:
            class model_config:
                model_is_mrope = False

        class _VerifyInput:
            draft_token = strided_tokens
            positions = strided_tokens + 1000
            prepared_out_cache_loc = strided_tokens + 2000
            prepared_mrope_positions = None

        verifier._model_runner = _Model()
        packed = verifier._pack(verify_input=_VerifyInput, batch=None, layout=layout)

        expected = []
        for req, length in enumerate(verify_lens.tolist()):
            expected += [100 + req * num_draft_tokens + j for j in range(length)]
        num_real = len(expected)
        self.assertEqual(packed.input_ids[:num_real].tolist(), expected)
        self.assertTrue(torch.all(packed.input_ids[num_real:] == 0).item())
        self.assertTrue(torch.all(packed.window_cols[num_real:] == -1).item())
        torch.testing.assert_close(
            packed.out_cache_loc[:num_real], packed.input_ids[:num_real] + 2000
        )

        unpacked = packed.input_ids[packed.strided_rows].view(bs, num_draft_tokens)
        steps = torch.arange(num_draft_tokens, device=device)
        source = torch.minimum(steps.unsqueeze(0), (verify_lens - 1).unsqueeze(1))
        reference = strided_tokens.view(bs, num_draft_tokens).gather(1, source.long())
        torch.testing.assert_close(unpacked, reference, rtol=0, atol=0)

    def test_dense_row_maps_round_trip_packed_rows(self):
        """Packed rows must land on their request's dense steps, empty dense steps
        and tier-leftover packed rows must read zeros, and dense -> packed must
        restore every real packed row."""
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 1, 4, 6], dtype=torch.int32, device=device)
        dense_rows, packed_rows = build_ragged_dense_row_maps(
            query_start_loc=query_start_loc, num_packed_rows=8, draft_token_num=4
        )
        self.assertEqual(dense_rows.tolist(), [0, 4, 5, 6, 8, 9, -1, -1])
        self.assertEqual(
            packed_rows.tolist(), [0, -1, -1, -1, 1, 2, 3, -1, 4, 5, -1, -1]
        )
        packed = torch.arange(8 * 3, device=device, dtype=torch.float32).view(8, 3) + 1
        dense = gather_rows_masked(packed, packed_rows)
        self.assertTrue(torch.all(dense[packed_rows < 0] == 0).item())
        restored = gather_rows_masked(dense, dense_rows)
        torch.testing.assert_close(restored[:6], packed[:6], rtol=0, atol=0)
        self.assertTrue(torch.all(restored[6:] == 0).item())

    def test_truncated_chain_ends_at_last_verified_slot(self):
        device = torch.device("cuda")
        verifier = EagleAdaptiveVerifier.__new__(EagleAdaptiveVerifier)
        verifier.num_draft_tokens = 4
        verifier.device = device
        chain = torch.tensor([[1, 2, 3, -1]] * 3, device=device)
        truncated = verifier._truncate_chain(
            retrieve_next_token=chain,
            verify_lens=torch.tensor([1, 3, 4], dtype=torch.int32, device=device),
        )
        self.assertEqual(
            truncated.tolist(), [[-1, -1, -1, -1], [1, 2, -1, -1], [1, 2, 3, -1]]
        )


if __name__ == "__main__":
    unittest.main()
