import unittest

import torch

from sglang.kernels.ops.speculative.acceptance_estimator import (
    NUM_BINS,
    accumulate_acceptance_counts,
    accumulate_acceptance_counts_torch,
    predict_acceptance,
    predict_acceptance_torch,
    refit_acceptance,
    refit_acceptance_torch,
)
from sglang.srt.speculative.online_acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _predict(fn, logits, temperatures, slope, intercepts):
    bs, num_positions, _ = logits.shape
    features = torch.empty(bs, num_positions, device=logits.device)
    predictions = torch.empty_like(features)
    fn(
        logits=logits,
        temperatures=temperatures,
        slope=slope,
        intercepts=intercepts,
        features_out=features,
        predictions_out=predictions,
    )
    return features, predictions


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestAcceptanceEstimatorKernels(CustomTestCase):
    def test_predict_matches_reference(self):
        torch.manual_seed(0)
        slope = torch.tensor([0.7], device="cuda")
        intercepts = torch.tensor([0.3, -0.1, -0.6, 0.2, 1.0], device="cuda")
        for vocab, dtype in (
            (129280, torch.bfloat16),
            (1000, torch.float32),
            (7, torch.float32),
        ):
            with self.subTest(vocab=vocab, dtype=dtype):
                # A strided [bs, 5, vocab] view of a wider buffer.
                logits = (torch.randn(6, 7, vocab, device="cuda") * 4).to(dtype)[:, 1:6]
                logits[0, 0] = -1e4
                logits[0, 0, 3] = 0.0
                logits[1, 1, vocab // 2 :] = float("-inf")
                temperatures = torch.tensor(
                    [1.0, 0.0, 0.3, 1.7, -1.0, 1e-5], device="cuda"
                )
                got = _predict(
                    predict_acceptance, logits, temperatures, slope, intercepts
                )
                want = _predict(
                    predict_acceptance_torch, logits, temperatures, slope, intercepts
                )
                for g, w in zip(got, want):
                    torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-4)

    def test_accumulate_matches_reference(self):
        torch.manual_seed(1)
        num_positions, bs = 5, 300
        features = (torch.randn(bs, num_positions + 2, device="cuda") * 10)[
            :, :num_positions
        ]
        num_correct = torch.randint(0, num_positions + 1, (bs,), device="cuda")
        num_graded = torch.randint(0, num_positions + 1, (bs,), device="cuda")
        counts = [
            torch.zeros(num_positions, NUM_BINS, 2, device="cuda") for _ in range(2)
        ]
        for fn, out in zip(
            (accumulate_acceptance_counts, accumulate_acceptance_counts_torch), counts
        ):
            fn(
                counts=out,
                features=features,
                num_correct_drafts=num_correct,
                num_graded_drafts=num_graded,
            )
        torch.testing.assert_close(counts[0], counts[1], rtol=0, atol=0)

    def test_refit_matches_reference(self):
        torch.manual_seed(2)
        num_positions = 5
        graded = torch.randint(0, 50, (num_positions, NUM_BINS), device="cuda").float()
        graded[3] = 0  # a position without labels
        correct = torch.floor(graded * torch.rand_like(graded))
        round_counts = torch.stack([graded, correct], dim=-1)
        history = round_counts.flip(1) * 0.5
        results = []
        for fn in (refit_acceptance, refit_acceptance_torch):
            state = dict(
                round_counts=round_counts.clone(),
                history_counts=history.clone(),
                slope=torch.tensor([0.5], device="cuda"),
                intercepts=torch.full((num_positions,), -0.2, device="cuda"),
            )
            fn(**state, decay=0.75, prior_weight=1.0, num_iters=8)
            results.append(state)
        got, want = results
        self.assertEqual(float(got["round_counts"].abs().sum()), 0.0)
        torch.testing.assert_close(got["history_counts"], want["history_counts"])
        torch.testing.assert_close(got["slope"], want["slope"], rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(
            got["intercepts"], want["intercepts"], rtol=1e-3, atol=1e-4
        )
        self.assertNotEqual(float(got["slope"]), 0.5)

    def test_estimator_converges_on_cuda(self):
        torch.manual_seed(3)
        true_slope, true_intercepts = 0.55, torch.tensor([0.6, 0.2, -0.3, -0.8])
        est = OnlineAcceptanceEstimator(
            num_positions=4, device=torch.device("cuda"), log_rank=False
        )
        bs = 256
        for _ in range(1500):
            features = torch.rand(bs, 4, device="cuda") * 6.0
            logits = torch.stack([features, torch.zeros_like(features)], dim=-1)
            x, _ = est.predict(
                logits=logits, temperatures=torch.ones(bs, device="cuda")
            )
            accept_prob = torch.sigmoid(true_slope * x + true_intercepts.cuda())
            correct = torch.rand_like(x) < accept_prob
            est.observe(
                features=x,
                num_correct_drafts=torch.cumprod(correct.long(), dim=1).sum(dim=1),
                num_graded_drafts=torch.full((bs,), 4, device="cuda"),
            )
        self.assertAlmostEqual(float(est.slope), true_slope, delta=0.03)
        torch.testing.assert_close(
            est.intercepts.cpu(), true_intercepts, atol=0.08, rtol=0
        )

    def test_predict_is_graph_capturable(self):
        est = OnlineAcceptanceEstimator(
            num_positions=3, device=torch.device("cuda"), log_rank=False
        )
        logits = torch.randn(4, 3, 1000, device="cuda")
        temperatures = torch.ones(4, device="cuda")
        features = torch.empty(4, 3, device="cuda")
        predictions = torch.empty_like(features)
        kwargs = dict(
            logits=logits,
            temperatures=temperatures,
            features_out=features,
            predictions_out=predictions,
        )
        est.predict(**kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            est.predict(**kwargs)
        est.slope.fill_(2.0)
        est.intercepts.fill_(0.5)
        graph.replay()
        torch.testing.assert_close(predictions, torch.sigmoid(2.0 * features + 0.5))


if __name__ == "__main__":
    unittest.main()
