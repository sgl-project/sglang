from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)

import unittest

import torch

from sglang.srt.speculative.online_acceptance_estimator import OnlineAcceptanceEstimator
from sglang.test.test_utils import CustomTestCase

_MAX_LOG_ODDS = 40.0
_INIT_SLOPE = 0.5
_INIT_BIAS = -0.2
_DAMPING = 50.0
_L2 = 1e-3
_MAX_SLOPE_HALVINGS = 8


def _ref_feature(scores: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """logit(q_max) of the temperature-scaled rows, in float64."""
    temps = temperatures.reshape(-1).double()
    temps = torch.where(temps > 0, temps, torch.ones_like(temps))
    s = scores.double() / temps[:, None, None]
    top, arg = s.max(dim=-1, keepdim=True)
    rest = torch.exp(s - top).scatter(-1, arg, 0.0).sum(dim=-1)
    return (-torch.log(rest)).clamp(-_MAX_LOG_ODDS, _MAX_LOG_ODDS)


class _RefEstimator:
    """Float64 reference of the estimator round: observation rule, IRLS
    statistics, damped Newton refit and the reset after every refit."""

    def __init__(
        self,
        num_positions: int,
        refit_interval: int,
        init_slope: float = _INIT_SLOPE,
        max_slope_halvings: int = _MAX_SLOPE_HALVINGS,
    ):
        self.k = num_positions
        self.refit_interval = refit_interval
        self.max_slope_halvings = max_slope_halvings
        self.slope = torch.tensor(init_slope, dtype=torch.float64)
        self.bias = torch.full((num_positions,), _INIT_BIAS, dtype=torch.float64)
        self.stats = torch.zeros(num_positions, 6, dtype=torch.float64)
        self.calls = 0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.slope * x.double() + self.bias)

    def observe(self, x, num_verified, num_accepted):
        x = x.double().cpu()
        verified = num_verified.cpu().long()[:, None]
        accepted = num_accepted.cpu().long()[:, None]
        pos = torch.arange(self.k)[None, :]
        observed = (pos <= accepted) & (pos < verified)
        label = (pos < accepted).double()
        p = self.predict(x)
        w = torch.where(observed, p * (1 - p), 0.0)
        r = torch.where(observed, label - p, 0.0)
        x = torch.where(observed, x, 0.0)
        self.stats += torch.stack(
            [
                (w * x * x).sum(0),
                (w * x).sum(0),
                w.sum(0),
                (r * x).sum(0),
                r.sum(0),
                observed.double().sum(0),
            ],
            dim=1,
        )
        self.calls += 1
        if self.calls >= self.refit_interval:
            self._refit()

    def _refit(self):
        a, b, c, g0, g1, n = self.stats.unbind(1)
        # Dense solve of (X^T W X + L2 I) d = X^T r for the slope step.
        hess = torch.zeros(self.k + 1, self.k + 1, dtype=torch.float64)
        hess[0, 0] = a.sum()
        hess[0, 1:] = b
        hess[1:, 0] = b
        hess[1:, 1:] = torch.diag(c)
        hess += _L2 * torch.eye(self.k + 1, dtype=torch.float64)
        grad = torch.cat([g0.sum()[None], g1])
        step_slope = torch.linalg.solve(hess, grad)[0]
        step_slope = step_slope * n.sum() / (n.sum() + _DAMPING)
        # Backtrack until the slope stays positive, else drop the step.
        for _ in range(self.max_slope_halvings):
            if self.slope + step_slope > 0:
                break
            step_slope = step_slope / 2
        if not torch.isfinite(step_slope) or self.slope + step_slope <= 0:
            step_slope = torch.zeros((), dtype=torch.float64)
        # Intercepts: exact Newton step given the (damped) slope step, damped.
        step_bias = (g1 - b * step_slope) / (c + _L2) * n / (n + _DAMPING)
        step_bias = torch.where(torch.isfinite(step_bias), step_bias, 0.0)
        self.slope = self.slope + step_slope
        self.bias = self.bias + step_bias
        self.stats.zero_()
        self.calls = 0


def _random_step(gen, bs, k, true_slope, true_bias, device):
    """One verify step of synthetic drafts: features, then a verify outcome
    drawn from the true logistic model, with some requests verifying fewer than
    k drafts."""
    x = (torch.randn(bs, k, generator=gen) * 3.0 + 2.0).clamp(
        -_MAX_LOG_ODDS, _MAX_LOG_ODDS
    )
    p = torch.sigmoid(true_slope * x + true_bias)
    hits = (torch.rand(bs, k, generator=gen) < p).int()
    leading = torch.cumprod(hits, dim=1).sum(dim=1)
    verified = torch.randint(0, k + 1, (bs,), generator=gen)
    verified = torch.where(torch.rand(bs, generator=gen) < 0.7, k, verified)
    accepted = torch.minimum(leading, verified)
    return (
        x.to(device),
        verified.to(device=device, dtype=torch.int32),
        accepted.to(device=device, dtype=torch.int32),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestOnlineAcceptanceEstimator(CustomTestCase):
    device = torch.device("cuda")

    def _estimator(self, k=7, max_num_reqs=128, refit_interval=100):
        return OnlineAcceptanceEstimator(
            num_positions=k,
            max_num_reqs=max_num_reqs,
            device=self.device,
            refit_interval=refit_interval,
        )

    def test_feature_matches_reference(self):
        gen = torch.Generator().manual_seed(0)
        for bs, k, n_cand, dtype in [
            (1, 7, 1, torch.float32),
            (5, 7, 16, torch.float32),
            (33, 3, 100, torch.bfloat16),
            (4, 2, 5000, torch.float32),
            (3, 1, 10000, torch.float16),
        ]:
            scores = (torch.randn(bs, k, n_cand, generator=gen) * 4.0).to(dtype)
            temps = torch.rand(bs, 1, generator=gen) * 1.5 + 0.1
            temps[0] = 0.0  # greedy rows are scored at T = 1
            est = self._estimator(k=k)
            x = est.feature_from_scores(scores.to(self.device), temps.to(self.device))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(tuple(x.shape), (bs, k))
            torch.testing.assert_close(
                x.cpu().double(), _ref_feature(scores, temps), rtol=1e-5, atol=1e-4
            )

    def test_feature_edge_rows(self):
        neg_inf = float("-inf")
        rows = torch.tensor(
            [
                [0.0, 0.0, neg_inf, neg_inf],  # two-way tie: q_max = 1/2
                [20.0, 0.0, 0.0, 0.0],  # confident but far from the clamp
                [200.0, 0.0, 0.0, 0.0],  # beyond the clamp
                [0.0, 0.0, 0.0, 0.0],  # uniform: q_max = 1/4
                [neg_inf, neg_inf, 3.0, neg_inf],  # one finite candidate
            ]
        )
        scores = rows[:, None, :]
        temps = torch.ones(rows.shape[0])
        x = self._estimator(k=1).feature_from_scores(
            scores.to(self.device), temps.to(self.device)
        )
        expected = torch.tensor(
            [0.0, 20.0 - torch.log(torch.tensor(3.0)).item(), 40.0, -1.0986123, 40.0]
        )
        torch.testing.assert_close(x[:, 0].cpu(), expected, rtol=0, atol=1e-5)

    def test_feature_strided_scores(self):
        gen = torch.Generator().manual_seed(1)
        base = torch.randn(6, 9, 40, generator=gen)
        scores = base[::2, 1:8, ::3]  # non-contiguous on every dim
        temps = torch.full((3,), 0.7)
        x = self._estimator(k=7).feature_from_scores(
            scores.to(self.device), temps.to(self.device)
        )
        torch.testing.assert_close(
            x.cpu().double(), _ref_feature(scores, temps), rtol=1e-5, atol=1e-4
        )

    def test_feature_and_predict_matches_separate_calls(self):
        gen = torch.Generator().manual_seed(4)
        k = 7
        est = self._estimator(k=k)
        est.slope.fill_(1.3)
        est.intercepts.copy_(torch.linspace(-1, 1, k))
        for bs, n_cand, dtype in [(5, 16, torch.float32), (33, 300, torch.bfloat16)]:
            scores = (torch.randn(bs, k, n_cand, generator=gen) * 4.0).to(dtype)
            temps = torch.rand(bs, generator=gen) + 0.1
            scores, temps = scores.to(self.device), temps.to(self.device)
            x, p = est.feature_and_predict(scores, temps)
            self.assertTrue(torch.equal(x, est.feature_from_scores(scores, temps)))
            torch.testing.assert_close(p, est.predict(x), rtol=0, atol=1e-6)
        x, p = est.feature_and_predict(
            torch.empty(0, k, 8, device=self.device), torch.empty(0, device=self.device)
        )
        self.assertEqual((tuple(x.shape), tuple(p.shape)), ((0, k), (0, k)))

    def test_predict_is_stateless_logistic(self):
        est = self._estimator(k=7)
        x = torch.randn(9, 7, device=self.device) * 5
        p = est.predict(x)
        torch.testing.assert_close(p, torch.sigmoid(_INIT_SLOPE * x + _INIT_BIAS))
        est.slope.fill_(1.3)
        est.intercepts.copy_(torch.linspace(-1, 1, 7))
        stats, calls = est._stats.clone(), est._num_calls.clone()
        p = est.predict(x)
        torch.testing.assert_close(p, torch.sigmoid(1.3 * x + est.intercepts))
        self.assertTrue(torch.equal(est._stats, stats))
        self.assertTrue(torch.equal(est._num_calls, calls))

    def test_observation_rule(self):
        k = 5
        est = self._estimator(k=k, refit_interval=1000)
        ref = _RefEstimator(k, refit_interval=1000)
        x = torch.randn(7, k, device=self.device) * 2
        # (num_verified, num_accepted): nothing verified, first draft rejected,
        # all accepted, trimmed then all accepted, rejection mid-way, trimmed
        # with a rejection, one draft accepted.
        cases = [(0, 0), (3, 0), (5, 5), (3, 3), (5, 2), (4, 1), (1, 1)]
        verified = torch.tensor([v for v, _ in cases], device=self.device)
        accepted = torch.tensor([a for _, a in cases], device=self.device)
        est.observe(x, verified, accepted)
        ref.observe(x, verified, accepted)
        torch.testing.assert_close(
            est._stats.cpu().double(), ref.stats, rtol=1e-5, atol=1e-6
        )
        # Observed counts per position: k <= accepted and k < verified.
        self.assertEqual(est._stats[:, 5].tolist(), [6.0, 4.0, 3.0, 1.0, 1.0])

    def test_padding_rows_are_ignored(self):
        k, bs, pad = 7, 40, 24
        est = self._estimator(k=k, max_num_reqs=bs + pad, refit_interval=5)
        padded = self._estimator(k=k, max_num_reqs=bs + pad, refit_interval=5)
        gen = torch.Generator().manual_seed(5)
        nan_x = torch.full((pad, k), float("nan"), device=self.device)
        zeros = torch.zeros(pad, dtype=torch.int32, device=self.device)
        for _ in range(12):  # crosses two refits
            x, verified, accepted = _random_step(
                gen, bs, k, 1.0, torch.zeros(k), self.device
            )
            est.observe(x, verified, accepted)
            # Padding rows verify nothing, so their features are never read.
            padded.observe(
                torch.cat([x, nan_x]),
                torch.cat([verified, zeros]),
                torch.cat([accepted, zeros]),
            )
        self.assertNotEqual(est.slope.item(), _INIT_SLOPE)
        self.assertTrue(torch.equal(padded.slope, est.slope))
        self.assertTrue(torch.equal(padded.intercepts, est.intercepts))
        self.assertTrue(torch.equal(padded._stats, est._stats))

    def test_matches_float64_reference_over_refits(self):
        k = 7
        true_slope = 1.3
        true_bias = torch.tensor([0.8, 0.5, 0.2, 0.0, -0.3, -0.5, -0.9])
        # 300 requests take two passes of the observe kernel's request loop.
        for refit_interval, bs, max_num_reqs, steps in [
            (100, 128, 128, 650),
            (7, 300, 512, 120),
        ]:
            gen = torch.Generator().manual_seed(refit_interval)
            est = self._estimator(
                k=k, max_num_reqs=max_num_reqs, refit_interval=refit_interval
            )
            ref = _RefEstimator(k, refit_interval)
            for step in range(steps):
                x, verified, accepted = _random_step(
                    gen, bs, k, true_slope, true_bias, self.device
                )
                torch.testing.assert_close(
                    est.predict(x).cpu().double(),
                    ref.predict(x.cpu()),
                    atol=1e-5,
                    rtol=1e-5,
                )
                est.observe(x, verified, accepted)
                ref.observe(x, verified, accepted)
                torch.testing.assert_close(
                    est.slope.cpu().double()[0], ref.slope, atol=1e-4, rtol=1e-4
                )
                torch.testing.assert_close(
                    est.intercepts.cpu().double(), ref.bias, atol=1e-4, rtol=1e-4
                )
            self.assertEqual(est._num_calls.item(), steps % refit_interval)
            if refit_interval == 100:
                # Six rounds of ~50k observations recover the generating model.
                self.assertAlmostEqual(est.slope.item(), true_slope, delta=0.1)
                torch.testing.assert_close(
                    est.intercepts.cpu(), true_bias, atol=0.25, rtol=0
                )

    def test_refit_interval_and_reset(self):
        k = 4
        est = self._estimator(k=k, refit_interval=3)
        gen = torch.Generator().manual_seed(2)
        for call in range(1, 4):
            x, verified, accepted = _random_step(
                gen, 16, k, 1.0, torch.zeros(k), self.device
            )
            est.observe(x, verified, accepted)
            if call < 3:
                self.assertEqual(est.slope.item(), _INIT_SLOPE)
                self.assertGreater(est._stats[:, 5].sum().item(), 0)
                self.assertEqual(est._num_calls.item(), call)
        self.assertNotEqual(est.slope.item(), _INIT_SLOPE)
        self.assertEqual(est._stats.abs().sum().item(), 0.0)
        self.assertEqual(est._num_calls.item(), 0)

    def test_slope_step_backtracks_to_stay_positive(self):
        k = 2
        # Confident drafts are always rejected, unsure ones always accepted, so
        # the full Newton step points to a negative slope.
        x = torch.tensor([[8.0, 8.0]] * 64 + [[-8.0, -8.0]] * 64, device=self.device)
        verified = torch.full((128,), 2, device=self.device)
        accepted = torch.tensor([0] * 64 + [2] * 64, device=self.device)
        # From 0.5 the step is halved until the slope stays positive. From 1e-4
        # it still crosses zero after every allowed halving and is dropped.
        for init_slope, slope_moves in [(0.5, True), (1e-4, False)]:
            est = self._estimator(k=k, refit_interval=1)
            est.slope.fill_(init_slope)
            ref = _RefEstimator(k, refit_interval=1, init_slope=est.slope.item())
            before = est.slope.clone()
            est.observe(x, verified, accepted)
            ref.observe(x, verified, accepted)
            torch.testing.assert_close(
                est.slope.cpu().double()[0], ref.slope, atol=1e-5, rtol=1e-5
            )
            # The intercepts take the Newton step given the slope step taken.
            torch.testing.assert_close(
                est.intercepts.cpu().double(), ref.bias, atol=1e-4, rtol=1e-4
            )
            if slope_moves:
                self.assertGreater(est.slope.item(), 0.0)
                self.assertLess(est.slope.item(), init_slope)
            else:
                self.assertTrue(torch.equal(est.slope, before))
            self.assertFalse(
                torch.equal(est.intercepts.cpu(), torch.full((k,), _INIT_BIAS))
            )

    def test_saturated_slope_keeps_moving(self):
        # An overconfident slope saturates the predictions: the curvature is
        # small and every full Newton step crosses zero. Dropping those steps
        # pins the slope; backtracking brings it to the generating model.
        k, bs, refit_interval, rounds = 7, 128, 4, 10
        init_slope, true_slope = 3.0, 1.0
        true_bias = torch.linspace(0.6, -0.6, k)
        est = self._estimator(k=k, max_num_reqs=bs, refit_interval=refit_interval)
        est.slope.fill_(init_slope)
        ref = _RefEstimator(k, refit_interval, init_slope=init_slope)
        dropped = _RefEstimator(
            k, refit_interval, init_slope=init_slope, max_slope_halvings=0
        )
        gen = torch.Generator().manual_seed(6)
        for _ in range(rounds * refit_interval):
            x, verified, accepted = _random_step(
                gen, bs, k, true_slope, true_bias, self.device
            )
            for e in (est, ref, dropped):
                e.observe(x, verified, accepted)
            torch.testing.assert_close(
                est.slope.cpu().double()[0], ref.slope, atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(
                est.intercepts.cpu().double(), ref.bias, atol=1e-4, rtol=1e-4
            )
        self.assertEqual(dropped.slope.item(), init_slope)
        self.assertAlmostEqual(est.slope.item(), true_slope, delta=0.15)

    def test_empty_batch_counts_as_a_call(self):
        est = self._estimator(k=3, refit_interval=2)
        empty = torch.empty(0, 3, device=self.device)
        none = torch.empty(0, dtype=torch.int32, device=self.device)
        est.observe(empty, none, none)
        self.assertEqual(est._num_calls.item(), 1)
        est.observe(empty, none, none)
        self.assertEqual(est._num_calls.item(), 0)
        # No observations: the refit leaves the coefficients alone.
        self.assertEqual(est.slope.item(), _INIT_SLOPE)
        self.assertTrue(torch.equal(est.intercepts.cpu(), torch.full((3,), _INIT_BIAS)))
        x = est.feature_from_scores(
            torch.empty(0, 3, 8, device=self.device), torch.empty(0, device=self.device)
        )
        self.assertEqual(tuple(x.shape), (0, 3))

    def test_no_host_sync(self):
        k = 7
        est = self._estimator(k=k, refit_interval=2)
        scores = torch.randn(16, k, 16, device=self.device)
        temps = torch.ones(16, 1, device=self.device)
        verified = torch.full((16,), k, dtype=torch.int32, device=self.device)
        accepted = torch.randint(0, k + 1, (16,), dtype=torch.int32, device=self.device)
        # Compile the kernels first.
        est.observe(est.feature_from_scores(scores, temps), verified, accepted)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):  # crosses a refit
                x = est.feature_from_scores(scores, temps)
                est.predict(x)
                est.feature_and_predict(scores, temps)
                est.observe(x, verified, accepted)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_cuda_graph_replay_matches_eager(self):
        k, bs, steps = 7, 48, 230
        true_bias = torch.linspace(0.6, -0.6, k)
        eager = self._estimator(k=k, max_num_reqs=bs, refit_interval=100)
        graphed = self._estimator(k=k, max_num_reqs=bs, refit_interval=100)
        scores = torch.zeros(bs, k, 16, device=self.device)
        temps = torch.ones(bs, device=self.device)
        verified = torch.zeros(bs, dtype=torch.int32, device=self.device)
        accepted = torch.zeros(bs, dtype=torch.int32, device=self.device)

        def body(est):
            x, fused_prob = est.feature_and_predict(scores, temps)
            prob = est.predict(est.feature_from_scores(scores, temps))
            est.observe(x, verified, accepted)
            return prob, fused_prob

        warm = self._estimator(k=k, max_num_reqs=bs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body(warm)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_probs = body(graphed)

        gen = torch.Generator().manual_seed(3)
        for _ in range(steps):
            x, v, a = _random_step(gen, bs, k, 1.0, true_bias, self.device)
            # Two-candidate rows [x, 0]: their log-odds feature is |x|.
            scores.zero_()
            scores[:, :, 0] = x
            scores[:, :, 1:] = float("-inf")
            scores[:, :, 1] = 0.0
            verified.copy_(v)
            accepted.copy_(a)
            graph.replay()
            for graph_prob, eager_prob in zip(graph_probs, body(eager)):
                self.assertTrue(torch.equal(graph_prob, eager_prob))
        self.assertTrue(torch.equal(graphed.slope, eager.slope))
        self.assertTrue(torch.equal(graphed.intercepts, eager.intercepts))
        self.assertNotEqual(eager.slope.item(), _INIT_SLOPE)


if __name__ == "__main__":
    unittest.main()
