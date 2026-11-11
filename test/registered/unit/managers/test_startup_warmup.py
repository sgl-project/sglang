"""--enable-decode-warmup: the round plan, fail-closed startup, what the rounds
leave behind for served requests, and the refusals."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers import startup_warmup as W
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.speculative.spec_utils import simulated_acc_len_generator
from sglang.srt.speculative.staged_next_draft import StagedDraftTracker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _plan(batch_sizes=(16, 4, 2, 1), prompt_len=2048, **kw):
    args = dict(
        max_running_requests=128,
        max_prefill_tokens=16384,
        chunked_prefill_size=16384,
        page_size=64,
        max_req_input_len=262144,
    )
    args.update(kw)
    return W._plan(batch_sizes=list(batch_sizes), prompt_len=prompt_len, **args)


class TestDecodeWarmupPlan(CustomTestCase):
    def test_rounds_fit_the_prefill_budget_and_running_cap(self):
        self.assertEqual(
            _plan(prompt_len=12288),
            [(16, 1024), (4, 4096), (2, 8192), (1, 12288)],
        )
        # A running cap below the largest size dedups the capped sizes.
        self.assertEqual(
            _plan(prompt_len=12288, max_running_requests=3),
            [(3, 5440), (2, 8192), (1, 12288)],
        )
        # A PD decoder may set neither limit: the default budget applies.
        self.assertEqual(
            _plan(prompt_len=8192, max_prefill_tokens=None, chunked_prefill_size=None),
            [(16, 512), (4, 2048), (2, 4096), (1, 8192)],
        )

    def test_prompt_is_page_aligned_and_below_the_input_limit(self):
        for bs, prompt in _plan(prompt_len=5000, max_req_input_len=3000, page_size=64):
            self.assertEqual(prompt % 64, 0, bs)
            self.assertLess(prompt, 3000)
            self.assertLessEqual(bs * prompt, 16384)
        # Below one page the prompt is kept as is rather than rounded to 0.
        self.assertEqual(_plan(batch_sizes=[1], prompt_len=10), [(1, 10)])


def _scheduler():
    return SimpleNamespace(
        model_worker=SimpleNamespace(),
        max_running_requests=4,
        max_prefill_tokens=8192,
        chunked_prefill_size=8192,
        page_size=1,
        max_req_input_len=10000,
        enable_overlap=False,
    )


class TestDecodeWarmupFailsClosed(CustomTestCase):
    def _run(self, *, round_fn, pools):
        restored = []
        with (
            mock.patch.object(W, "_refusal", lambda s, w: None),
            mock.patch.object(W, "_save_rng", lambda worker: "rng"),
            mock.patch.object(
                W, "_restore", lambda worker, saved: restored.append(saved)
            ),
            mock.patch.object(W, "_round", round_fn),
            mock.patch.object(W, "_pool_text", lambda s: pools.pop(0)),
        ):
            try:
                W.run_decode_warmup(
                    _scheduler(), batch_sizes=[4, 1], prompt_len=64, sampling=None
                )
            finally:
                self.assertEqual(restored, ["rng"])

    def test_completes_when_every_slot_comes_back(self):
        calls = []

        def round_fn(scheduler, worker, *, bs, prompt_len, sampling, tag):
            calls.append((bs, prompt_len, sampling, torch.is_grad_enabled()))
            return 0.1, 0.2

        self._run(round_fn=round_fn, pools=["same", "same"])
        self.assertEqual(calls, [(4, 64, {}, False), (1, 64, {}, False)])

    def test_a_failed_round_stops_the_server_after_restoring_rng(self):
        def round_fn(scheduler, worker, **kw):
            raise ValueError("boom")

        with self.assertRaisesRegex(RuntimeError, "Decode warm-up failed"):
            self._run(round_fn=round_fn, pools=["same", "same"])

    def test_a_pool_that_does_not_come_back_stops_the_server(self):
        with self.assertRaisesRegex(RuntimeError, "Decode warm-up failed"):
            self._run(round_fn=lambda *a, **k: (0.0, 0.0), pools=["before", "after"])

    def test_a_refused_server_skips_with_a_warning(self):
        with (
            mock.patch.object(
                W, "_refusal", lambda s, w: "a PD prefill server runs no decode round"
            ),
            mock.patch.object(W, "_round") as round_fn,
            self.assertLogs(W.logger, level="WARNING"),
        ):
            W.run_decode_warmup(
                _scheduler(), batch_sizes=[1], prompt_len=64, sampling=None
            )
        round_fn.assert_not_called()


class TestDecodeWarmupRestore(CustomTestCase):
    """What the rounds drew or noted is undone before the first served round."""

    def _worker(self, sim_rng):
        tracker = StagedDraftTracker(width=4, graph_sizes=[1])
        tracker.noted = ("last round's note",)
        return SimpleNamespace(
            _simulated_acc_len_rng=sim_rng,
            _post_verify_graphs=None,
            _staged_draft=tracker,
        )

    def _save_draw_restore(self, worker):
        with (
            mock.patch.object(torch.cuda, "get_rng_state", lambda: "cuda"),
            mock.patch.object(torch.cuda, "set_rng_state", lambda state: None),
            mock.patch.object(torch.cuda, "synchronize", lambda: None),
        ):
            saved = W._save_rng(worker)
            if worker._simulated_acc_len_rng is not None:
                torch.randint(0, 4, (8,), generator=worker._simulated_acc_len_rng)
            torch.rand(8)
            W._restore(worker, saved)

    def test_forced_acceptance_draws_are_the_unwarmed_ones(self):
        worker = self._worker(simulated_acc_len_generator(35))
        cpu_first = torch.rand(4, generator=torch.Generator().manual_seed(1))
        torch.manual_seed(1)
        self._save_draw_restore(worker)
        unwarmed = torch.randint(0, 4, (8,), generator=simulated_acc_len_generator(35))
        served = torch.randint(0, 4, (8,), generator=worker._simulated_acc_len_rng)
        self.assertTrue(torch.equal(served, unwarmed))
        self.assertTrue(torch.equal(torch.rand(4), cpu_first))
        self.assertIsNone(worker._staged_draft.noted)

    def test_without_its_own_generator_the_cpu_generator_is_restored(self):
        worker = self._worker(None)
        torch.manual_seed(2)
        expected = torch.rand(4, generator=torch.Generator().manual_seed(2))
        self._save_draw_restore(worker)
        self.assertTrue(torch.equal(torch.rand(4), expected))


class TestDecodeWarmupRelease(CustomTestCase):
    def test_the_dummies_go_back_through_the_request_retirer(self):
        retirer, tree_cache = mock.Mock(), mock.Mock()
        scheduler = SimpleNamespace(request_retirer=retirer, tree_cache=tree_cache)
        worker = SimpleNamespace(_post_verify_graphs=None)
        lock = object()
        reqs = [SimpleNamespace(lock=None), SimpleNamespace(lock=lock)]
        with mock.patch.object(torch.cuda, "synchronize", lambda: None):
            W._release(scheduler, worker, reqs)
        self.assertEqual(
            retirer.release.call_args_list,
            [mock.call(req, checkpoint=False) for req in reqs],
        )
        # A request that never got KV still holds its tree lock.
        tree_cache.unlock.assert_called_once_with(lock)
        self.assertIsNone(reqs[1].lock)


class TestDecodeWarmupRefusal(CustomTestCase):
    def test_pd_prefill_and_non_eagle_v2_workers_are_refused(self):
        worker = SimpleNamespace()
        prefill = SimpleNamespace(disaggregation_mode=DisaggregationMode.PREFILL)
        self.assertIn("PD prefill", W._refusal(prefill, worker))
        plain = SimpleNamespace(disaggregation_mode=DisaggregationMode.NULL)
        self.assertIn("EAGLE v2", W._refusal(plain, worker))


class TestDecodeWarmupSampling(CustomTestCase):
    def test_without_preferred_params_the_dummies_sample_as_served_requests(self):
        """Served requests that send no sampling parameters run temperature 1.0,
        top_p 1.0 and no top-k; the warm-up rounds must take the same sampling
        mode so the kernels and graphs they warm are the served ones."""
        params = SamplingParams(**{"max_new_tokens": 64, "ignore_eos": True, **{}})
        self.assertEqual(
            (params.temperature, params.top_p, params.top_k, params.min_p),
            (1.0, 1.0, TOP_K_ALL, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
