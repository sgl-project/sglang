"""The folded DSpark draft sampler takes the one-kernel sharded Markov chain
(--enable-dsv41-low-latency-decode) for the batches it supports and the per-step
path otherwise; the chain's noise offset advances once per sampled block, as the
per-step path does; without the switch no chain is installed."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.dsv41_low_latency.dspark import sharded_markov
from sglang.srt.runtime_context import get_context, get_exec, reset_context
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
from sglang.test.test_utils import CustomTestCase

GAMMA, MAX_BS, VOCAB, HIDDEN = 5, 8, 64, 4


class _Kernel:
    """DSparkShardedMarkovSampler's surface, recording launches."""

    max_rows = MAX_BS

    def __init__(self, supported):
        self.supported, self.calls, self.noise_draws = supported, [], 0

    def fork_noise(self, **kwargs):
        self.noise_draws += 1

    def join_noise(self):
        pass

    def supports(self, base_logits):
        return self.supported

    def sample(self, *, draft_tokens, logits_cache, **kwargs):
        draft_tokens.fill_(7)
        self.calls.append(logits_cache)


def _sampler(kernel, greedy_draft):
    sampler = ds.DsparkDraftSampler.__new__(ds.DsparkDraftSampler)
    sampler.model = SimpleNamespace(
        compute_base_logits=lambda h: (torch.zeros(h.shape[0], VOCAB), None),
        lm_head=SimpleNamespace(org_vocab_size=VOCAB),
        markov_head=SimpleNamespace(
            vocab_shard=None, step_logits_dtype=lambda d: torch.float32
        ),
    )
    sampler.markov_head = sampler.model.markov_head
    sampler.gamma = sampler.query_token_num = GAMMA
    sampler.sample_from_anchor = True
    sampler.max_bs = MAX_BS
    sampler.out = torch.zeros(MAX_BS * GAMMA, dtype=torch.int64)
    sampler.confidence_out = None
    sampler.greedy_draft = greedy_draft
    sampler._noise = SimpleNamespace(key=1, offset=torch.zeros(1, dtype=torch.int64))
    sampler.params = SimpleNamespace(step_temperatures=torch.zeros(MAX_BS))
    sampler.step_logits = None
    sampler._tp_sync = None
    sampler.markov_chain = (
        None if kernel is None else sharded_markov.ShardedMarkovChain(kernel)
    )
    return sampler


def _run(sampler):
    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        sampler(torch.zeros(GAMMA, HIDDEN), torch.zeros(GAMMA, dtype=torch.int64))


def _per_step(calls):
    def per_step(**kwargs):
        calls.append(kwargs)
        return torch.full((1, GAMMA), 3, dtype=torch.int64), None

    return per_step


class TestShardedMarkovDispatch(CustomTestCase):
    def test_supported_batches_take_the_chain_kernel(self):
        for greedy_draft in (False, True):
            with self.subTest(greedy_draft=greedy_draft):
                kernel = _Kernel(supported=True)
                sampler = _sampler(kernel, greedy_draft)
                with patch.object(ds, "draft_markov_block", side_effect=AssertionError):
                    _run(sampler)
                self.assertTrue(sampler.out[:GAMMA].eq(7).all())
                self.assertEqual(len(kernel.calls), 1)
                # A greedy draft keeps no step logits and draws no noise.
                self.assertEqual(kernel.calls[0] is None, greedy_draft)
                expected = 0 if greedy_draft else 1
                self.assertEqual(kernel.noise_draws, expected)
                self.assertEqual(int(sampler._noise.offset), expected)

    def test_unsupported_batches_and_no_chain_keep_the_per_step_path(self):
        for kernel in (_Kernel(supported=False), None):
            with self.subTest(chain=kernel is not None):
                calls = []
                sampler = _sampler(kernel, greedy_draft=True)
                with patch.object(ds, "draft_markov_block", _per_step(calls)):
                    _run(sampler)
                self.assertEqual(len(calls), 1)
                self.assertTrue(sampler.out[:GAMMA].eq(3).all())
                if kernel is not None:
                    self.assertFalse(kernel.calls)


class TestSwitchGate(CustomTestCase):
    def setUp(self):
        reset_context()
        self._published = get_context().override_server_args(model_path="dummy")
        self._published.install()

    def tearDown(self):
        self._published.restore()
        reset_context()

    def test_chain_installed_only_with_the_switch(self):
        model = SimpleNamespace(
            markov_head=object(), confidence_head=None, sample_from_anchor=True
        )
        for on in (False, True):
            with (
                self.subTest(switch=on),
                get_exec().features.override(enable_dsv41_low_latency_decode=on),
                patch.object(sharded_markov, "install_draft_sampler") as install,
            ):
                ds.maybe_build_draft_sampler(
                    draft_model=model,
                    gamma=GAMMA,
                    max_bs=MAX_BS,
                    device="cpu",
                    tp_rank=1,
                    noise=None,
                    greedy_draft=True,
                    tp_sync=None,
                )
                self.assertEqual(install.called, on)


if __name__ == "__main__":
    unittest.main()
