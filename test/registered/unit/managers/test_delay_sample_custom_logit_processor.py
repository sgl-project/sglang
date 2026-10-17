"""Deterministic regression for delayed sampling with custom logit processors
under overlap scheduling.

Background: `DeepseekOCRNoRepeatNGramLogitProcessor` rebuilds the token
history from `req.origin_input_ids + req.output_ids`. Under the default
overlap scheduler, the previous batch's sampled tokens are only committed to
`req.output_ids` after the current batch's forward has been launched
(`event_loop_overlap`: run_batch -> pop_and_process ->
launch_batch_sample_if_needed). Sampling eagerly inside
`TpModelWorker.forward_batch_generation` lets the processor see a history that
lags one token behind, shifting the phase of the banned n-gram positions so
period >= 2 repetition loops escape the ban entirely.

The invariant under test: when sampling the k-th output token, the request
history visible to the processor contains exactly the first k-1 output
tokens. These tests drive the real `TpModelWorker.forward_batch_generation`
gating condition and its `sample_batch_func` closure; only the model forward
(`model_runner`) is replaced with a stub that records how much history a
`__req__`-reading processor would observe at sample time.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.tp_worker import TpModelWorker

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")

VOCAB = 8
SAMPLED_TOKEN = 7


def _make_req(origin_input_ids):
    """A minimal stand-in for the Req a custom logit processor reads via __req__."""
    return SimpleNamespace(
        origin_input_ids=array("q", origin_input_ids), output_ids=array("q", [])
    )


def _make_forward_batch(req, has_processor=True, grammars=None, is_prefill_only=False):
    sampling_info = SimpleNamespace(
        grammars=grammars,
        has_custom_logit_processor=has_processor,
        is_prefill_only=is_prefill_only,
        # Mirrors how the sampler resolves per-row processor params.
        custom_params=[
            {"__req__": req, "ngram_size": 2, "window_size": 100},
        ],
    )
    return SimpleNamespace(
        sampling_info=sampling_info,
        is_prefill_only=is_prefill_only,
        return_logprob=False,
        input_ids=torch.zeros(1, dtype=torch.long),
        seq_lens=torch.ones(1, dtype=torch.long),
        apply_deprecated_skip_attn_backend_init=lambda *args, **kwargs: None,
    )


def _make_worker(req, sample_history):
    """TpModelWorker with only the model runner stubbed.

    `model_runner.sample` records `len(req.output_ids)` at the moment sampling
    runs — exactly the history a processor reads via `__req__` — and returns a
    fixed token. `model_runner` is a read-only property on TpModelWorker, so
    tests must patch it with a PropertyMock; this helper returns both.
    """
    worker = TpModelWorker.__new__(TpModelWorker)
    worker.enable_overlap = True
    worker.enable_spec = False
    worker.dllm_algorithm = None
    worker.pp_group = SimpleNamespace(is_last_rank=True)

    model_runner = MagicMock()

    def fake_forward(forward_batch, pp_proxy_tensors=None):
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, VOCAB)),
            can_run_graph=False,
            expert_distribution_metrics=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            kv_loc_plan=None,
        )

    def fake_sample(logits_output, forward_batch):
        sample_history.append(len(req.output_ids))
        return torch.tensor([[SAMPLED_TOKEN]])

    model_runner.forward.side_effect = fake_forward
    model_runner.sample.side_effect = fake_sample
    model_runner._elastic_cuda_graph_enabled.return_value = False
    return worker, model_runner


class TestDelaySampleCustomLogitProcessor(CustomTestCase):
    def test_sample_sees_fully_committed_history(self):
        """A processor batch must defer sampling past the previous batch's commit.

        Fails on pre-fix code: sampling ran eagerly inside
        forward_batch_generation, so next_token_ids was already set here and no
        delay_sample_func existed.
        """
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)
        forward_batch = _make_forward_batch(req)

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            result = worker.forward_batch_generation(None, forward_batch)

            self.assertIsNone(result.next_token_ids)
            self.assertIsNotNone(result.delay_sample_func)
            # event_loop_overlap commits the previous batch's token
            # (process_batch_result_decode) before launch_batch_sample_if_needed
            # runs the deferred sampling.
            req.output_ids.append(42)
            result.delay_sample_func()

        # The processor must have seen the committed output token: output_ids
        # holds exactly the k-1 = 1 previously sampled token, not a stale
        # history that lags one token behind.
        self.assertEqual(sample_history, [1])
        self.assertEqual(result.next_token_ids.tolist(), [[SAMPLED_TOKEN]])

    def test_two_overlap_iterations_keep_phase(self):
        """Emulates two overlap iterations: sampling iteration k must observe
        every output token committed by iterations < k.

        Fails on pre-fix code, where iteration k's sample ran before iteration
        k-1's commit and the recorded history stayed at 0 (the phase shift that
        let period >= 2 n-gram loops escape the ban).
        """
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            # Iteration 1 (extend): first output token, nothing committed yet.
            result_1 = worker.forward_batch_generation(
                None, _make_forward_batch(req, is_prefill_only=False)
            )
            result_1.delay_sample_func()
            req.output_ids.append(SAMPLED_TOKEN)

            # Iteration 2 (decode): the commit above happens before this
            # sampling runs.
            result_2 = worker.forward_batch_generation(
                None, _make_forward_batch(req, is_prefill_only=False)
            )
            result_2.delay_sample_func()

        self.assertEqual(sample_history, [0, 1])
        self.assertEqual(result_2.next_token_ids.tolist(), [[SAMPLED_TOKEN]])

    def test_no_processor_samples_eagerly(self):
        """Without a processor (and with the env flag off) the eager sampling
        path is unchanged: no delay function, next_token_ids set immediately."""
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)
        forward_batch = _make_forward_batch(req, has_processor=False)

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            result = worker.forward_batch_generation(None, forward_batch)

        self.assertIsNone(result.delay_sample_func)
        self.assertIsNotNone(result.next_token_ids)
        self.assertEqual(sample_history, [0])

    def test_grammar_requests_still_delayed(self):
        """The existing grammar delay-sampling behavior is preserved."""
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)
        forward_batch = _make_forward_batch(req, has_processor=False, grammars=["x"])

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            result = worker.forward_batch_generation(None, forward_batch)

        self.assertIsNone(result.next_token_ids)
        self.assertIsNotNone(result.delay_sample_func)

    def test_prefill_only_batch_not_delayed(self):
        """Prefill-only requests never sample, so they must not be deferred."""
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)
        forward_batch = _make_forward_batch(req, is_prefill_only=True)

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            result = worker.forward_batch_generation(None, forward_batch)

        self.assertIsNone(result.delay_sample_func)
        self.assertEqual(sample_history, [])
        # Dummy zeros for prefill-only, one per sequence.
        self.assertEqual(result.next_token_ids.tolist(), [0])

    def test_delay_sample_env_flag_still_works(self):
        """SGLANG_ENABLE_DELAY_SAMPLE keeps deferring all non-prefill-only
        batches regardless of processors."""
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)
        forward_batch = _make_forward_batch(req, has_processor=False)

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            with patch.object(
                envs.SGLANG_ENABLE_DELAY_SAMPLE, "get", return_value=True
            ):
                result = worker.forward_batch_generation(None, forward_batch)

        self.assertIsNone(result.next_token_ids)
        self.assertIsNotNone(result.delay_sample_func)

    def test_speculative_never_delayed(self):
        """Spec decoding owns its verify/sample flow; the delay path must stay
        disabled there."""
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)
        worker.enable_spec = True
        forward_batch = _make_forward_batch(req)

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            result = worker.forward_batch_generation(None, forward_batch)

        self.assertIsNone(result.delay_sample_func)
        self.assertEqual(sample_history, [0])

    def test_no_overlap_never_delayed(self):
        """With overlap scheduling off, sampling is already in-order and the
        eager path applies."""
        req = _make_req([1, 2])
        sample_history = []
        worker, model_runner = _make_worker(req, sample_history)
        worker.enable_overlap = False
        forward_batch = _make_forward_batch(req)

        with patch.object(
            TpModelWorker, "model_runner", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = model_runner
            result = worker.forward_batch_generation(None, forward_batch)

        self.assertIsNone(result.delay_sample_func)
        self.assertEqual(sample_history, [0])


if __name__ == "__main__":
    unittest.main()
