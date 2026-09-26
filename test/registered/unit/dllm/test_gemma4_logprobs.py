"""Final-step denoiser logprobs, independent of model weights."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_gemma4_renoise import _batch, _config, _FakeRunner
from test_gemma4_uniform_lifecycle import _Batch, _result, _Scheduler

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.algorithm.gemma4_renoise import Gemma4Renoise
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ChangingRunner(_FakeRunner):
    """Reuse one logits buffer, like CUDA graph replay, with staggered convergence."""

    def forward(self, forward_batch, pp_proxy_tensors=None):
        out = super().forward(forward_batch, pp_proxy_tensors)
        if not hasattr(self, "buffer"):
            self.buffer = torch.zeros_like(out.logits_output.full_logits)
        step = len(self.records)
        buffer = self.buffer[: forward_batch.input_ids.numel()]
        rows = buffer.view(forward_batch.batch_size, 3, 4)
        rows.zero_()
        # First row converges immediately; its logits change on later forwards.
        for i in range(forward_batch.batch_size):
            if forward_batch.rids[i] == "early":
                rows[i, :, 0 if step == 1 else 1] = 40
            else:
                rows[i].copy_(
                    torch.tensor(
                        [
                            [0.0, 1.0, 2.0, 3.0],
                            [3.0, 1.0, 0.0, 2.0],
                            [0.0, 3.0, 1.0, 2.0],
                        ]
                    )
                    * step
                )
        out.logits_output.full_logits = buffer
        return out


class TestGemma4Logprobs(unittest.TestCase):
    def run_sampler(self, fdfo, enabled=True):
        algo = Gemma4Renoise(
            _config(fdfo=fdfo, max_denoising_steps=3, stability_threshold=0, seed=42)
        )
        batch = _batch(["early", "late"], 3)
        batch.return_logprob = enabled
        batch.token_ids_logprobs = [[3, 0, 3], [2, 1]]
        runner = _ChangingRunner()
        outputs = {}
        if fdfo:
            states = None
            for _ in range(3):
                out, tokens, lengths, states, _ = algo.run(runner, batch, states)
                keep = []
                for i, rid in enumerate(batch.rids):
                    if lengths[i]:
                        outputs[rid] = (
                            tokens[i],
                            out.next_token_logprobs[i].clone() if enabled else None,
                            out.next_token_token_ids_logprobs_val[i].clone()
                            if enabled
                            else None,
                        )
                    else:
                        keep.append(i)
                if not keep:
                    break
                states = [states[i] for i in keep]
                ids = [batch.token_ids_logprobs[i] for i in keep]
                batch = _batch([batch.rids[i] for i in keep], 3)
                batch.return_logprob = enabled
                batch.token_ids_logprobs = ids
        else:
            out, tokens, _, _, _ = algo.run(runner, batch)
            for i, rid in enumerate(batch.rids):
                outputs[rid] = (
                    tokens[i].tolist(),
                    out.next_token_logprobs[i] if enabled else None,
                    out.next_token_token_ids_logprobs_val[i] if enabled else None,
                )
        return outputs, runner

    def test_sync_and_fdfo_keep_each_rows_final_active_step(self):
        expected = {
            "early": torch.tensor([[40.0, 0.0, 0.0, 0.0]] * 3),
            "late": torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, 2.0], [0.0, 3.0, 1.0, 2.0]]
            )
            * 3,
        }
        for fdfo in (False, True):
            outputs, runner = self.run_sampler(fdfo)
            self.assertEqual(len(runner.records), 3)
            for rid, candidates in (("early", [3, 0, 3]), ("late", [2, 1])):
                tokens, selected, values = outputs[rid]
                reference = expected[rid].log_softmax(-1)
                self.assertEqual(tokens, expected[rid].argmax(-1).tolist())
                torch.testing.assert_close(
                    selected,
                    reference.gather(-1, torch.tensor(tokens)[:, None]).squeeze(-1),
                )
                torch.testing.assert_close(values, reference[:, candidates])

    def test_logprobs_do_not_change_sampling_or_extra_forward_count(self):
        for fdfo in (False, True):
            scored, scored_runner = self.run_sampler(fdfo)
            plain, plain_runner = self.run_sampler(fdfo, False)
            self.assertEqual(len(scored_runner.records), len(plain_runner.records))
            for rid in scored:
                self.assertEqual(scored[rid][0], plain[rid][0])
            for scored_step, plain_step in zip(
                scored_runner.records, plain_runner.records
            ):
                torch.testing.assert_close(
                    scored_step["input_ids"], plain_step["input_ids"]
                )
                torch.testing.assert_close(
                    scored_step["input_embeds"], plain_step["input_embeds"]
                )

    def test_selected_only_and_empty_candidates(self):
        for candidates in (None, [None], [[]]):
            algo = Gemma4Renoise(_config(max_denoising_steps=1))
            batch = _batch(["one"], 3)
            batch.return_logprob = True
            batch.token_ids_logprobs = candidates
            out, _, _, _, _ = algo.run(_FakeRunner(), batch)
            self.assertEqual(out.next_token_logprobs.shape, (1, 3))
            self.assertEqual(out.next_token_token_ids_logprobs_val, [None])

    def test_unfinished_fdfo_does_not_publish_candidate_scores(self):
        algo = Gemma4Renoise(
            _config(fdfo=True, max_denoising_steps=3, confidence_threshold=0)
        )
        batch = _batch(["one"], 3)
        batch.return_logprob = True
        batch.token_ids_logprobs = [[0, 1]]
        out, _, lengths, states, _ = algo.run(_FakeRunner(), batch)
        self.assertEqual(lengths, [0])
        self.assertNotIn("output_logprobs", states[0])
        self.assertEqual(out.next_token_token_ids_logprobs_val, [None])


class TestGemma4LogprobResponses(unittest.TestCase):
    def request(self, scheduler, *, enabled=True, **params):
        req = Req(
            "scores",
            "",
            array("q", [7, 8]),
            SamplingParams(**params),
            return_logprob=enabled,
            token_ids_logprob=[3, 1] if enabled else None,
            dllm_config=scheduler.dllm_config,
        )
        req.dllm_block_offset = 2
        req.sampling_params.normalize(None)
        req.full_untruncated_fill_ids.extend([0] * 4)
        req.set_extend_range(2, 6)
        req.stream = True
        return req

    def emit(self, req):
        stream = _GenerationStreamAccumulator(
            return_logprob=True,
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=1,
            get_cached_tokens_details=lambda req: None,
            current_weight_version=None,
        )
        stream.accept(req=req)
        return stream

    def process(self, scheduler, reqs, *, pending=False):
        batch = _Batch(reqs)
        batch.forward_mode = _batch(["one"], 4).forward_mode
        batch.return_logprob = True
        fdfo = scheduler.dllm_config.first_done_first_out_mode
        tokens = [3, 2, 1, 0]
        result = _result(
            [tokens if fdfo else torch.tensor(tokens) for _ in reqs],
            accept_lengths=[0 if pending else 4] * len(reqs) if fdfo else None,
            algo_states=[{}] * len(reqs) if fdfo else None,
        )
        result.logits_output = SimpleNamespace(
            next_token_logprobs=torch.tensor([[-0.1, -0.2, -0.3, -0.4]] * len(reqs)),
            next_token_token_ids_logprobs_val=[
                torch.tensor([[-0.1, -2.0], [-0.2, -3.0], [-0.3, -4.0], [-0.4, -5.0]])
            ]
            * len(reqs),
        )
        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache"):
            scheduler.process_batch_result_dllm(batch, result)

    def test_response_truncation_and_mixed_logprob_flags(self):
        for fdfo in (False, True):
            for params in (
                {"max_new_tokens": 2},
                {"max_new_tokens": 8, "stop_token_ids": {2}},
            ):
                with self.subTest(fdfo=fdfo, params=params):
                    scheduler = _Scheduler(fdfo=fdfo)
                    req = self.request(scheduler, **params)
                    plain = self.request(scheduler, enabled=False, **params)
                    self.process(scheduler, [req, plain])
                    stream = self.emit(req)
                    self.assertEqual(list(stream.output_ids[0]), [3, 2])
                    self.assertEqual(stream.output_token_logprobs_idx, [[3, 2]])
                    self.assertEqual(
                        stream.output_token_ids_logprobs_idx, [[[3, 1], [3, 1]]]
                    )
                    self.assertEqual(len(stream.output_token_ids_logprobs_val[0]), 2)
                    self.assertEqual(stream.input_token_logprobs_val, [[]])
                    self.assertIsNone(plain.logprob.output_token_logprobs_val)

    def test_streaming_multiblock_offsets_and_unresolved_fdfo(self):
        for fdfo in (False, True):
            scheduler = _Scheduler(fdfo=fdfo)
            req = self.request(scheduler, max_new_tokens=6)
            if fdfo:
                self.process(scheduler, [req], pending=True)
                self.assertEqual(req.logprob.output_token_logprobs_val, [])
            self.process(scheduler, [req])
            first = self.emit(req)
            self.assertEqual(first.output_token_logprobs_idx, [[3, 2, 1, 0]])
            req.full_untruncated_fill_ids.extend([0] * 4)
            req.set_extend_range(6, 10)
            self.process(scheduler, [req])
            second = self.emit(req)
            self.assertEqual(second.output_token_logprobs_idx, [[3, 2]])
            self.assertEqual(len(second.output_token_ids_logprobs_val[0]), 2)
            self.assertEqual(req.send_output_token_logprobs_offset, 6)

    def test_validation(self):
        scheduler = _Scheduler(fdfo=False)
        req = self.request(scheduler, max_new_tokens=5)
        for start in (-1, 2):
            self.assertIsNone(
                scheduler.validate_dllm_request(
                    req, SimpleNamespace(logprob_start_len=start)
                )
            )
        self.assertIn(
            "output denoiser",
            scheduler.validate_dllm_request(req, SimpleNamespace(logprob_start_len=0)),
        )
        req.sampling_params.max_new_tokens = 0
        self.assertIn("no denoising", Gemma4Renoise.validate_request(req))
        req.sampling_params.max_new_tokens = 5
        req.return_logprob = False
        self.assertIn("without return_logprob", Gemma4Renoise.validate_request(req))


if __name__ == "__main__":
    unittest.main()
