import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.models.dspark import run_markov_block
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    DraftForwardResult,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
    _resolve_folded_sampling,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _sampling_info():
    return SimpleNamespace(
        temperatures=torch.tensor([[0.7], [1.5], [1.0]]),
        top_ks=torch.tensor([2, 3, 1], dtype=torch.int32),
        top_ps=torch.tensor([0.8, 0.9, 1.0]),
        is_all_greedy=False,
        is_any_greedy=True,
    )


def _reference_probs(logits, sampling_info, temperature=None, top_k=None, top_p=None):
    rows = []
    for row, scores in enumerate(logits):
        temp = (
            float(sampling_info.temperatures[row])
            if temperature is None
            else temperature
        )
        target_k = int(sampling_info.top_ks[row])
        k = target_k if top_k is None else (TOP_K_ALL if top_k == -1 else top_k)
        p = sampling_info.top_ps[row] if top_p is None else top_p
        if temp == 0 or target_k <= 1 or k <= 1:
            probs = torch.zeros_like(scores, dtype=torch.float32)
            probs[scores.argmax()] = 1.0
        else:
            probs = (scores.float() / temp).softmax(-1)
            sorted_probs, indices = probs.sort(descending=True)
            sorted_probs[k:] = 0
            sorted_probs /= sorted_probs.sum()
            remove = sorted_probs.cumsum(-1) - sorted_probs > p
            sorted_probs[remove] = 0
            sorted_probs /= sorted_probs.sum()
            probs = torch.zeros_like(probs).scatter_(-1, indices, sorted_probs)
        rows.append(probs)
    return torch.stack(rows)


class _MarkovHead:
    def apply_step_logits(self, logits, *, token_ids, hidden_states):
        # Promote a token outside the base top-k. Its identity changes with
        # the sampled prefix, so pre-correction cutoffs cannot pass this test.
        correction = torch.zeros_like(logits)
        correction.scatter_(1, ((token_ids + 3) % logits.shape[-1])[:, None], 4.0)
        return logits + correction

    def sample_block(self, base_logits, **kwargs):
        return run_markov_block(self, base_logits, **kwargs)


def _tp_sync():
    return SimpleNamespace(sync=lambda _site, values: values)


def _base_logits():
    return torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0]).repeat(3, 2, 1)


class TestDsparkDraftSampling(unittest.TestCase):
    def setUp(self):
        self.spec = SimpleNamespace(
            speculative_draft_temperature=None,
            speculative_draft_top_k=None,
            speculative_draft_top_p=None,
        )
        patcher = mock.patch(
            "sglang.srt.runtime_context.get_spec", return_value=self.spec
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _check_block(
        self, tokens, logits, probs, info, temperature=None, top_k=None, top_p=None
    ):
        for step in range(tokens.shape[1]):
            expected = _reference_probs(
                logits[:, step], info, temperature, top_k, top_p
            )
            torch.testing.assert_close(probs[:, step], expected)
            self.assertTrue(
                bool((probs[:, step].gather(1, tokens[:, step, None]) > 0).all())
            )

    def test_eager_saves_corrected_truncated_distribution_for_both_samplers(self):
        info = _sampling_info()
        for fast in (False, True):
            for override, top_k, top_p in (
                (None, None, None),
                (0.4, 2, 0.5),
                (None, -1, 1.0),
                (None, 1, None),
                (0.0, -1, 1.0),
            ):
                with self.subTest(
                    fast=fast, override=override, top_k=top_k, top_p=top_p
                ):
                    self.spec.speculative_draft_temperature = override
                    self.spec.speculative_draft_top_k = top_k
                    self.spec.speculative_draft_top_p = top_p
                    with envs.SGLANG_DSPARK_FAST_SAMPLING.override(fast):
                        result = sample_draft_block(
                            base_logits=_base_logits(),
                            anchor_tokens=torch.zeros(3, dtype=torch.long),
                            draft_hidden=torch.zeros(3, 2, 1),
                            sampling_info=info,
                            markov_head=_MarkovHead(),
                            device=torch.device("cpu"),
                            tp_sync=_tp_sync(),
                        )
                    self._check_block(
                        result.draft_tokens,
                        result.corrected_logits,
                        result.draft_probs,
                        info,
                        override,
                        top_k,
                        top_p,
                    )
                    # Token 3 is outside the uncorrected top-2, yet the
                    # correction puts it inside the actual proposal support.
                    self.assertGreater(float(result.draft_probs[0, 0, 3]), 0)
                    torch.testing.assert_close(
                        result.greedy_mask,
                        torch.tensor([override == 0.0 or top_k == 1] * 2 + [True]),
                    )

    def test_folded_sampler_retains_q_and_resets_graph_padding(self):
        base_logits = _base_logits()
        model = SimpleNamespace(
            sample_from_anchor=True,
            markov_head=_MarkovHead(),
            lm_head=SimpleNamespace(org_vocab_size=5, weight=torch.empty(1)),
            compute_base_logits=lambda hidden: (hidden, None),
        )
        sampler = DsparkDraftSampler(
            model=model, gamma=2, max_bs=3, device="cpu", tp_sync=_tp_sync()
        )
        info = _sampling_info()
        probs_address = sampler.probs_out.data_ptr()
        for override, top_k, top_p in (
            (None, None, None),
            (None, -1, 1.0),
            (0.4, 2, 0.5),
            (None, 1, None),
            (0.0, -1, 1.0),
        ):
            self.spec.speculative_draft_temperature = override
            self.spec.speculative_draft_top_k = top_k
            self.spec.speculative_draft_top_p = top_p
            sampler.stage_sampling_params(bs=3, sampling_info=info)
            sampler(base_logits.reshape(6, 5), torch.zeros(6, dtype=torch.long))
            self._check_block(
                sampler.out.view(3, 2),
                sampler.corrected_out.view(3, 2, 5),
                sampler.probs_out.view(3, 2, 5),
                info,
                override,
                top_k,
                top_p,
            )
        self.spec.speculative_draft_temperature = None
        sampler.stage_sampling_params(bs=1, sampling_info=info)
        self.assertEqual(sampler.probs_out.data_ptr(), probs_address)
        torch.testing.assert_close(sampler.temperatures[1:], torch.ones(2))
        torch.testing.assert_close(sampler.sampling_params.top_ps[1:], torch.ones(2))
        self.assertTrue(bool((sampler.sampling_params.top_ks[1:] == TOP_K_ALL).all()))
        self.assertFalse(bool(sampler.greedy_mask[1:].any()))

    def test_folded_memory_budget_includes_saved_probabilities(self):
        model = SimpleNamespace(
            lm_head=SimpleNamespace(org_vocab_size=1024, weight=torch.empty(1)),
            markov_head=SimpleNamespace(supports_sharded_greedy=False),
        )
        # Noise and corrected logits fit above the 1 GiB floor; adding q does not.
        with envs.SGLANG_DSPARK_FOLDED_SAMPLING.override(
            DsparkFoldedSampling.AUTO.value
        ):
            self.assertFalse(
                _resolve_folded_sampling(
                    model=model,
                    gamma=5,
                    max_bs=64,
                    device="cpu",
                    tp_rank=1,
                    available_memory_gb=1.002,
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_folded_cuda_graph_replays_changed_sampling_policy(self):
        hidden = _base_logits().to("cuda").reshape(6, 5)
        input_ids = torch.zeros(6, dtype=torch.long, device="cuda")
        model = SimpleNamespace(
            sample_from_anchor=True,
            markov_head=_MarkovHead(),
            lm_head=SimpleNamespace(org_vocab_size=5, weight=hidden),
            compute_base_logits=lambda values: (values, None),
        )
        sampler = DsparkDraftSampler(
            model=model, gamma=2, max_bs=3, device="cuda", tp_sync=_tp_sync()
        )
        info = _sampling_info()
        for name in ("temperatures", "top_ks", "top_ps"):
            setattr(info, name, getattr(info, name).to("cuda"))
        sampler.stage_sampling_params(bs=3, sampling_info=info)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sampler(hidden, input_ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler(hidden, input_ids)
        for override, top_k, top_p in (
            (None, None, None),
            (0.4, 2, 0.5),
            (None, -1, 1.0),
            (None, 1, None),
            (0.0, -1, 1.0),
        ):
            self.spec.speculative_draft_temperature = override
            self.spec.speculative_draft_top_k = top_k
            self.spec.speculative_draft_top_p = top_p
            sampler.stage_sampling_params(bs=3, sampling_info=info)
            graph.replay()
            self._check_block(
                sampler.out.view(3, 2),
                sampler.corrected_out.view(3, 2, 5),
                sampler.probs_out.view(3, 2, 5),
                info,
                override,
                top_k,
                top_p,
            )

    def test_proposer_passes_folded_q_to_verification(self):
        sampler = SimpleNamespace(
            folded_sampling=True,
            out=torch.arange(6),
            corrected_out=torch.randn(6, 5),
            probs_out=torch.randn(6, 5).softmax(-1),
            temperatures=torch.ones(3),
            greedy_mask=torch.zeros(3, dtype=torch.bool),
            confidence_out=None,
        )
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer.sample_from_anchor = True
        proposer.gamma = 2
        proposer._draft_sampler = sampler
        proposer._run_forward = mock.Mock(
            return_value=DraftForwardResult(
                draft_block_ids=torch.zeros(2, 2, dtype=torch.long),
                raw_hidden=torch.zeros(4, 1),
                draft_hidden_3d=torch.zeros(2, 2, 1),
                can_run_graph=True,
            )
        )
        target = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(5, 1))
        with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(True):
            result = proposer.propose(
                batch=None,
                draft_input=None,
                verify_window=None,
                bs=2,
                device="cpu",
                target_model=target,
                sampling_info=_sampling_info(),
            )
        self.assertTrue(result.folded)
        self.assertEqual(
            result.draft_block.draft_probs.data_ptr(), sampler.probs_out.data_ptr()
        )
        torch.testing.assert_close(
            result.draft_block.draft_probs, sampler.probs_out[:4].view(2, 2, 5)
        )

    def test_mixed_verification_uses_saved_q_and_target_greedy_mask(self):
        info = _sampling_info()
        probs = torch.zeros(3, 2, 5)
        probs[:, :, 3] = 1.0
        draft = DraftBlockResult(
            draft_tokens=torch.full((3, 2), 3),
            corrected_logits=torch.zeros_like(probs),
            greedy_mask=torch.ones(3, dtype=torch.bool),
            temperatures=torch.zeros(3),
            draft_probs=probs,
        )
        zeros = torch.zeros(3, dtype=torch.int32)
        selected = SimpleNamespace(correct_len=zeros, bonus=zeros, cap_trim_lens=zeros)
        with (
            mock.patch.object(
                dspark_verify.AcceptGreedy, "execute", return_value=(zeros,) * 3
            ),
            mock.patch.object(
                dspark_verify.AcceptSampling, "execute", return_value=(zeros,) * 3
            ) as sampling,
            mock.patch.object(
                dspark_verify.SelectMixedAccept, "execute", return_value=selected
            ) as select,
        ):
            dspark_verify.accept_draft_tokens(
                candidates=torch.zeros(3, 3, dtype=torch.long),
                target_logits=torch.zeros(9, 5),
                draft_block=draft,
                sampling_info=info,
                draft_input=None,
                gamma=2,
                verify_num_draft_tokens=3,
            )
        self.assertIs(sampling.call_args.kwargs["draft_probs"], probs)
        torch.testing.assert_close(
            select.call_args.kwargs["greedy_mask"], torch.tensor([False, False, True])
        )


if __name__ == "__main__":
    unittest.main()
