import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.models.dspark import run_markov_block
from sglang.srt.sampling.draft_sampling import DraftSamplingParams, build_draft_probs
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
    _resolve_folded_sampling,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

# (temperature, top_k, top_p) draft overrides: inherit, sharpen, greedy draft.
_OVERRIDES = ((None, None, None), (0.4, 2, 0.5), (0.0, -1, 1.0))


def _sampling_info(device="cpu"):
    return SimpleNamespace(
        temperatures=torch.tensor([[0.7], [1.5], [1.0]], device=device),
        top_ks=torch.tensor([2, 3, 1], dtype=torch.int32, device=device),
        top_ps=torch.tensor([0.8, 0.9, 1.0], device=device),
        is_all_greedy=False,
        is_any_greedy=True,
    )


class _MarkovHead:
    def apply_step_logits(self, logits, *, token_ids, hidden_states):
        # Promote a token outside the base top-2. Its identity depends on the
        # sampled prefix, so q built before the correction cannot match.
        correction = torch.zeros_like(logits)
        correction.scatter_(1, ((token_ids + 3) % logits.shape[-1])[:, None], 4.0)
        return logits + correction

    def sample_block(self, base_logits, **kwargs):
        return run_markov_block(self, base_logits, **kwargs)


def _tp_sync():
    return SimpleNamespace(sync=lambda _site, values: values)


def _base_logits(device="cpu"):
    return torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0], device=device).repeat(3, 2, 1)


def _folded_sampler(device):
    model = SimpleNamespace(
        sample_from_anchor=True,
        markov_head=_MarkovHead(),
        lm_head=SimpleNamespace(org_vocab_size=5, weight=torch.empty(1)),
        compute_base_logits=lambda hidden: (hidden, None),
    )
    return DsparkDraftSampler(
        model=model, gamma=2, max_bs=3, device=device, tp_sync=_tp_sync()
    )


class TestDsparkDraftSampling(unittest.TestCase):
    def setUp(self):
        self.spec = SimpleNamespace()
        patcher = mock.patch(
            "sglang.srt.runtime_context.get_spec", return_value=self.spec
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _set_overrides(self, temperature, top_k, top_p):
        self.spec.speculative_draft_temperature = temperature
        self.spec.speculative_draft_top_k = top_k
        self.spec.speculative_draft_top_p = top_p

    def _check_block(self, *, tokens, corrected_logits, probs, info):
        params = DraftSamplingParams.from_sampling_info(info)
        for step in range(tokens.shape[1]):
            torch.testing.assert_close(
                probs[:, step], build_draft_probs(corrected_logits[:, step], params)
            )
        self.assertTrue(bool((probs.gather(-1, tokens[..., None]) > 0).all()))
        # Token 3 enters request 0's top-2 only after the Markov correction.
        self.assertGreater(float(probs[0, 0, 3]), 0)

    def test_eager_q_follows_the_markov_correction(self):
        info = _sampling_info()
        for fast in (False, True):
            for overrides in _OVERRIDES:
                self._set_overrides(*overrides)
                with (
                    self.subTest(fast=fast, overrides=overrides),
                    envs.SGLANG_DSPARK_FAST_SAMPLING.override(fast),
                ):
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
                        tokens=result.draft_tokens,
                        corrected_logits=result.corrected_logits,
                        probs=result.draft_probs,
                        info=info,
                    )
                    # greedy_mask picks the accept rule, so it stays the
                    # target's even when the draft itself is greedy.
                    self.assertEqual(result.greedy_mask.tolist(), [False, False, True])

    def test_folded_sampler_retains_q(self):
        sampler, info = _folded_sampler("cpu"), _sampling_info()
        for overrides in _OVERRIDES:
            self._set_overrides(*overrides)
            sampler.stage_sampling_params(bs=3, sampling_info=info)
            sampler(_base_logits().reshape(6, 5), torch.zeros(6, dtype=torch.long))
            self._check_block(
                tokens=sampler.out.view(3, 2),
                corrected_logits=sampler.corrected_out.view(3, 2, 5),
                probs=sampler.probs_out.view(3, 2, 5),
                info=info,
            )
            self.assertEqual(sampler.greedy_mask.tolist(), [False, False, True])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_folded_cuda_graph_replays_changed_overrides(self):
        sampler, info = _folded_sampler("cuda"), _sampling_info("cuda")
        hidden = _base_logits("cuda").reshape(6, 5)
        input_ids = torch.zeros(6, dtype=torch.long, device="cuda")
        self._set_overrides(*_OVERRIDES[0])
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
        for overrides in _OVERRIDES:
            self._set_overrides(*overrides)
            sampler.stage_sampling_params(bs=3, sampling_info=info)
            graph.replay()
            self._check_block(
                tokens=sampler.out.view(3, 2),
                corrected_logits=sampler.corrected_out.view(3, 2, 5),
                probs=sampler.probs_out.view(3, 2, 5),
                info=info,
            )

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

    def test_sampling_verification_uses_saved_q(self):
        # Rebuilding q from corrected logits would drop the draft cutoffs.
        probs = torch.zeros(3, 2, 5)
        probs[:, :, 3] = 1.0
        draft = DraftBlockResult(
            draft_tokens=torch.full((3, 2), 3),
            corrected_logits=torch.zeros_like(probs),
            greedy_mask=torch.tensor([False, False, True]),
            temperatures=torch.ones(3),
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
            ),
        ):
            dspark_verify.accept_draft_tokens(
                candidates=torch.zeros(3, 3, dtype=torch.long),
                target_logits=torch.zeros(9, 5),
                draft_block=draft,
                sampling_info=_sampling_info(),
                draft_input=None,
                gamma=2,
                verify_num_draft_tokens=3,
            )
        self.assertIs(sampling.call_args.kwargs["draft_probs"], probs)


if __name__ == "__main__":
    unittest.main()
