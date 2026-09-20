"""Exercise the real DSpark eager/graph sampling wrappers with a tiny model.

Only the backbone/LM-head forward is replaced with known input logits. Markov
heads, capability selection, parameter staging, graph sampling and eager
sampling are production implementations. No checkpoint or server claim follows
from these integration tests.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")


def _model(device, sample_from_anchor=True, gated=False):
    from sglang.srt.models.dspark import GatedMarkovHead, VanillaMarkov

    args = dict(vocab_size=7, draft_vocab_size=5, markov_rank=3, logit_scale=1)
    head = GatedMarkovHead(**args, hidden_size=5) if gated else VanillaMarkov(**args)
    head = head.to(device)
    w1 = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1, 1, 0],
        ],
        device=device,
        dtype=torch.float32,
    )
    w2 = torch.tensor(
        [
            [1, 1 / 16, 1 / 32],
            [1 / 8, 2, 1 / 4],
            [1 / 16, 1 / 8, 3],
            [-1, -1 / 2, -1 / 4],
            [1 / 2, 1, 3 / 2],
        ],
        device=device,
    )
    mapping = torch.tensor([4, 1, 6, 0, 3], device=device)
    offsets = mapping - torch.arange(5, device=device)
    with torch.no_grad():
        head.markov_w1.weight.copy_(w1)
        head.markov_w2.weight.copy_(w2)
        if gated:
            head.gate_proj.weight.zero_()
            head.gate_proj.bias.zero_()
    head.configure_target_vocab(7, offsets)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    config = SimpleNamespace(
        vocab_size=7,
        draft_vocab_size=5,
        markov_rank=3,
        markov_head_type="gated" if gated else "vanilla",
        logit_scale=1,
        sample_from_anchor=sample_from_anchor,
    )
    model = SimpleNamespace(
        markov_head=head,
        config=config,
        sample_from_anchor=sample_from_anchor,
        target_vocab_size=7,
        draft_vocab_size=5,
        logit_scale=1,
        d2t=offsets,
        draft_id_to_target_id=offsets,
        lm_head=SimpleNamespace(
            weight=torch.zeros(5, 5, device=device), org_vocab_size=5
        ),
        compute_base_logits=lambda hidden: (hidden, None),
    )
    return model


def _base(device):
    return (
        torch.tensor(
            [
                [[2, 1, 0, -1, -2], [0, 2, 1, -1, -2], [-1, 0, 2, 1, -2]],
                [[-2, -1, 0, 1, 2], [2, -2, 1, 0, -1], [0, 1, -1, 2, -2]],
            ],
            dtype=torch.float32,
            device=device,
        )
        / 4
    )


def _sync():
    from sglang.srt.speculative.spec_tp_sync import SpecTpSync

    return SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0))


def _assert_scores(tc, tokens, logits, base, anchor, model, k, m, greedy):
    # Re-derive supports and raw scores, independently of optimized candidates.
    w1 = model.markov_head.markov_w1.weight.detach().cpu()
    w2 = model.markov_head.markov_w2.weight.detach().cpu()
    mapping = (torch.arange(5) + model.d2t.cpu()).tolist()
    alpha = 0.5 if model.config.markov_head_type == "gated" else 1.0
    for row in range(base.shape[0]):
        prev = int(anchor[row])
        for step in range(base.shape[1]):
            raw = base[row, step].cpu().tolist()
            bias = [
                sum(float(x) * float(y) for x, y in zip(w1[prev], weight))
                for weight in w2
            ]
            a = sorted(range(5), key=lambda v: (-raw[v], v))[:k]
            h = sorted(range(5), key=lambda v: (-alpha * bias[v], v))[:m]
            expected = {mapping[v]: raw[v] + alpha * bias[v] for v in set(a) | set(h)}
            actual = logits[row, step].cpu()
            tc.assertEqual(
                set(torch.isfinite(actual).nonzero().flatten().tolist()), set(expected)
            )
            for token, score in expected.items():
                tc.assertAlmostEqual(float(actual[token]), score, places=5)
            chosen = int(tokens[row, step])
            tc.assertIn(chosen, expected)
            if bool(greedy[row]):
                peak = max(expected.values())
                tc.assertEqual(
                    chosen, min(v for v, score in expected.items() if score == peak)
                )
            prev = chosen


class TestDsparkCandidateFallback(unittest.TestCase):
    def test_disabled_cpu_tp_and_gated_use_functional_dense_path(self):
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            sample_draft_block,
        )
        from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
            initialize_markov_candidate_sampler,
        )

        base = _base("cpu")
        anchor = torch.tensor([0, 4])
        for k, tp, gated in ((0, 1, False), (3, 1, False), (3, 2, False), (3, 1, True)):
            with self.subTest(k=k, tp=tp, gated=gated):
                model = _model("cpu", gated=gated)
                candidate = initialize_markov_candidate_sampler(
                    model=model,
                    draft_hf_config=model.config,
                    gamma=3,
                    capacity=2,
                    tp_size=tp,
                    markov_topk=k,
                    markov_bias_topk=2,
                )
                self.assertIsNone(candidate)
                self.assertIsNone(model.markov_candidate_sampler)
                result = sample_draft_block(
                    base_logits=base,
                    anchor_tokens=anchor,
                    draft_hidden=base,
                    sampling_info=None,
                    markov_head=model.markov_head,
                    device=torch.device("cpu"),
                    tp_sync=_sync(),
                    candidate_sampler=candidate,
                )
                _assert_scores(
                    self,
                    result.draft_tokens,
                    result.corrected_logits,
                    base,
                    anchor,
                    model,
                    5,
                    0,
                    torch.ones(2, dtype=torch.bool),
                )


@unittest.skipUnless(
    torch.cuda.is_available(), "requires NVIDIA CUDA runtime graph validation"
)
class TestDsparkCandidateRuntimeCuda(unittest.TestCase):
    def test_real_wrapper_graph_stages_inputs_and_eager_cache_contract(self):
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            sample_draft_block,
        )
        from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
            DsparkDraftSampler,
            initialize_markov_candidate_sampler,
        )

        for from_anchor in (True, False):
            for folded in (True, False):
                with self.subTest(sample_from_anchor=from_anchor, folded=folded):
                    model = _model("cuda", sample_from_anchor=from_anchor)
                    candidate = initialize_markov_candidate_sampler(
                        model=model,
                        draft_hf_config=model.config,
                        gamma=3,
                        capacity=2,
                        tp_size=1,
                        markov_topk=3,
                        markov_bias_topk=2,
                    )
                    self.assertIsNotNone(candidate)
                    self.assertEqual(candidate.path, "triton")

                    # The callback encodes the actual predecessor chain, making
                    # an incorrect confidence input visible without claiming
                    # confidence calibration has been evaluated.
                    def confidence_fn(
                        *, draft_hidden, anchor_tokens, draft_tokens, confidence_tap
                    ):
                        return torch.cat(
                            (anchor_tokens[:, None], draft_tokens[:, :-1]), dim=1
                        ).float()

                    wrapper = DsparkDraftSampler(
                        model=model,
                        gamma=3,
                        max_bs=2,
                        device="cuda",
                        tp_sync=_sync(),
                        folded_sampling=folded,
                        confidence_fn=confidence_fn,
                    )
                    self.assertIsNone(wrapper.exp_noise)
                    self.assertEqual(wrapper.corrected_out.dtype, torch.float32)
                    self.assertEqual(
                        wrapper.corrected_out.data_ptr(),
                        candidate.corrected_logits.data_ptr(),
                    )
                    query = 3 if from_anchor else 4
                    hidden = torch.zeros(2, query, 5, device="cuda")
                    input_ids = torch.zeros(2, query, device="cuda", dtype=torch.int64)
                    base = _base("cuda")

                    def stage(iteration, valid):
                        current = base.roll(iteration, dims=2)
                        hidden[:, -3:].copy_(current)
                        if not from_anchor:
                            hidden[:, 0].fill_(1234)  # Must never become base step 0.
                        anchors = torch.tensor(
                            [iteration % 7, (iteration + 4) % 7], device="cuda"
                        )
                        if valid == 1:
                            anchors[1] = -1  # Must be masked before any W1 access.
                        input_ids[:, 0].copy_(anchors)
                        request_topks = (
                            ([1, -1] if iteration % 2 == 0 else [-1, 1])
                            if folded
                            else [1, 1]
                        )
                        requests = [
                            SamplingParams(temperature=t, top_k=k, top_p=1.0)
                            for t, k in zip((0.75 + iteration / 4, 1.5), request_topks)
                        ]
                        temperatures = torch.tensor(
                            [r.temperature for r in requests[:valid]], device="cuda"
                        )
                        # Request top_k=-1 is normalized by SamplingParams to
                        # TOP_K_ALL before the real runtime sees top_ks. Passing
                        # raw -1 here would accidentally exercise greedy rows.
                        topks = torch.tensor(
                            [r.top_k for r in requests[:valid]], device="cuda"
                        )
                        info = SimpleNamespace(
                            temperatures=temperatures[:, None],
                            top_ks=topks,
                            is_all_greedy=not folded,
                        )
                        wrapper.stage_sampling_params(bs=valid, sampling_info=info)
                        return current[:valid], anchors[:valid], info

                    stage(0, 2)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            wrapper(hidden.reshape(-1, 5), input_ids.flatten())
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        wrapper(hidden.reshape(-1, 5), input_ids.flatten())
                    for iteration, valid in enumerate((2, 1, 2, 1, 2)):
                        current, anchor, info = stage(iteration, valid)
                        graph.replay()
                        tokens = wrapper.out.view(2, 3)[:valid]
                        cache = wrapper.corrected_out.view(2, 3, 7)[:valid]
                        _assert_scores(
                            self,
                            tokens,
                            cache,
                            current,
                            anchor,
                            model,
                            3,
                            2,
                            info.top_ks <= 1,
                        )
                        expected_predecessors = torch.cat(
                            (anchor[:, None], tokens[:, :-1]), dim=1
                        )
                        torch.testing.assert_close(
                            wrapper.confidence_out[:valid],
                            expected_predecessors.float(),
                        )
                        if valid < 2:
                            self.assertFalse(
                                wrapper.corrected_out.softmax(-1).isnan().any()
                            )
                        # Consume graph q before calling eager on the same
                        # physical cache. The following graph replay must clear
                        # columns that the intervening eager call published.
                        result = sample_draft_block(
                            base_logits=current,
                            anchor_tokens=anchor,
                            draft_hidden=current,
                            sampling_info=info,
                            markov_head=model.markov_head,
                            device=torch.device("cuda"),
                            tp_sync=_sync(),
                            candidate_sampler=candidate,
                        )
                        _assert_scores(
                            self,
                            result.draft_tokens,
                            result.corrected_logits,
                            current,
                            anchor,
                            model,
                            3,
                            2,
                            info.top_ks <= 1,
                        )
                        self.assertEqual(
                            result.corrected_logits.data_ptr(),
                            candidate.corrected_logits.data_ptr(),
                        )
                    if folded:
                        observed = set()
                        for _ in range(32):
                            graph.replay()
                            observed.add(
                                tuple(wrapper.out.view(2, 3)[1].cpu().tolist())
                            )
                        self.assertGreater(
                            len(observed),
                            1,
                            "sampling row stayed frozen or became greedy",
                        )

    def test_cuda_unsupported_dispatch_keeps_dense_sampling_working(self):
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            sample_draft_block,
        )
        from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
            initialize_markov_candidate_sampler,
        )

        base = _base("cuda")
        anchor = torch.tensor([0, 4], device="cuda")
        for k, tp, gated in ((0, 1, False), (3, 2, False), (3, 1, True)):
            model = _model("cuda", gated=gated)
            candidate = initialize_markov_candidate_sampler(
                model=model,
                draft_hf_config=model.config,
                gamma=3,
                capacity=2,
                tp_size=tp,
                markov_topk=k,
                markov_bias_topk=2,
            )
            self.assertIsNone(candidate)
            result = sample_draft_block(
                base_logits=base,
                anchor_tokens=anchor,
                draft_hidden=base,
                sampling_info=None,
                markov_head=model.markov_head,
                device=torch.device("cuda"),
                tp_sync=_sync(),
                candidate_sampler=candidate,
            )
            _assert_scores(
                self,
                result.draft_tokens,
                result.corrected_logits,
                base,
                anchor,
                model,
                5,
                0,
                torch.ones(2, dtype=torch.bool),
            )
        # TP=2 here tests dispatch only. It does not claim a multi-process
        # collective or TP model execution test.


if __name__ == "__main__":
    unittest.main()
