"""Independent small-vocabulary oracles for candidate Markov proposals.

CPU tests exercise the readable reference; CUDA tests exercise the production
sampler and the actual chain rejection kernel. A CUDA skip is not GPU validation.
Run directly or with pytest. Fixed sample counts and a family-wise Bernstein
bound are chosen before observing samples; failures must not be rerun to pass.
"""

import itertools
import math
import unittest
from collections import defaultdict

import torch

from sglang.kernels.ops.speculative.dspark.dspark_markov_topk_reference import (
    markov_candidates_reference,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


def _oracle_step(base, w1, w2, prev, k, m, alpha, temperature, mapping):
    """Scalar, float64 arithmetic from original inputs, no production scores."""
    bias = [sum(float(x) * float(y) for x, y in zip(w1[prev], row)) for row in w2]
    a = sorted(range(len(base)), key=lambda v: (-float(base[v]), v))[:k]
    h = sorted(range(len(base)), key=lambda v: (-alpha * bias[v], v))[:m]
    candidates = set(a) | set(h)
    scores = {mapping[v]: float(base[v]) + alpha * bias[v] for v in candidates}
    peak = max(scores.values())
    weights = {v: math.exp((s - peak) / temperature) for v, s in scores.items()}
    total = sum(weights.values())
    return scores, {v: weight / total for v, weight in weights.items()}


def _fixture(dtype=torch.float32, device="cpu"):
    # Powers of two separate semantic failures from reduction-order rounding.
    # Vd=5, Vt=7, R=3 exercise tails and a nonidentity injective mapping.
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
        dtype=dtype,
        device=device,
    )
    w2 = torch.tensor(
        [
            [1, 1 / 16, 1 / 32],
            [1 / 8, 2, 1 / 4],
            [1 / 16, 1 / 8, 3],
            [-1, -1 / 2, -1 / 4],
            [1 / 2, 1, 3 / 2],
        ],
        dtype=dtype,
        device=device,
    )
    base = (
        torch.tensor(
            [
                [[2, 1, 0, -1, -2], [0, 2, 1, -1, -2], [-1, 0, 2, 1, -2]],
                [[-2, -1, 0, 1, 2], [2, -2, 1, 0, -1], [0, 1, -1, 2, -2]],
            ],
            dtype=dtype,
            device=device,
        )
        / 4
    )
    return base, w1, w2, [4, 1, 6, 0, 3]


def _inputs(base, mapping, anchors=(0, 4), greedy=(True, False)):
    batch, gamma, _ = base.shape
    return dict(
        anchor=torch.tensor(anchors, device=base.device, dtype=torch.int64),
        temperatures=torch.tensor([0.75, 1.5][:batch], device=base.device),
        greedy_mask=torch.tensor(greedy, device=base.device),
        d2t_offset=torch.tensor(mapping, device=base.device)
        - torch.arange(len(mapping), device=base.device),
        seeds=torch.arange(batch * gamma, device=base.device).reshape(batch, gamma)
        + 1701,
    )


def _assert_chain(tc, result, base, w1, w2, mapping, inputs, k, m, alpha):
    scores = result.corrected_logits.cpu()
    tokens = result.tokens.cpu()
    for b in range(base.shape[0]):
        prev = int(inputs["anchor"][b])
        for step in range(base.shape[1]):
            tc.assertEqual(int(result.prev_tokens[b, step]), prev)
            expected, _ = _oracle_step(
                base[b, step].cpu(),
                w1.cpu(),
                w2.cpu(),
                prev,
                k,
                m,
                alpha,
                float(inputs["temperatures"][b]),
                mapping,
            )
            actual = scores[b, step]
            tc.assertEqual(
                set(torch.isfinite(actual).nonzero().flatten().tolist()), set(expected)
            )
            for token, value in expected.items():
                tc.assertAlmostEqual(float(actual[token]), value, places=5)
            chosen = int(tokens[b, step])
            tc.assertIn(chosen, expected)
            if bool(inputs["greedy_mask"][b]):
                best = max(expected.values())
                tc.assertEqual(chosen, min(v for v, s in expected.items() if s == best))
            prev = chosen  # Never substitute a dense winner or an anchor here.


def _assert_frequencies(tc, counts, probabilities, samples, family=256):
    # Two-sided Bernstein + union bound: family-wise false rejection <= 1e-6.
    log_term = math.log(2 * family / 1e-6)
    tc.assertEqual(sum(counts.values()), samples)
    for key in set(counts) | set(probabilities):
        p = probabilities.get(key, 0.0)
        count = counts.get(key, 0)
        if p == 0:
            tc.assertEqual(count, 0, f"impossible outcome {key}")
        else:
            bound = math.sqrt(2 * p * (1 - p) * log_term / samples)
            bound += 2 * log_term / (3 * samples)
            tc.assertLessEqual(abs(count / samples - p), bound, (key, count, p))


def _enumerate_verify(q_at_prefix, p_at_prefix, gamma, supplied_q=None):
    """Enumerate proposal, accept/reject and bonus outcomes, including zeros.

    supplied_q allows a negative control: proposals still come from true q,
    while acceptance/residual consume a different q, as a stale cache would.
    """
    outcomes = defaultdict(float)

    def walk(prefix, mass):
        p = p_at_prefix(prefix)
        if len(prefix) == gamma:
            for token, prob in enumerate(p):
                outcomes[prefix + (token,)] += mass * prob
            return
        true_q = q_at_prefix(prefix)
        used_q = (supplied_q or q_at_prefix)(prefix)
        residual = [max(a - b, 0.0) for a, b in zip(p, used_q)]
        norm = sum(residual)
        for token, proposal_prob in enumerate(true_q):
            if proposal_prob == 0:
                continue
            acceptance = min(1.0, p[token] / used_q[token])
            walk(prefix + (token,), mass * proposal_prob * acceptance)
            rejected = mass * proposal_prob * (1 - acceptance)
            if rejected:
                for bonus, prob in enumerate(residual):
                    outcomes[prefix + (bonus,)] += rejected * prob / norm

    walk((), 1.0)
    return dict(outcomes)


class TestMarkovCandidateReference(unittest.TestCase):
    def test_static_budget_refuses_oversized_temporary_and_m_zero_builds_nothing(self):
        from sglang.kernels.ops.speculative.dspark.dspark_markov_topk import (
            CandidateCapacityError,
            build_static_candidates,
        )

        _, w1, w2, _ = _fixture(dtype=torch.float16)
        with self.assertRaises(CandidateCapacityError):
            build_static_candidates(w1, w2, 2, 1.0, memory_budget_bytes=1)
        ids, bias, info = build_static_candidates(
            w1,
            w2,
            0,
            1.0,
            memory_budget_bytes=0,
        )
        self.assertEqual(ids.shape, (7, 0))
        self.assertEqual(bias.shape, (7, 0))
        self.assertEqual(info["temporary_bytes"], 0)

    def test_staged_cache_slot_reuse_and_padding(self):
        from sglang.kernels.ops.speculative.dspark.dspark_markov_topk import (
            MarkovCandidateSampler,
        )

        base, w1, w2, mapping = _fixture()
        inputs = _inputs(base, mapping)
        sampler = MarkovCandidateSampler(
            w1,
            w2,
            alpha=-0.5,
            topk=2,
            bias_topk=2,
            target_vocab_size=7,
            gamma=3,
            capacity=3,
            d2t_offset=inputs["d2t_offset"],
        )
        sampler.corrected_logits[2].fill_(777)
        for iteration in range(4):
            current = base.flip(0).roll(iteration, dims=2).contiguous()
            result = sampler.sample(
                current,
                inputs["anchor"],
                inputs["temperatures"],
                inputs["greedy_mask"],
                seeds=inputs["seeds"],
            )
            _assert_chain(self, result, current, w1, w2, mapping, inputs, 2, 2, -0.5)
        inputs["anchor"][1] = -1
        result = sampler.sample(
            base,
            inputs["anchor"],
            inputs["temperatures"],
            inputs["greedy_mask"],
            seeds=inputs["seeds"],
            num_valid=torch.tensor(1, dtype=torch.int32),
        )
        self.assertEqual(result.tokens[1].tolist(), [0, 0, 0])
        self.assertTrue(torch.isfinite(result.corrected_logits[1, :, 0]).all())
        self.assertTrue(torch.isneginf(result.corrected_logits[1, :, 1:]).all())
        self.assertFalse(result.corrected_logits.softmax(-1).isnan().any())
        self.assertTrue((sampler.corrected_logits[2] == 777).all())

    def test_chain_union_scale_mapping_and_mixed_sampling(self):
        base, w1, w2, mapping = _fixture()
        inputs = _inputs(base, mapping)
        for k, m, alpha in itertools.product((1, 3, 5), (0, 2, 5), (-0.5, 0.0, 1.0)):
            with self.subTest(k=k, m=m, alpha=alpha):
                result = markov_candidates_reference(
                    base,
                    inputs["anchor"],
                    w1,
                    w2,
                    alpha=alpha,
                    topk=k,
                    bias_topk=m,
                    target_vocab_size=7,
                    **{key: value for key, value in inputs.items() if key != "anchor"},
                )
                self.assertEqual(result.corrected_logits.dtype, torch.float32)
                _assert_chain(self, result, base, w1, w2, mapping, inputs, k, m, alpha)

    def test_duplicate_union_has_one_probability_per_token(self):
        # Both top lists contain token 0. Double-counting it changes q0 from
        # exp(2)/(exp(2)+1) to 2*exp(2)/(2*exp(2)+1).
        base = torch.tensor([[[1.0, 0.0, -2.0]]])
        result = markov_candidates_reference(
            base,
            torch.tensor([0]),
            torch.tensor([[1.0]] * 3),
            torch.tensor([[1.0], [0.0], [-1.0]]),
            alpha=1.0,
            topk=2,
            bias_topk=1,
            target_vocab_size=3,
            temperatures=torch.ones(1),
            greedy_mask=torch.ones(1, dtype=torch.bool),
        )
        expected = torch.tensor(
            [math.exp(2) / (math.exp(2) + 1), 1 / (math.exp(2) + 1), 0.0]
        )
        torch.testing.assert_close(result.corrected_logits[0, 0].softmax(0), expected)

    def test_tie_break_uses_target_id_not_candidate_slot(self):
        result = markov_candidates_reference(
            torch.zeros(1, 1, 3),
            torch.tensor([0]),
            torch.zeros(4, 1),
            torch.zeros(3, 1),
            alpha=1,
            topk=3,
            bias_topk=3,
            target_vocab_size=4,
            temperatures=torch.ones(1),
            greedy_mask=torch.ones(1, dtype=torch.bool),
            d2t_offset=torch.tensor([3, -1, 0]),
        )
        self.assertEqual(result.tokens.tolist(), [[0]])

    def test_full_coverage_matches_dense_chain(self):
        base, w1, w2, mapping = _fixture()
        inputs = _inputs(base, mapping, greedy=(True, True))
        result = markov_candidates_reference(
            base,
            inputs["anchor"],
            w1,
            w2,
            alpha=0.5,
            topk=5,
            bias_topk=2,
            target_vocab_size=7,
            **{key: value for key, value in inputs.items() if key != "anchor"},
        )
        _assert_chain(self, result, base, w1, w2, mapping, inputs, 5, 2, 0.5)

    def test_rejection_enumeration_and_negative_control(self):
        p = lambda prefix: [0.1, 0.2, 0.3, 0.4]
        q = lambda prefix: [0.75, 0.25, 0.0, 0.0]
        expected = _enumerate_verify(q, p, gamma=2)
        self.assertAlmostEqual(sum(expected.values()), 1.0)
        first = [
            sum(prob for path, prob in expected.items() if path[0] == v)
            for v in range(4)
        ]
        torch.testing.assert_close(torch.tensor(first), torch.tensor(p(())))
        wrong = _enumerate_verify(q, p, 2, lambda prefix: [0.5, 0.5, 0, 0])
        wrong_first = [
            sum(prob for path, prob in wrong.items() if path[0] == v) for v in range(4)
        ]
        self.assertGreater(max(abs(x - y) for x, y in zip(first, wrong_first)), 0.05)


@unittest.skipUnless(
    torch.cuda.is_available(), "requires NVIDIA CUDA; not validated on CPU"
)
class TestMarkovCandidateCuda(unittest.TestCase):
    def _sampler(self, base, w1, w2, mapping, k, m, alpha=1, capacity=None):
        from sglang.kernels.ops.speculative.dspark.dspark_markov_topk import (
            MarkovCandidateSampler,
        )

        return MarkovCandidateSampler(
            w1,
            w2,
            alpha=alpha,
            topk=k,
            bias_topk=m,
            target_vocab_size=w1.shape[0],
            gamma=base.shape[1],
            capacity=capacity or base.shape[0],
            d2t_offset=torch.tensor(mapping, device="cuda")
            - torch.arange(len(mapping), device="cuda"),
        )

    def test_actual_kernel_chain_and_cache_match_independent_oracle(self):
        for dtype, k, m, alpha in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16),
            (1, 3, 5),
            (0, 2, 5),
            (-0.5, 0.0, 1.0),
        ):
            with self.subTest(dtype=dtype, k=k, m=m, alpha=alpha):
                base, w1, w2, mapping = _fixture(dtype, "cuda")
                inputs = _inputs(base, mapping)
                sampler = self._sampler(base, w1, w2, mapping, k, m, alpha)
                result = sampler.sample(
                    base,
                    inputs["anchor"],
                    inputs["temperatures"],
                    inputs["greedy_mask"],
                    seeds=inputs["seeds"],
                )
                _assert_chain(self, result, base, w1, w2, mapping, inputs, k, m, alpha)

    def test_cache_reorder_shrink_replace_and_separate_tiers(self):
        base, w1, w2, mapping = _fixture(device="cuda")
        samplers = [
            self._sampler(base, w1, w2, mapping, 1, 1, capacity=c) for c in (2, 4)
        ]
        for iteration in range(6):
            # Alternate physical storage buffers, reorder/replace slot contents,
            # and remove a request. Expected support scans the *entire* vocab.
            current = base.flip(0).roll(iteration, dims=2).contiguous()
            batch = 1 if iteration % 3 == 1 else 2
            current = current[:batch]
            inputs = _inputs(
                current, mapping, anchors=(6, 1)[:batch], greedy=(True, False)[:batch]
            )
            sampler = samplers[iteration % 2]
            result = sampler.sample(
                current,
                inputs["anchor"],
                inputs["temperatures"],
                inputs["greedy_mask"],
                seeds=inputs["seeds"],
            )
            _assert_chain(self, result, current, w1, w2, mapping, inputs, 1, 1, 1)

    def test_padding_canaries_degenerate_rows_and_strides(self):
        base, w1, w2, mapping = _fixture(device="cuda")
        inputs = _inputs(base, mapping)
        sampler = self._sampler(base, w1, w2, mapping, 2, 2, capacity=3)
        # Pointer-offset guards catch writes before/after the cache allocation.
        count = sampler.corrected_logits.numel()
        guarded = torch.full((count + 2,), 777.0, device="cuda")
        sampler.corrected_logits = guarded[1:-1].view_as(sampler.corrected_logits)
        sampler.corrected_logits[:2].fill_(-torch.inf)
        padded_base = torch.empty(2, 3, 7, device="cuda")
        padded_base[:, :, :5].copy_(base)
        view = padded_base[:, :, :5]
        self.assertFalse(view.is_contiguous())
        inputs["anchor"][1] = -1
        result = sampler.sample(
            view,
            inputs["anchor"],
            inputs["temperatures"],
            inputs["greedy_mask"],
            seeds=inputs["seeds"],
            num_valid=torch.tensor(1, device="cuda", dtype=torch.int32),
        )
        self.assertEqual(result.tokens[1].cpu().tolist(), [0, 0, 0])
        self.assertTrue(torch.isfinite(result.corrected_logits[1, :, 0]).all())
        self.assertTrue(torch.isneginf(result.corrected_logits[1, :, 1:]).all())
        self.assertTrue((sampler.corrected_logits[2] == 777).all())
        self.assertEqual(guarded[[0, -1]].cpu().tolist(), [777, 777])
        view[0].fill_(-torch.inf)
        result = sampler.sample(
            view,
            inputs["anchor"],
            inputs["temperatures"],
            inputs["greedy_mask"],
            seeds=inputs["seeds"],
            num_valid=torch.tensor(1, device="cuda", dtype=torch.int32),
        )
        self.assertFalse(result.corrected_logits.softmax(-1).isnan().any())
        self.assertTrue((torch.isfinite(result.corrected_logits).sum(-1) == 1).all())

    def test_rank_tiles_and_main_candidate_budget(self):
        # Rank tails beyond the first BLOCK_R, and the requested K32/M128
        # specialization. Full static coverage avoids ambiguous ranking ties.
        generator = torch.Generator(device="cuda").manual_seed(5811)
        for rank in (33, 65, 1024):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                w1 = (
                    torch.randint(
                        -2, 3, (131, rank), generator=generator, device="cuda"
                    ).to(dtype)
                    / 8
                )
                w2 = (
                    torch.randint(
                        -2, 3, (131, rank), generator=generator, device="cuda"
                    ).to(dtype)
                    / 8
                )
                # Use the small vocabulary for exact scalar oracle coverage
                # through rank=1024; the production case tests K32/M128 below.
                base = torch.arange(7, device="cuda", dtype=dtype).reshape(1, 1, 7) / 8
                sampler = self._sampler(base, w1[:7], w2[:7], list(range(7)), 3, 7)
                anchor = torch.tensor([3], device="cuda")
                temp = torch.tensor([0.75], device="cuda")
                greedy = torch.tensor([True], device="cuda")
                result = sampler.sample(base, anchor, temp, greedy)
                inputs = dict(anchor=anchor, temperatures=temp, greedy_mask=greedy)
                _assert_chain(
                    self, result, base, w1[:7], w2[:7], list(range(7)), inputs, 3, 7, 1
                )
        # The serving configuration is exercised with non-tied FP32 bases and
        # independent raw-weight reference, not kernel realized scores.
        base = torch.randn(4, 8, 131, generator=generator, device="cuda")
        w1 = torch.randn(131, 33, generator=generator, device="cuda") / 8
        w2 = torch.randn(131, 33, generator=generator, device="cuda") / 8
        sampler = self._sampler(base, w1, w2, list(range(131)), 32, 128)
        anchor = torch.arange(4, device="cuda")
        temp = torch.full((4,), 0.75, device="cuda")
        greedy = torch.ones(4, dtype=torch.bool, device="cuda")
        result = sampler.sample(base, anchor, temp, greedy)
        expected = markov_candidates_reference(
            base.cpu(),
            anchor.cpu(),
            w1.cpu(),
            w2.cpu(),
            alpha=1,
            topk=32,
            bias_topk=128,
            target_vocab_size=131,
            temperatures=temp.cpu(),
            greedy_mask=greedy.cpu(),
        )
        torch.testing.assert_close(result.tokens.cpu(), expected.tokens, rtol=0, atol=0)
        torch.testing.assert_close(
            result.corrected_logits.cpu(),
            expected.corrected_logits,
            rtol=2e-4,
            atol=2e-5,
        )

    def test_cuda_graph_updates_inputs_and_advances_rng(self):
        base, w1, w2, mapping = _fixture(device="cuda")
        inputs = _inputs(base, mapping, greedy=(False, False))
        sampler = self._sampler(base, w1, w2, mapping, 3, 2)
        kwargs = (base, inputs["anchor"], inputs["temperatures"], inputs["greedy_mask"])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sampler.sample(*kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = sampler.sample(*kwargs)
        seen = set()
        for iteration in range(16):
            inputs["anchor"].copy_(
                torch.tensor([iteration % 7, (iteration + 3) % 7], device="cuda")
            )
            inputs["temperatures"].fill_(0.75 + (iteration % 3) / 2)
            graph.replay()
            seen.add(tuple(result.tokens.flatten().cpu().tolist()))
            _assert_chain(self, result, base, w1, w2, mapping, inputs, 3, 2, 1)
        self.assertGreater(len(seen), 1)
        # Fixed input replay specifically detects capture-time frozen noise.
        seen.clear()
        for _ in range(32):
            graph.replay()
            seen.add(tuple(result.tokens.flatten().cpu().tolist()))
        self.assertGreater(len(seen), 1)

    def test_graph_and_eager_with_identical_seed_state(self):
        base, w1, w2, mapping = _fixture(device="cuda")
        inputs = _inputs(base, mapping)
        sampler = self._sampler(base, w1, w2, mapping, 3, 2)

        def run():
            return sampler.sample(
                base,
                inputs["anchor"],
                inputs["temperatures"],
                inputs["greedy_mask"],
                seeds=inputs["seeds"],
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for _ in range(3):
            inputs["seeds"].add_(101)
            eager = run()
            expected_tokens = eager.tokens.clone()
            expected_logits = eager.corrected_logits.clone()
            graph.replay()
            torch.testing.assert_close(captured.tokens, expected_tokens, rtol=0, atol=0)
            torch.testing.assert_close(
                captured.corrected_logits, expected_logits, rtol=0, atol=0
            )

    def test_candidate_permutation_keeps_token_keyed_noise(self):
        base, w1, w2, mapping = _fixture(device="cuda")
        inputs = _inputs(base, mapping, greedy=(False, False))
        sampler = self._sampler(base, w1, w2, mapping, 3, 2)
        values, ids = sampler.prepare_topk(base)
        args = (inputs["anchor"], inputs["temperatures"], inputs["greedy_mask"])
        first = sampler.sample_prepared(base, values, ids, *args, seeds=inputs["seeds"])
        expected_tokens = first.tokens.clone()
        expected_logits = first.corrected_logits.clone()
        sampler.static_ids = sampler.static_ids.flip(-1).contiguous()
        sampler.static_bias = sampler.static_bias.flip(-1).contiguous()
        second = sampler.sample_prepared(
            base,
            values.flip(-1).contiguous(),
            ids.flip(-1).contiguous(),
            *args,
            seeds=inputs["seeds"],
        )
        torch.testing.assert_close(second.tokens, expected_tokens, rtol=0, atol=0)
        torch.testing.assert_close(
            second.corrected_logits, expected_logits, rtol=0, atol=0
        )

    def test_actual_proposal_frequencies(self):
        total, batch = 131072, 1024
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base = torch.tensor([[[0.5, 0.25, 0, -1]]], device="cuda", dtype=dtype)
            base = base.repeat(batch, 1, 1)
            w1 = torch.ones(4, 1, device="cuda", dtype=dtype)
            w2 = torch.tensor([[0.25], [0], [0.5], [-1]], device="cuda", dtype=dtype)
            sampler = self._sampler(base, w1, w2, list(range(4)), 2, 2)
            anchor = torch.zeros(batch, dtype=torch.int64, device="cuda")
            temperatures = torch.full((batch,), 0.75, device="cuda")
            greedy = torch.zeros(batch, dtype=torch.bool, device="cuda")
            _, expected = _oracle_step(
                base[0, 0].cpu(), w1.cpu(), w2.cpu(), 0, 2, 2, 1, 0.75, list(range(4))
            )
            counts = torch.zeros(4, dtype=torch.int64, device="cuda")
            generator = torch.Generator(device="cuda").manual_seed(62343)
            for _ in range(total // batch):
                seeds = torch.randint(
                    0, 2**31, (batch, 1), device="cuda", generator=generator
                )
                result = sampler.sample(base, anchor, temperatures, greedy, seeds=seeds)
                counts += torch.bincount(result.tokens.flatten(), minlength=4)
            _assert_frequencies(
                self, dict(enumerate(counts.cpu().tolist())), expected, total
            )

    def test_actual_rejection_distribution_and_wrong_q_negative_control(self):
        from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        total, batch, gamma, vocab = 131072, 1024, 2, 4
        base = torch.tensor([[[0.5, 0, -1, -2], [0, 0.5, -1, -2]]], device="cuda")
        base = base.repeat(batch, 1, 1)
        w1 = torch.tensor([[1.0], [-1.0], [0.5], [0]], device="cuda")
        w2 = torch.tensor([[0.5], [0.25], [-1], [-2]], device="cuda")
        mapping = list(range(vocab))
        sampler = self._sampler(base, w1, w2, mapping, 2, 0)
        anchor = torch.zeros(batch, dtype=torch.int64, device="cuda")
        temperatures = torch.full((batch,), 0.75, device="cuda")
        greedy = torch.zeros(batch, dtype=torch.bool, device="cuda")
        p = lambda prefix: [0.1, 0.2, 0.3, 0.4]

        def q(prefix):
            _, probs = _oracle_step(
                base[0, len(prefix)].cpu(),
                w1.cpu(),
                w2.cpu(),
                prefix[-1] if prefix else 0,
                2,
                0,
                1,
                0.75,
                mapping,
            )
            return [probs.get(v, 0) for v in range(vocab)]

        expected = _enumerate_verify(q, p, gamma)
        counts, wrong_counts = defaultdict(int), defaultdict(int)
        # Proposal, acceptance, and final/residual sampling use independent RNGs.
        generators = [
            torch.Generator(device="cuda").manual_seed(seed)
            for seed in (5919, 12315, 6717)
        ]
        indices = torch.arange(batch * (gamma + 1), device="cuda", dtype=torch.int32)
        indices = indices.reshape(batch, gamma + 1)
        target = torch.tensor(p(()), device="cuda").expand(batch, gamma + 1, vocab)
        for _ in range(total // batch):
            result = sampler.sample(
                base,
                anchor,
                temperatures,
                greedy,
                seeds=torch.randint(
                    0, 2**31, (batch, gamma), device="cuda", generator=generators[0]
                ),
            )
            candidates = torch.cat((anchor[:, None], result.tokens), dim=1)
            probs = SoftmaxTemp.execute(
                logits=result.corrected_logits.reshape(batch * gamma, vocab),
                temperatures=temperatures,
                rows_per_request=gamma,
            ).reshape(batch, gamma, vocab)
            coin = torch.rand(batch, gamma, device="cuda", generator=generators[1])
            final_coin = torch.rand(batch, device="cuda", generator=generators[2])
            for wrong, tally in ((False, counts), (True, wrong_counts)):
                # Corrupt q by duplicating token 0's probability and normalizing.
                supplied = probs.clone()
                if wrong:
                    supplied[:, :, 0] *= 2
                    supplied /= supplied.sum(-1, keepdim=True)
                predicts = torch.empty(
                    batch * (gamma + 1), device="cuda", dtype=torch.int32
                )
                accept_index = torch.empty_like(indices)
                accepted = torch.empty(batch, device="cuda", dtype=torch.int32)
                chain_speculative_sampling_triton(
                    predicts,
                    accept_index,
                    accepted,
                    candidates,
                    indices,
                    indices,
                    indices,
                    coin,
                    final_coin,
                    target,
                    supplied,
                    1.0,
                    1.0,
                    True,
                )
                out = predicts.reshape(batch, gamma + 1).cpu().tolist()
                for row, length in zip(out, accepted.cpu().tolist()):
                    tally[tuple(row[: length + 1])] += 1
        _assert_frequencies(self, counts, expected, total)
        with self.assertRaises(AssertionError):
            _assert_frequencies(self, wrong_counts, expected, total)


if __name__ == "__main__":
    unittest.main()
