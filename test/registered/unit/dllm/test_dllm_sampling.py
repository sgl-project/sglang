"""Unit tests for srt/dllm/sampling.py -- no server, no model loading.

Synthetic logits stand in for a model so every step count and token is
reproducible. GPU output cannot serve as the oracle here: on a real dLLM server
the same greedy prompt decodes differently at batch size 1 and 8, so
"a greedy row is unaffected by a sampled neighbour" is only checkable on CPU.
"""

import unittest

import torch

from sglang.srt.dllm import sampling as dllm_sampling
from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.sampling import DllmSamplingPlan
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BLOCK_SIZE = 8
VOCAB_SIZE = 64
MASK_ID = 63
GREEDY_TOP_K = 1
# The sentinel SamplingParams.normalize writes for an unset top_k.
NO_TOP_K = 1 << 30


class _FakeForwardBatch:
    def __init__(self, input_ids, batch_size, sampling_info):
        self.input_ids = input_ids
        self.batch_size = batch_size
        self.sampling_info = sampling_info


def _sampling_info(top_ks, temperatures, top_ps, min_ps=None, device="cpu"):
    min_ps = min_ps or [0.0] * len(top_ks)
    return SamplingBatchInfo(
        temperatures=torch.tensor(temperatures, dtype=torch.float, device=device).view(
            -1, 1
        ),
        top_ps=torch.tensor(top_ps, dtype=torch.float, device=device),
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device=device),
        min_ps=torch.tensor(min_ps, dtype=torch.float, device=device),
        is_all_greedy=all(k <= 1 for k in top_ks),
        is_any_greedy=any(k <= 1 for k in top_ks),
        need_top_p_sampling=any(p != 1.0 for p in top_ps),
        need_top_k_sampling=any(k != NO_TOP_K for k in top_ks),
        need_min_p_sampling=any(p > 0 for p in min_ps),
        vocab_size=VOCAB_SIZE,
    )


def _algorithm(name, **algorithm_config):
    config = DllmConfig(
        algorithm=name,
        algorithm_config=algorithm_config,
        block_size=BLOCK_SIZE,
        mask_id=MASK_ID,
        max_running_requests=8,
        first_done_first_out_mode=False,
    )
    return {"JointThreshold": JointThreshold, "LowConfidence": LowConfidence}[name](
        config
    )


def _denoise_block(algorithm, batch_size, sampling_info, seed=0):
    """Denoise one all-masked block to completion; returns (tokens, num_steps)."""
    torch.manual_seed(seed)
    # The same logits are replayed every step, so the model holds one fixed
    # opinion and any extra step is attributable to the algorithm, not to drift.
    logits = (
        torch.randn(
            batch_size * BLOCK_SIZE,
            VOCAB_SIZE,
            generator=torch.Generator().manual_seed(1234),
        )
        * 3.0
    )
    logits[:, MASK_ID] = -1e4

    input_ids = torch.full((batch_size * BLOCK_SIZE,), MASK_ID, dtype=torch.int64)
    forward_batch = _FakeForwardBatch(input_ids, batch_size, sampling_info)
    states = algorithm.init_step_state(forward_batch)

    steps = 0
    for _ in range(algorithm.max_steps(BLOCK_SIZE)):
        steps += 1
        if all(algorithm.step(forward_batch, logits.clone(), states)):
            break
    return input_ids.view(batch_size, BLOCK_SIZE).tolist(), steps


# (label, algorithm name, extra algorithm_config) for every decision path that
# reads sampling_info: JointThreshold ships per-row on CUDA and vectorized on NPU.
DECODE_PATHS = [
    ("JointThreshold/per-row", "JointThreshold", {}),
    ("JointThreshold/vectorized", "JointThreshold", {"vectorized_decoding": True}),
    ("LowConfidence", "LowConfidence", {}),
]


class TestDllmSampling(CustomTestCase):
    def test_greedy_batch_matches_no_sampling_info(self):
        """An all-greedy batch must not take the sampling path at all."""
        for label, name, config in DECODE_PATHS:
            with self.subTest(label):
                algorithm = _algorithm(name, **config)
                baseline = _denoise_block(algorithm, 2, None)
                greedy = _denoise_block(
                    algorithm,
                    2,
                    _sampling_info([GREEDY_TOP_K] * 2, [1.0] * 2, [1.0] * 2),
                )
                self.assertEqual(greedy, baseline)

    def test_top_k_one_stays_greedy_at_high_temperature(self):
        """temperature alone cannot make a row sample.

        SamplingParams.normalize rewrites temperature == 0 to (1.0, top_k=1), so
        top_k <= 1 -- not temperature -- is the greedy predicate. A row asking
        for temperature=2.0 with top_k=1 has exactly one candidate and must stay
        bitwise greedy however hot the distribution is made.
        """
        for label, name, config in DECODE_PATHS:
            with self.subTest(label):
                algorithm = _algorithm(name, **config)
                greedy = _denoise_block(
                    algorithm,
                    2,
                    _sampling_info([GREEDY_TOP_K] * 2, [1.0] * 2, [1.0] * 2),
                )
                hot = _denoise_block(
                    algorithm,
                    2,
                    _sampling_info([GREEDY_TOP_K] * 2, [2.0] * 2, [1.0] * 2),
                )
                self.assertEqual(hot, greedy)

    def test_greedy_row_isolated_from_sampled_neighbour(self):
        """Row 0 greedy, row 1 sampled: row 0 must match an all-greedy batch.

        The sampling path gathers, samples and scatters back only the non-greedy
        rows, so an off-by-one in that gather/scatter -- or a request index that
        indexes the sampling params by position within the gathered subset
        instead of by request -- silently corrupts the greedy neighbour.
        """
        for label, name, config in DECODE_PATHS:
            with self.subTest(label):
                algorithm = _algorithm(name, **config)
                greedy, _ = _denoise_block(
                    algorithm,
                    2,
                    _sampling_info([GREEDY_TOP_K] * 2, [1.0] * 2, [1.0] * 2),
                )
                mixed, _ = _denoise_block(
                    algorithm,
                    2,
                    _sampling_info([GREEDY_TOP_K, NO_TOP_K], [1.0, 1.5], [1.0, 1.0]),
                )
                self.assertEqual(mixed[0], greedy[0])
                self.assertNotEqual(mixed[1], greedy[1])

    def test_sampled_tokens_are_not_reverted_to_argmax(self):
        """Regression: the drawn token must survive to the end of the block.

        JointThreshold's token-to-token pass rewrites any committed position whose
        token differs from this step's prediction. Left unguarded it overwrites
        every sampled token with the argmax one on the next forward, so sampling
        converges back to greedy output and only the forward count changes. The
        guard must also not be a hill climb ("edit only towards a higher-scoring
        token"), which reverts just as surely, one position at a time.
        """
        for label, name, config in DECODE_PATHS:
            with self.subTest(label):
                algorithm = _algorithm(name, **config)
                greedy, _ = _denoise_block(
                    algorithm,
                    2,
                    _sampling_info([GREEDY_TOP_K] * 2, [1.0] * 2, [1.0] * 2),
                )
                sampled, _ = _denoise_block(
                    algorithm, 2, _sampling_info([NO_TOP_K] * 2, [1.5] * 2, [1.0] * 2)
                )
                self.assertNotEqual(sampled, greedy)

    def test_sampled_block_terminates_independently_of_the_edit_budget(self):
        """Regression: a sampled block must still reach a fixed point.

        With edit_threshold=0 the token-to-token predicate reduces to "the
        prediction differs from the committed token", which redrawing every step
        makes permanently true. The block then has no fixed point and only ever
        stops by running out of max_post_edit_steps, so raising that budget buys
        more forwards and nothing else -- which is what this compares.
        """
        sampling_info = _sampling_info([NO_TOP_K] * 2, [1.5] * 2, [0.9] * 2)
        default_budget = _denoise_block(
            _algorithm("JointThreshold", max_post_edit_steps=16), 2, sampling_info
        )
        large_budget = _denoise_block(
            _algorithm("JointThreshold", max_post_edit_steps=64), 2, sampling_info
        )
        self.assertEqual(large_budget, default_budget)


class TestDllmSamplingPlan(CustomTestCase):
    def test_relative_confidence_semantics(self):
        """``exp((z_t - z_max) / T)`` is what lets one threshold serve both paths.

        It is 1 for a draw that lands on the argmax token, so the committed
        confidence ``min(p_argmax, r)`` degenerates to exactly greedy's own; and
        it rises with temperature, so a hotter request defers fewer of its draws.
        An absolute probability has neither property.
        """
        logits = torch.randn(4, VOCAB_SIZE, generator=torch.Generator().manual_seed(7))
        req_ids = torch.zeros(4, dtype=torch.int64)
        argmax_ids = logits.argmax(dim=-1)
        runner_up = logits.topk(2, dim=-1).indices[:, 1]

        cold = DllmSamplingPlan.maybe_build(_sampling_info([NO_TOP_K], [1.0], [1.0]))
        hot = DllmSamplingPlan.maybe_build(_sampling_info([NO_TOP_K], [2.0], [1.0]))

        torch.testing.assert_close(
            cold.confidence(logits=logits, req_ids=req_ids, token_ids=argmax_ids),
            torch.ones(4),
        )
        self.assertTrue(
            torch.all(
                hot.confidence(logits=logits, req_ids=req_ids, token_ids=runner_up)
                > cold.confidence(logits=logits, req_ids=req_ids, token_ids=runner_up)
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "flashinfer kernels need CUDA")
    def test_min_p_neighbour_keeps_joint_top_k_top_p_support(self):
        """A min-p row in the batch must not change another row's filtering.

        With top_k=2, top_p=0.5 over [.4, .3, .2, .1], joint filtering keeps
        tokens 0 and 1 (token 1's prefix mass .4 is under .5). Applying top-p
        after top-k renormalization instead sees a prefix of .57 and drops
        token 1.
        """
        num_draws = 4096
        probs = torch.zeros(num_draws, VOCAB_SIZE, device="cuda")
        probs[:, :4] = torch.tensor([0.4, 0.3, 0.2, 0.1])
        # Every draw comes from request 0; request 1 only sets need_min_p_sampling.
        req_ids = torch.zeros(num_draws, dtype=torch.int64, device="cuda")
        for min_ps in ([0.0], [0.0, 0.9]):
            num_reqs = len(min_ps)
            sampling_info = _sampling_info(
                [2] * num_reqs, [1.0] * num_reqs, [0.5] * num_reqs, min_ps, "cuda"
            )
            tokens = dllm_sampling._flashinfer_sample(
                probs=probs, sampling_info=sampling_info, req_ids=req_ids
            )
            counts = torch.bincount(tokens.long(), minlength=VOCAB_SIZE).tolist()
            # Support is {0, 1} with token 1 at 3/7, ~1755 draws.
            self.assertEqual(counts[0] + counts[1], num_draws, f"{min_ps=}")
            self.assertGreater(counts[1], num_draws // 4, f"{min_ps=}")


if __name__ == "__main__":
    unittest.main()
