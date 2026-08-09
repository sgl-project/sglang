from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.test_utils import CustomTestCase

VOCAB = 8
NUM_DRAFT = 6
BATCH = 8192
REPEATS = 4
# Deliberately skewed so a greedy verifier has an unambiguous argmax to collapse to.
TARGET_ROW = torch.tensor([0.35, 0.25, 0.20, 0.10, 0.05, 0.03, 0.015, 0.005])


class TestSpecSamplingDispatch(CustomTestCase):
    def test_hip_target_only_uses_triton_tree(self):
        from sglang.kernels.ops.speculative.tree_sampling import (
            tree_speculative_sampling_target_only_triton,
        )
        from sglang.srt.speculative import eagle_utils

        with patch.object(eagle_utils, "_is_hip", True):
            sampling_fn = eagle_utils._get_spec_sampling_verify_fn(False)
        self.assertIs(sampling_fn, tree_speculative_sampling_target_only_triton)

    def test_rejection_sampling_uses_triton_chain(self):
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )
        from sglang.srt.speculative import eagle_utils

        sampling_fn = eagle_utils._get_spec_sampling_verify_fn(True)
        self.assertIs(sampling_fn, chain_speculative_sampling_triton)


def _chain_topology(batch_size: int, num_draft: int, device: torch.device):
    """Linear chain, i.e. `--speculative-eagle-topk 1`: slot i's child is slot i+1."""
    retrive_index = torch.arange(
        batch_size * num_draft, dtype=torch.int32, device=device
    ).reshape(batch_size, num_draft)
    retrive_next_token = torch.full(
        (batch_size, num_draft), -1, dtype=torch.int32, device=device
    )
    retrive_next_token[:, :-1] = torch.arange(
        1, num_draft, dtype=torch.int32, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_draft), -1, dtype=torch.int32, device=device
    )
    return retrive_index, retrive_next_token, retrive_next_sibling


def _make_inputs(device: torch.device):
    """Same target distribution at every draft position, draft proposing a different one.

    With one shared target distribution the aggregate distribution of emitted
    tokens is directly comparable to it, whatever the accept pattern is.
    """
    target_row = TARGET_ROW.to(device)
    draft_row = torch.flip(target_row, dims=[0])
    target_probs = (
        target_row.view(1, 1, -1).expand(BATCH, NUM_DRAFT, VOCAB).contiguous()
    )
    draft_probs = draft_row.view(1, 1, -1).expand(BATCH, NUM_DRAFT, VOCAB).contiguous()
    candidates = (
        torch.multinomial(draft_row.expand(BATCH * NUM_DRAFT, VOCAB), 1)
        .reshape(BATCH, NUM_DRAFT)
        .to(torch.int32)
    )
    return target_probs, draft_probs, candidates


def _empty_outputs(device: torch.device):
    predicts = torch.full((BATCH * NUM_DRAFT,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((BATCH, NUM_DRAFT), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((BATCH,), dtype=torch.int32, device=device)
    return predicts, accept_index, accept_token_num


@unittest.skipUnless(torch.cuda.is_available(), "GPU is required for this test.")
class TestSpecVerifyDistribution(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def _emitted_histogram(self, verifier: str) -> torch.Tensor:
        histogram = torch.zeros(VOCAB, dtype=torch.float64, device=self.device)
        for _ in range(REPEATS):
            target_probs, draft_probs, candidates = _make_inputs(self.device)
            retrive_index, retrive_next_token, retrive_next_sibling = _chain_topology(
                BATCH, NUM_DRAFT, self.device
            )
            predicts, accept_index, accept_token_num = _empty_outputs(self.device)

            if verifier == "greedy":
                from sgl_kernel import verify_tree_greedy

                verify_tree_greedy(
                    predicts=predicts,
                    accept_index=accept_index,
                    accept_token_num=accept_token_num,
                    candidates=candidates.long(),
                    retrive_index=retrive_index.long(),
                    retrive_next_token=retrive_next_token.long(),
                    retrive_next_sibling=retrive_next_sibling.long(),
                    target_predict=target_probs.argmax(dim=-1),
                )
            else:
                chain_speculative_sampling_triton(
                    predicts=predicts,
                    accept_index=accept_index,
                    accept_token_num=accept_token_num,
                    candidates=candidates,
                    retrive_index=retrive_index,
                    retrive_next_token=retrive_next_token,
                    retrive_next_sibling=retrive_next_sibling,
                    uniform_samples=torch.rand_like(candidates, dtype=torch.float32),
                    uniform_samples_for_final_sampling=torch.rand(
                        (BATCH,), dtype=torch.float32, device=self.device
                    ),
                    target_probs=target_probs,
                    draft_probs=draft_probs,
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    deterministic=True,
                )
            torch.cuda.synchronize()

            emitted = predicts[predicts >= 0].long()
            histogram += torch.bincount(emitted, minlength=VOCAB).to(torch.float64)
        return histogram / histogram.sum()

    def test_chain_sampler_preserves_target_distribution(self):
        emitted = self._emitted_histogram("rejection")
        target = TARGET_ROW.to(self.device).double()
        total_variation = 0.5 * (emitted - target).abs().sum().item()
        self.assertLess(
            total_variation,
            0.02,
            f"emitted distribution {emitted.tolist()} deviates from target "
            f"{target.tolist()} (total variation {total_variation:.4f})",
        )

    def test_greedy_verify_collapses_to_argmax(self):
        """Guards the gate in `eagle_sample` that picks between the two verifiers.

        The greedy verifier emits the target argmax and nothing else, so it is not
        a valid substitute for a non-greedy request: falling back to it silently
        discards `temperature` / `top_p` / `top_k`.
        """
        emitted = self._emitted_histogram("greedy")
        argmax_token = int(TARGET_ROW.argmax())
        self.assertAlmostEqual(emitted[argmax_token].item(), 1.0, places=6)


@unittest.skipUnless(torch.cuda.is_available(), "GPU is required for this test.")
class TestPortableSpecRenorm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_top_k_scalar_and_per_row(self):
        from sglang.kernels.ops.sampling.renorm import top_k_renorm_probs_torch

        probs = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
            dtype=torch.float32,
            device=self.device,
        )
        expected_top1 = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        torch.testing.assert_close(
            top_k_renorm_probs_torch(probs, 1),
            expected_top1,
        )

        top_ks = torch.tensor([2, 4], dtype=torch.int32, device=self.device)
        renorm = top_k_renorm_probs_torch(probs, top_ks)
        torch.testing.assert_close(
            renorm.sum(dim=-1),
            torch.ones(2, device=self.device),
        )
        self.assertEqual(torch.count_nonzero(renorm[0]).item(), 2)
        torch.testing.assert_close(renorm[1], probs[1])

    def test_top_p_per_row_and_zero_mass(self):
        from sglang.kernels.ops.sampling.renorm import top_p_renorm_probs_torch

        probs = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )
        top_ps = torch.tensor([0.5, 0.95], dtype=torch.float32, device=self.device)
        renorm = top_p_renorm_probs_torch(probs, top_ps)
        expected = torch.tensor(
            [[4.0 / 7.0, 3.0 / 7.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )
        torch.testing.assert_close(renorm, expected)


@unittest.skipUnless(torch.cuda.is_available(), "GPU is required for this test.")
class TestSpecRenormFallbacks(CustomTestCase):
    """The torch renorm fallbacks must match the kernels they stand in for.

    `top_k_renorm_prob` / `top_p_renorm_prob` come from flashinfer's `renorm.cu`,
    which is not part of the ROCm sgl-kernel build, so `_get_spec_renorm_fns()`
    substitutes torch implementations there. This checks the substitutes against
    the kernels wherever both are available.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")
        # The python wrappers import fine on ROCm; only the underlying torch op is
        # missing, so probe with a real call rather than trusting the import.
        try:
            from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

            probe = torch.full((1, 4), 0.25, dtype=torch.float32, device=cls.device)
            top_k_renorm_prob(
                probe.clone(), torch.ones(1, dtype=torch.int32, device=cls.device)
            )
            top_p_renorm_prob(
                probe.clone(), torch.ones(1, dtype=torch.float32, device=cls.device)
            )
        except (ImportError, AttributeError, NotImplementedError, RuntimeError) as exc:
            raise unittest.SkipTest(f"sgl_kernel renorm ops are unavailable: {exc}")
        cls.top_k_renorm_prob = staticmethod(top_k_renorm_prob)
        cls.top_p_renorm_prob = staticmethod(top_p_renorm_prob)

    def _probs(self, batch_size: int, vocab_size: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        logits = torch.randn(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        return torch.softmax(logits, dim=-1)

    def _assert_close_in_distribution(self, fallback, kernel):
        """Compare as distributions rather than element-wise.

        The two implementations locate the cutoff differently (sort-and-scan vs a
        pivot search), so a probability sitting exactly at the boundary may be kept
        by one and dropped by the other. That moves a negligible amount of mass,
        while a genuine semantic error (wrong cutoff or missing renormalization)
        moves an order of magnitude more.
        """
        total_variation = 0.5 * (fallback - kernel).abs().sum(dim=-1)
        self.assertLess(total_variation.max().item(), 1e-2)
        torch.testing.assert_close(
            fallback.sum(dim=-1), kernel.sum(dim=-1), rtol=1e-5, atol=1e-5
        )

    def test_top_k_fallback_matches_kernel(self):
        from sglang.kernels.ops.sampling.renorm import top_k_renorm_probs_torch

        probs = self._probs(64, 4096, seed=0)
        for top_k in (1, 8, 512, 4096):
            with self.subTest(top_k=top_k):
                top_ks = torch.full(
                    (probs.shape[0],), top_k, dtype=torch.int32, device=self.device
                )
                self._assert_close_in_distribution(
                    top_k_renorm_probs_torch(probs.clone(), top_ks),
                    self.top_k_renorm_prob(probs.clone(), top_ks),
                )

    def test_top_p_fallback_matches_kernel(self):
        from sglang.kernels.ops.sampling.renorm import top_p_renorm_probs_torch

        probs = self._probs(64, 4096, seed=1)
        for top_p in (0.5, 0.9, 0.95, 1.0):
            with self.subTest(top_p=top_p):
                top_ps = torch.full(
                    (probs.shape[0],), top_p, dtype=torch.float32, device=self.device
                )
                self._assert_close_in_distribution(
                    top_p_renorm_probs_torch(probs.clone(), top_ps),
                    self.top_p_renorm_prob(probs.clone(), top_ps),
                )


if __name__ == "__main__":
    unittest.main()
