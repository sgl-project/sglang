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
class TestTritonTopPFastPath(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def _distribution_with_nucleus(self, nucleus_size: int, vocab_size: int = 128):
        weights = torch.arange(
            vocab_size,
            0,
            step=-1,
            dtype=torch.float32,
            device=self.device,
        )
        probs = weights / weights.sum()
        cumulative = probs.cumsum(dim=0)
        lower = cumulative[nucleus_size - 2] if nucleus_size > 1 else 0.0
        top_p = (lower + cumulative[nucleus_size - 1]) / 2
        return probs.unsqueeze(0), top_p.unsqueeze(0)

    def _assert_matches_baseline(self, probs, top_ps):
        from sglang.kernels.ops.sampling import top_p_renorm_probs
        from sglang.kernels.ops.sampling.renorm_triton import (
            top_p_renorm_probs_triton,
            top_p_renorm_probs_triton_baseline,
            top_p_renorm_probs_triton_hierarchical,
            top_p_renorm_probs_triton_scale_fast,
            top_p_renorm_probs_triton_scatter_fast,
        )

        expected = top_p_renorm_probs_triton_baseline(probs, top_ps)
        functions = [
            top_p_renorm_probs_triton_scale_fast,
            top_p_renorm_probs_triton_scatter_fast,
            top_p_renorm_probs_triton_hierarchical,
            top_p_renorm_probs_triton,
        ]
        if torch.version.hip is not None:
            functions.append(top_p_renorm_probs)
        for fn in functions:
            got = fn(probs, top_ps)
            self.assertTrue(torch.equal(got > 0, expected > 0))
            torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(
                got.sum(dim=-1), expected.sum(dim=-1), rtol=1e-6, atol=1e-6
            )

    def test_fast_nucleus_sizes(self):
        from sglang.kernels.ops.sampling.renorm import top_p_fast_prefix

        for nucleus_size in (1, 15, 31):
            with self.subTest(nucleus_size=nucleus_size):
                probs, top_ps = self._distribution_with_nucleus(nucleus_size)
                *_, fast_path = top_p_fast_prefix(probs, top_ps)
                self.assertTrue(bool(fast_path.all()))
                self._assert_matches_baseline(probs, top_ps)

    def test_boundary_and_cross_prefix_ties_fall_back(self):
        from sglang.kernels.ops.sampling.renorm import top_p_fast_prefix

        boundary_probs, boundary_top_ps = self._distribution_with_nucleus(32)
        tied_probs = torch.zeros((1, 128), dtype=torch.float32, device=self.device)
        tied_probs[:, :70] = 1.0 / 70
        tied_top_ps = torch.tensor([0.5], device=self.device)

        for probs, top_ps in (
            (boundary_probs, boundary_top_ps),
            (tied_probs, tied_top_ps),
        ):
            *_, fast_path = top_p_fast_prefix(probs, top_ps)
            self.assertFalse(bool(fast_path.any()))
            self._assert_matches_baseline(probs, top_ps)

    def test_ties_contained_inside_prefix_use_fast_path(self):
        from sglang.kernels.ops.sampling.renorm import top_p_fast_prefix

        probs = torch.full((1, 128), 0.1 / 113, dtype=torch.float32, device=self.device)
        probs[:, :15] = 0.9 / 15
        top_ps = torch.tensor([0.5], device=self.device)
        *_, fast_path = top_p_fast_prefix(probs, top_ps)
        self.assertTrue(bool(fast_path.all()))
        self._assert_matches_baseline(probs, top_ps)

    def test_mixed_p_one_and_zero_rows_fall_back(self):
        from sglang.kernels.ops.sampling.renorm import top_p_fast_prefix

        fast_probs = torch.full(
            (1, 128), 0.04 / 127, dtype=torch.float32, device=self.device
        )
        fast_probs[:, 0] = 0.96
        full_probs = torch.arange(
            128, 0, -1, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        full_probs /= full_probs.sum(dim=-1, keepdim=True)
        zero_probs = torch.zeros_like(full_probs)
        probs = torch.cat((fast_probs, full_probs, zero_probs))
        top_ps = torch.tensor([0.95, 1.0, 0.95], device=self.device)

        *_, fast_path = top_p_fast_prefix(probs, top_ps)
        self.assertEqual(fast_path.tolist(), [True, False, False])
        self._assert_matches_baseline(probs, top_ps)

    def test_small_vocab_threshold_forms_and_non_contiguous_input(self):
        from sglang.kernels.ops.sampling.renorm import top_p_fast_prefix
        from sglang.kernels.ops.sampling.renorm_triton import (
            top_p_renorm_probs_triton,
            top_p_renorm_probs_triton_baseline,
        )

        base = torch.rand((2, 64), dtype=torch.float32, device=self.device)
        probs = base[:, ::2]
        probs /= probs.sum(dim=-1, keepdim=True)
        self.assertFalse(probs.is_contiguous())

        for top_p in (
            0.0,
            0.9,
            torch.tensor([0.9], device=self.device),
            torch.tensor([0.5, 0.95], device=self.device),
        ):
            expected = top_p_renorm_probs_triton_baseline(probs, top_p)
            got = top_p_renorm_probs_triton(probs, top_p)
            self.assertTrue(torch.equal(got > 0, expected > 0))
            torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)

        top_ps = torch.tensor([1.0, 1.0], device=self.device)
        *_, fast_path = top_p_fast_prefix(probs.contiguous(), top_ps)
        self.assertTrue(bool(fast_path.all()))

    def test_hierarchical_selector_values(self):
        from sglang.kernels.ops.sampling.top_p_select_triton import (
            top_p_select_hierarchical_triton,
        )

        for rows, vocab_size, chunk_size in (
            (1, 127, 512),
            (6, 1549, 512),
            (2, 154880, 1024),
        ):
            with self.subTest(rows=rows, vocab_size=vocab_size, chunk_size=chunk_size):
                generator = torch.Generator(device=self.device).manual_seed(vocab_size)
                probs = torch.softmax(
                    torch.randn(
                        (rows, vocab_size),
                        dtype=torch.float32,
                        device=self.device,
                        generator=generator,
                    )
                    * 8,
                    dim=-1,
                )
                expected_values, _ = torch.topk(probs, 32, dim=-1, sorted=True)
                values = top_p_select_hierarchical_triton(probs, chunk_size=chunk_size)
                torch.testing.assert_close(values, expected_values, rtol=0, atol=0)

    def test_hierarchical_matches_topk_path_at_large_vocab_boundaries(self):
        from sglang.kernels.ops.sampling.renorm_triton import (
            top_p_renorm_probs_triton_hierarchical,
            top_p_renorm_probs_triton_scale_fast,
        )

        vocab_size = 154880
        probs = torch.zeros((6, vocab_size), dtype=torch.float32, device=self.device)
        indices = torch.tensor(
            [3, 2051, 4099, 8197, 16391, 32771, 65537, 131071],
            dtype=torch.int64,
            device=self.device,
        )
        masses = torch.tensor(
            [0.50, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.01, 0.005625],
            dtype=torch.float32,
            device=self.device,
        )
        row_indices = (
            indices.unsqueeze(0)
            + torch.arange(6, dtype=torch.int64, device=self.device).unsqueeze(1) * 17
        ) % vocab_size
        probs.scatter_(1, row_indices, masses.expand(6, -1))
        top_ps = torch.tensor(
            [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375],
            dtype=torch.float32,
            device=self.device,
        )

        expected = top_p_renorm_probs_triton_scale_fast(probs, top_ps)
        got = top_p_renorm_probs_triton_hierarchical(probs, top_ps)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(got > 0, expected > 0))

    @unittest.skipUnless(torch.version.hip is not None, "ROCm-only dispatch test")
    def test_hierarchical_selector_dispatches_on_rocm(self):
        from sglang.kernels.ops.sampling import renorm_triton

        probs = torch.softmax(
            torch.randn((1, 1549), dtype=torch.float32, device=self.device) * 8,
            dim=-1,
        )
        top_ps = torch.full((1,), 0.95, dtype=torch.float32, device=self.device)
        with patch.object(
            renorm_triton,
            "top_p_renorm_probs_triton_hierarchical",
            wraps=renorm_triton.top_p_renorm_probs_triton_hierarchical,
        ) as hierarchical:
            renorm_triton.top_p_renorm_probs_triton(probs, top_ps)
        hierarchical.assert_called_once()


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
