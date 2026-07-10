import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.speculative.dflash_utils as dflash_utils
from sglang.srt.speculative.dspark_components.kernels.accept_sampling import (
    _chain_uniform_samples,
    _reference_chain_accept,
)
from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs
from sglang.srt.speculative.dflash_utils import build_seeded_dflash_sampling_uniforms
from sglang.srt.speculative.dspark_components.dspark_draft import sample_draft_block
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeMarkovHead:
    def sample_block(self, base_logits, first_prev_tokens, hidden_states, sampler):
        del first_prev_tokens, hidden_states
        tokens = []
        corrected_logits = []
        for step in range(base_logits.shape[1]):
            step_logits = base_logits[:, step, :]
            tokens.append(sampler(step_logits, step))
            corrected_logits.append(step_logits)
        return torch.stack(tokens, dim=1), torch.stack(corrected_logits, dim=1)


def _top_k_renorm_prob(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(probs)
    for row, top_k in enumerate(top_ks.tolist()):
        top_k = min(max(int(top_k), 1), probs.shape[-1])
        values, indices = torch.topk(probs[row], k=top_k)
        normalizer = values.sum().clamp_min(torch.finfo(probs.dtype).tiny)
        out[row, indices] = values / normalizer
    return out


def _top_p_renorm_prob(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(probs)
    for row, top_p in enumerate(top_ps.tolist()):
        sorted_probs, sorted_indices = torch.sort(probs[row], descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumsum - sorted_probs) <= float(top_p)
        kept = sorted_probs * keep.to(sorted_probs.dtype)
        normalizer = kept.sum().clamp_min(torch.finfo(probs.dtype).tiny)
        out[row, sorted_indices] = kept / normalizer
    return out


class TestDFlashVerifyTargetProbs(unittest.TestCase):
    def test_reference_chain_accept_handles_reject_and_cutoff(self):
        candidates = torch.tensor([[10, 1, 2, 3]], dtype=torch.int64)
        target_probs = torch.tensor(
            [
                [
                    [0.0, 0.6, 0.1, 0.1, 0.2],
                    [0.0, 0.0, 0.1, 0.2, 0.7],
                    [0.1, 0.1, 0.1, 0.6, 0.1],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                ]
            ],
            dtype=torch.float32,
        )
        draft_probs = torch.tensor(
            [
                [
                    [0.0, 0.5, 0.1, 0.2, 0.2],
                    [0.0, 0.0, 0.8, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.6, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        uniform = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        uniform_final = torch.tensor([[0.9, 0.2, 0.3, 0.4]], dtype=torch.float32)

        correct, bonus, cap_trim = _reference_chain_accept(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            uniform_samples=uniform,
            uniform_samples_final=uniform_final,
            gamma=3,
        )
        torch.testing.assert_close(correct, torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([4], dtype=torch.int64))
        torch.testing.assert_close(cap_trim, torch.tensor([0], dtype=torch.int32))

        correct, bonus, cap_trim = _reference_chain_accept(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            uniform_samples=uniform,
            uniform_samples_final=uniform_final,
            gamma=3,
            cutoff_verify_lens=torch.tensor([1], dtype=torch.int32),
        )
        torch.testing.assert_close(correct, torch.tensor([0], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([1], dtype=torch.int64))
        torch.testing.assert_close(cap_trim, torch.tensor([1], dtype=torch.int32))

    def test_seeded_chain_uniform_samples_are_position_deterministic(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA/ROCm for murmur_hash32 Triton kernel")

        device = torch.device("cuda")
        sampling_info = SimpleNamespace(
            sampling_seed=torch.tensor([1234, 5678], dtype=torch.int64, device=device)
        )
        positions = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int64,
            device=device,
        )

        accept_a, final_a = _chain_uniform_samples(
            sampling_info=sampling_info,
            positions_2d=positions,
            bs=2,
            gamma=3,
            verify_num_draft_tokens=4,
            device=device,
        )
        accept_b, final_b = _chain_uniform_samples(
            sampling_info=sampling_info,
            positions_2d=positions,
            bs=2,
            gamma=3,
            verify_num_draft_tokens=4,
            device=device,
        )

        torch.testing.assert_close(accept_a, accept_b)
        torch.testing.assert_close(final_a, final_b)
        self.assertTrue(torch.all((accept_a >= 0.0) & (accept_a <= 1.0)).item())
        self.assertTrue(torch.all((final_a >= 0.0) & (final_a <= 1.0)).item())

        shifted = positions.clone()
        shifted[0, 0] += 1
        accept_shifted, final_shifted = _chain_uniform_samples(
            sampling_info=sampling_info,
            positions_2d=shifted,
            bs=2,
            gamma=3,
            verify_num_draft_tokens=4,
            device=device,
        )
        self.assertNotEqual(float(accept_a[0, 0]), float(accept_shifted[0, 0]))
        self.assertNotEqual(float(final_a[0, 0]), float(final_shifted[0, 0]))

    def test_seeded_chain_uniform_requires_positions(self):
        sampling_info = SimpleNamespace(sampling_seed=torch.tensor([1234]))

        with self.assertRaisesRegex(RuntimeError, "needs positions_2d"):
            _chain_uniform_samples(
                sampling_info=sampling_info,
                positions_2d=None,
                bs=1,
                gamma=3,
                verify_num_draft_tokens=4,
                device=torch.device("cpu"),
            )

    def test_seeded_dflash_uniforms_validate_positions_shape(self):
        sampling_info = SimpleNamespace(sampling_seed=torch.tensor([1234]))

        with self.assertRaisesRegex(RuntimeError, "positions_2d shaped at least"):
            build_seeded_dflash_sampling_uniforms(
                sampling_info=sampling_info,
                positions_2d=torch.tensor([[10, 11]], dtype=torch.int64),
                draft_token_num=3,
            )

    def test_seeded_draft_sampling_is_position_deterministic(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA/ROCm for seeded multinomial Triton kernel")

        device = torch.device("cuda")
        base_logits = torch.tensor(
            [
                [[2.0, 1.0, 0.2, -0.5], [0.1, 1.5, 0.3, 0.0], [0.2, 0.4, 1.6, 0.1]],
                [[0.3, 1.1, 0.7, 0.0], [1.7, 0.2, 0.1, 0.5], [0.1, 0.2, 0.3, 1.4]],
            ],
            dtype=torch.float32,
            device=device,
        )
        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            top_ks=torch.full((2,), TOP_K_ALL, dtype=torch.int32, device=device),
            temperatures=torch.ones((2, 1), dtype=torch.float32, device=device),
            sampling_seed=torch.tensor([1234, 5678], dtype=torch.int64, device=device),
        )
        draft_positions = torch.tensor(
            [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.envs."
            "SGLANG_DSPARK_FAST_SAMPLING.get",
            return_value=False,
        ):
            result_a = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=torch.tensor([7, 9], dtype=torch.int64, device=device),
                draft_hidden=torch.zeros((2, 3, 1), dtype=torch.float32, device=device),
                sampling_info=sampling_info,
                markov_head=_FakeMarkovHead(),
                device=device,
                draft_positions=draft_positions,
            )
            result_b = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=torch.tensor([7, 9], dtype=torch.int64, device=device),
                draft_hidden=torch.zeros((2, 3, 1), dtype=torch.float32, device=device),
                sampling_info=sampling_info,
                markov_head=_FakeMarkovHead(),
                device=device,
                draft_positions=draft_positions,
            )

        torch.testing.assert_close(result_a.draft_tokens, result_b.draft_tokens)

    def test_min_p_filters_and_renormalizes_target_probs(self):
        logits = torch.log(
            torch.tensor(
                [
                    [0.60, 0.30, 0.09, 0.01],
                    [0.50, 0.24, 0.23, 0.03],
                ],
                dtype=torch.float32,
            )
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), dtype=torch.float32),
            top_ks=torch.tensor([4], dtype=torch.int32),
            top_ps=torch.tensor([1.0], dtype=torch.float32),
            min_ps=torch.tensor([0.2], dtype=torch.float32),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=True,
        )

        out = build_dflash_verify_target_probs(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=2,
            bs=1,
            use_sparse_topk=False,
        )

        expected = torch.tensor(
            [
                [
                    [0.60 / 0.90, 0.30 / 0.90, 0.0, 0.0],
                    [0.50 / 0.97, 0.24 / 0.97, 0.23 / 0.97, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_sparse_top_k_path_applies_min_p_after_top_k_renormalization(self):
        logits = torch.log(
            torch.tensor(
                [
                    [0.50, 0.20, 0.15, 0.10, 0.05],
                    [0.35, 0.30, 0.20, 0.10, 0.05],
                ],
                dtype=torch.float32,
            )
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), dtype=torch.float32),
            top_ks=torch.tensor([3], dtype=torch.int32),
            top_ps=torch.tensor([1.0], dtype=torch.float32),
            min_ps=torch.tensor([0.5], dtype=torch.float32),
            need_top_k_sampling=True,
            need_top_p_sampling=False,
            need_min_p_sampling=True,
        )

        out = build_dflash_verify_target_probs(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=2,
            bs=1,
            max_top_k=3,
            uniform_top_k_value=3,
            use_sparse_topk=True,
        )

        expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        0.35 / 0.85,
                        0.30 / 0.85,
                        0.20 / 0.85,
                        0.0,
                        0.0,
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_dense_path_applies_top_k_top_p_then_min_p(self):
        logits = torch.log(
            torch.tensor([[0.45, 0.25, 0.15, 0.10, 0.05]], dtype=torch.float32)
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), dtype=torch.float32),
            top_ks=torch.tensor([4], dtype=torch.int32),
            top_ps=torch.tensor([0.8], dtype=torch.float32),
            min_ps=torch.tensor([0.5], dtype=torch.float32),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=True,
        )

        with patch.object(
            dflash_utils, "top_k_renorm_prob", _top_k_renorm_prob
        ), patch.object(dflash_utils, "top_p_renorm_prob", _top_p_renorm_prob):
            out = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=1,
                bs=1,
                use_sparse_topk=False,
            )

        expected = torch.tensor(
            [[[0.45 / 0.70, 0.25 / 0.70, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_dense_path_uses_torch_fallback_when_renorm_kernels_unavailable(self):
        logits = torch.log(
            torch.tensor([[0.45, 0.25, 0.15, 0.10, 0.05]], dtype=torch.float32)
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), dtype=torch.float32),
            top_ks=torch.tensor([4], dtype=torch.int32),
            top_ps=torch.tensor([0.8], dtype=torch.float32),
            min_ps=torch.tensor([0.5], dtype=torch.float32),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=True,
        )

        with patch.object(dflash_utils, "top_k_renorm_prob", None), patch.object(
            dflash_utils, "top_p_renorm_prob", None
        ):
            out = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=1,
                bs=1,
                use_sparse_topk=False,
            )

        expected = torch.tensor(
            [[[0.45 / 0.70, 0.25 / 0.70, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_sparse_and_dense_paths_match_heterogeneous_sampling_params(self):
        logits = torch.log(
            torch.tensor(
                [
                    [0.42, 0.21, 0.16, 0.10, 0.07, 0.04],
                    [0.31, 0.25, 0.19, 0.11, 0.08, 0.06],
                    [0.36, 0.20, 0.17, 0.13, 0.08, 0.06],
                    [0.28, 0.23, 0.18, 0.15, 0.10, 0.06],
                ],
                dtype=torch.float32,
            )
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((2, 1), dtype=torch.float32),
            top_ks=torch.tensor([2, 4], dtype=torch.int32),
            top_ps=torch.tensor([0.95, 0.70], dtype=torch.float32),
            min_ps=torch.tensor([0.10, 0.35], dtype=torch.float32),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=True,
        )

        with patch.object(
            dflash_utils, "top_k_renorm_prob", _top_k_renorm_prob
        ), patch.object(dflash_utils, "top_p_renorm_prob", _top_p_renorm_prob):
            dense = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=2,
                bs=2,
                use_sparse_topk=False,
            )
            sparse = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=2,
                bs=2,
                max_top_k=4,
                uniform_top_k_value=None,
                use_sparse_topk=True,
            )

        torch.testing.assert_close(sparse, dense, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
