import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.tree_sampling import (
    tree_speculative_sampling_target_only_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=45, stage="stage-b", runner_config="1-gpu-small-amd")


def _tree_topology(batch_size: int, device: torch.device):
    num_draft_tokens = 6
    retrieve_index = torch.arange(
        batch_size * num_draft_tokens,
        dtype=torch.int64,
        device=device,
    ).view(batch_size, num_draft_tokens)
    retrieve_next_token = (
        torch.tensor(
            [1, 2, -1, 4, 5, -1],
            dtype=torch.int64,
            device=device,
        )
        .expand(batch_size, -1)
        .contiguous()
    )
    retrieve_next_sibling = (
        torch.tensor(
            [-1, 3, -1, -1, -1, -1],
            dtype=torch.int64,
            device=device,
        )
        .expand(batch_size, -1)
        .contiguous()
    )
    return retrieve_index, retrieve_next_token, retrieve_next_sibling


def _random_inputs(
    *,
    batch_size: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    num_draft_tokens = 6
    candidates = torch.randint(
        0,
        vocab_size,
        (batch_size, num_draft_tokens),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    target_probs = torch.softmax(
        torch.randn(
            (batch_size, num_draft_tokens, vocab_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        dim=-1,
    )
    uniform_samples = torch.rand(
        (batch_size, num_draft_tokens),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    uniform_samples_for_final_sampling = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topology = _tree_topology(batch_size, device)
    return (
        candidates,
        *topology,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
    )


def _run_sampler(
    sampling_fn,
    *,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    max_tree_depth: int = 4,
):
    batch_size, num_draft_tokens = candidates.shape
    predicts = torch.full(
        (batch_size * num_draft_tokens,),
        -1,
        dtype=torch.int32,
        device=candidates.device,
    )
    accept_index = torch.full(
        (batch_size, max_tree_depth),
        -1,
        dtype=torch.int32,
        device=candidates.device,
    )
    num_correct_drafts = torch.zeros(
        (batch_size,),
        dtype=torch.int32,
        device=candidates.device,
    )
    draft_probs = torch.zeros_like(target_probs)
    sampling_fn(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=num_correct_drafts,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )
    torch.cuda.synchronize()
    return predicts, accept_index, num_correct_drafts, draft_probs


def _tree_target_only_reference(
    *,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    max_tree_depth: int,
):
    candidates = candidates.cpu()
    retrieve_index = retrieve_index.cpu()
    retrieve_next_token = retrieve_next_token.cpu()
    retrieve_next_sibling = retrieve_next_sibling.cpu()
    uniform_samples = uniform_samples.cpu()
    uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.cpu()
    target_probs = target_probs.cpu()
    draft_probs = torch.zeros_like(target_probs)

    batch_size, num_draft_tokens = candidates.shape
    vocab_size = target_probs.shape[-1]
    predicts = torch.full((batch_size * num_draft_tokens,), -1, dtype=torch.int32)
    accept_index = torch.full((batch_size, max_tree_depth), -1, dtype=torch.int32)
    num_correct_drafts = torch.zeros((batch_size,), dtype=torch.int32)
    safe_threshold_acc = max(threshold_acc, 1e-9)

    for batch_idx in range(batch_size):
        current_tree_idx = 0
        current_prob_row = 0
        last_accept_global_idx = int(retrieve_index[batch_idx, 0])
        accept_index[batch_idx, 0] = last_accept_global_idx
        coin = float(uniform_samples[batch_idx, 0])

        for _ in range(1, max_tree_depth):
            current_tree_idx = int(retrieve_next_token[batch_idx, current_tree_idx])
            sibling_prob_acc = 0.0
            accept_draft = False
            while current_tree_idx != -1:
                draft_token = int(candidates[batch_idx, current_tree_idx])
                target_prob = float(
                    target_probs[batch_idx, current_prob_row, draft_token]
                )
                sibling_prob_acc += target_prob
                accept_draft = (
                    coin <= sibling_prob_acc / safe_threshold_acc
                    or target_prob >= threshold_single
                )
                if accept_draft:
                    draft_global_idx = int(retrieve_index[batch_idx, current_tree_idx])
                    predicts[last_accept_global_idx] = draft_token
                    num_correct_drafts[batch_idx] += 1
                    accept_index[batch_idx, int(num_correct_drafts[batch_idx])] = (
                        draft_global_idx
                    )
                    last_accept_global_idx = draft_global_idx
                    current_prob_row = current_tree_idx
                    coin = float(uniform_samples[batch_idx, current_tree_idx])
                    break

                draft_probs[batch_idx, current_prob_row, draft_token] = target_prob
                current_tree_idx = int(
                    retrieve_next_sibling[batch_idx, current_tree_idx]
                )
            if not accept_draft:
                break

        all_drafts_accept = int(num_correct_drafts[batch_idx]) == max_tree_depth - 1
        residual = target_probs[batch_idx, current_prob_row].clone()
        if not all_drafts_accept:
            residual.sub_(draft_probs[batch_idx, current_prob_row]).clamp_(min=0)
        target_cdf = float(uniform_samples_for_final_sampling[batch_idx]) * float(
            residual.sum()
        )
        cumulative = residual.cumsum(dim=0)
        positive = residual > 0
        matches = positive & (cumulative > target_cdf)
        if matches.any():
            bonus_token = int(matches.to(torch.int32).argmax())
        elif positive.any():
            bonus_token = int(torch.nonzero(positive).flatten()[-1].item())
        else:
            bonus_token = vocab_size - 1
        predicts[last_accept_global_idx] = bonus_token

    return predicts, accept_index, num_correct_drafts, draft_probs


@unittest.skipUnless(torch.cuda.is_available(), "GPU is required for this test.")
class TestTreeSpeculativeSamplingTriton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_matches_torch_oracle(self):
        for threshold_single, threshold_acc in (
            (1.0, 1.0),
            (0.5, 0.7),
            (0.0, 0.0),
        ):
            with self.subTest(
                threshold_single=threshold_single,
                threshold_acc=threshold_acc,
            ):
                inputs = _random_inputs(
                    batch_size=4,
                    vocab_size=17,
                    seed=7,
                    device=self.device,
                )
                actual = _run_sampler(
                    tree_speculative_sampling_target_only_triton,
                    candidates=inputs[0],
                    retrieve_index=inputs[1],
                    retrieve_next_token=inputs[2],
                    retrieve_next_sibling=inputs[3],
                    uniform_samples=inputs[4],
                    uniform_samples_for_final_sampling=inputs[5],
                    target_probs=inputs[6],
                    threshold_single=threshold_single,
                    threshold_acc=threshold_acc,
                )
                expected = _tree_target_only_reference(
                    candidates=inputs[0],
                    retrieve_index=inputs[1],
                    retrieve_next_token=inputs[2],
                    retrieve_next_sibling=inputs[3],
                    uniform_samples=inputs[4],
                    uniform_samples_for_final_sampling=inputs[5],
                    target_probs=inputs[6],
                    threshold_single=threshold_single,
                    threshold_acc=threshold_acc,
                    max_tree_depth=4,
                )
                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(
                        actual_tensor.cpu(),
                        expected_tensor,
                    )

    @unittest.skipIf(
        torch.version.hip is not None,
        "CUDA AOT oracle is unavailable on ROCm.",
    )
    def test_matches_cuda_aot_oracle(self):
        from sgl_kernel import tree_speculative_sampling_target_only

        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                inputs = _random_inputs(
                    batch_size=4,
                    vocab_size=257,
                    seed=seed,
                    device=self.device,
                )
                kwargs = dict(
                    candidates=inputs[0],
                    retrieve_index=inputs[1],
                    retrieve_next_token=inputs[2],
                    retrieve_next_sibling=inputs[3],
                    uniform_samples=inputs[4],
                    uniform_samples_for_final_sampling=inputs[5],
                    target_probs=inputs[6],
                    threshold_single=1.0,
                    threshold_acc=1.0,
                )
                triton_result = _run_sampler(
                    tree_speculative_sampling_target_only_triton,
                    **kwargs,
                )
                cuda_result = _run_sampler(
                    tree_speculative_sampling_target_only,
                    **kwargs,
                )
                for triton_tensor, cuda_tensor in zip(
                    triton_result,
                    cuda_result,
                ):
                    torch.testing.assert_close(triton_tensor, cuda_tensor)

    def test_seeded_inputs_are_repeatable(self):
        inputs = _random_inputs(
            batch_size=8,
            vocab_size=128,
            seed=123,
            device=self.device,
        )
        kwargs = dict(
            candidates=inputs[0],
            retrieve_index=inputs[1],
            retrieve_next_token=inputs[2],
            retrieve_next_sibling=inputs[3],
            uniform_samples=inputs[4],
            uniform_samples_for_final_sampling=inputs[5],
            target_probs=inputs[6],
            threshold_single=1.0,
            threshold_acc=1.0,
        )
        first = _run_sampler(
            tree_speculative_sampling_target_only_triton,
            **kwargs,
        )
        second = _run_sampler(
            tree_speculative_sampling_target_only_triton,
            **kwargs,
        )
        for first_tensor, second_tensor in zip(first, second):
            torch.testing.assert_close(first_tensor, second_tensor)

    def test_dflash_sampling_integration_matches_oracle(self):
        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_sampling_correct_drafts_and_bonus,
            is_dflash_sampling_verify_available,
        )

        if not is_dflash_sampling_verify_available():
            self.skipTest("DFLASH sampling verification is unavailable.")

        batch_size = 2
        num_draft_tokens = 3
        vocab_size = 7
        generator = torch.Generator(device=self.device).manual_seed(11)
        candidates = torch.randint(
            0,
            vocab_size,
            (batch_size, num_draft_tokens),
            dtype=torch.int64,
            device=self.device,
            generator=generator,
        )
        target_probs = torch.softmax(
            torch.randn(
                (batch_size, num_draft_tokens, vocab_size),
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            ),
            dim=-1,
        )
        uniform_samples = torch.rand(
            (batch_size, num_draft_tokens),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        final_coins = torch.rand(
            (batch_size,),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones(
                (batch_size, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            top_ks=torch.full(
                (batch_size,),
                vocab_size,
                dtype=torch.int32,
                device=self.device,
            ),
            top_ps=torch.ones(
                (batch_size,),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        num_correct_drafts, bonus_tokens = (
            compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=target_probs.log().view(
                    batch_size * num_draft_tokens,
                    vocab_size,
                ),
                sampling_info=sampling_info,
                uniform_samples=uniform_samples,
                uniform_samples_for_final_sampling=final_coins,
                threshold_single=1.0,
                threshold_acc=1.0,
                use_sparse_topk=False,
            )
        )

        retrieve_index = torch.arange(
            batch_size * num_draft_tokens,
            dtype=torch.int64,
            device=self.device,
        ).view(batch_size, num_draft_tokens)
        retrieve_next_token = torch.tensor(
            [1, 2, -1],
            dtype=torch.int64,
            device=self.device,
        ).expand(batch_size, -1)
        retrieve_next_sibling = torch.full(
            (batch_size, num_draft_tokens),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        reference = _tree_target_only_reference(
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=final_coins,
            target_probs=target_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            max_tree_depth=num_draft_tokens,
        )
        reference_predicts, reference_accept_index, reference_num_correct, _ = reference
        row_ids = torch.arange(batch_size)
        reference_bonus = reference_predicts[
            reference_accept_index[
                row_ids,
                reference_num_correct.to(torch.long),
            ]
        ]
        torch.testing.assert_close(
            num_correct_drafts.cpu(),
            reference_num_correct,
        )
        torch.testing.assert_close(
            bonus_tokens.cpu(),
            reference_bonus.to(torch.int64),
        )

    def test_branched_tree_preserves_target_distribution(self):
        batch_size = 8192
        num_draft_tokens = 4
        vocab_size = 5
        target_row = torch.tensor(
            [0.35, 0.25, 0.20, 0.10, 0.10],
            dtype=torch.float32,
            device=self.device,
        )
        target_probs = (
            target_row.view(1, 1, -1)
            .expand(batch_size, num_draft_tokens, vocab_size)
            .contiguous()
        )
        candidates = torch.tensor(
            [4, 0, 1, 2],
            dtype=torch.int64,
            device=self.device,
        ).expand(batch_size, -1)
        retrieve_index = torch.arange(
            batch_size * num_draft_tokens,
            dtype=torch.int64,
            device=self.device,
        ).view(batch_size, num_draft_tokens)
        retrieve_next_token = torch.tensor(
            [1, -1, -1, -1],
            dtype=torch.int64,
            device=self.device,
        ).expand(batch_size, -1)
        retrieve_next_sibling = torch.tensor(
            [-1, 2, 3, -1],
            dtype=torch.int64,
            device=self.device,
        ).expand(batch_size, -1)
        generator = torch.Generator(device=self.device).manual_seed(42)
        uniform_samples = torch.rand(
            (batch_size, num_draft_tokens),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        uniform_samples_for_final_sampling = torch.rand(
            (batch_size,),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        predicts, _, _, _ = _run_sampler(
            tree_speculative_sampling_target_only_triton,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=(uniform_samples_for_final_sampling),
            target_probs=target_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            max_tree_depth=2,
        )
        emitted = predicts[predicts >= 0].to(torch.int64)
        distribution = torch.bincount(
            emitted,
            minlength=vocab_size,
        ).to(torch.float64)
        distribution.div_(distribution.sum())
        total_variation = (
            0.5 * (distribution - target_row.to(torch.float64)).abs().sum()
        )
        self.assertLess(total_variation.item(), 0.02)


if __name__ == "__main__":
    unittest.main()
