"""CPU contract tests use an explicit verifier double; CUDA tests use real kernels."""

import importlib.util
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import sampler as sampler_impl
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative import eagle_utils, eagle_worker_v2
from sglang.srt.speculative.eagle_utils import (
    _sample_spec_opt_out_roots,
    eagle_sample,
    validate_spec_opt_out_request,
)
from sglang.srt.speculative.eagle_worker_common import _compact_accept_to_front
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def fixture(flags=(False, True, False), device="cpu"):
    bs, width, vocab = len(flags), 4, 8
    info = SimpleNamespace(
        temperatures=torch.ones(bs, 1, device=device),
        top_ks=torch.ones(bs, dtype=torch.int32, device=device),
        top_ps=torch.ones(bs, device=device),
        min_ps=torch.zeros(bs, device=device),
        is_all_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        has_watermark_candidates=False,
        has_custom_logit_processor=False,
        max_top_k=8,
        sampling_mask_batch_indices=None,
        sampling_support_logprobs_capture_indices=None,
        grammars=None,
        has_spec_opt_out=any(flags),
        is_all_spec_opt_out=bool(flags) and all(flags),
        spec_opt_out_mask=torch.tensor(flags, device=device),
    )
    batch = SimpleNamespace(
        device=device,
        seq_lens=torch.arange(10, 10 + bs, device=device),
        sampling_info=info,
        forward_mode=ForwardMode.TARGET_VERIFY,
        has_grammar=False,
        hisparse_coordinator=None,
        req_pool_indices=torch.arange(bs, device=device),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    verify = SimpleNamespace(
        draft_token=torch.full((bs, width), 3, dtype=torch.int64, device=device),
        draft_token_num=width,
        max_tree_depth=width,
        tree_topk=1,
        retrieve_index=torch.arange(bs * width, device=device).reshape(bs, width),
        retrieve_next_token=torch.tensor([[1, 2, 3, -1]] * bs, device=device),
        retrieve_next_sibling=torch.full((bs, width), -1, device=device),
        custom_mask=None,
        positions=None,
        draft_probs=None,
    )
    logits = torch.zeros(bs * width, vocab, device=device)
    logits[:, 3] = 10
    return batch, verify, LogitsProcessorOutput(next_token_logits=logits)


def full_acceptance(**kw):
    """Explicitly model a valid fully accepted chain; never rely on kernel stubs."""
    kw["predicts"].fill_(3)
    kw["accept_index"].copy_(kw["retrieve_index"])
    kw["accept_token_num"].fill_(kw["candidates"].shape[1] - 1)
    return kw["predicts"], kw["accept_index"], kw["accept_token_num"]


def parallel():
    return SimpleNamespace(tp_group=SimpleNamespace(world_size=1, device_group=None))


class TestPerRequestSpecOptOut(CustomTestCase):
    def sample(self, batch, verify, output, root_token=3, real_kernel=False):
        target_sampler = Mock(
            return_value=torch.full_like(batch.seq_lens, root_token, dtype=torch.int32)
        )
        with (
            patch.object(
                eagle_utils,
                "get_spec",
                return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
            ),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel()),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            nullcontext()
            if real_kernel
            else patch.object(
                eagle_utils, "verify_tree_greedy_func", side_effect=full_acceptance
            ),
        ):
            result = eagle_sample(verify, batch, output, target_sampler=target_sampler)
        return result, target_sampler

    def test_spec_opt_out_truncates_both_acceptance_representations(self):
        batch, verify, output = fixture()
        (predict, lengths, indices), sampler = self.sample(batch, verify, output)
        self.assertEqual(lengths.tolist(), [4, 1, 4])
        self.assertEqual(
            indices.tolist(), [[0, 1, 2, 3], [4, -1, -1, -1], [8, 9, 10, 11]]
        )
        # HiSparse derives commit counts from index validity, not lengths.
        self.assertEqual((indices >= 0).sum(1).tolist(), lengths.tolist())
        bonus = predict[indices.gather(1, (lengths - 1).long()[:, None]).flatten()]
        self.assertEqual(bonus.tolist(), [3, 3, 3])
        self.assertEqual((lengths - 1).tolist(), [3, 0, 3])
        self.assertEqual((torch.arange(3) * 4 + lengths - 1).tolist(), [3, 4, 11])
        sampler.assert_called_once()

    def test_root_sample_replaces_accepted_proposal_and_logprob_matches(self):
        batch, verify, output = fixture()
        # A verifier accepting proposal 3 must not override root sample 5.
        (predict, lengths, indices), _ = self.sample(
            batch, verify, output, root_token=5
        )
        self.assertEqual(predict[indices[:, 0]].tolist(), [3, 5, 3])
        compute_spec_logprobs(batch, output, predict, accept_index=indices)
        expected = output.next_token_logits[4].log_softmax(-1)[5]
        torch.testing.assert_close(output.next_token_logprobs[1, 0], expected)
        compact = _compact_accept_to_front(predict, indices, 3, num_draft_tokens=4)
        emitted = [
            compact[i * 4 : i * 4 + length].tolist()
            for i, length in enumerate(lengths.tolist())
        ]
        self.assertEqual(emitted, [[3] * 4, [5], [3] * 4])

    def test_rejection_sampling_cannot_retain_an_opt_out_draft_proposal(self):
        batch, verify, output = fixture()
        batch.sampling_info.is_all_greedy = False
        verify.draft_probs = torch.full((3, 3, 8), 1 / 8)
        config = SimpleNamespace(
            speculative_use_rejection_sampling=True,
            speculative_use_block_verification=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )

        def accept_proposals(**kw):
            kw["predicts"].fill_(3)
            kw["accept_index"].copy_(kw["retrive_index"])
            kw["accept_token_num"].fill_(3)

        target_sampler = Mock(return_value=torch.tensor([5, 5, 5], dtype=torch.int32))
        with (
            patch.object(eagle_utils, "get_spec", return_value=config),
            patch.object(eagle_utils, "_verify_uses_greedy", return_value=False),
            patch.object(
                eagle_utils,
                "chain_speculative_sampling_triton",
                side_effect=accept_proposals,
            ) as reject_kernel,
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel()),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            predict, lengths, indices = eagle_sample(
                verify, batch, output, target_sampler=target_sampler
            )
        reject_kernel.assert_called_once()
        self.assertEqual(predict[indices[:, 0]].tolist(), [3, 5, 3])
        self.assertEqual(lengths.tolist(), [4, 1, 4])
        self.assertEqual(indices[1].tolist(), [4, -1, -1, -1])
        torch.testing.assert_close(target_sampler.call_args.args[-1], batch.seq_lens)

    def test_all_opt_out_keeps_full_width_and_only_commits_roots(self):
        batch, verify, output = fixture((True, True, True))
        (predict, lengths, indices), _ = self.sample(batch, verify, output)
        self.assertEqual(lengths.tolist(), [1, 1, 1])
        compact = _compact_accept_to_front(predict, indices, 3, num_draft_tokens=4)
        self.assertEqual(compact.shape, (12,))
        self.assertEqual(indices[:, 0].tolist(), [0, 4, 8])
        self.assertTrue(torch.all(indices[:, 1:] == -1))

    def test_default_requests_do_not_call_extra_sampler(self):
        batch, verify, output = fixture((False, False, False))
        (_, lengths, _), sampler = self.sample(batch, verify, output)
        self.assertEqual(lengths.tolist(), [4, 4, 4])
        sampler.assert_not_called()

    def test_root_sampler_matches_ordinary_sampling(self):
        self._check_root_sampling(seeded=False)

    def test_greedy_root_sampler_matches_ordinary_logprob_sampling(self):
        self._check_root_sampling(seeded=False, greedy=True)

    def test_greedy_root_sampler_exact_ties_choose_lowest_token_id(self):
        # BF16 warm-decode divergence had exact ties, not a sampling error.
        # Include a near tie that must NOT be rounded into an exact tie.
        self._check_root_sampling(seeded=False, greedy=True, ties=True)

    def test_root_sampler_matches_ordinary_seeded_sampling(self):
        if (
            not torch.cuda.is_available()
            and importlib.util.find_spec("sgl_kernel") is None
        ):
            self.skipTest("seeded sampling requires native sgl_kernel on CPU or CUDA")
        self._check_root_sampling(seeded=True)

    def _check_root_sampling(self, seeded, greedy=False, ties=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch, verify, output = fixture(device=device)
        output.next_token_logits.copy_(
            torch.randn(12, 8, generator=torch.Generator().manual_seed(42)).to(device)
        )
        info = batch.sampling_info
        info.is_all_greedy = greedy
        info.top_ks.fill_(8)
        info.sampling_seed = (
            torch.tensor([11, 22, 33], device=device) if seeded else None
        )
        config = SimpleNamespace(
            deterministic=SimpleNamespace(
                rl_on_policy_target=None, enable_deterministic_inference=True
            ),
            kernel=SimpleNamespace(sampling_backend="pytorch"),
            features=SimpleNamespace(sampling_mask_max_tokens=32),
        )
        roots = verify.retrieve_index[:, 0]
        if ties:
            output.next_token_logits[roots] = torch.tensor(
                [
                    [0, 22.875, 22.875, 0, 0, 0, 0, 0],
                    [0, 24.625, 0, 24.625, 0, 0, 0, 0],
                    [0, 22.75, 22.875, 0, 0, 0, 0, 0],
                ],
                device=device,
            )
        original_logits = output.next_token_logits.clone()
        with (
            patch.object(sampler_impl, "get_parallel", return_value=parallel()),
            patch.object(sampler_impl, "get_exec", return_value=config),
            patch.object(sampler_impl, "is_dp_attention_enabled", return_value=False),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel()),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            sampler = sampler_impl.Sampler()
            # The ordinary sampler supports seeded top-k/top-p, but explicitly
            # rejects seeded min-p. Cover min-p only on its unseeded path.
            for filtered in (False, True):
                info.need_top_k_sampling = info.need_top_p_sampling = filtered
                info.need_min_p_sampling = filtered and not seeded
                if filtered:
                    info.top_ks.fill_(4)
                    info.top_ps.fill_(0.9)
                    info.min_ps.fill_(0.0 if seeded else 0.01)
                torch.manual_seed(1234)
                actual = _sample_spec_opt_out_roots(
                    output.next_token_logits, verify, batch, sampler
                )
                torch.manual_seed(4321 if greedy else 1234)
                expected = sampler(
                    LogitsProcessorOutput(
                        next_token_logits=original_logits[roots].clone()
                    ),
                    info,
                    greedy,
                    [],
                    [],
                    batch.seq_lens,
                )
                torch.testing.assert_close(actual, expected)
                if ties:
                    self.assertEqual(actual.tolist(), [1, 1, 2])
                torch.testing.assert_close(output.next_token_logits, original_logits)

    def test_mixed_all_out_mixed_keeps_draft_runtime_layout(self):
        worker = object.__new__(eagle_worker_v2.EAGLEWorkerV2)
        verify = SimpleNamespace(is_verify_input=lambda: True, iteration_ids=None)
        worker._draft_worker = SimpleNamespace(
            parallel_layout=None,
            draft=Mock(return_value=verify),
            _draft_extend_for_decode=Mock(),
        )
        worker.speculative_num_steps = 3
        worker.speculative_num_draft_tokens = 4
        worker.activate_step_by_batch = Mock()
        worker.verify = Mock(return_value=SimpleNamespace())
        worker._post_verify_graphs = None
        worker._build_trivial_verify_input = Mock(
            side_effect=AssertionError("wrong width")
        )
        with (
            patch.object(
                eagle_worker_v2, "draft_tp_context", return_value=nullcontext()
            ),
            patch.object(
                eagle_worker_v2,
                "speculative_moe_backend_context",
                return_value=nullcontext(),
            ),
            patch.object(
                eagle_worker_v2,
                "speculative_moe_a2a_backend_context",
                return_value=nullcontext(),
            ),
            patch.object(
                eagle_worker_v2, "spec_stage_span", return_value=nullcontext()
            ),
            patch.object(
                eagle_worker_v2, "get_parallel", return_value=SimpleNamespace(pp_size=1)
            ),
        ):
            for flags in (
                (False, True, False),
                (True, True, True),
                (True, False, True),
            ):
                batch, _, _ = fixture(flags)
                batch.forward_mode = ForwardMode.DECODE
                batch.is_extend_in_batch = False
                batch.spec_info = SimpleNamespace(is_verify_input=lambda: False)
                worker.forward_batch_generation(batch)
                self.assertEqual(worker.speculative_num_draft_tokens, 4)
        self.assertEqual(worker._draft_worker.draft.call_count, 3)
        self.assertEqual(worker._draft_worker._draft_extend_for_decode.call_count, 3)
        worker._build_trivial_verify_input.assert_not_called()

    def test_unsupported_algorithms_are_rejected_not_silently_ignored(self):
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(disable_speculative_decoding=True),
            custom_logit_processor=None,
        )
        supported = {
            SpeculativeAlgorithm.NONE,
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.STANDALONE,
        }
        for algo in SpeculativeAlgorithm:
            with self.subTest(algorithm=algo):
                self.assertEqual(
                    validate_spec_opt_out_request(req, algo) is None, algo in supported
                )
        req.custom_logit_processor = "custom"
        self.assertIn(
            "custom logit",
            validate_spec_opt_out_request(req, SpeculativeAlgorithm.EAGLE),
        )
        req.sampling_params.disable_speculative_decoding = False
        self.assertIsNone(
            validate_spec_opt_out_request(req, SpeculativeAlgorithm.DFLASH)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA verifier")
    def test_cuda_mixed_acceptance_uses_real_verifier(self):
        batch, verify, output = fixture(device="cuda")
        (predict, lengths, indices), _ = self.sample(
            batch, verify, output, real_kernel=True
        )
        self.assertEqual(lengths.tolist(), [4, 1, 4])
        self.assertEqual((indices >= 0).sum(1).tolist(), [4, 1, 4])
        self.assertEqual(predict[indices[:, 0]].tolist(), [3, 3, 3])


if __name__ == "__main__":
    unittest.main()
