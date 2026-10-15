"""Draft q and target p must agree on CUDA for the same logits and cutoffs."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.draft_sampling import DraftSamplingParams, build_draft_probs
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDraftSamplingCUDA(unittest.TestCase):
    def test_same_logits_and_params_match_dense_target(self):
        torch.manual_seed(7)
        for width in (16, 4096, 129280):
            with self.subTest(width=width):
                logits = torch.randn(3, width, device="cuda")
                params = DraftSamplingParams(
                    torch.tensor([0.7, 1.0, 1.5], device="cuda"),
                    torch.tensor([7, 12, TOP_K_ALL], dtype=torch.int32, device="cuda"),
                    torch.tensor([0.7, 0.95, 1.0], device="cuda"),
                    apply_top_k=True,
                    apply_top_p=True,
                )
                info = SimpleNamespace(
                    temperatures=params.temperatures[:, None],
                    top_ks=params.top_ks,
                    top_ps=params.top_ps,
                    need_top_k_sampling=True,
                    need_top_p_sampling=True,
                )
                target = build_speculative_verify_target_probs(
                    next_token_logits=logits,
                    sampling_info=info,
                    draft_token_num=1,
                    bs=3,
                    use_sparse_topk=False,
                )[:, 0]
                proposal = build_draft_probs(logits, params)
                self.assertTrue(torch.equal(proposal > 0, target > 0))
                torch.testing.assert_close(proposal, target, atol=1e-6, rtol=1e-5)

    def test_out_slot_takes_the_fused_softmax_without_touching_neighbours(self):
        # A graph runner hands over one step's slot of [bs, steps, vocab]; the
        # fused no-cutoff path must honour that row stride.
        torch.manual_seed(3)
        logits = torch.randn(4, 8192, device="cuda")
        params = DraftSamplingParams(
            torch.tensor([0.7, 1.0, 1.5, 1.0], device="cuda"),
            torch.full((4,), TOP_K_ALL, dtype=torch.int32, device="cuda"),
            torch.ones(4, device="cuda"),
        )
        buffer = torch.full((4, 3, 8192), -1.0, device="cuda")
        out = build_draft_probs(logits, params, out=buffer[:, 1])
        self.assertEqual(out.data_ptr(), buffer[:, 1].data_ptr())
        torch.testing.assert_close(
            out, build_draft_probs(logits, params), atol=1e-6, rtol=1e-5
        )
        self.assertTrue(bool((buffer[:, [0, 2]] == -1.0).all()))

    def test_fused_prepare_top_k_staging_matches_native(self):
        # The Triton prepare fill and its native fallback stage the same
        # temperatures and draft top_k from the same sources.
        from sglang.kernels.ops.speculative.multi_layer_eagle import (
            fill_draft_extend_prepare_buffers_native,
            fill_draft_extend_prepare_buffers_triton,
        )
        from sglang.test.draft_sampling_common import run_prepare_fill

        src_top_ks = torch.tensor([1, 50], dtype=torch.int32)
        src_temperatures = torch.tensor([[0.7], [1.2]])
        results = [
            run_prepare_fill(
                impl,
                device="cuda",
                src_top_ks=src_top_ks,
                src_temperatures=src_temperatures,
                sampling_top_k=20,
            )
            for impl in (
                fill_draft_extend_prepare_buffers_triton,
                fill_draft_extend_prepare_buffers_native,
            )
        ]
        (triton_temps, triton_top_ks), (native_temps, native_top_ks) = results
        torch.testing.assert_close(triton_temps, native_temps)
        self.assertEqual(triton_top_ks.tolist(), native_top_ks.tolist())
        self.assertEqual(triton_top_ks.tolist(), [1, 20, 20, 7])


if __name__ == "__main__":
    unittest.main()
