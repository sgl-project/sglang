import unittest
from types import SimpleNamespace as NS
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

import torch

from sglang.srt.sampling.verify_probs import build_verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci


def rank_top_k(probs, ks):
    values, indices = probs.sort(dim=-1, descending=True)
    values = values.masked_fill(
        torch.arange(probs.shape[-1], device=probs.device) >= ks[:, None], 0
    )
    values = values / values.sum(-1, keepdim=True)
    return torch.zeros_like(probs).scatter(1, indices, values)


def top_p(probs, ps):
    values, indices = probs.sort(dim=-1, descending=True)
    values = values.masked_fill((values.cumsum(-1) - values) > ps[:, None], 0)
    values = values / values.sum(-1, keepdim=True)
    return torch.zeros_like(probs).scatter(1, indices, values)


register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class SharedVerifyProbsTest(TestCase):
    def test_min_p_after_top_k_top_p_and_without_other_filters(self):
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
        for filters in (False, True):
            info = NS(
                temperatures=torch.tensor([[0.7]]),
                top_ks=torch.tensor([3]),
                top_ps=torch.tensor([0.9]),
                min_ps=torch.tensor([0.4]),
                need_top_k_sampling=filters,
                need_top_p_sampling=filters,
                need_min_p_sampling=True,
            )
            expected = (logits / 0.7).softmax(-1)
            if filters:
                expected = top_p(
                    rank_top_k(expected, info.top_ks.repeat(2)), info.top_ps.repeat(2)
                )
            expected = expected.masked_fill(
                expected < expected.amax(-1, keepdim=True) * 0.4, 0
            )
            expected /= expected.sum(-1, keepdim=True)
            actual = build_verify_target_probs(
                next_token_logits=logits,
                sampling_info=info,
                draft_token_num=2,
                bs=1,
                max_top_k=3,
                renorm_top_k=rank_top_k,
                renorm_top_p=top_p,
            )
            torch.testing.assert_close(actual[0], expected)

    def test_rank_sparse_and_dense_match_reference(self):
        torch.manual_seed(42)
        logits = torch.randn(6, 129)
        info = NS(
            temperatures=torch.tensor([[0.7], [1.3]]),
            top_ks=torch.tensor([7, 50]),
            top_ps=torch.tensor([0.6, 0.95]),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
        )
        expected = logits / info.temperatures.repeat_interleave(3, 0)
        expected = rank_top_k(expected.softmax(-1), info.top_ks.repeat_interleave(3))
        expected = top_p(expected, info.top_ps.repeat_interleave(3)).view(2, 3, 129)
        for sparse in (False, True):
            actual = build_verify_target_probs(
                next_token_logits=logits,
                sampling_info=info,
                draft_token_num=3,
                bs=2,
                max_top_k=64,
                use_sparse_topk=sparse,
                renorm_top_k=rank_top_k,
                renorm_top_p=top_p,
            )
            torch.testing.assert_close(actual, expected)

    def test_dense_unfiltered_and_nan_probes(self):
        info = NS(
            temperatures=torch.ones(1, 1),
            top_ks=torch.tensor([2]),
            top_ps=torch.tensor([0.9]),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
        )
        logits = torch.tensor([[1.0, -float("inf"), 2.0]])
        probe = Mock()
        actual = build_verify_target_probs(
            next_token_logits=logits,
            sampling_info=info,
            bs=1,
            draft_token_num=1,
            use_sparse_topk=False,
            renorm_top_k=rank_top_k,
            renorm_top_p=top_p,
            probe=probe,
        )
        torch.testing.assert_close(actual[0], logits.softmax(-1))
        self.assertEqual(probe.call_count, 1)
        info.need_top_k_sampling = info.need_top_p_sampling = True
        probe.reset_mock()
        build_verify_target_probs(
            next_token_logits=logits,
            sampling_info=info,
            bs=1,
            draft_token_num=1,
            use_sparse_topk=False,
            renorm_top_k=rank_top_k,
            renorm_top_p=top_p,
            probe=probe,
        )
        self.assertEqual(probe.call_count, 3)

    def test_invalid_layout_and_policy(self):
        info = NS(temperatures=torch.ones(2, 1))
        for logits, bs, width in (
            (torch.zeros(5, 8), 2, 3),
            (torch.zeros(6, 8), 2, 0),
            (torch.zeros(2, 3, 8), 2, 3),
            (torch.zeros(6, 8), 3, 2),
        ):
            with self.subTest(bs=bs, width=width), self.assertRaises(ValueError):
                build_verify_target_probs(
                    next_token_logits=logits,
                    sampling_info=info,
                    bs=bs,
                    draft_token_num=width,
                )
        with self.assertRaisesRegex(ValueError, "Unknown sparse top-k mode"):
            build_verify_target_probs(
                next_token_logits=torch.zeros(6, 8),
                sampling_info=info,
                bs=2,
                draft_token_num=3,
                sparse_top_k_mode="invalid",
            )

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sparse_capture_requires_host_bound(self):
        info = NS(
            temperatures=torch.ones(1, 1, device="cuda"), need_top_k_sampling=True
        )
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(ValueError, "requires max_top_k"),
        ):
            build_verify_target_probs(
                next_token_logits=torch.zeros(3, 129, device="cuda"),
                sampling_info=info,
                bs=1,
                draft_token_num=3,
            )

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_ties_and_graph_replay(self):
        from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

        torch.manual_seed(43)
        logits = torch.randn(6, 129, device="cuda")
        info = NS(
            temperatures=torch.ones(2, 1, device="cuda"),
            top_ks=torch.tensor([7, 50], dtype=torch.int32, device="cuda"),
            top_ps=torch.ones(2, device="cuda"),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
        )
        for mode in ("rank", "threshold"):

            def run():
                return build_verify_target_probs(
                    next_token_logits=logits,
                    sampling_info=info,
                    bs=2,
                    draft_token_num=3,
                    max_top_k=64,
                    sparse_top_k_mode=mode,
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for tied in (False, True):
                logits.normal_()
                if tied:
                    logits.zero_()
                for p in (1.0, 0.95):
                    info.top_ps.fill_(p)
                    info.temperatures.copy_(torch.tensor([[0.7], [1.3]], device="cuda"))
                    graph.replay()
                    torch.testing.assert_close(captured, run(), atol=0, rtol=0)
                    torch.testing.assert_close(
                        captured.sum(-1), torch.ones(2, 3, device="cuda")
                    )
                    if mode == "threshold":
                        expected = (
                            logits / info.temperatures.repeat_interleave(3, 0)
                        ).softmax(-1)
                        expected = top_k_renorm_probs(
                            expected, info.top_ks.repeat_interleave(3)
                        )
                        expected = top_p_renorm_probs(
                            expected, info.top_ps.repeat_interleave(3)
                        )
                        torch.testing.assert_close(
                            captured.view(6, 129), expected, atol=2e-6, rtol=2e-5
                        )
                    if tied and p == 1.0:
                        support = (captured > 0).sum(-1)
                        expected_support = (
                            torch.full((2, 3), 129, device="cuda")
                            if mode == "threshold"
                            else info.top_ks[:, None].expand(2, 3)
                        )
                        torch.testing.assert_close(
                            support, expected_support, check_dtype=False
                        )


if __name__ == "__main__":
    unittest.main()
