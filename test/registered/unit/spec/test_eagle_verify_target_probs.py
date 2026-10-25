"""Verify that speculative target filtering matches the configured sampler.

Requires a GPU: the top-k / top-p renormalization uses FlashInfer kernels.
"""

from importlib.util import find_spec
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

from sglang.srt.layers import sampler
from sglang.srt.speculative.eagle_utils import _filter_verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DRAFT_TOKEN_NUM = 2
VOCAB = 8
PROBS = [0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]
HAS_FLASHINFER = find_spec("flashinfer") is not None


def _sampling_info(top_ks, top_ps, min_ps):
    return SimpleNamespace(
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor(top_ps, dtype=torch.float32, device="cuda"),
        min_ps=torch.tensor(min_ps, dtype=torch.float32, device="cuda"),
        need_top_k_sampling=any(k < VOCAB for k in top_ks),
        need_top_p_sampling=any(p < 1.0 for p in top_ps),
        need_min_p_sampling=any(p > 0.0 for p in min_ps),
    )


def _expanded_info(info):
    def per_row(values):
        return values.repeat_interleave(DRAFT_TOKEN_NUM)

    return SimpleNamespace(
        top_ks=per_row(info.top_ks),
        top_ps=per_row(info.top_ps),
        min_ps=per_row(info.min_ps),
        need_top_k_sampling=info.need_top_k_sampling,
        need_top_p_sampling=info.need_top_p_sampling,
        need_min_p_sampling=info.need_min_p_sampling,
        sampling_mask_batch_indices=torch.arange(
            info.top_ks.numel() * DRAFT_TOKEN_NUM, dtype=torch.int64, device="cuda"
        ),
        sampling_seed=None,
        max_top_k=None,
    )


def _filtered(info, probs, backend):
    return _filter_verify_target_probs(
        probs,
        info,
        DRAFT_TOKEN_NUM,
        sampling_backend=backend,
        top_k_renorm_prob=top_k_renorm_prob,
        top_p_renorm_prob=top_p_renorm_prob,
    )


def _sampler_distribution(info, probs, backend):
    """Normalized weights captured by the actual Sampler branch."""
    with patch.object(
        sampler,
        "get_exec",
        return_value=SimpleNamespace(kernel=SimpleNamespace(sampling_backend=backend)),
    ):
        _, captured = sampler.Sampler._sample_from_probs(
            None,
            probs.clone(),
            _expanded_info(info),
            torch.zeros(probs.shape[0], dtype=torch.int64, device="cuda"),
            simple_sampling_case=False,
        )
    dist = captured.weights
    if captured.token_ids is not None:
        dist = torch.zeros_like(probs).scatter_(1, captured.token_ids.long(), dist)
    return dist / dist.sum(dim=-1, keepdim=True)


class TestVerifyTargetProbs(CustomTestCase):
    def _probs(self, num_reqs):
        return torch.tensor(
            [PROBS] * (num_reqs * DRAFT_TOKEN_NUM), dtype=torch.float32, device="cuda"
        )

    def test_matches_the_sampler_for_every_filter(self):
        # Request 0: top-k/top-p with min-p=0; 1: min-p; 2: all; 3: none.
        info = _sampling_info(
            top_ks=[3, VOCAB, 4, VOCAB],
            top_ps=[0.6, 1.0, 0.7, 1.0],
            min_ps=[0.0, 0.45, 0.4, 0.0],
        )
        probs = self._probs(4)
        backends = ("pytorch", "flashinfer") if HAS_FLASHINFER else ("pytorch",)
        for backend in backends:
            with self.subTest(backend=backend):
                got = _filtered(info, probs, backend)
                torch.testing.assert_close(
                    got, _sampler_distribution(info, probs, backend)
                )
                torch.testing.assert_close(got[6:], probs[6:])
                if backend == "flashinfer":
                    # Batch-level min-p selects sequential filtering even for
                    # this request with min-p=0.
                    self.assertEqual((got[0] > 0).nonzero().flatten().tolist(), [0, 1])

    def test_without_min_p_uses_joint_filtering(self):
        info = _sampling_info(top_ks=[3], top_ps=[0.6], min_ps=[0.0])
        probs = self._probs(1)
        backends = ("pytorch", "flashinfer") if HAS_FLASHINFER else ("pytorch",)
        for backend in backends:
            with self.subTest(backend=backend):
                got = _filtered(info, probs, backend)
                torch.testing.assert_close(
                    got, _sampler_distribution(info, probs, backend)
                )
                self.assertEqual((got[0] > 0).nonzero().flatten().tolist(), [0, 1, 2])

    def test_min_p_drops_tokens_below_its_share_of_the_top_token(self):
        if not HAS_FLASHINFER:
            self.skipTest("FlashInfer is not installed")
        info = _sampling_info(top_ks=[VOCAB], top_ps=[1.0], min_ps=[0.45])
        got = _filtered(info, self._probs(1), "flashinfer")
        # threshold 0.45 * 0.3 = 0.135: tokens 0, 1, 2 stay
        self.assertEqual((got[1] > 0).nonzero().flatten().tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
