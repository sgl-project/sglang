"""SynthID Triton kernels against the torch paths of the SynthID lib."""

import functools
import unittest

import torch

from sglang.kernels.ops.sampling.synthid.triton import (
    synthid_sample_triton,
    watermark_topk_scores_triton,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.synthid.algorithm import (
    REFERENCE_SYNTHID_KEYS,
    SynthIDConfig,
    watermark_topk_scores_torch,
)
from sglang.srt.sampling.watermarking.synthid.sampler import (
    SynthIDSampler,
    top_k_top_p_min_p_filter_probs,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_STATE_FIELDS = ("context", "context_history", "history_cursor", "num_calls")


def _sampler(device, *, top_k=40, skip_first=False):
    config = SynthIDConfig(
        keys=REFERENCE_SYNTHID_KEYS[:4],
        top_k=top_k,
        context_history_size=4,
        skip_first_ngram_calls=skip_first,
    )
    sampler = SynthIDSampler(config, 8, device)
    slots = torch.tensor([1, 5], device=device)
    # Slot 5 replays two periods of a cycle as long as the context, so its next
    # context is already in its history and its first row is not watermarked.
    sampler.state.seed(slots, [[11, 12], [21, 22, 23, 24] * 2])
    return sampler, slots


class TestSynthIDKernels(CustomTestCase):
    def test_tournament_kernel_matches_torch(self):
        for rows, top_k, depth in (
            (1, 3, 2),
            (8, 40, 30),
            (64, 64, 30),
            (8, 1024, 30),
            (1, 1024, 64),
            (1, 65536, 30),
        ):
            with self.subTest(rows=rows, top_k=top_k, depth=depth):
                torch.manual_seed(7)
                scores = torch.randn(rows, top_k, device="cuda")
                tokens = torch.randint(0, 200_000, scores.shape, device="cuda")
                hashes = torch.arange(rows, device="cuda", dtype=torch.int64)
                keys = torch.arange(1, depth + 1, device="cuda")
                mask = torch.arange(rows, device="cuda") % 3 != 1
                temperatures = torch.linspace(0.5, 1.3, rows, device="cuda")[:, None]
                scaled = scores / temperatures
                expected = watermark_topk_scores_torch(
                    scaled, tokens, hashes, keys, mask
                ).softmax(dim=-1)
                actual = watermark_topk_scores_triton(
                    scaled, tokens, hashes, keys, mask
                ).softmax(dim=-1)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_fused_sample_applies_temperature_and_leaves_state(self):
        for top_k, skip_first in ((3, False), (40, True), (1024, False)):
            with self.subTest(top_k=top_k, skip_first=skip_first):
                reference, slots = _sampler("cpu", top_k=top_k, skip_first=skip_first)
                actual, gpu_slots = _sampler("cuda", top_k=top_k, skip_first=skip_first)
                torch.manual_seed(7)
                scores = torch.randn(2, top_k, device="cuda")
                tokens = torch.stack(
                    [torch.randperm(4096, device="cuda")[:top_k] for _ in range(2)]
                )
                temperatures = torch.tensor([[0.5], [1.3]], device="cuda")
                noise = torch.empty_like(scores).exponential_()
                hashes, mask = reference.state.current(slots)
                expected = watermark_topk_scores_torch(
                    (scores / temperatures).cpu(),
                    tokens.cpu(),
                    hashes,
                    reference.keys,
                    mask,
                ).softmax(dim=-1)
                selected = (expected / noise.cpu()).argmax(dim=-1, keepdim=True)
                expected_tokens = tokens.cpu().gather(1, selected).view(-1)

                state = actual.state
                sampled = synthid_sample_triton(
                    scores,
                    tokens,
                    temperatures,
                    gpu_slots,
                    state.context,
                    state.context_history,
                    state.num_calls,
                    actual.keys,
                    noise,
                    hash_iv=actual.config.hash_iv,
                    skip_first_ngram_calls=skip_first,
                )
                self.assertEqual(sampled.cpu().tolist(), expected_tokens.tolist())
                self.assert_state_equal(reference, actual)

                logits = torch.randn(2, 4096, device="cuda")
                self.assertTrue(actual.can_fuse_sample(logits))
                sampled = actual.sample(logits, temperatures, gpu_slots)
                candidates = logits.topk(top_k).indices
                self.assertTrue((candidates == sampled[:, None]).any(dim=1).all())
                self.assert_state_equal(reference, actual)

    def test_projection_transforms_and_commit_match_cpu(self):
        for skip_first in (False, True):
            for stage in ("current", "apply", "verify", "commit"):
                with self.subTest(skip_first=skip_first, stage=stage):
                    reference, slots = _sampler("cpu", skip_first=skip_first)
                    actual, gpu_slots = _sampler("cuda", skip_first=skip_first)
                    if stage == "current":
                        hashes, mask = reference.state.current(slots)
                        self.assertFalse(mask[1].item())
                        torch.testing.assert_close(
                            tuple(t.cpu() for t in actual.state.current(gpu_slots)),
                            (hashes, mask),
                        )
                    elif stage == "commit":
                        for tokens, lengths in (
                            (torch.tensor([101, 201]), torch.tensor([0, 1])),
                            (torch.tensor([102, 202]), None),
                            (
                                torch.tensor([[101, 102, 103], [201, 202, 0]]),
                                torch.tensor([3, 2]),
                            ),
                            (torch.tensor([[104, 105], [203, 204]]), None),
                        ):
                            reference.state.commit(slots, tokens, lengths)
                            actual.state.commit(
                                gpu_slots,
                                tokens.cuda(),
                                None if lengths is None else lengths.cuda(),
                            )
                            self.assert_state_equal(reference, actual)
                    else:
                        width = 3 if stage == "verify" else 1
                        # Unique logits avoid CPU/CUDA top-k tie-order differences.
                        logits = torch.stack(
                            [
                                torch.randperm(4096).float() / 4096
                                for _ in range(2 * width)
                            ]
                        )
                        temperatures = torch.tensor([[0.5], [1.3]])
                        expected = logits.clone()
                        watermarked = logits.cuda()
                        if stage == "verify":
                            verify_tokens = torch.tensor(
                                [[12, 101, 102], [24, 201, 202]]
                            )
                            reference.apply_verify(
                                expected, temperatures, slots, verify_tokens
                            )
                            actual.apply_verify(
                                watermarked,
                                temperatures.cuda(),
                                gpu_slots,
                                verify_tokens.cuda(),
                            )
                        else:
                            reference.apply(expected, temperatures, slots)
                            actual.apply(watermarked, temperatures.cuda(), gpu_slots)
                        self.assert_state_equal(reference, actual)
                        row_temperatures = temperatures.repeat_interleave(width, 0)
                        torch.testing.assert_close(
                            (watermarked.cpu() / row_temperatures).softmax(-1),
                            (expected / row_temperatures).softmax(-1),
                            rtol=2e-4,
                            atol=2e-5,
                        )

    def test_filters_precede_watermark(self):
        for top_k, top_p, min_p in (
            (2, 1.0, 0.0),
            (TOP_K_ALL, 0.75, 0.0),
            (TOP_K_ALL, 1.0, 0.5),
            (2, 0.55, 0.0),
            (2, 0.55, 0.5),
        ):
            with self.subTest(top_k=top_k, top_p=top_p, min_p=min_p):
                config = SynthIDConfig(keys=(9,), top_k=3)
                sampler = SynthIDSampler(config, 1, "cuda")
                slot = torch.tensor([0], device="cuda")
                sampler.state.seed(slot, [[]])
                logits = torch.tensor([[0.5, 0.3, 0.2]], device="cuda").log()
                sampler.apply(
                    logits,
                    torch.ones(1, 1, device="cuda"),
                    slot,
                    filter_probs=functools.partial(
                        top_k_top_p_min_p_filter_probs,
                        top_ks=torch.tensor([top_k], device="cuda", dtype=torch.int32),
                        top_ps=torch.tensor([top_p], device="cuda"),
                        min_ps=torch.tensor([min_p], device="cuda"),
                    ),
                )
                torch.testing.assert_close(
                    logits.softmax(-1),
                    torch.tensor([[5 / 8, 3 / 8, 0.0]], device="cuda"),
                    rtol=2e-5,
                    atol=2e-6,
                )

    def assert_state_equal(self, reference, actual):
        for name in _STATE_FIELDS:
            torch.testing.assert_close(
                getattr(actual.state, name).cpu(), getattr(reference.state, name)
            )


if __name__ == "__main__":
    unittest.main()
