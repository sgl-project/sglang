import unittest

import torch

from sglang.kernels.ops.sampling.synthid_triton import (
    synthid_sample_triton,
    synthid_tournament_triton,
)
from sglang.srt.sampling.watermarking.synthid.algorithm import (
    SynthIDConfig,
    synthid_tournament_torch,
)
from sglang.srt.sampling.watermarking.synthid.watermark import SynthIDWatermark
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.synthid_utils import (
    SYNTHID_TEST_KEYS,
    assert_synthid_state_equal,
    synthid_state,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _watermark(device, *, num_candidates=40, skip_first=False):
    config = SynthIDConfig(
        keys=SYNTHID_TEST_KEYS[:4],
        num_candidates=num_candidates,
        context_history_size=4,
        skip_first_ngram_calls=skip_first,
    )
    watermark = SynthIDWatermark(config, 8, device)
    slots = torch.tensor([1, 5], device=device)
    # Slot 5 replays two periods of a cycle as long as the context, so its next
    # context is already in its history and its first row is not watermarked.
    watermark.state.seed(slots, [[11, 12], [21, 22, 23, 24] * 2])
    return watermark, slots


class TestSynthIDKernels(CustomTestCase):
    def test_tournament_kernel_matches_torch(self):
        for rows, num_candidates, depth in (
            (1, 3, 2),
            (8, 1024, 30),
            (1, 65536, 64),
        ):
            with self.subTest(rows=rows, num_candidates=num_candidates, depth=depth):
                torch.manual_seed(7)
                scores = torch.randn(rows, num_candidates, device="cuda")
                # A request filter leaves -inf scores; keep one finite per row.
                scores[:, 1:][torch.rand_like(scores[:, 1:]) < 0.25] = -float("inf")
                tokens = torch.randint(0, 200_000, scores.shape, device="cuda")
                hashes = torch.arange(rows, device="cuda", dtype=torch.int64)
                keys = torch.arange(1, depth + 1, device="cuda")
                mask = torch.arange(rows, device="cuda") % 3 != 1
                temperatures = torch.linspace(0.5, 1.3, rows, device="cuda")[:, None]
                scaled = scores / temperatures
                expected = synthid_tournament_torch(
                    scaled, tokens, hashes, keys, mask
                ).softmax(dim=-1)
                actual = synthid_tournament_triton(
                    scaled, tokens, hashes, keys, mask
                ).softmax(dim=-1)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_fused_sample_applies_temperature_and_leaves_state(self):
        for num_candidates, skip_first in ((40, True), (1024, False)):
            with self.subTest(num_candidates=num_candidates, skip_first=skip_first):
                reference, slots = _watermark(
                    "cpu", num_candidates=num_candidates, skip_first=skip_first
                )
                actual, gpu_slots = _watermark(
                    "cuda", num_candidates=num_candidates, skip_first=skip_first
                )
                torch.manual_seed(7)
                scores = torch.randn(2, num_candidates, device="cuda")
                tokens = torch.stack(
                    [
                        torch.randperm(4096, device="cuda")[:num_candidates]
                        for _ in range(2)
                    ]
                )
                temperatures = torch.tensor([[0.5], [1.3]], device="cuda")
                noise = torch.empty_like(scores).exponential_()
                hashes, mask = reference.state.current(slots)
                expected = synthid_tournament_torch(
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
                assert_synthid_state_equal(
                    synthid_state(actual.state), synthid_state(reference.state)
                )

    def test_projection_transforms_and_commit_match_cpu(self):
        for skip_first in (False, True):
            for stage in ("apply", "verify", "commit"):
                with self.subTest(skip_first=skip_first, stage=stage):
                    reference, slots = _watermark("cpu", skip_first=skip_first)
                    actual, gpu_slots = _watermark("cuda", skip_first=skip_first)
                    if stage == "commit":
                        for tokens, lengths in (
                            (torch.tensor([101, 201]), torch.tensor([0, 1])),
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
                            assert_synthid_state_equal(
                                synthid_state(actual.state),
                                synthid_state(reference.state),
                            )
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
                        assert_synthid_state_equal(
                            synthid_state(actual.state), synthid_state(reference.state)
                        )
                        row_temperatures = temperatures.repeat_interleave(width, 0)
                        torch.testing.assert_close(
                            (watermarked.cpu() / row_temperatures).softmax(-1),
                            (expected / row_temperatures).softmax(-1),
                            rtol=2e-4,
                            atol=2e-5,
                        )


if __name__ == "__main__":
    unittest.main()
