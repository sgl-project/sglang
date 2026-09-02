import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.sampler import (
    Sampler,
    build_effective_probs,
    sample_textseal_from_probs,
)
from sglang.srt.sampling.watermark import (
    WatermarkBatchInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _watermark(enabled=(True, True)):
    return WatermarkBatchInfo(
        enabled=torch.tensor(enabled),
        all_enabled=all(enabled),
        key_a=torch.tensor([741852963, 741852963]),
        key_b=torch.tensor([963852741, 963852741]),
        mixing_probabilities=torch.tensor([1.0, 0.0]),
        ngrams=torch.tensor([2, 2], dtype=torch.int32),
        contexts=torch.tensor([[1, 2], [2, 1]]),
        nonces=torch.tensor([11, 12]),
    )


class TestTextSealSampler(CustomTestCase):
    def test_no_attachment_uses_only_ordinary_path(self):
        sampler = MagicMock()
        sampler._sample_from_probs_ordinary.return_value = torch.tensor([3, 4])
        sampling_info = SimpleNamespace(watermark=None)

        selected = Sampler._sample_from_probs(
            sampler,
            torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
            sampling_info,
            torch.tensor([1, 1]),
            False,
        )

        torch.testing.assert_close(selected, torch.tensor([3, 4]))
        sampler._sample_from_probs_ordinary.assert_called_once()

    def test_mixed_batch_overwrites_only_enabled_rows(self):
        sampler = MagicMock()
        sampler._sample_from_probs_ordinary.return_value = torch.tensor([3, 4])
        watermark = _watermark(enabled=(True, False))
        sampling_info = SimpleNamespace(
            watermark=watermark,
            top_ks=torch.tensor([4, 4]),
            top_ps=torch.ones(2),
            min_ps=torch.zeros(2),
            watermark_max_top_k=4,
        )
        with patch(
            "sglang.srt.layers.sampler.sample_textseal_from_probs",
            return_value=torch.tensor([1]),
        ) as textseal_sampler:
            selected = Sampler._sample_from_probs(
                sampler,
                torch.full((2, 4), 0.25),
                sampling_info,
                torch.tensor([1, 1]),
                False,
            )

        torch.testing.assert_close(selected, torch.tensor([1, 4]))
        self.assertEqual(textseal_sampler.call_args.args[0].shape, (1, 4))

    def test_fully_watermarked_batch_skips_ordinary_sampler(self):
        sampler = MagicMock()
        sampling_info = SimpleNamespace(
            watermark=_watermark(),
            top_ks=torch.tensor([4, 4]),
            top_ps=torch.ones(2),
            min_ps=torch.zeros(2),
            watermark_max_top_k=4,
        )
        with patch(
            "sglang.srt.layers.sampler.sample_textseal_from_probs",
            return_value=torch.tensor([1, 2]),
        ):
            selected = Sampler._sample_from_probs(
                sampler,
                torch.full((2, 4), 0.25),
                sampling_info,
                torch.tensor([1, 1]),
                False,
            )

        torch.testing.assert_close(selected, torch.tensor([1, 2]))
        sampler._sample_from_probs_ordinary.assert_not_called()

    def test_effective_support_matches_top_k_top_p_min_p(self):
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.5, 0.2, 0.2, 0.1]])
        effective = build_effective_probs(
            probs,
            torch.tensor([3, 4]),
            torch.tensor([0.65, 1.0]),
            torch.tensor([0.0, 0.5]),
        )
        torch.testing.assert_close(effective[0], torch.tensor([4 / 7, 3 / 7, 0.0, 0.0]))
        torch.testing.assert_close(effective[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_textseal_selection_uses_effective_support(self):
        probs = torch.tensor([[0.55, 0.25, 0.15, 0.05], [0.6, 0.1, 0.2, 0.1]])
        selected = sample_textseal_from_probs(
            probs,
            torch.tensor([4, 4]),
            torch.ones(2),
            torch.zeros(2),
            _watermark(),
            torch.tensor([4, 9]),
            4,
        )
        self.assertEqual(selected.tolist(), [0, 1])

    def test_collapsed_support_returns_only_admissible_token(self):
        selected = sample_textseal_from_probs(
            torch.tensor([[0.6, 0.4], [0.6, 0.4]]),
            torch.tensor([1, 2]),
            torch.ones(2),
            torch.zeros(2),
            _watermark(),
            torch.tensor([1, 1]),
            2,
        )
        self.assertEqual(selected[0].item(), 0)

    def test_finite_top_k_bounds_textseal_candidates(self):
        """Finite top-k must avoid passing the full vocabulary to selection."""
        probs = torch.tensor([[0.05, 0.4, 0.1, 0.3, 0.15], [0.3, 0.1, 0.2, 0.05, 0.35]])
        with patch(
            "sglang.srt.layers.sampler.select_textseal_tokens",
            return_value=torch.tensor([1, 4]),
        ) as selector:
            sample_textseal_from_probs(
                probs,
                torch.tensor([2, 2]),
                torch.ones(2),
                torch.zeros(2),
                _watermark(),
                torch.tensor([1, 1]),
                2,
            )

        effective_probs = selector.call_args.args[0]
        token_ids = selector.call_args.kwargs["token_ids"]
        self.assertEqual(effective_probs.shape, (2, 2))
        torch.testing.assert_close(token_ids, torch.tensor([[1, 3], [4, 0]]))


if __name__ == "__main__":
    unittest.main()
