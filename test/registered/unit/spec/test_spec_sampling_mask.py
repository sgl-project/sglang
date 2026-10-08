"""Per-token sampling supports for chain EAGLE verify (return_sampling_mask)."""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import SamplingMaskStatus
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_mask import SamplingMaskRows
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_sampling_mask import (
    joint_filtered_verify_probs,
    spec_sampling_mask_unsupported_reason,
    verify_sampling_mask_output,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _top_k_renorm(probs, top_ks):
    out = torch.zeros_like(probs)
    for i, k in enumerate(top_ks.tolist()):
        values, idx = probs[i].topk(int(k))
        out[i, idx] = values / values.sum()
    return out


def _top_p_renorm(probs, top_ps):
    out = torch.zeros_like(probs)
    for i, p in enumerate(top_ps.tolist()):
        values, idx = probs[i].sort(descending=True)
        keep = (values.cumsum(0) - values) < p
        out[i, idx[keep]] = values[keep] / values[keep].sum()
    return out


class TestSpecSamplingMask(CustomTestCase):
    def test_joint_filter_keeps_what_the_non_speculative_sampler_keeps(self):
        """Top-p over the top-k-renormalized distribution (verify's order) drops token 2."""
        probs = torch.tensor([[0.40, 0.30, 0.20, 0.10]] * 2)
        rows, joint = joint_filtered_verify_probs(
            target_probs=probs,
            mask_req_rows=torch.tensor([1]),
            top_ks=torch.tensor([4, 3]),
            top_ps=torch.tensor([1.0, 0.75]),
            draft_token_num=1,
            top_k_renorm_prob=_top_k_renorm,
            top_p_renorm_prob=_top_p_renorm,
        )
        sequential = _top_p_renorm(
            _top_k_renorm(probs[1:], torch.tensor([3])), torch.tensor([0.75])
        )
        self.assertEqual(rows.tolist(), [1])
        self.assertEqual((joint[0] > 0).tolist(), [True, True, True, False])
        self.assertEqual((sequential[0] > 0).tolist(), [True, True, False, False])
        torch.testing.assert_close(joint[0, :3], probs[1, :3] / 0.9)

    def test_supports_follow_accept_index_rows(self):
        """Opted-in requests 0 and 2 of 3; padding past the accept run; out-of-support is invalid."""
        draft_token_num = 2
        mask_probs = torch.tensor(
            [
                [0.0, 0.6, 0.3, 0.1],  # req 0, position 0
                [1.0, 0.0, 0.0, 0.0],  # req 0, position 1
                [0.5, 0.5, 0.0, 0.0],  # req 2, position 0
                [0.0, 0.0, 1.0, 0.0],  # req 2, position 1
            ]
        )
        accept_index = torch.tensor([[0, 1], [2, 3], [4, -1]], dtype=torch.int32)
        predict = torch.tensor([2, 0, 9, 9, 3, 7], dtype=torch.int32)
        out = verify_sampling_mask_output(
            mask_req_rows=torch.tensor([0, 2]),
            mask_probs=mask_probs,
            predict=predict,
            accept_index=accept_index,
            draft_token_num=draft_token_num,
            max_tokens=16,
            support_capture_indices=None,
            sync_groups=(),
        )
        self.assertEqual(out.lengths.tolist(), [3, 1, 2, 0])
        self.assertEqual(sorted(out.token_ids[0, :3].tolist()), [1, 2, 3])
        self.assertAlmostEqual(out.selected_logprobs[0].item(), math.log(0.3), places=5)
        self.assertEqual(
            out.statuses.tolist(),
            [SamplingMaskStatus.OK, SamplingMaskStatus.OK]
            + [SamplingMaskStatus.INVALID, SamplingMaskStatus.OK],
        )

    def test_materialize_one_support_per_position(self):
        mask = SimpleNamespace(
            support_logprobs=None,
            token_ids=torch.tensor([[5, 6], [7, 0], [8, 0], [0, 0]], dtype=torch.int32),
            lengths=torch.tensor([2, 1, 1, 0]),
            selected_logprobs=torch.tensor([-0.5, 0.0, -0.25, 0.0]),
            statuses=torch.tensor([0, 0, 0, SamplingMaskStatus.OVERFLOW]),
        )
        output = SimpleNamespace(sampling_mask_output=mask)
        reqs = [
            SimpleNamespace(return_sampling_mask=x, sampling_logprobs_mode="selected")
            for x in (True, False, True)
        ]
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(reqs, output)
        self.assertEqual(
            [
                [row.tolist() for row in output.next_token_sampling_mask_idx[0]],
                *output.next_token_sampling_mask_idx[1:],
            ],
            [[[5, 6], [7]], None, None],
        )
        self.assertEqual(
            [row.tolist() for row in output.next_token_sampling_logprobs[0]],
            [[-0.5], [0.0]],
        )
        self.assertEqual(
            output.next_token_sampling_mask_status,
            [SamplingMaskStatus.OK, None, SamplingMaskStatus.OVERFLOW],
        )

    def test_mixed_selected_and_support_modes_follow_verify_positions(self):
        for greedy in (False, True):
            with self.subTest(greedy=greedy):
                probs = torch.tensor([[0.75, 0.25], [0.0, 1.0]] * 2)
                output = verify_sampling_mask_output(
                    mask_req_rows=torch.tensor([0, 2]),
                    mask_probs=None if greedy else probs,
                    predict=torch.tensor([0, 1, 0, 0, 1, 1]),
                    accept_index=torch.tensor([[0, 1], [2, -1], [4, -1]]),
                    draft_token_num=2,
                    max_tokens=2,
                    support_capture_indices=torch.tensor([1]),
                    sync_groups=(),
                )
                self.assertEqual(output.statuses.tolist(), [0, 0, 0, 0])
                self.assertEqual(
                    output.lengths.tolist(), [1, 1, 1, 0] if greedy else [2, 1, 2, 0]
                )
                reqs = [
                    SimpleNamespace(
                        return_sampling_mask=mode is not None,
                        sampling_logprobs_mode=mode,
                        sampling_mask_rows=SamplingMaskRows(),
                    )
                    for mode in ("selected", None, "support")
                ]
                host = SimpleNamespace(sampling_mask_output=output)
                SchedulerBatchResultProcessor.materialize_sampling_mask_output(
                    reqs, host
                )
                processor = SchedulerBatchResultProcessor.__new__(
                    SchedulerBatchResultProcessor
                )
                processor.add_sampling_mask_return_values(0, reqs[0], host, 2)
                processor.add_sampling_mask_return_values(2, reqs[2], host, 1)
                masks, logprobs = reqs[0].sampling_mask_rows.take().to_lists(False)
                self.assertEqual(masks, [[0], [1]] if greedy else [[0, 1], [1]])
                self.assertAlmostEqual(
                    logprobs[0], 0 if greedy else math.log(0.75), places=6
                )
                self.assertEqual(logprobs[1], 0)
                masks, logprobs = reqs[2].sampling_mask_rows.take().to_lists(True)
                self.assertEqual(masks, [[1]] if greedy else [[0, 1]])
                torch.testing.assert_close(
                    torch.tensor(logprobs),
                    torch.tensor([[0.0]])
                    if greedy
                    else torch.tensor([[0.75, 0.25]]).log(),
                )

    def test_overflow_and_padding_statuses(self):
        output = verify_sampling_mask_output(
            mask_req_rows=torch.tensor([0]),
            mask_probs=torch.tensor([[0.5, 0.5], [1.0, 0.0]]),
            predict=torch.tensor([0, 0]),
            accept_index=torch.tensor([[0, -1]]),
            draft_token_num=2,
            max_tokens=1,
            support_capture_indices=None,
            sync_groups=(),
        )
        self.assertEqual(
            output.statuses.tolist(),
            [SamplingMaskStatus.OVERFLOW, SamplingMaskStatus.OK],
        )
        self.assertEqual(output.lengths.tolist(), [1, 0])

    def test_only_exact_chain_eagle_verify_is_supported(self):
        supported = dict(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            eagle_topk=1,
            use_rejection_sampling=False,
            accept_thresholds=(1.0, 1.0),
            min_p=0.0,
            cuda=True,
            simulate_acceptance=False,
        )
        self.assertIsNone(spec_sampling_mask_unsupported_reason(**supported))
        for field, value in (
            ("spec_algorithm", SpeculativeAlgorithm.NGRAM),
            ("spec_algorithm", SpeculativeAlgorithm.FROZEN_KV_MTP),
            ("eagle_topk", 4),
            ("use_rejection_sampling", True),
            ("accept_thresholds", (0.9, 1.0)),
            ("min_p", 0.05),
            ("cuda", False),
            ("simulate_acceptance", True),
        ):
            with self.subTest(field=field, value=value):
                self.assertIsNotNone(
                    spec_sampling_mask_unsupported_reason(**{**supported, field: value})
                )


if __name__ == "__main__":
    unittest.main()
