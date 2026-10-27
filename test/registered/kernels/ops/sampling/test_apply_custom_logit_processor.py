import dataclasses
import unittest

import torch

from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.sampling.sampling_batch_info import ProcessorEntry, SamplingBatchInfo
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestApplyCustomLogitProcessorCUDA(CustomTestCase):
    def _make_info(self):
        def processor(logits, params):
            for row, param in zip(logits, params, strict=True):
                row.narrow(0, param["token_id"], 1).fill_(-float("inf"))
            return logits

        return SamplingBatchInfo(
            temperatures=torch.ones(3, 1, device="cuda"),
            top_ps=torch.ones(3, device="cuda"),
            top_ks=torch.zeros(3, dtype=torch.int32, device="cuda"),
            min_ps=torch.zeros(3, device="cuda"),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=4,
            has_custom_logit_processor=True,
            custom_params=[{"token_id": 1}, None, {"token_id": 2}],
            custom_logit_processor={
                0: ProcessorEntry(
                    processor=processor,
                    rows=[0, 2],
                    indices=torch.tensor([0, 2], device="cuda"),
                )
            },
            device="cuda",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cached_decode_does_not_read_row_indices_back_to_cpu(self):
        info = self._make_info()
        for width in (1, 3):
            with self.subTest(width=width):
                logits = torch.zeros(3 * width, 4, device="cuda")
                apply_custom_logit_processor(logits, info, width)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    apply_custom_logit_processor(logits, info, width)
                names = {event.key for event in profile.key_averages()}
                self.assertNotIn("aten::nonzero", names)
                self.assertNotIn("aten::_local_scalar_dense", names)
                expected = torch.zeros(3, 4)
                expected[0, 1] = -float("inf")
                expected[2, 2] = -float("inf")
                self.assertTrue(
                    torch.equal(logits.cpu(), expected.repeat_interleave(width, dim=0))
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_casts_processor_result_to_logits_dtype(self):
        def processor(logits, params):
            return logits.float() + 0.1

        info = self._make_info()
        info.custom_logit_processor = {
            0: dataclasses.replace(info.custom_logit_processor[0], processor=processor)
        }
        for width in (1, 3):
            with self.subTest(width=width):
                logits = torch.zeros(3 * width, 4, dtype=torch.bfloat16, device="cuda")
                apply_custom_logit_processor(logits, info, width)
                expected = torch.zeros(3, 4, dtype=torch.bfloat16)
                expected[0] = 0.1
                expected[2] = 0.1
                self.assertEqual(logits.dtype, torch.bfloat16)
                self.assertTrue(
                    torch.equal(logits.cpu(), expected.repeat_interleave(width, dim=0))
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_filter_does_not_read_processor_membership_back_to_cpu(self):
        for keep in ([2, 1], [1]):
            with self.subTest(keep=keep):
                info = self._make_info()
                self._make_info()._filter_batch_custom_logit_processor(keep)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    info._filter_batch_custom_logit_processor(keep)
                names = {event.key for event in profile.key_averages()}
                self.assertNotIn("aten::nonzero", names)
                self.assertNotIn("aten::_local_scalar_dense", names)
                if keep == [2, 1]:
                    entry = info.custom_logit_processor[0]
                    self.assertEqual(entry.rows, [0])
                    self.assertEqual(entry.indices.tolist(), [0])
                    self.assertEqual(info.custom_params, [{"token_id": 2}, None])
                    self.assertEqual(set(info.custom_logit_processor), {0})
                else:
                    self.assertIsNone(info.custom_logit_processor)
                    self.assertFalse(info.has_custom_logit_processor)


if __name__ == "__main__":
    unittest.main()
