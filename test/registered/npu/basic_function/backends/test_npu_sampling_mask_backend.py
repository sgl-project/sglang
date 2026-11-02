import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ascend.sampling_mask_reference import (
    assert_distribution,
    reference_distribution,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=60, suite="nightly-1-npu-a3", nightly=True)


@unittest.skipUnless(sampler_module.is_npu(), "Requires an Ascend NPU")
class TestAscendSamplingMaskKernel(CustomTestCase):
    def test_mask_and_logprobs_match_cpu_reference(self):
        """Real fused/fallback kernels, packing and scheduler vs a CPU oracle."""
        sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(sampler)
        sampler.use_ascend_backend = True
        sampler.sampling_mask_max_tokens = 2048
        sampler.tp_sync_group = sampler.cp_sync_group = None
        modes = ["support", None, "selected", "support"]
        reqs = [
            SimpleNamespace(
                return_sampling_mask=m is not None, sampling_logprobs_mode=m
            )
            for m in modes
        ]
        cases = [
            (False, 4, 1.0, 0.0),
            (False, 6, 0.72, 0.0),
            (False, 6, 1.0, 0.35),
            (False, 6, 0.72, 0.35),
            (False, 1025, 0.72, 0.0),
            (True, 4, 1.0, 0.0),
            (True, 6, 0.72, 0.0),
            (True, 2, 1.0, 0.0),
        ]
        tolerances = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 3e-2}
        for dtype, (fused, k, p, m) in product(tolerances, cases):
            ties = k == 2
            with self.subTest(dtype=dtype, fused=fused, k=k, p=p, min_p=m):
                values = (
                    torch.tensor([3.0, 2.0, 2.0, 0.0])
                    if ties
                    else torch.cat(
                        [
                            torch.tensor([0.2, 3.0, 1.7, -1.3, 2.3, 0.8, -0.4, 1.1]),
                            torch.linspace(-8, -16, 2040),
                        ]
                    )
                )
                logits = torch.stack([values.roll(i * 17) for i in range(4)]).to(
                    device="npu", dtype=dtype
                )
                info = SimpleNamespace(
                    sampling_seed=None,
                    temperatures=torch.ones(4, 1, device="npu", dtype=dtype),
                    top_ks=torch.full((4,), k, device="npu", dtype=torch.int32),
                    top_ps=torch.full((4,), p, device="npu", dtype=dtype),
                    min_ps=torch.full((4,), m, device="npu", dtype=dtype),
                    need_min_p_sampling=m > 0,
                    npu_top_k_top_p_eligible=fused,
                    sampling_mask_batch_indices=torch.tensor([0, 2, 3], device="npu"),
                )
                sampled, _, capture = sampler._forward_ascend_backend(
                    logits.clone(),
                    info,
                    False,
                    False,
                    torch.zeros(4, device="npu", dtype=torch.int64),
                )
                self.assertEqual(capture.weights.dtype, dtype)
                packed = sampler._build_sampling_mask_output(
                    sampled, capture, torch.tensor([0, 2], device="npu")
                )
                self.assertEqual(packed.selected_logprobs.dtype, torch.float32)
                self.assertEqual(packed.support_logprobs.dtype, torch.float32)
                self.assertEqual(packed.statuses.cpu().tolist(), [0, 0, 0])
                packed.map_device_tensors(lambda tensor: tensor.cpu())
                output = LogitsProcessorOutput(
                    next_token_logits=None, sampling_mask_output=packed
                )
                SchedulerBatchResultProcessor.materialize_sampling_mask_output(
                    reqs, output
                )
                self.assertIsNone(output.next_token_sampling_mask_idx[1])
                self.assertIsNone(output.next_token_sampling_logprobs[1])
                # Quantize first, then independently evaluate on CPU in FP64.
                raw = logits.cpu().double().log_softmax(-1)
                for row in (0, 2, 3):
                    expected = reference_distribution(
                        dict(enumerate(raw[row].tolist())),
                        k,
                        float(info.top_ps[0]),
                        float(info.min_ps[0]),
                        cutoff_ties=fused,
                    )
                    if ties and row == 0:
                        self.assertEqual(set(expected), {0, 1, 2})
                    actual = output.next_token_sampling_logprobs[row]
                    actual = actual.item() if modes[row] == "selected" else actual
                    assert_distribution(
                        self,
                        int(sampled[row]),
                        output.next_token_sampling_mask_idx[row],
                        actual,
                        expected,
                        modes[row],
                        tolerances[dtype],
                    )


if __name__ == "__main__":
    unittest.main()
