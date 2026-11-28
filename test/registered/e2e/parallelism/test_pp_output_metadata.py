"""PP metadata must be ready for CPU serialization before switching streams."""

import pickle
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed.parallel_state import _split_tensor_dict
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestPPOutputMetadata(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_nested_cuda_views_are_ready_before_communication(self):
        forward = torch.cuda.Stream()
        comm = torch.cuda.Stream()
        with torch.cuda.stream(forward):
            ids = torch.tensor([7], device="cuda")
            values = torch.arange(100000, device="cuda", dtype=torch.float32)
            view = values.view(10000, 10)[5:7, :2]
            result = GenerationBatchResult(
                next_token_ids=ids,
                logits_output=LogitsProcessorOutput(
                    next_token_logits=None,
                    next_token_token_ids_logprobs_val=[7, view],
                ),
            )
            wire = SchedulerPPMixin()._pp_prepare_tensor_dict(
                result, SimpleNamespace(return_logprob=True)
            )
            values.zero_()
        with torch.cuda.stream(comm):
            metadata, tensors = _split_tensor_dict(wire)
            self.assertIs(tensors[0], ids)
            self.assertEqual(wire["next_token_token_ids_logprobs_val"][0], 7)
            frozen = wire["next_token_token_ids_logprobs_val"][1]
            self.assertEqual(frozen.device.type, "cpu")
            self.assertEqual(frozen.shape, (2, 2))
            self.assertEqual(frozen.untyped_storage().nbytes(), 16)
            payload = pickle.dumps(metadata)
            restored = dict(pickle.loads(payload))  # Locally generated test metadata.
            self.assertEqual(
                restored["next_token_token_ids_logprobs_val"][1].tolist(),
                [[50, 51], [60, 61]],
            )
        forward.synchronize()


if __name__ == "__main__":
    unittest.main()
