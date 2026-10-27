"""Exercise shuffled-weight metadata through real OffloaderV2 CPU onloads."""

import unittest

import torch
from torch import nn

from sglang.srt.utils.offloader import _hook_module_forward_raw, _ModuleOffloader
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class WeightConsumer(nn.Module):
    def __init__(self, is_shuffled):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4),
            requires_grad=False,
        )
        if is_shuffled is not None:
            self.weight.is_shuffled = is_shuffled

    def forward(self, inputs):
        return self.weight @ inputs, getattr(self.weight, "is_shuffled", False)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA or ROCm device")
class TestShuffledWeightOnload(unittest.TestCase):
    def test_onload_forward_and_host_parameter_restoration(self):
        for is_shuffled in (True, False, None):
            with self.subTest(is_shuffled=is_shuffled):
                module = WeightConsumer(is_shuffled)
                inputs = torch.arange(4, dtype=torch.float32, device="cuda")
                expected, _ = module(inputs)
                host_parameter = module.weight
                module_offloader = _ModuleOffloader(
                    mode="cpu",
                    module=module,
                    alt_stream=torch.cuda.Stream(),
                    whitelist_param_names=["weight"],
                    resident_parameter_ids=set(),
                )
                self.assertEqual(host_parameter.device.type, "cpu")
                self.assertTrue(host_parameter.is_pinned())
                _hook_module_forward_raw(
                    module,
                    on_forward_end=module_offloader.offload,
                    get_parameter_and_buffer_dicts=(
                        module_offloader.wait_and_get_device_tensors
                    ),
                )

                for _ in range(2):
                    module_offloader.start_onload()
                    output, observed_tag = module(inputs)

                    self.assertEqual(output.device.type, "cuda")
                    self.assertEqual(observed_tag, bool(is_shuffled))
                    torch.testing.assert_close(output, expected)
                    self.assertIs(module.weight, host_parameter)
                    self.assertEqual(module.weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
