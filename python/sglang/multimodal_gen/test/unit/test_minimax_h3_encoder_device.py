import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.models.encoders import minimax_h3_qwen3vl
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)


class TestMiniMaxH3EncoderDevice(unittest.TestCase):
    """`device` must name the compute side, not the parameter storage side.

    `--text-encoder-cpu-offload` loads this encoder under an FSDP CPU offload
    policy: the sharded parameters sit on CPU and are all-gathered to the
    accelerator for the forward. Reporting the parameter device there sent
    `encode_ids` to build `input_ids`/`attention_mask`/`position_ids` on CPU
    while the forward ran on the accelerator, and the rope matmul died with
    "Expected all tensors to be on the same device ... mat2 is on cpu".
    """

    def _encoder_with_param_on(self, device: torch.device) -> MiniMaxH3Qwen3VLEncoder:
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.register_parameter(
            "offloaded", torch.nn.Parameter(torch.zeros(1, device=device))
        )
        return encoder

    def test_device_ignores_cpu_offloaded_parameters(self):
        encoder = self._encoder_with_param_on(torch.device("cpu"))
        compute_device = torch.device("cuda", 3)

        with mock.patch.object(
            minimax_h3_qwen3vl, "get_local_torch_device", return_value=compute_device
        ):
            self.assertEqual(encoder.device, compute_device)

        # The parameter really is on CPU: the property is not just echoing it back.
        self.assertEqual(next(encoder.parameters()).device.type, "cpu")

    def test_device_follows_local_device_on_cpu_only_platforms(self):
        encoder = self._encoder_with_param_on(torch.device("cpu"))
        cpu = torch.device("cpu")

        with mock.patch.object(
            minimax_h3_qwen3vl, "get_local_torch_device", return_value=cpu
        ):
            self.assertEqual(encoder.device, cpu)


if __name__ == "__main__":
    unittest.main()
