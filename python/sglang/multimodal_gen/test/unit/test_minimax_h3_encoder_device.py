import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.models.encoders import minimax_h3_qwen3vl
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)


class TestMiniMaxH3EncoderDevice(unittest.TestCase):
    """`device` must name the compute side, not the parameter storage side.

    Component offload stores parameters on CPU between uses. Input construction
    must still target the accelerator selected for the forward rather than infer
    the compute device from an offloaded parameter.
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
