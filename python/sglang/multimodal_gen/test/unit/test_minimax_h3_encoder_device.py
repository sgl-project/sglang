import unittest
from types import SimpleNamespace
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

    def test_vision_attention_kernels_are_built_once_on_the_compute_device(self):
        encoder = self._encoder_with_param_on(torch.device("cpu"))
        visual = mock.Mock(dtype=torch.bfloat16, device=torch.device("cpu"))
        visual.patch_embed = SimpleNamespace(
            in_channels=3, temporal_patch_size=2, patch_size=16
        )
        encoder.model = SimpleNamespace(visual=visual)

        # CPU-offloaded tower: nothing to build yet
        with mock.patch.object(
            minimax_h3_qwen3vl,
            "get_local_torch_device",
            return_value=torch.device("cuda", 0),
        ):
            encoder.build_vision_attention_kernels()
        visual.assert_not_called()

        with mock.patch.object(
            minimax_h3_qwen3vl, "get_local_torch_device", return_value=visual.device
        ):
            encoder.build_vision_attention_kernels()
            encoder.build_vision_attention_kernels()
        shapes = [
            (tuple(call.args[0].shape), call.kwargs["grid_thw"].tolist())
            for call in visual.call_args_list
        ]
        self.assertEqual(
            shapes,
            [
                ((t * h * w, 1536), [[t, h, w]])
                for t, h, w in minimax_h3_qwen3vl._VISION_ATTENTION_KERNEL_GRIDS
            ],
        )


if __name__ == "__main__":
    unittest.main()
