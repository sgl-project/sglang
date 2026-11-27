# SPDX-License-Identifier: Apache-2.0
"""Online quantization must not materialize a CPU-offloaded DiT on CUDA."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config
from sglang.multimodal_gen.runtime.loader import fsdp_load
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    process_model_weights_after_loading,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


class _TwoLayerModel(nn.Module):
    param_names_mapping = {}
    _fsdp_forward_methods: tuple[str, ...] = ()

    def __init__(self, quant_config):
        super().__init__()
        self.layers = nn.ModuleList(
            ReplicatedLinear(128, 128, bias=False, quant_config=quant_config)
            for _ in range(2)
        )
        self.register_buffer("inv_freq", torch.empty(4), persistent=False)

    def post_load_weights(self):
        self.inv_freq = torch.ones(4, device=self.layers[0].weight.device)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA quantization")
class TestOnlineFp8LayerwiseLoad(unittest.TestCase):
    def test_bounded_loading_matches_resident_quantized_weights(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        generator = torch.Generator().manual_seed(7)
        checkpoint = {
            f"layers.{i}.weight": torch.randn(
                128, 128, dtype=torch.bfloat16, generator=generator
            )
            for i in range(2)
        }
        device = torch.device("cuda:0")
        for config_cls in (Fp8Config, MXFP8Config):
            with self.subTest(quantization=config_cls.__name__):

                def load(plan, starts_on_cpu):
                    return fsdp_load.maybe_load_fsdp_model(
                        model_cls=_TwoLayerModel,
                        init_params={"quant_config": config_cls()},
                        weight_dir_list=[],
                        device=device,
                        hsdp_replicate_dim=1,
                        hsdp_shard_dim=1,
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        component_starts_on_cpu=starts_on_cpu,
                        weight_load_plan=plan,
                        weights_iterator=iter(checkpoint.items()),
                    )

                reference = load(WeightLoadPlan(checkpoint_load_device=device), False)
                expected = {
                    name: value.detach().cpu().contiguous().view(torch.uint8)
                    for name, value in reference.state_dict().items()
                }
                del reference
                visited = []

                def bounded_postprocess(model, process_device):
                    self.assertEqual(process_device, device)
                    for layer in model.layers:
                        original = layer.quant_method.process_weights_after_loading

                        def check(current, original=original):
                            self.assertEqual(current.weight.device, device)
                            for other in model.layers:
                                if other is not current:
                                    self.assertEqual(other.weight.device.type, "cpu")
                            visited.append(current)
                            original(current)

                        layer.quant_method.process_weights_after_loading = check
                    return process_model_weights_after_loading(model, process_device)

                plan = WeightLoadPlan.for_component(
                    checkpoint_load_device=torch.device("cpu"),
                    needs_device_weight_postprocess=True,
                    component_starts_on_cpu=True,
                    layerwise_postprocess_device=device,
                )
                with patch.object(
                    fsdp_load,
                    "process_model_weights_after_loading",
                    bounded_postprocess,
                ):
                    model = load(plan, True)
                self.assertEqual(len(visited), 2)
                for tensor in (*model.parameters(), *model.buffers()):
                    self.assertEqual(tensor.device.type, "cpu")
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value.detach().contiguous().view(torch.uint8),
                        expected[name],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
