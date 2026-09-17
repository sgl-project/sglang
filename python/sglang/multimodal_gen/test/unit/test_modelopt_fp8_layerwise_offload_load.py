# SPDX-License-Identifier: Apache-2.0
"""Serialized ModelOpt FP8 checkpoints must postprocess correctly even when
layerwise offload moves the component back to CPU after loading."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.loader import fsdp_load
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

_IN_FEATURES = 32
_SHARD_OUT = 16


class _FusedFp8Model(nn.Module):
    """Minimal stand-in for a serialized ModelOpt FP8 DiT: one fused linear
    with per-shard scales, plus a cosmos3-style non-checkpoint meta buffer."""

    param_names_mapping = {}
    _fsdp_forward_methods: tuple[str, ...] = ()

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        super().__init__()
        self.qkv = MergedColumnParallelLinear(
            input_size=_IN_FEATURES,
            output_sizes=[_SHARD_OUT, _SHARD_OUT],
            bias=False,
            quant_config=quant_config,
            prefix="qkv",
        )
        self.register_buffer("inv_freq", torch.empty(4), persistent=False)

    def post_load_weights(self) -> None:
        if self.inv_freq.is_meta:
            device = next(self.parameters()).device
            self.register_buffer(
                "inv_freq",
                torch.ones(4, dtype=torch.float32, device=device),
                persistent=False,
            )


def _make_serialized_fp8_checkpoint() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Quantize a reference weight shard-by-shard, as ModelOpt exports do."""
    torch.manual_seed(0)
    weight_ref = torch.randn(2 * _SHARD_OUT, _IN_FEATURES, dtype=torch.float32) * 0.05

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    shard_scales = []
    fp8_shards = []
    for shard in weight_ref.split(_SHARD_OUT, dim=0):
        scale = shard.abs().max() / fp8_max
        shard_scales.append(scale)
        fp8_shards.append((shard / scale).to(torch.float8_e4m3fn))

    state_dict = {
        "qkv.weight": torch.cat(fp8_shards, dim=0),
        "qkv.weight_scale": torch.stack(shard_scales),
        "qkv.input_scale": torch.tensor([0.5, 0.5], dtype=torch.float32),
    }
    return state_dict, weight_ref


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for scaled_fp8_quant")
class TestModelOptFp8LayerwiseOffloadLoad(unittest.TestCase):
    def test_serialized_checkpoint_loads_with_component_starting_on_cpu(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

        # The plan a layerwise-offload component gets when
        # _needs_device_weight_postprocess() returns True: load and postprocess
        # on GPU, then defer the CPU placement.
        load_plan = WeightLoadPlan.for_component(
            checkpoint_load_device=torch.device("cuda"),
            needs_device_weight_postprocess=True,
            component_starts_on_cpu=True,
        )
        self.assertTrue(load_plan.defer_cpu_placement)

        for cutlass_supported in (False, True):
            with self.subTest(cutlass_supported=cutlass_supported):
                state_dict, weight_ref = _make_serialized_fp8_checkpoint()
                checkpoint_weight = state_dict["qkv.weight"].clone()
                checkpoint_scales = state_dict["qkv.weight_scale"].clone()
                expected_max_scale = checkpoint_scales.max()

                with patch(
                    "sglang.multimodal_gen.runtime.layers.quantization."
                    "modelopt_quant.cutlass_fp8_supported",
                    return_value=cutlass_supported,
                ):
                    model = fsdp_load.maybe_load_fsdp_model(
                        model_cls=_FusedFp8Model,
                        init_params={
                            "quant_config": ModelOptFp8Config(
                                is_checkpoint_fp8_serialized=True
                            )
                        },
                        weight_dir_list=[],
                        device=torch.device("cuda"),
                        hsdp_replicate_dim=1,
                        hsdp_shard_dim=1,
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        component_starts_on_cpu=True,
                        weight_load_plan=load_plan,
                        weights_iterator=iter(state_dict.items()),
                    )

                # Both paths rebind the runtime weight transposed. CUTLASS can
                # consume a channelwise scale, so it preserves the checkpoint's
                # FP8 shards; the fallback requantizes them to one max scale.
                weight = model.qkv.weight
                self.assertEqual(weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(tuple(weight.shape), (_IN_FEATURES, 2 * _SHARD_OUT))
                weight_scale = model.qkv.weight_scale.flatten()
                if cutlass_supported:
                    expected_scales = torch.repeat_interleave(
                        checkpoint_scales, _SHARD_OUT
                    )
                    self.assertTrue(torch.equal(weight.t(), checkpoint_weight))
                else:
                    expected_scales = expected_max_scale.expand(weight_scale.numel())
                torch.testing.assert_close(
                    weight_scale,
                    expected_scales,
                    check_device=False,
                )
                torch.testing.assert_close(
                    model.qkv.input_scale.flatten().max(),
                    torch.tensor(0.5),
                    check_device=False,
                )

                # The round trip stays close to the source. Loose on purpose:
                # this guards against garbage (wrong scale or shard order), not
                # FP8 precision.
                dequant = weight.t().float().cpu() * weight_scale.float().cpu().view(
                    -1, 1
                )
                torch.testing.assert_close(
                    dequant,
                    weight_ref,
                    rtol=0.5,
                    atol=float(expected_max_scale) * 8,
                )

                # Layerwise offload contract: the component lands on CPU
                # afterwards, with the non-checkpoint buffer rebuilt.
                self.assertEqual(weight.device.type, "cpu")
                self.assertFalse(model.inv_freq.is_meta)
                self.assertEqual(model.inv_freq.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
