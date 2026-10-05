"""Deferred block-FP8 parameter creation uses the actual layer's partition."""

import unittest

import torch

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def topology(rank=0, dp=1):
    reset_context()
    publish(
        ServerArgs(
            model_path="dummy",
            device="cuda",
            tp_size=4,
            attn_dp_size=dp,
            fp8_gemm_runner_backend="triton",
        ),
        role="test",
        ranks=SpawnRanks(world_rank=rank),
    )


def config():
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "block",
                        "block_structure": [128, 128],
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                }
            },
        }
    )


def build(kind, group="tp", invalid=False):
    # Build the native owner first; its real scheme may create weights later.
    kwargs = dict(bias=False, params_dtype=torch.bfloat16)
    with torch.device("cuda"):
        if kind == "replicated":
            layer = ReplicatedLinear(256, 384, **kwargs)
        elif kind == "row":
            layer = RowParallelLinear(
                384 if invalid else 1024, 256, parallel_group=group, **kwargs
            )
        elif kind == "column":
            layer = ColumnParallelLinear(
                256, 384 if invalid else 1024, parallel_group=group, **kwargs
            )
        elif kind == "merged":
            layer = MergedColumnParallelLinear(
                256, [512, 512], parallel_group=group, **kwargs
            )
        elif kind == "qkv":
            layer = QKVParallelLinear(256, 128, 8, 8, parallel_group=group, **kwargs)
        else:
            layer = MergedColumnParallelLinear(
                256,
                [96, 128] if invalid else [128, 96],
                parallel_group="replicated",
                **kwargs,
            )
    layer.quant_method = config().get_quant_method(layer, prefix="projection")
    return layer


def create(layer):
    input_partition = layer.weight.shape[-1]
    parts = getattr(layer, "output_partition_sizes", [layer.output_size])
    with torch.device("cuda"):
        layer.quant_method.create_weights(
            layer,
            input_size_per_partition=input_partition,
            output_partition_sizes=parts,
            input_size=layer.input_size,
            output_size=layer.output_size,
            params_dtype=layer.params_dtype,
            weight_loader=(
                layer.weight_loader
                if isinstance(layer, ReplicatedLinear)
                else layer.weight_loader_v2
            ),
        )
    return layer


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFp8BlockLayoutDefaults(CustomTestCase):
    def tearDown(self):
        reset_context()

    def test_replicated_scheme_creation_without_published_context(self):
        topology(rank=3, dp=2)
        layer = build("replicated")
        reset_context()
        create(layer)
        self.assertEqual(layer.weight.shape, (384, 256))
        self.assertEqual(layer.weight_scale.shape, (3, 2))
        codes = (torch.arange(384 * 256, device="cuda") % 7 - 3).reshape(384, 256)
        codes = codes.to(torch.float8_e4m3fn)
        scales = torch.arange(6, device="cuda").reshape(3, 2).float() + 1
        layer.weight.weight_loader(layer.weight, codes)
        layer.weight_scale.weight_loader(layer.weight_scale, scales)
        torch.testing.assert_close(
            layer.weight.view(torch.uint8), codes.view(torch.uint8)
        )
        torch.testing.assert_close(layer.weight_scale, scales)

    def test_sharded_schemes_keep_owner_shapes_after_context_reset(self):
        for group in ("tp", "attn_tp", "replicated"):
            for kind in ("column", "row", "merged", "qkv"):
                with self.subTest(group=group, kind=kind):
                    topology(rank=3, dp=2)
                    layer = build(kind, group)
                    shape = layer.weight.shape
                    reset_context()
                    create(layer)
                    self.assertEqual(layer.weight.shape, shape)
                    self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)

    def test_partition_rejections_and_last_merged_tail_are_preserved(self):
        for kind in ("column", "row", "tail"):
            with self.subTest(kind=kind):
                topology(rank=3)
                layer = build(kind, invalid=True)
                before = layer.weight
                reset_context()
                with self.assertRaisesRegex(ValueError, "is not divisible"):
                    create(layer)
                self.assertIs(layer.weight, before)
        topology(rank=3)
        layer = build("tail")
        reset_context()
        create(layer)
        self.assertEqual(layer.weight.shape, (224, 256))


if __name__ == "__main__":
    unittest.main()
