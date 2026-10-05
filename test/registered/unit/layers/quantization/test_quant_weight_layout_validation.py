"""Block quantization checks use the weight owner's constructed partition."""

import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.blockwise_int8 import (
    BlockInt8Config,
    BlockInt8LinearMethod,
    BlockInt8MoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

LINEAR_METHODS = {"int8": BlockInt8LinearMethod, "fp8": Fp8LinearMethod}
MOE_METHODS = {
    "int8": BlockInt8MoEMethod,
    "fp8": Fp8MoEMethod,
    "compressed": CompressedTensorsW8A8Fp8MoE,
}


def quant_config(kind, block=(128, 128)):
    if kind == "int8":
        return BlockInt8Config(
            is_checkpoint_int8_serialized=True, weight_block_size=list(block)
        )
    if kind == "fp8":
        return Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=list(block)
        )
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "ignore": [],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "block",
                        "block_structure": list(block),
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


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return get_parallel().override(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


@contextmanager
def callback_scope(method, changed):
    original = method.create_weights

    def create_weights(self, *args, **kwargs):
        with loading_scope(changed):
            return original(self, *args, **kwargs)

    with patch.object(method, "create_weights", create_weights):
        yield


def build_linear(kind, layout, group="tp", *, changed=False, invalid=False):
    config = quant_config(kind)
    kwargs = dict(
        quant_config=config,
        params_dtype=torch.bfloat16,
        bias=False,
        parallel_group=group,
    )
    with callback_scope(LINEAR_METHODS[kind], changed):
        if layout == "column":
            return ColumnParallelLinear(256, 384 if invalid else 1024, **kwargs)
        if layout == "row":
            return RowParallelLinear(384 if invalid else 1024, 256, **kwargs)
        if layout == "merged":
            return MergedColumnParallelLinear(256, [512, 512], **kwargs)
        if layout == "qkv":
            return QKVParallelLinear(256, 128, 8, 2, **kwargs)
        kwargs.pop("parallel_group")
        return ReplicatedLinear(256, 384, **kwargs)


def build_moe(kind, *, changed=False, invalid=False):
    size = get_parallel().moe_tp_size
    block = (32, 64) if invalid else (128, 128)
    intermediate = (96 if invalid else 256) * size
    with callback_scope(MOE_METHODS[kind], changed):
        return FusedMoE(
            4,
            256,
            intermediate,
            0,
            top_k=2,
            params_dtype=torch.bfloat16,
            quant_config=quant_config(kind, block),
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestQuantWeightLayoutValidation(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def topology(self, rank=0, dp=1, ep=1):
        reset_context()
        publish(
            ServerArgs(
                model_path="dummy",
                device="cuda",
                tp_size=4,
                attn_dp_size=dp,
                ep_size=ep,
                moe_runner_backend="triton",
                fp8_gemm_runner_backend="triton",
            ),
            role="test",
            ranks=SpawnRanks(world_rank=rank),
        )

    def check_shapes(self, changed):
        with torch.inference_mode(), torch.device("cuda"):
            for rank in range(4):
                for dp in (1, 2):
                    self.topology(rank, dp)
                    for kind in LINEAR_METHODS:
                        for group in ("tp", "attn_tp", "replicated"):
                            for layout in (
                                "column",
                                "row",
                                "merged",
                                "qkv",
                                "replicated",
                            ):
                                layer = build_linear(
                                    kind, layout, group, changed=changed
                                )
                                scale = layer.weight_scale_inv
                                expected = (
                                    (layer.weight.shape[0] + 127) // 128,
                                    (layer.weight.shape[1] + 127) // 128,
                                )
                                self.assertEqual(tuple(scale.shape), expected)
                                self.assertEqual(
                                    layer.weight.dtype,
                                    torch.int8
                                    if kind == "int8"
                                    else torch.float8_e4m3fn,
                                )
                for ep in (1, 2, 4):
                    self.topology(rank, ep=ep)
                    for kind in MOE_METHODS:
                        layer = build_moe(kind, changed=changed)
                        self.assertEqual(layer.w13_weight.shape, (4 // ep, 512, 256))
                        self.assertEqual(layer.w2_weight.shape, (4 // ep, 256, 256))

    def test_native_parameter_shapes(self):
        self.check_shapes(False)

    def test_native_parameter_shapes_after_scope_change(self):
        self.check_shapes(True)

    def test_attention_partition_alignment_is_checked(self):
        self.topology(dp=2)
        for kind in LINEAR_METHODS:
            for layout in ("column", "row"):
                for changed in (False, True):
                    with self.assertRaisesRegex(ValueError, "not divisible"):
                        build_linear(
                            kind, layout, "attn_tp", changed=changed, invalid=True
                        )

    def test_explicit_fp8_shape_check_skip_is_preserved(self):
        self.topology(dp=2)
        with torch.device("cuda"):
            layer = ColumnParallelLinear(
                256,
                384,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=quant_config("fp8"),
                parallel_group="attn_tp",
                skip_block_quant_check=True,
            )
            self.assertEqual(tuple(layer.weight.shape), (192, 256))
            self.assertEqual(tuple(layer.weight_scale_inv.shape), (2, 2))

    def test_moe_partition_alignment_survives_scope_change(self):
        with torch.device("cuda"):
            for ep in (1, 2):
                self.topology(ep=ep)
                for kind in MOE_METHODS:
                    with self.assertRaisesRegex(ValueError, "not divisible"):
                        build_moe(kind, changed=True, invalid=True)


if __name__ == "__main__":
    unittest.main()
