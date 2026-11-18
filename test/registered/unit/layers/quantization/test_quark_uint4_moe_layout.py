"""Layout conversion and load checks for Quark packed uint4 MoE weights.

The checkpoint side is packed here in AutoAWQ's nibble order, which the checkpoint's
pack_method "reorder" follows; the packer is written for the test, not taken from the
loader.
"""

import unittest

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.quark.schemes.quark_w4a16_uint4_moe import (
    QuarkW4A16Uint4MoE,
    convert_quark_uint4_tensor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# AutoAWQ's packing, which Quark's pack_method "reorder" follows:
# nibble i of a word holds code _AWQ_NIBBLE_ORDER[i].
_AWQ_NIBBLE_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_words(codes: torch.Tensor) -> torch.Tensor:
    """[rows, cols] 4-bit codes -> [rows, cols // 8] int32, packed as the checkpoint."""
    rows, cols = codes.shape
    nibbles = (codes.long() & 0xF).view(rows, cols // 8, 8)[..., _AWQ_NIBBLE_ORDER]
    words = (nibbles << (4 * torch.arange(8))).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _unpack_kernel_bytes(packed: torch.Tensor, *, axis: int) -> torch.Tensor:
    """Kernel layout (two codes per byte along `axis`, low nibble first) -> codes."""
    low, high = packed & 0xF, packed >> 4
    return torch.stack((low, high), dim=axis + 1).flatten(axis, axis + 1)


class TestConvertQuarkUint4Tensor(CustomTestCase):
    def test_codes_reach_the_kernel_layout_in_checkpoint_order(self):
        g = torch.Generator().manual_seed(0)
        codes = torch.randint(0, 16, (24, 16), generator=g, dtype=torch.int32)
        packed = _pack_words(codes.T.contiguous())

        weight = convert_quark_uint4_tensor(packed, tensor_type="weight")
        self.assertEqual(weight.shape, (24, 8))
        self.assertTrue(torch.equal(_unpack_kernel_bytes(weight, axis=1).int(), codes))
        # Zero points pair rows, not columns: [out // 2, groups].
        zero_point = convert_quark_uint4_tensor(packed, tensor_type="zero_point")
        self.assertEqual(zero_point.shape, (12, 16))
        self.assertTrue(
            torch.equal(_unpack_kernel_bytes(zero_point, axis=0).int(), codes)
        )


NUM_EXPERTS = 2


def _expert_layer(
    *,
    intermediate: int = 192,
    group_size: int = 64,
    tp_size: int = 1,
    tp_rank: int = 0,
    use_presharded_weights: bool = False,
) -> FusedMoE:
    """A FusedMoE with only what weight creation and loading read, for one TP rank."""
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.scheme = QuarkW4A16Uint4MoE(weight_config={"group_size": group_size})
    layer.quant_method = layer.scheme
    layer.quant_config = None
    layer.moe_runner_config = MoeRunnerConfig(activation="silu")
    layer.intermediate_size_per_partition = intermediate // tp_size
    layer.moe_tp_size, layer.moe_tp_rank = tp_size, tp_rank
    layer.num_local_experts = NUM_EXPERTS
    layer.uses_expert_location = False
    layer._expert_storage_rank, layer._has_fused_shared = 0, False
    layer._num_local_routed = layer._num_global_routed = NUM_EXPERTS
    layer.use_presharded_weights = use_presharded_weights
    layer.use_padded_loading = False
    layer.use_flashinfer_trtllm_moe = False
    return layer


def _create_weights(
    layer: FusedMoE,
    *,
    hidden_size: int = 128,
    params_dtype: torch.dtype = torch.bfloat16,
    weight_loader=None,
    **attrs,
) -> None:
    layer.scheme.create_weights(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=hidden_size,
        intermediate_size_per_partition=layer.intermediate_size_per_partition,
        params_dtype=params_dtype,
        weight_loader=weight_loader or layer.weight_loader,
        **attrs,
    )


def _random_projection(
    g: torch.Generator, *, out_features: int, in_features: int, group_size: int
):
    """Checkpoint tensors of one projection and the [out, in] weight they encode."""
    groups = in_features // group_size
    codes = torch.randint(0, 16, (out_features, in_features), generator=g)
    zero_point = torch.randint(0, 16, (out_features, groups), generator=g)
    # Odd multiples of 1/512 below 1/2: eight significant bits, exact in bf16.
    scale = (2 * torch.randint(0, 128, (out_features, groups), generator=g) + 1) / 512
    expand = lambda t: t.repeat_interleave(group_size, dim=1)  # noqa: E731
    tensors = {
        "weight": _pack_words(codes.T.contiguous()),
        "weight_scale": scale.T.contiguous().to(torch.bfloat16),
        "weight_zero_point": _pack_words(zero_point.T.contiguous()),
    }
    return tensors, expand(scale) * (codes - expand(zero_point))


def _loaded_weight(layer: FusedMoE, prefix: str, expert: int) -> torch.Tensor:
    """The [out, in] weight the kernel would compute from the layer's parameters."""
    codes = _unpack_kernel_bytes(getattr(layer, f"{prefix}_weight")[expert], axis=1)
    zero_point = _unpack_kernel_bytes(
        getattr(layer, f"{prefix}_weight_zero_point")[expert], axis=0
    )
    scale = getattr(layer, f"{prefix}_weight_scale")[expert].float()
    expand = lambda t: t.repeat_interleave(layer.group_size, dim=1)  # noqa: E731
    return expand(scale) * (codes.int() - expand(zero_point.int()))


class TestQuarkUint4MoeLoad(CustomTestCase):
    def test_each_tp_rank_gets_its_slice_of_every_expert(self):
        # (hidden, intermediate, checkpoint group, TP size) -> the layer's group size.
        # Two ranks of 96 rows split 64-wide groups once; four ranks of 160 rows
        # split 128-wide groups twice.
        cases = {
            (128, 192, 64, 1): 64,
            (128, 192, 64, 2): 32,
            (256, 640, 128, 4): 32,
        }
        for (hidden, intermediate, group, tp_size), layer_group in cases.items():
            g = torch.Generator().manual_seed(0)
            shapes = {
                "w1": (intermediate, hidden),
                "w3": (intermediate, hidden),
                "w2": (hidden, intermediate),
            }
            checkpoint = {
                (expert, shard_id): _random_projection(
                    g, out_features=shape[0], in_features=shape[1], group_size=group
                )
                for expert in range(NUM_EXPERTS)
                for shard_id, shape in shapes.items()
            }
            # The exported bf16 scales, or the same values as fp32.
            scale_dtype = torch.float32 if tp_size == 2 else torch.bfloat16
            for tp_rank in range(tp_size):
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank, group=group):
                    layer = _expert_layer(
                        intermediate=intermediate,
                        group_size=group,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                    )
                    _create_weights(
                        layer, hidden_size=hidden, params_dtype=torch.float16
                    )
                    self.assertEqual(layer.group_size, layer_group)
                    self.assertEqual(layer.w13_weight_scale.dtype, torch.float16)
                    for (expert, shard_id), (tensors, _) in checkpoint.items():
                        prefix = "w2" if shard_id == "w2" else "w13"
                        for suffix, tensor in tensors.items():
                            if suffix == "weight_scale":
                                tensor = tensor.to(scale_dtype)
                            param = getattr(layer, f"{prefix}_{suffix}")
                            param.weight_loader(
                                param, tensor, f"{prefix}_{suffix}", shard_id, expert
                            )
                    local = intermediate // tp_size
                    rows = slice(tp_rank * local, (tp_rank + 1) * local)
                    for expert in range(NUM_EXPERTS):
                        w1, w3, w2 = (
                            checkpoint[expert, shard_id][1]
                            for shard_id in ("w1", "w3", "w2")
                        )
                        self.assertTrue(
                            torch.equal(
                                _loaded_weight(layer, "w13", expert),
                                torch.cat((w1[rows], w3[rows])),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                _loaded_weight(layer, "w2", expert), w2[:, rows]
                            )
                        )

    def test_a_tensor_of_another_extent_type_or_shard_is_rejected(self):
        layer = _expert_layer()
        _create_weights(layer)
        # down projection, 128 outputs and 3 groups of 64 inputs
        tensors = {
            "w2_weight": torch.zeros(192, 16, dtype=torch.int32),
            "w2_weight_scale": torch.ones(3, 128, dtype=torch.bfloat16),
            "w2_weight_zero_point": torch.zeros(3, 16, dtype=torch.int32),
        }

        def load(name: str, tensor: torch.Tensor, shard_id: str = "w2") -> None:
            param = getattr(layer, name)
            param.weight_loader(param, tensor, name, shard_id, 1)

        for name, tensor in tensors.items():
            load(name, tensor)
            # One row less; for the zero points that is one group where three are
            # expected, which would broadcast into all of them.
            with self.subTest(name=name, wrong="extent"):
                with self.assertRaisesRegex(ValueError, "expected"):
                    load(name, tensor[:-1])
            wrong_dtype = tensor.int() if tensor.is_floating_point() else tensor.float()
            with self.subTest(name=name, wrong="dtype"):
                with self.assertRaisesRegex(ValueError, "expected"):
                    load(name, wrong_dtype)
        # A gate tensor of the right extent under a shard name that is not one.
        with self.assertRaisesRegex(ValueError, "shard_id"):
            load(
                "w13_weight_zero_point",
                torch.zeros(2, 24, dtype=torch.int32),
                shard_id="w13",
            )

    def test_unsupported_layouts_fail_at_construction(self):
        with self.assertRaisesRegex(ValueError, "pre-sharded"):
            _create_weights(_expert_layer(use_presharded_weights=True))
        with self.assertRaisesRegex(ValueError, "expert biases"):
            _create_weights(_expert_layer(), with_bias=True)
        layer = _expert_layer()
        with self.assertRaisesRegex(ValueError, "stacked expert tensors"):
            _create_weights(layer, weight_loader=layer.weight_loader_fused)
        # 160 and 96 are multiples of 32 but not of the checkpoint's group size.
        with self.assertRaisesRegex(ValueError, "checkpoint group size 64"):
            _create_weights(_expert_layer(), hidden_size=160)
        with self.assertRaisesRegex(ValueError, "checkpoint group size 64"):
            _create_weights(_expert_layer(intermediate=96))
        # Four ranks get 48 rows each, which no group of 32 or more divides.
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            _create_weights(_expert_layer(tp_size=4))
        for config in (
            MoeRunnerConfig(activation="gelu"),
            MoeRunnerConfig(activation="silu", is_gated=False),
            MoeRunnerConfig(activation="silu", gemm1_alpha=1.702),
        ):
            with self.assertRaisesRegex(ValueError, "plain gated SiLU"):
                layer.scheme.create_moe_runner(layer, config)


if __name__ == "__main__":
    unittest.main()
