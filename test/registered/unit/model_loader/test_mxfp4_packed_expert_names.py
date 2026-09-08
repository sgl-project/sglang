"""CPU contracts for compressed-tensors MXFP4 expert checkpoint names."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import TestCase, mock

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _normalize_mxfp4_packed_expert_weight_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _quant_config(*, name="compressed_tensors", quant_format="mxfp4-pack-quantized"):
    return SimpleNamespace(
        get_name=lambda: name,
        quant_format=quant_format,
    )


GLM53_DATAFREE_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "config_groups": {
        "config_group_0": {
            "format": "mxfp4-pack-quantized",
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        }
    },
    "ignore": [],
}


class _DummyFusedMoE:
    pass


def _pattern(shape, offset, modulus=256):
    return (
        torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64)
        .add(offset)
        .remainder(modulus)
        .to(torch.uint8)
        .reshape(shape)
    )


def _make_expert_checkpoint(num_experts, hidden_size, intermediate_size):
    checkpoint = {}
    for expert_id in range(num_experts):
        prefix = f"model.layers.3.mlp.experts.{expert_id}."
        for projection, shape, offset in (
            (
                "gate_proj",
                (intermediate_size, hidden_size // 2),
                17 + expert_id * 41,
            ),
            (
                "up_proj",
                (intermediate_size, hidden_size // 2),
                71 + expert_id * 43,
            ),
            (
                "down_proj",
                (hidden_size, intermediate_size // 2),
                131 + expert_id * 47,
            ),
        ):
            checkpoint[f"{prefix}{projection}.weight_packed"] = _pattern(
                shape, offset
            )

        for projection, shape, offset in (
            (
                "gate_proj",
                (intermediate_size, hidden_size // 32),
                expert_id,
            ),
            (
                "up_proj",
                (intermediate_size, hidden_size // 32),
                3 + expert_id,
            ),
            (
                "down_proj",
                (hidden_size, intermediate_size // 32),
                7 + expert_id,
            ),
        ):
            # Keep exponents in a compact finite range while retaining distinct
            # bytes for every projection and expert.
            checkpoint[f"{prefix}{projection}.weight_scale"] = _pattern(
                shape, offset, modulus=15
            ).add_(120)

    return checkpoint


def _make_candidate_loader(
    *, ep_rank, tp_rank, num_experts, hidden_size, intermediate_size
):
    num_local_experts = num_experts // 2
    intermediate_per_tp = intermediate_size // 2
    method = object.__new__(Mxfp4MoEMethod)
    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.quant_method = method
    layer.scheme = None
    layer.quant_config = _quant_config()
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.moe_tp_rank = tp_rank
    layer.moe_tp_size = 2
    layer.moe_ep_size = 2
    layer._expert_storage_rank = ep_rank
    layer._num_local_routed = num_local_experts
    layer._num_global_routed = num_experts
    layer._has_fused_shared = False
    layer.num_local_experts = num_local_experts
    layer.num_fused_shared_experts = 0
    layer.use_flashinfer_trtllm_moe = False
    layer.use_triton_kernels = False
    layer.use_presharded_weights = False
    # Exercise the production GPU-style exact-shape loader without making the
    # test depend on the platform selected by the CPU CI process.
    layer.__dict__["use_padded_loading"] = False

    params = {
        "model.layers.3.mlp.experts.w13_weight": torch.nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_per_tp,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
        "model.layers.3.mlp.experts.w13_weight_scale": torch.nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_per_tp,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
        "model.layers.3.mlp.experts.w2_weight": torch.nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_per_tp // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
        "model.layers.3.mlp.experts.w2_weight_scale": torch.nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_per_tp // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    }
    params["model.layers.3.mlp.experts.w13_weight_scale"].quant_method = "group"
    params["model.layers.3.mlp.experts.w2_weight_scale"].quant_method = "group"
    for qualified_name, param in params.items():
        setattr(layer, qualified_name.rsplit(".", 1)[-1], param)
    return layer, params


def _load_candidate_checkpoint(layer, params, checkpoint, num_experts):
    mappings = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )
    matched = set()
    with mock.patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer."
        "get_global_expert_location_metadata",
        return_value=None,
    ):
        for checkpoint_name, loaded_weight in checkpoint.items():
            for param_name, weight_name, expert_id, shard_id in mappings:
                if weight_name not in checkpoint_name:
                    continue
                normalized = _normalize_mxfp4_packed_expert_weight_name(
                    checkpoint_name, layer.quant_config
                )
                target_name = normalized.replace(weight_name, param_name)
                if target_name not in params:
                    continue
                layer.weight_loader(
                    params[target_name],
                    loaded_weight,
                    target_name,
                    shard_id,
                    expert_id,
                )
                matched.add(checkpoint_name)
                break

    if matched != checkpoint.keys():
        raise AssertionError(
            f"candidate loader missed checkpoint keys: {checkpoint.keys() - matched}"
        )


def _baseline_reference_shards(
    checkpoint, *, ep_rank, tp_rank, num_experts, hidden_size, intermediate_size
):
    """Independent reference for the deleted dedicated W4A8 scheme."""
    num_local_experts = num_experts // 2
    intermediate_per_tp = intermediate_size // 2
    result = {
        "w13_weight": torch.zeros(
            num_local_experts,
            2 * intermediate_per_tp,
            hidden_size // 2,
            dtype=torch.uint8,
        ),
        "w13_weight_scale": torch.zeros(
            num_local_experts,
            2 * intermediate_per_tp,
            hidden_size // 32,
            dtype=torch.uint8,
        ),
        "w2_weight": torch.zeros(
            num_local_experts,
            hidden_size,
            intermediate_per_tp // 2,
            dtype=torch.uint8,
        ),
        "w2_weight_scale": torch.zeros(
            num_local_experts,
            hidden_size,
            intermediate_per_tp // 32,
            dtype=torch.uint8,
        ),
    }
    first_expert = ep_rank * num_local_experts
    for local_expert, expert_id in enumerate(
        range(first_expert, first_expert + num_local_experts)
    ):
        prefix = f"model.layers.3.mlp.experts.{expert_id}."
        row_start = tp_rank * intermediate_per_tp
        packed_col_start = tp_rank * (intermediate_per_tp // 2)
        scale_col_start = tp_rank * (intermediate_per_tp // 32)
        for projection, row_offset in (("gate_proj", 0), ("up_proj", 1)):
            result["w13_weight"][
                local_expert,
                row_offset * intermediate_per_tp : (row_offset + 1)
                * intermediate_per_tp,
            ].copy_(
                checkpoint[f"{prefix}{projection}.weight_packed"].narrow(
                    0, row_start, intermediate_per_tp
                )
            )
            result["w13_weight_scale"][
                local_expert,
                row_offset * intermediate_per_tp : (row_offset + 1)
                * intermediate_per_tp,
            ].copy_(
                checkpoint[f"{prefix}{projection}.weight_scale"].narrow(
                    0, row_start, intermediate_per_tp
                )
            )
        result["w2_weight"][local_expert].copy_(
            checkpoint[f"{prefix}down_proj.weight_packed"].narrow(
                1, packed_col_start, intermediate_per_tp // 2
            )
        )
        result["w2_weight_scale"][local_expert].copy_(
            checkpoint[f"{prefix}down_proj.weight_scale"].narrow(
                1, scale_col_start, intermediate_per_tp // 32
            )
        )
    return result


def _baseline_reference_mega_weights(reference):
    """Encode the baseline SM90 MegaMoE layout without production helpers."""

    def interleave_gate_up(tensor, granularity=8):
        experts, rows, *tail = tensor.shape
        half = rows // 2
        gate = tensor[:, :half].reshape(
            experts, half // granularity, granularity, *tail
        )
        up = tensor[:, half:].reshape(
            experts, half // granularity, granularity, *tail
        )
        return torch.stack((gate, up), dim=2).reshape(experts, rows, *tail)

    def pack_ue8m0(scale_bytes):
        experts, rows, groups = scale_bytes.shape
        if groups % 4 != 0:
            raise AssertionError(f"scale group count must be divisible by 4: {groups}")
        return (
            scale_bytes.contiguous()
            .reshape(experts, rows, groups // 4, 4)
            .view(torch.int32)
            .reshape(experts, rows, groups // 4)
            .contiguous()
        )

    return (
        (
            interleave_gate_up(reference["w13_weight"]).view(torch.int8),
            pack_ue8m0(interleave_gate_up(reference["w13_weight_scale"])),
        ),
        (
            reference["w2_weight"].contiguous().view(torch.int8),
            pack_ue8m0(reference["w2_weight_scale"]),
        ),
    )


class TestMxfp4PackedExpertNames(TestCase):
    def test_actual_glm_quant_config_selects_generic_mxfp4_method(self):
        config = CompressedTensorsConfig.from_config(GLM53_DATAFREE_QUANT_CONFIG)
        layer = _DummyFusedMoE()

        with mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.FusedMoE",
            _DummyFusedMoE,
        ), mock.patch(
            "sglang.srt.layers.quantization.mxfp4." "Mxfp4MoEMethod",
            return_value="generic-mxfp4",
        ) as generic_method:
            method = config.get_quant_method(layer, "model.layers.3.mlp.experts")

        self.assertEqual(method, "generic-mxfp4")
        generic_method.assert_called_once_with(prefix="model.layers.3.mlp.experts")

    def test_actual_glm_quant_config_normalizes_packed_expert_weight(self):
        config = CompressedTensorsConfig.from_config(GLM53_DATAFREE_QUANT_CONFIG)
        name = "model.layers.3.mlp.experts.0.gate_proj.weight_packed"

        self.assertEqual(
            _normalize_mxfp4_packed_expert_weight_name(name, config),
            "model.layers.3.mlp.experts.0.gate_proj.weight",
        )

    def test_actual_glm_gate_up_down_keys_hit_generic_mxfp4_params(self):
        mappings = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=1,
        )
        prefix = "model.layers.3.mlp."
        cases = {
            "gate_proj": "w13",
            "up_proj": "w13",
            "down_proj": "w2",
        }

        for projection, fused in cases.items():
            for suffix in ("weight_packed", "weight_scale"):
                checkpoint_name = f"{prefix}experts.0.{projection}.{suffix}"
                matched = []
                for param_name, weight_name, _, _ in mappings:
                    if weight_name not in checkpoint_name:
                        continue
                    normalized = _normalize_mxfp4_packed_expert_weight_name(
                        checkpoint_name, _quant_config()
                    )
                    matched.append(normalized.replace(weight_name, param_name))

                expected_suffix = "weight" if suffix == "weight_packed" else suffix
                self.assertEqual(
                    matched,
                    [f"{prefix}experts.{fused}_{expected_suffix}"],
                )

    def test_generic_loader_matches_dedicated_w4a8_scheme_for_all_tp_ep_ranks(self):
        num_experts = 4
        hidden_size = 128
        intermediate_size = 256
        checkpoint = _make_expert_checkpoint(
            num_experts, hidden_size, intermediate_size
        )

        for ep_rank in range(2):
            for tp_rank in range(2):
                with self.subTest(ep_rank=ep_rank, tp_rank=tp_rank):
                    layer, params = _make_candidate_loader(
                        ep_rank=ep_rank,
                        tp_rank=tp_rank,
                        num_experts=num_experts,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                    )
                    _load_candidate_checkpoint(
                        layer, params, checkpoint, num_experts
                    )
                    reference = _baseline_reference_shards(
                        checkpoint,
                        ep_rank=ep_rank,
                        tp_rank=tp_rank,
                        num_experts=num_experts,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                    )

                    for name, expected in reference.items():
                        actual = params[f"model.layers.3.mlp.experts.{name}"].data
                        self.assertEqual(actual.dtype, torch.uint8)
                        self.assertTrue(
                            torch.equal(actual, expected),
                            f"pre-transform bytes differ for {name}",
                        )

                    deep_gemm = ModuleType("deep_gemm")
                    deep_gemm.fp8_fp4_mega_moe = mock.MagicMock()
                    deep_gemm.mega_moe_pre_dispatch_sm90 = mock.MagicMock()
                    deep_gemm._C = SimpleNamespace(
                        fp8_fp4_mega_moe_sm90=mock.MagicMock()
                    )
                    layer.quant_method.use_marlin = False
                    layer.quant_method.use_deep_gemm = False
                    layer.quant_method.use_mega_moe = True
                    with (
                        mock.patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                        mock.patch(
                            "sglang.srt.layers.quantization.mxfp4.get_platform",
                            return_value=SimpleNamespace(is_sm90=True),
                        ),
                    ):
                        layer.quant_method.process_weights_after_loading(layer)
                    candidate_l1 = layer.mega_l1_weights
                    candidate_l2 = layer.mega_l2_weights
                    baseline_l1, baseline_l2 = _baseline_reference_mega_weights(
                        reference
                    )
                    for label, candidate_pair, baseline_pair in (
                        ("mega_l1_weights", candidate_l1, baseline_l1),
                        ("mega_l2_weights", candidate_l2, baseline_l2),
                    ):
                        for candidate_tensor, baseline_tensor in zip(
                            candidate_pair, baseline_pair
                        ):
                            self.assertEqual(
                                candidate_tensor.dtype, baseline_tensor.dtype
                            )
                            self.assertTrue(
                                torch.equal(candidate_tensor, baseline_tensor),
                                f"post-transform bytes differ for {label}",
                            )

    def test_non_mxfp4_checkpoint_name_is_unchanged(self):
        name = "model.layers.3.mlp.experts.0.gate_proj.weight_packed"
        self.assertEqual(
            _normalize_mxfp4_packed_expert_weight_name(
                name, _quant_config(quant_format="pack-quantized")
            ),
            name,
        )

    def test_mxfp4_scale_name_is_unchanged(self):
        name = "model.layers.3.mlp.experts.0.gate_proj.weight_scale"
        self.assertEqual(
            _normalize_mxfp4_packed_expert_weight_name(name, _quant_config()),
            name,
        )


if __name__ == "__main__":
    unittest.main()
