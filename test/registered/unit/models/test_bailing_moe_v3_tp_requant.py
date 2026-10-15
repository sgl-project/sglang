"""CPU unit tests for Bailing serialized blockwise-FP8 TP requantization."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.layers.quantization.fp8 as fp8_module  # noqa: E402
import sglang.srt.layers.quantization.fp8_utils as fp8_utils_module  # noqa: E402
import sglang.srt.models.bailing_moe_v3 as bailing_module  # noqa: E402
from sglang.kernels.ops.quantization.fp8_kernel import block_quant_fp8  # noqa: E402
from sglang.srt.layers.moe import utils as moe_utils  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: E402
from sglang.srt.layers.moe.utils import (  # noqa: E402
    MoeA2ABackend,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod  # noqa: E402
from sglang.srt.models.bailing_moe_v3 import (  # noqa: E402
    BailingMLP,
    BailingMoE,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FixedGateUpProjection(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, _hidden_states):
        return self.output, None


class _CaptureDownProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, hidden_states):
        self.last_input = hidden_states
        rows = hidden_states[0].shape[0] if isinstance(hidden_states, tuple) else 0
        return torch.zeros(rows, 1), None


class _CpuSiluAndMul(torch.nn.Module):
    def forward(self, hidden_states):
        d = hidden_states.shape[-1] // 2
        return torch.nn.functional.silu(hidden_states[..., :d]) * hidden_states[..., d:]


class TestBailingMoeV3TpRequant(CustomTestCase):
    @staticmethod
    def _empty_moe():
        moe = BailingMoE.__new__(BailingMoE)
        torch.nn.Module.__init__(moe)
        return moe

    def _supports_tail(
        self,
        *,
        opt_in=True,
        runner=MoeRunnerBackend.TRITON,
        moe_ep_size=1,
        cpu_offload_gb=0,
    ):
        method = object.__new__(Fp8MoEMethod)
        method.quant_config = SimpleNamespace(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
        )
        method.use_mxfp8 = False
        method.is_fp4_expert = False

        layer = SimpleNamespace(
            allow_fp8_blockwise_tail=opt_in,
            moe_ep_size=moe_ep_size,
            moe_tp_size=8,
            moe_tp_rank=3,
        )
        parallel = SimpleNamespace(
            tp_size=8,
            tp_rank=3,
            moe_dp_size=1,
        )
        runtime = SimpleNamespace(
            moe=SimpleNamespace(
                enable_eplb=False,
                ep_num_redundant_experts=0,
                init_expert_location="trivial",
            ),
            offload=SimpleNamespace(
                cpu_offload_gb=cpu_offload_gb,
                offload_group_size=-1,
            ),
        )
        a2a = MoeA2ABackend.NONE
        with (
            patch.object(fp8_module, "_is_cuda", True),
            patch.object(fp8_module, "_is_hip", False),
            patch.object(fp8_module, "get_parallel", return_value=parallel),
            patch.object(fp8_module, "get_exec", return_value=runtime),
            patch.object(fp8_module, "get_moe_runner_backend", return_value=runner),
            patch.object(fp8_module, "get_moe_a2a_backend", return_value=a2a),
            patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a),
        ):
            return method._supports_blockwise_fp8_tail(layer)

    def test_fp8_moe_tail_shape_gate_matches_bailing_runtime_boundary(self):
        for runner in (MoeRunnerBackend.TRITON, MoeRunnerBackend.AUTO):
            with self.subTest(supported_runner=runner.value):
                self.assertTrue(self._supports_tail(runner=runner))

        cases = (
            ("not opted in", {"opt_in": False}),
            ("cutlass", {"runner": MoeRunnerBackend.CUTLASS}),
            ("expert parallel", {"moe_ep_size": 2}),
            ("weight offload", {"cpu_offload_gb": 1}),
        )
        for name, kwargs in cases:
            with self.subTest(rejected=name):
                self.assertFalse(self._supports_tail(**kwargs))

    def test_dequantize_uses_checkpoint_global_block_offsets(self):
        quantized = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        checkpoint_scale = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32
        )

        output = BailingMoE._dequantize_serialized_fp8_tp_shard(
            quantized,
            checkpoint_scale,
            (2, 2),
            row_offset=1,
            column_offset=1,
        )

        # Global row indices are [1, 2] -> scale rows [0, 1], while global
        # columns [1, 2, 3] -> scale columns [0, 1, 1].
        expected = torch.tensor([[1, 4, 6], [16, 25, 30]], dtype=torch.bfloat16)
        torch.testing.assert_close(output, expected)

        with self.assertRaisesRegex(ValueError, "exceeds its checkpoint scale"):
            BailingMoE._dequantize_serialized_fp8_tp_shard(
                quantized,
                checkpoint_scale[:1, :1],
                (2, 2),
                row_offset=1,
                column_offset=1,
            )

    def test_cache_keeps_global_scales_until_all_shards_are_loaded(self):
        moe = self._empty_moe()
        moe.experts = SimpleNamespace(
            num_local_experts=2,
            w13_weight=torch.empty(1),
            _map_global_expert_id_to_local_expert_id=lambda expert_id: (
                1 if expert_id == 7 else -1
            ),
        )
        loaded_scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        with patch.object(
            BailingMoE, "_requires_serialized_fp8_tp_requant", return_value=True
        ):
            self.assertTrue(
                moe.cache_serialized_fp8_scale_for_tp_requant(loaded_scale, "w1", 7)
            )
            self.assertTrue(
                moe.cache_serialized_fp8_scale_for_tp_requant(loaded_scale, "w1", 99)
            )
            self.assertFalse(
                moe.cache_serialized_fp8_scale_for_tp_requant(
                    loaded_scale, "not-an-expert-shard", 7
                )
            )

            self.assertEqual(
                tuple(moe._serialized_fp8_tp_scale_cache["w1"].shape), (2, 3, 2)
            )
            torch.testing.assert_close(
                moe._serialized_fp8_tp_scale_cache["w1"][1], loaded_scale
            )
            self.assertEqual(moe._serialized_fp8_tp_scale_loaded["w1"], {1})

            with self.assertRaisesRegex(ValueError, "Inconsistent.*scale shapes"):
                moe.cache_serialized_fp8_scale_for_tp_requant(torch.ones(4, 2), "w1", 7)
            with self.assertRaisesRegex(ValueError, "scales must be 2D"):
                moe.cache_serialized_fp8_scale_for_tp_requant(
                    torch.ones(1, 1, 1), "w2", 7
                )

    def test_cache_maps_the_fused_shared_expert_global_id(self):
        moe = self._empty_moe()
        experts = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(experts)
        experts._expert_storage_rank = 0
        experts._num_local_routed = 512
        experts._num_global_routed = 512
        experts._has_fused_shared = True
        experts.num_local_experts = 513
        experts.w13_weight = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        moe.experts = experts
        shared_expert_scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        self.assertEqual(experts._map_global_expert_id_to_local_expert_id(512), 512)

        with patch.object(
            BailingMoE, "_requires_serialized_fp8_tp_requant", return_value=True
        ):
            self.assertTrue(
                moe.cache_serialized_fp8_scale_for_tp_requant(
                    shared_expert_scale, "w1", 512
                )
            )

        self.assertEqual(moe._serialized_fp8_tp_scale_loaded["w1"], {512})
        torch.testing.assert_close(
            moe._serialized_fp8_tp_scale_cache["w1"][512], shared_expert_scale
        )

    def test_mxfp4_routed_experts_skip_serialized_fp8_requant(self):
        """MXFP4 routed experts sit under a model-level blockwise-FP8 config
        that only describes the dense/shared projections. Their TP-local
        intermediate (320 at TP4, 160 at TP8) is not [128, 128]-aligned, but
        there is no FP8 scale grid to requantize, so the gate must not fire."""
        for tp_size in (4, 8):
            for is_fp4_experts in (True, False):
                with self.subTest(tp_size=tp_size, is_fp4_experts=is_fp4_experts):
                    moe = self._empty_moe()
                    moe.layer_id = 20
                    moe.intermediate_size = 1280
                    moe.moe_tp_size = tp_size
                    moe.quant_config = Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        weight_block_size=[128, 128],
                        is_fp4_experts=is_fp4_experts,
                    )
                    validated = []
                    moe._validate_serialized_fp8_tp_requant_support = lambda: (
                        validated.append(True)
                    )
                    self.assertEqual(
                        moe._requires_serialized_fp8_tp_requant(),
                        not is_fp4_experts,
                    )
                    # FP8 experts still reach the runtime validation; MXFP4
                    # experts never consult it.
                    self.assertEqual(bool(validated), not is_fp4_experts)

    def _standalone_tail_supported(self, *, scale_fmt=None, dispatched=None):
        parallel = SimpleNamespace(
            tp_size=8,
            tp_rank=3,
            moe_ep_size=1,
            moe_dp_size=1,
            moe_tp_size=8,
            moe_tp_rank=3,
        )
        runtime = SimpleNamespace(
            moe=SimpleNamespace(
                enable_eplb=False,
                ep_num_redundant_experts=0,
                init_expert_location="trivial",
            ),
            offload=SimpleNamespace(cpu_offload_gb=0, offload_group_size=-1),
        )
        deepgemm = bailing_module.deepgemm_w8a8_block_fp8_linear_with_fallback

        def dispatch(weight_block_size=None, act_scale_ue8m0=False):
            if dispatched is not None:
                return dispatched
            return (
                partial(deepgemm, act_scale_ue8m0=True) if act_scale_ue8m0 else deepgemm
            )

        quant_config = SimpleNamespace(
            get_name=lambda: "fp8",
            use_mxfp8=False,
            weight_block_size=[128, 128],
            is_checkpoint_fp8_serialized=True,
            scale_fmt=scale_fmt,
        )
        with (
            patch.object(bailing_module, "_is_cuda", True),
            patch.object(bailing_module, "_is_hip", False),
            patch.object(bailing_module, "get_parallel", return_value=parallel),
            patch.object(bailing_module, "get_exec", return_value=runtime),
            patch.object(
                bailing_module,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch.object(bailing_module.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(
                bailing_module.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False
            ),
            patch.object(
                bailing_module, "dispatch_w8a8_block_fp8_linear", side_effect=dispatch
            ),
        ):
            return BailingMLP.validate_standalone_fp8_tail(
                hidden_size=4096,
                intermediate_size=1280,
                quant_config=quant_config,
                tp_size=8,
            )

    def test_standalone_tail_gate_accepts_tp8_logical_160(self):
        self.assertTrue(self._standalone_tail_supported())

    def test_standalone_tail_gate_sees_through_ue8m0_activation_policy(self):
        # Dispatch wraps DeepGEMM in a partial for ue8m0 checkpoints; both the
        # gate and the down-projection pre-quant switch must still recognise it.
        self.assertTrue(self._standalone_tail_supported(scale_fmt="ue8m0"))
        deepgemm = bailing_module.deepgemm_w8a8_block_fp8_linear_with_fallback
        self.assertTrue(
            BailingMLP._is_deepgemm_block_linear(
                partial(deepgemm, act_scale_ue8m0=True)
            )
        )
        with self.assertRaisesRegex(NotImplementedError, "native Hopper DeepGEMM"):
            self._standalone_tail_supported(
                dispatched=fp8_utils_module.triton_w8a8_block_fp8_linear
            )

    def test_ep_keeps_the_padded_shared_mlp_without_consulting_the_tail_gate(self):
        quant_config = SimpleNamespace(
            get_name=lambda: "fp8",
            use_mxfp8=False,
            weight_block_size=[128, 128],
            is_checkpoint_fp8_serialized=True,
        )
        kwargs = dict(
            hidden_size=4096,
            intermediate_size=1280,
            quant_config=quant_config,
            tp_size=8,
        )
        with patch.object(
            BailingMLP,
            "validate_standalone_fp8_tail",
            side_effect=NotImplementedError("tail gate consulted"),
        ):
            for moe_ep_size, enable_a2a_moe in ((8, False), (1, True), (8, True)):
                with self.subTest(moe_ep_size=moe_ep_size, a2a=enable_a2a_moe):
                    self.assertFalse(
                        BailingMoE._allow_shared_fp8_blockwise_tail(
                            moe_ep_size=moe_ep_size,
                            enable_a2a_moe=enable_a2a_moe,
                            **kwargs,
                        )
                    )
            with self.assertRaisesRegex(NotImplementedError, "tail gate consulted"):
                BailingMoE._allow_shared_fp8_blockwise_tail(
                    moe_ep_size=1, enable_a2a_moe=False, **kwargs
                )

    def test_standalone_tail_gate_rejects_unsupported_runtime(self):
        quant_config = SimpleNamespace(
            get_name=lambda: "fp8",
            use_mxfp8=False,
            weight_block_size=[128, 128],
            is_checkpoint_fp8_serialized=True,
        )
        with (
            patch.object(bailing_module, "_is_cuda", False),
            self.assertRaisesRegex(NotImplementedError, "native Hopper DeepGEMM"),
        ):
            BailingMLP.validate_standalone_fp8_tail(
                hidden_size=4096,
                intermediate_size=1280,
                quant_config=quant_config,
                tp_size=8,
            )

    def test_standalone_requant_uses_global_offsets_for_tp8_logical_160(self):
        hidden_size = 256
        intermediate_size = 1280
        tp_size = 8
        tp_rank = 5
        local_size = intermediate_size // tp_size
        fp8_dtype = torch.float8_e4m3fn

        mlp = BailingMLP.__new__(BailingMLP)
        torch.nn.Module.__init__(mlp)
        mlp.hidden_size = hidden_size
        mlp.intermediate_size = intermediate_size
        mlp.tp_size = tp_size
        mlp.tp_rank = tp_rank
        mlp.quant_config = SimpleNamespace(weight_block_size=[128, 128])

        gate_up = torch.nn.Module()
        gate_up.weight = torch.nn.Parameter(
            ((torch.arange(2 * local_size * hidden_size) % 8) + 1)
            .reshape(2 * local_size, hidden_size)
            .to(fp8_dtype),
            requires_grad=False,
        )
        gate_up.weight_scale_inv = torch.nn.Parameter(
            torch.empty(3, 2), requires_grad=False
        )
        down = torch.nn.Module()
        down.weight = torch.nn.Parameter(
            ((torch.arange(hidden_size * local_size) % 8) + 1)
            .reshape(hidden_size, local_size)
            .to(fp8_dtype),
            requires_grad=False,
        )
        down.weight_scale_inv = torch.nn.Parameter(
            torch.empty(2, 2), requires_grad=False
        )
        mlp.gate_up_proj = gate_up
        mlp.down_proj = down

        gate_scale = torch.linspace(0.01, 0.20, 20).reshape(10, 2)
        up_scale = torch.linspace(0.02, 0.40, 20).reshape(10, 2)
        down_scale = torch.linspace(0.01, 0.20, 20).reshape(2, 10)
        tp_offset = tp_rank * local_size
        expected_gate = BailingMLP._dequantize_serialized_fp8_tp_shard(
            gate_up.weight[:local_size],
            gate_scale,
            (128, 128),
            row_offset=tp_offset,
        )
        expected_up = BailingMLP._dequantize_serialized_fp8_tp_shard(
            gate_up.weight[local_size:],
            up_scale,
            (128, 128),
            row_offset=tp_offset,
        )
        expected_down = BailingMLP._dequantize_serialized_fp8_tp_shard(
            down.weight,
            down_scale,
            (128, 128),
            column_offset=tp_offset,
        )
        gate_scale_param = gate_up.weight_scale_inv
        down_scale_param = down.weight_scale_inv

        with patch.object(
            BailingMLP, "_requires_serialized_fp8_tp_requant", return_value=True
        ):
            self.assertTrue(
                mlp.cache_serialized_fp8_scale_for_tp_requant(gate_scale, "gate_proj")
            )
            self.assertTrue(
                mlp.cache_serialized_fp8_scale_for_tp_requant(up_scale, "up_proj")
            )
            self.assertTrue(
                mlp.cache_serialized_fp8_scale_for_tp_requant(down_scale, "down_proj")
            )
            self.assertTrue(mlp.requantize_serialized_fp8_tp_weights())

        self.assertEqual(tuple(gate_up.weight.shape), (320, 256))
        self.assertEqual(tuple(down.weight.shape), (256, 160))
        self.assertIs(gate_up.weight_scale_inv, gate_scale_param)
        self.assertIs(down.weight_scale_inv, down_scale_param)
        self.assertEqual(tuple(gate_up.weight_scale_inv.shape), (3, 2))
        self.assertEqual(tuple(down.weight_scale_inv.shape), (2, 2))
        self.assertFalse(hasattr(mlp, "_serialized_fp8_tp_scale_cache"))

        def dequantize_local(weight, scale):
            expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
            return weight.float() * expanded[: weight.shape[0], : weight.shape[1]]

        torch.testing.assert_close(
            dequantize_local(gate_up.weight, gate_up.weight_scale_inv),
            torch.cat((expected_gate, expected_up)).float(),
            rtol=0.05,
            atol=0.02,
        )
        torch.testing.assert_close(
            dequantize_local(down.weight, down.weight_scale_inv),
            expected_down.float(),
            rtol=0.05,
            atol=0.02,
        )

    def test_standalone_down_prequant_keeps_activation_k160(self):
        rows, local_size = 3, 160
        gate_up_output = torch.linspace(-1, 1, rows * 2 * local_size).reshape(
            rows, 2 * local_size
        )
        mlp = BailingMLP.__new__(BailingMLP)
        torch.nn.Module.__init__(mlp)
        mlp.swiglu_limit = None
        mlp.padded_size_per_partition = None
        mlp.hidden_size = 256
        mlp.act_fn = _CpuSiluAndMul()
        mlp.gate_up_proj = _FixedGateUpProjection(gate_up_output)
        mlp.down_proj = _CaptureDownProjection()
        mlp.use_prequantized_fp8_tail = True
        mlp.quant_config = SimpleNamespace(weight_block_size=[128, 128])

        with patch.object(
            bailing_module.deep_gemm_wrapper,
            "get_mn_major_tma_aligned_tensor",
            side_effect=lambda scale: scale,
            create=True,
        ):
            mlp(torch.zeros(rows, 256))

        quantized, scale = mlp.down_proj.last_input
        self.assertEqual(tuple(quantized.shape), (rows, local_size))
        self.assertEqual(tuple(scale.shape), (rows, 2))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)

    def test_requantize_reblocks_tp_local_weights_on_cpu(self):
        moe = self._empty_moe()
        moe.layer_id = 0
        moe.intermediate_size = 6
        moe.hidden_size = 4
        moe.moe_tp_size = 2
        moe.moe_tp_rank = 1
        moe.quant_config = SimpleNamespace(weight_block_size=[2, 2])

        fp8_dtype = torch.float8_e4m3fn
        w13_weight = torch.tensor(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [2, 1, 4, 3],
                [3, 2, 5, 4],
                [4, 3, 6, 5],
            ],
            dtype=torch.float32,
        ).to(fp8_dtype)
        w2_weight = torch.tensor(
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
            dtype=torch.float32,
        ).to(fp8_dtype)

        experts = SimpleNamespace(
            num_local_experts=1,
            intermediate_size_per_partition=3,
            w13_weight=torch.nn.Parameter(w13_weight.unsqueeze(0), requires_grad=False),
            w2_weight=torch.nn.Parameter(w2_weight.unsqueeze(0), requires_grad=False),
            # Serialized loading creates shard-local scale layouts. Requant must
            # preserve these Parameter objects while rebinding their data.
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.empty(1, 4, 2), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.empty(1, 2, 2), requires_grad=False
            ),
            quant_method=SimpleNamespace(load_up_proj_weight_first=False),
        )
        moe.experts = experts
        moe._serialized_fp8_tp_scale_cache = {
            "w1": torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32),
            "w3": torch.tensor([[[2, 1], [4, 3], [6, 5]]], dtype=torch.float32),
            "w2": torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
        }
        moe._serialized_fp8_tp_scale_loaded = {
            "w1": {0},
            "w2": {0},
            "w3": {0},
        }

        expected_w13 = torch.cat(
            (
                BailingMoE._dequantize_serialized_fp8_tp_shard(
                    w13_weight[:3],
                    moe._serialized_fp8_tp_scale_cache["w1"][0],
                    (2, 2),
                    row_offset=3,
                ),
                BailingMoE._dequantize_serialized_fp8_tp_shard(
                    w13_weight[3:],
                    moe._serialized_fp8_tp_scale_cache["w3"][0],
                    (2, 2),
                    row_offset=3,
                ),
            ),
            dim=0,
        )
        expected_w2 = BailingMoE._dequantize_serialized_fp8_tp_shard(
            w2_weight,
            moe._serialized_fp8_tp_scale_cache["w2"][0],
            (2, 2),
            column_offset=3,
        )
        original_w13_scale_param = experts.w13_weight_scale_inv
        original_w2_scale_param = experts.w2_weight_scale_inv

        with patch.object(
            BailingMoE, "_requires_serialized_fp8_tp_requant", return_value=True
        ):
            self.assertTrue(moe.requantize_serialized_fp8_tp_weights())

        self.assertIs(experts.w13_weight_scale_inv, original_w13_scale_param)
        self.assertIs(experts.w2_weight_scale_inv, original_w2_scale_param)
        self.assertEqual(tuple(experts.w13_weight_scale_inv.shape), (1, 3, 2))
        self.assertEqual(tuple(experts.w2_weight_scale_inv.shape), (1, 2, 2))
        self.assertFalse(experts.w13_weight_scale_inv.format_ue8m0)
        self.assertFalse(experts.w2_weight_scale_inv.format_ue8m0)
        self.assertFalse(hasattr(moe, "_serialized_fp8_tp_scale_cache"))
        self.assertFalse(hasattr(moe, "_serialized_fp8_tp_scale_loaded"))

        def dequantize_local(weight, scale):
            expanded_scale = scale.repeat_interleave(2, 0).repeat_interleave(2, 1)
            return weight.float() * expanded_scale[: weight.shape[0], : weight.shape[1]]

        torch.testing.assert_close(
            dequantize_local(experts.w13_weight[0], experts.w13_weight_scale_inv[0]),
            expected_w13.float(),
            # This comparison includes the expected second FP8 quantization;
            # E4M3's local rounding error can reach 1/21 (~4.76%).
            rtol=0.05,
            atol=0.1,
        )
        torch.testing.assert_close(
            dequantize_local(experts.w2_weight[0], experts.w2_weight_scale_inv[0]),
            expected_w2.float(),
            rtol=0.05,
            atol=0.1,
        )


_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _global_fp8_checkpoint(rows, columns, seed):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=generator)
    weight *= torch.rand(rows, 1, generator=generator) * 4
    return block_quant_fp8(weight.bfloat16(), (128, 128))


def _dequantize_global(quantized, scale):
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return quantized.float() * expanded[: quantized.shape[0], : quantized.shape[1]]


class TestBailingMoeV3TpRequantMatchesGlobalCheckpoint(CustomTestCase):
    """Every TP8 rank's requantized 160-wide shard must equal the slice of the
    globally dequantized checkpoint, re-blocked on the rank's own 128 grid."""

    hidden_size = 256
    intermediate_size = 1280
    tp_size = 8
    local_size = intermediate_size // tp_size

    def _assert_local_blocks_match(self, weight, scale, reference):
        reference_bf16 = reference.bfloat16().float()
        rows, columns = weight.shape
        for block_row in range(scale.shape[0]):
            for block_column in range(scale.shape[1]):
                row_slice = slice(block_row * 128, min((block_row + 1) * 128, rows))
                column_slice = slice(
                    block_column * 128, min((block_column + 1) * 128, columns)
                )
                expected_scale = (
                    reference_bf16[row_slice, column_slice].abs().amax() / _FP8_MAX
                )
                torch.testing.assert_close(
                    scale[block_row, block_column], expected_scale, rtol=1e-6, atol=0
                )
        dequantized = _dequantize_global(weight, scale)
        expanded_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        # E4M3 rounds normals to 3 mantissa bits (half-ulp <= 2^-4 relative)
        # and subnormals to a 2^-9 step of the block scale.
        bound = reference_bf16.abs() * 2**-4 + expanded_scale[:rows, :columns] * 2**-9
        self.assertTrue(
            bool(((dequantized - reference_bf16).abs() <= bound * 1.001).all())
        )

    def test_routed_experts_on_every_rank(self):
        num_experts = 2
        checkpoint = {
            shard: [
                _global_fp8_checkpoint(*shape, seed=17 * expert + index)
                for expert in range(num_experts)
            ]
            for index, (shard, shape) in enumerate(
                (
                    ("w1", (self.intermediate_size, self.hidden_size)),
                    ("w3", (self.intermediate_size, self.hidden_size)),
                    ("w2", (self.hidden_size, self.intermediate_size)),
                )
            )
        }

        for tp_rank in range(self.tp_size):
            with self.subTest(tp_rank=tp_rank):
                rows = slice(tp_rank * self.local_size, (tp_rank + 1) * self.local_size)
                moe = BailingMoE.__new__(BailingMoE)
                torch.nn.Module.__init__(moe)
                moe.layer_id = 0
                moe.hidden_size = self.hidden_size
                moe.intermediate_size = self.intermediate_size
                moe.moe_tp_size = self.tp_size
                moe.moe_tp_rank = tp_rank
                moe.quant_config = SimpleNamespace(weight_block_size=[128, 128])
                moe.experts = SimpleNamespace(
                    num_local_experts=num_experts,
                    intermediate_size_per_partition=self.local_size,
                    w13_weight=torch.nn.Parameter(
                        torch.stack(
                            [
                                torch.cat(
                                    (
                                        checkpoint["w1"][e][0][rows],
                                        checkpoint["w3"][e][0][rows],
                                    )
                                )
                                for e in range(num_experts)
                            ]
                        ),
                        requires_grad=False,
                    ),
                    w2_weight=torch.nn.Parameter(
                        torch.stack(
                            [
                                checkpoint["w2"][e][0][:, rows]
                                for e in range(num_experts)
                            ]
                        ),
                        requires_grad=False,
                    ),
                    w13_weight_scale_inv=torch.nn.Parameter(
                        torch.empty(num_experts, 4, 2), requires_grad=False
                    ),
                    w2_weight_scale_inv=torch.nn.Parameter(
                        torch.empty(num_experts, 2, 2), requires_grad=False
                    ),
                    quant_method=SimpleNamespace(load_up_proj_weight_first=False),
                    _map_global_expert_id_to_local_expert_id=lambda expert_id: (
                        expert_id
                    ),
                )

                with patch.object(
                    BailingMoE, "_requires_serialized_fp8_tp_requant", return_value=True
                ):
                    for shard in ("w1", "w3", "w2"):
                        for expert in range(num_experts):
                            self.assertTrue(
                                moe.cache_serialized_fp8_scale_for_tp_requant(
                                    checkpoint[shard][expert][1], shard, expert
                                )
                            )
                    self.assertTrue(moe.requantize_serialized_fp8_tp_weights())

                self.assertEqual(
                    tuple(moe.experts.w13_weight_scale_inv.shape), (num_experts, 3, 2)
                )
                for expert in range(num_experts):
                    reference = {
                        shard: _dequantize_global(*checkpoint[shard][expert])
                        for shard in ("w1", "w3", "w2")
                    }
                    self._assert_local_blocks_match(
                        moe.experts.w13_weight[expert],
                        moe.experts.w13_weight_scale_inv[expert],
                        torch.cat((reference["w1"][rows], reference["w3"][rows])),
                    )
                    self._assert_local_blocks_match(
                        moe.experts.w2_weight[expert],
                        moe.experts.w2_weight_scale_inv[expert],
                        reference["w2"][:, rows],
                    )

    def test_standalone_shared_mlp_on_every_rank(self):
        checkpoint = {
            projection: _global_fp8_checkpoint(*shape, seed=seed)
            for seed, (projection, shape) in enumerate(
                (
                    ("gate_proj", (self.intermediate_size, self.hidden_size)),
                    ("up_proj", (self.intermediate_size, self.hidden_size)),
                    ("down_proj", (self.hidden_size, self.intermediate_size)),
                )
            )
        }
        reference = {
            projection: _dequantize_global(*value)
            for projection, value in checkpoint.items()
        }

        for tp_rank in range(self.tp_size):
            with self.subTest(tp_rank=tp_rank):
                rows = slice(tp_rank * self.local_size, (tp_rank + 1) * self.local_size)
                mlp = BailingMLP.__new__(BailingMLP)
                torch.nn.Module.__init__(mlp)
                mlp.hidden_size = self.hidden_size
                mlp.intermediate_size = self.intermediate_size
                mlp.tp_size = self.tp_size
                mlp.tp_rank = tp_rank
                mlp.quant_config = SimpleNamespace(weight_block_size=[128, 128])
                mlp.gate_up_proj = torch.nn.Module()
                mlp.gate_up_proj.weight = torch.nn.Parameter(
                    torch.cat(
                        (
                            checkpoint["gate_proj"][0][rows],
                            checkpoint["up_proj"][0][rows],
                        )
                    ),
                    requires_grad=False,
                )
                mlp.gate_up_proj.weight_scale_inv = torch.nn.Parameter(
                    torch.empty(3, 2), requires_grad=False
                )
                mlp.down_proj = torch.nn.Module()
                mlp.down_proj.weight = torch.nn.Parameter(
                    checkpoint["down_proj"][0][:, rows].contiguous(),
                    requires_grad=False,
                )
                mlp.down_proj.weight_scale_inv = torch.nn.Parameter(
                    torch.empty(2, 2), requires_grad=False
                )

                with patch.object(
                    BailingMLP, "_requires_serialized_fp8_tp_requant", return_value=True
                ):
                    for projection, (_, scale) in checkpoint.items():
                        self.assertTrue(
                            mlp.cache_serialized_fp8_scale_for_tp_requant(
                                scale, projection
                            )
                        )
                    self.assertTrue(mlp.requantize_serialized_fp8_tp_weights())

                self._assert_local_blocks_match(
                    mlp.gate_up_proj.weight,
                    mlp.gate_up_proj.weight_scale_inv,
                    torch.cat(
                        (reference["gate_proj"][rows], reference["up_proj"][rows])
                    ),
                )
                self._assert_local_blocks_match(
                    mlp.down_proj.weight,
                    mlp.down_proj.weight_scale_inv,
                    reference["down_proj"][:, rows],
                )


if __name__ == "__main__":
    unittest.main()
