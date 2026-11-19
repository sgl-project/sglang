import math

import pytest

# TODO: use interface in cpu.py
import sgl_kernel  # noqa: F401
import torch
from torch._inductor import config as inductor_config

from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import CPUQuantMethod, call_fused_experts_cpu
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.model_executor import cpu_graph_runner

kernel = torch.ops.sgl_kernel

torch.manual_seed(1183)

dtype = torch.bfloat16
prepack = True
alpha = 1.702
limit = 7.0

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import (
    BLOCK_K,
    BLOCK_N,
    MXFP4QuantizeUtil,
    factor_for_scale,
    fp8_max,
    fp8_min,
    native_fp8_fused_moe,
    precision,
    scaled_weight,
    torch_naive_fused_moe,
    torch_naive_fused_moe_gptoss,
    torch_w8a8_per_column_fused_moe,
    unpack_and_dequant_awq,
)

register_cpu_ci(est_time=7, suite="extra-test-cpu-intel")


def run_fused_experts(
    a,
    w1,
    w2,
    topk_weight,
    topk_ids,
    *,
    quant=CPUQuantMethod.UNQUANT,
    w1_scale=None,
    w2_scale=None,
    w1_zp=None,
    w2_zp=None,
    a1_scale=None,
    block_size=None,
    w1_bias=None,
    w2_bias=None,
    alpha=None,
    limit=None,
    is_vnni=True,
    activation=None,
    inplace=False,
):
    return kernel.fused_experts_cpu(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids.to(torch.int32),
        inplace,
        quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        block_size,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        is_vnni,
        activation,
    )


def make_routing(m, e, topk, dtype, renormalize=False, score=None, return_score=False):
    if score is None:
        score = torch.randn((m, e), dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    if return_score:
        return topk_weight, topk_ids, score
    return topk_weight, topk_ids


def make_bf16_weights(e, out_dim, in_dim, with_bias=False):
    weight = torch.randn((e, out_dim, in_dim), dtype=dtype) / 10

    if not with_bias:
        return weight

    bias = torch.randn((e, out_dim), dtype=torch.float32) / 10
    return weight, bias


def make_int8_weights(e, out_dim, in_dim, int8_max=127, int8_min=-128):
    weight_fp32 = (torch.rand((e, out_dim, in_dim), dtype=torch.float32) - 0.5) * 2
    weight = (weight_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
    return weight


def make_fp8_weights(e, out_dim, in_dim):
    weight_fp32 = torch.randn(e, out_dim, in_dim)
    weight = (
        (weight_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    )

    weight_s = (
        torch.rand(e, math.ceil(out_dim / BLOCK_N), math.ceil(in_dim / BLOCK_K))
        * factor_for_scale
    )

    weight_scaled = scaled_weight(weight, weight_s)
    return weight, weight_s, weight_scaled


def make_mxfp4_weights(e, out_dim, in_dim, dtype, with_bias=False):
    weight_bf16 = torch.randn((e, out_dim, in_dim), dtype=dtype) / 10
    weight_q, weight_s = MXFP4QuantizeUtil.quantize(weight_bf16)
    weight_s = weight_s.reshape(e, out_dim, in_dim // 32)
    weight_dq = MXFP4QuantizeUtil.dequantize(weight_q, dtype, weight_s)

    weight_packed = kernel.convert_weight_packed(weight_q)
    weight_s_packed = kernel.convert_scale_packed(weight_s)

    if not with_bias:
        return weight_dq, weight_packed, weight_s_packed

    bias = torch.randn((e, out_dim), dtype=torch.float32) / 10
    return weight_dq, bias, weight_packed, weight_s_packed


def quantize_fp8_rows_reference(x):
    x = x.float()
    scales = (x.abs().amax(dim=-1) / 448).clamp_min(torch.finfo(torch.float32).eps)
    # FP32 reciprocal-then-multiply matters at FP8 rounding ties.
    scaled = x * scales.reciprocal().unsqueeze(-1)
    return scaled.clamp(-448, 448).to(torch.float8_e4m3fn), scales


def full_fp8_moe_reference(a, w1, w2, topk_weights, topk_ids, quantize_down=True):
    n = w2.shape[-1]
    topk = topk_ids.shape[1]
    intermediate = torch.empty(a.shape[0], topk, a.shape[1], dtype=torch.bfloat16)
    aq, scales = quantize_fp8_rows_reference(a)
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(topk_ids == expert)
        if tokens.numel() == 0:
            continue
        gate_up = (
            (aq.float()[tokens] @ w1[expert].T) * scales[tokens, None]
        ).bfloat16()
        hidden = (
            torch.nn.functional.silu(gate_up[:, :n].float()) * gate_up[:, n:].float()
        ).bfloat16()
        if quantize_down:
            hidden_q, hidden_scales = quantize_fp8_rows_reference(hidden)
            down = (
                (hidden_q.float() @ w2[expert].T) * hidden_scales[:, None]
            ).bfloat16()
        else:
            down = (hidden.float() @ w2[expert].T).bfloat16()
        intermediate[tokens, slots] = (
            down.float() * topk_weights[tokens, slots, None]
        ).bfloat16()
    return intermediate.float().sum(dim=1).bfloat16()


@pytest.fixture
def full_fp8_method(monkeypatch):
    monkeypatch.setattr(fp8_module, "_is_cpu", True)
    monkeypatch.setattr(fp8_module, "SGLANG_DEEPSEEK_FP8A8", True)
    config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    with envs.SGLANG_ENABLE_DEEPSEEK_MOE_FP8_DOWN.override(True):
        return Fp8MoEMethod(config)


def make_full_fp8_moe_weights(method, e, n, k):
    layer = torch.nn.Module()
    dequantized = []
    for name, rows, columns in (("w13", 2 * n, k), ("w2", k, n)):
        weight = (torch.randn(e, rows, columns) * 16).to(torch.float8_e4m3fn)
        scales = (torch.rand(e, rows // 128, columns // 128) + 0.5) * 0.01
        setattr(
            layer, name + "_weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        setattr(
            layer,
            name + "_weight_scale_inv",
            torch.nn.Parameter(scales, requires_grad=False),
        )
        dequantized.append(
            weight.float() * scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
        )
    method._pack_full_fp8_cpu_weights(layer)
    return layer, *dequantized


def run_full_fp8_moe(layer, a, scales, weights, ids, cpp_wrapper=False, **overrides):
    args = dict(
        inplace=False,
        moe_comp_method=CPUQuantMethod.FP8_W8A8_FULL,
        w1_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
        w1_zero=None,
        w2_zero=None,
        a1_scale=scales,
        block_size=[128, 128],
        w1_bias=None,
        w2_bias=None,
        alpha=None,
        limit=None,
        is_vnni=True,
        activation="silu",
    )
    args.update(overrides)
    return call_fused_experts_cpu(
        cpp_wrapper, a, layer.w13_weight, layer.w2_weight, weights, ids, **args
    )


class TestFullFp8MoE:
    @pytest.mark.parametrize(
        "is_cpu,deepseek,down,expected",
        [
            (True, True, None, True),
            (True, True, "0", False),
            (True, True, "1", True),
            (True, False, None, False),
            (True, False, "0", False),
            (False, True, None, False),
            (False, False, None, False),
        ],
    )
    def test_mode_selection(self, monkeypatch, is_cpu, deepseek, down, expected):
        monkeypatch.setattr(fp8_module, "_is_cpu", is_cpu)
        monkeypatch.setattr(fp8_module, "SGLANG_DEEPSEEK_FP8A8", deepseek)
        if down is None:
            monkeypatch.delenv("SGLANG_ENABLE_DEEPSEEK_MOE_FP8_DOWN", raising=False)
        else:
            monkeypatch.setenv("SGLANG_ENABLE_DEEPSEEK_MOE_FP8_DOWN", down)
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )
        method = Fp8MoEMethod(config)
        assert method.use_full_fp8_moe == expected

    @pytest.mark.parametrize(
        "m,n,k,e,topk,skewed",
        [
            (1, 128, 256, 8, 2, False),
            (3, 256, 128, 8, 2, True),
            (96, 512, 256, 32, 8, False),
            (121, 256, 256, 8, 4, True),
        ],
    )
    @pytest.mark.parametrize("zero_input", [False, True])
    def test_two_w8a8_gemms(
        self, full_fp8_method, m, n, k, e, topk, skewed, zero_input
    ):
        layer, w1, w2 = make_full_fp8_moe_weights(full_fp8_method, e, n, k)
        a = (torch.randn(m, k) / math.sqrt(k)).bfloat16()
        if zero_input:
            a.zero_()
        else:
            a[0, 0] = 4
            if m > 1:
                a[-1].zero_()
        weights, ids = make_routing(m, e, topk, dtype=dtype)
        ids = ids.to(torch.int32)
        if skewed:
            ids[:] = torch.arange(topk - 1, -1, -1, dtype=torch.int32)
        aq, scales = kernel._quantize_fp8e4m3_vec(a, True, None)
        reference = full_fp8_moe_reference(a, w1, w2, weights, ids)
        try:
            actual = run_full_fp8_moe(layer, aq, scales, weights, ids)
        except RuntimeError as error:
            if "requires CPU FP8 BRGEMM support" in str(error) or (
                "requires a PyTorch build with CPU FP8 BRGEMM support" in str(error)
            ):
                pytest.skip(str(error))
            raise
        torch.testing.assert_close(actual, reference, atol=0.0005, rtol=0.02)
        scalar_actual = run_full_fp8_moe(
            layer, aq, scales, weights, ids, cpp_wrapper=True
        )
        torch.testing.assert_close(scalar_actual, actual, atol=0, rtol=0)

    @pytest.mark.parametrize("num_threads", [1, 5])
    def test_threaded_gather_and_silu_scratch(self, full_fp8_method, num_threads):
        previous_threads = torch.get_num_threads()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1234)
                torch.set_num_threads(num_threads)
                m, n, k, e, topk = 97, 384, 256, 8, 3
                layer, w1, w2 = make_full_fp8_moe_weights(full_fp8_method, e, n, k)
                a = (torch.randn(m, k) / math.sqrt(k)).bfloat16()
                a[0].zero_()
                a[-1, 0] = 4
                weights, ids = make_routing(m, e, topk, dtype=dtype)
                aq, scales = kernel._quantize_fp8e4m3_vec(a, True, None)
                reference = full_fp8_moe_reference(a, w1, w2, weights, ids)
                try:
                    actual = run_full_fp8_moe(
                        layer, aq, scales, weights, ids.to(torch.int32)
                    )
                except RuntimeError as error:
                    if "requires CPU FP8 BRGEMM support" in str(error) or (
                        "requires a PyTorch build with CPU FP8 BRGEMM support"
                        in str(error)
                    ):
                        pytest.skip(str(error))
                    raise
                torch.testing.assert_close(actual, reference, atol=0.0005, rtol=0.02)
        finally:
            torch.set_num_threads(previous_threads)

    @pytest.mark.parametrize("m", [1, 96])
    def test_row_quantization(self, m):
        a = torch.randn(m, 256).bfloat16()
        a[0].zero_()
        if m > 1:
            a[1, 0] = 128
        actual_q, actual_scales = kernel._quantize_fp8e4m3_vec(a, True, None)
        expected_q, expected_scales = quantize_fp8_rows_reference(a)
        torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=0)
        torch.testing.assert_close(
            actual_q.float(), expected_q.float(), atol=0.016, rtol=0.13
        )

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"inplace": True}, "inplace=False"),
            ({"is_vnni": False}, "prepacked weights"),
            ({"activation": "gelu"}, "silu"),
            ({"w1_bias": torch.zeros(256)}, "biases"),
            ({"a1_scale": None}, "requires w1_scale"),
            ({"w2_scale": None}, "requires w1_scale"),
            ({"block_size": [64, 128]}, "block_size"),
            ({"a1_scale": torch.ones(2)}, "as.size"),
            ({"w2_scale": torch.ones(2, 4, 2, 32)}, "w2s.size"),
        ],
    )
    def test_invalid_full_mode(self, full_fp8_method, overrides, message):
        layer, _, _ = make_full_fp8_moe_weights(full_fp8_method, 2, 128, 128)
        a = torch.zeros(1, 128, dtype=torch.float8_e4m3fn)
        with pytest.raises(RuntimeError, match=message):
            run_full_fp8_moe(
                layer,
                a,
                torch.ones(1),
                torch.ones(1, 1),
                torch.zeros(1, 1, dtype=torch.int32),
                **overrides,
            )

    def test_mode_is_snapshotted(self, full_fp8_method):
        with envs.SGLANG_ENABLE_DEEPSEEK_MOE_FP8_DOWN.override(False):
            layer, _, _ = make_full_fp8_moe_weights(full_fp8_method, 2, 256, 128)
        assert full_fp8_method.use_full_fp8_moe
        assert layer.use_intel_amx_backend
        assert layer.w13_weight.shape == (2, 16, 1, 128, 32)
        assert layer.w2_weight.shape == (2, 4, 2, 128, 32)
        assert layer.w2_weight_scale_inv.shape == (2, 4, 2, 32)

    @pytest.mark.parametrize("m", [0, 3])
    def test_python_dispatch(self, full_fp8_method, monkeypatch, m):
        monkeypatch.delenv("SGLANG_ENABLE_DEEPSEEK_MOE_FP8_DOWN", raising=False)
        full_fp8_method = Fp8MoEMethod(full_fp8_method.quant_config)
        layer, w1, w2 = make_full_fp8_moe_weights(full_fp8_method, 2, 256, 128)
        full_fp8_method.create_moe_runner(layer, MoeRunnerConfig())
        a = (torch.randn(m, 128) / math.sqrt(128)).bfloat16()
        weights, ids = make_routing(m, 2, 1, dtype=dtype)
        dispatch = StandardDispatchOutput(
            hidden_states=a,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                weights, ids.to(torch.int32), torch.zeros(m, 2)
            ),
        )
        try:
            result = full_fp8_method.apply(layer, dispatch).hidden_states
        except RuntimeError as error:
            if "requires CPU FP8 BRGEMM support" in str(error) or (
                "requires a PyTorch build with CPU FP8 BRGEMM support" in str(error)
            ):
                pytest.skip(str(error))
            raise
        reference = full_fp8_moe_reference(a, w1, w2, weights, ids)
        torch.testing.assert_close(result, reference, atol=0.0005, rtol=0.02)

    def test_rejects_incompatible_config(self, full_fp8_method, monkeypatch):
        monkeypatch.setattr(fp8_module, "SGLANG_DEEPSEEK_FP8A8", False)
        with envs.SGLANG_ENABLE_DEEPSEEK_MOE_FP8_DOWN.override(True):
            with pytest.raises(ValueError, match="SGLANG_DEEPSEEK_FP8A8=1"):
                Fp8MoEMethod(full_fp8_method.quant_config)
        with pytest.raises(ValueError, match="silu"):
            full_fp8_method.create_moe_runner(
                torch.nn.Module(), MoeRunnerConfig(activation="gelu")
            )

    @pytest.mark.parametrize("n,k", [(128, 320), (352, 256)])
    def test_rejects_unaligned_weights(self, full_fp8_method, n, k):
        layer = torch.nn.Module()
        layer.w13_weight = torch.empty(2, 2 * n, k, dtype=torch.float8_e4m3fn)
        layer.w2_weight = torch.empty(2, k, n, dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="divisible by 128"):
            full_fp8_method._pack_full_fp8_cpu_weights(layer)

    @pytest.mark.parametrize(
        "params_dtype,with_bias", [(torch.float16, False), (torch.bfloat16, True)]
    )
    def test_rejects_unsupported_activation_dtype_and_bias(
        self, full_fp8_method, params_dtype, with_bias
    ):
        with pytest.raises(ValueError, match="BF16 activations and no expert biases"):
            full_fp8_method.create_weights(
                torch.nn.Module(), 2, 128, 256, params_dtype, with_bias=with_bias
            )

    @pytest.mark.parametrize("m", [1, 96])
    def test_existing_mixed_precision_path(self, full_fp8_method, m):
        e, n, k, topk = 8, 256, 256, 2
        layer, w1, _ = make_full_fp8_moe_weights(full_fp8_method, e, n, k)
        raw_w2 = (torch.randn(e, k, n) * 16).to(torch.float8_e4m3fn)
        w2_scales = (torch.rand(e, k // 128, n // 128) + 0.5) * 0.01
        w2 = raw_w2.float() * w2_scales.repeat_interleave(128, 1).repeat_interleave(
            128, 2
        )
        a = (torch.randn(m, k) / math.sqrt(k)).bfloat16()
        aq, scales = kernel._quantize_fp8e4m3_vec(a, True, None)
        weights, ids = make_routing(m, e, topk, dtype=dtype)
        # Probe the FP8 backend before exercising the unguarded legacy entry point.
        try:
            run_full_fp8_moe(layer, aq, scales, weights, ids)
        except RuntimeError as error:
            if "requires CPU FP8 BRGEMM support" in str(error) or (
                "requires a PyTorch build with CPU FP8 BRGEMM support" in str(error)
            ):
                pytest.skip(str(error))
            raise
        with envs.SGLANG_ENABLE_DEEPSEEK_MOE_FP8_DOWN.override(False):
            mixed_method = Fp8MoEMethod(full_fp8_method.quant_config)
        layer.w2_weight = torch.nn.Parameter(
            kernel.convert_weight_packed(raw_w2), requires_grad=False
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(w2_scales, requires_grad=False)
        mixed_method.create_moe_runner(layer, MoeRunnerConfig())
        dispatch = StandardDispatchOutput(
            hidden_states=a,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                weights, ids.to(torch.int32), torch.zeros(m, e)
            ),
        )
        actual = mixed_method.apply(layer, dispatch).hidden_states
        reference = full_fp8_moe_reference(a, w1, w2, weights, ids, quantize_down=False)
        torch.testing.assert_close(actual, reference, atol=0.0005, rtol=0.02)

    @pytest.mark.parametrize("m", [0, 3])
    def test_fullgraph_and_scalar_overload(self, full_fp8_method, m):
        layer, _, _ = make_full_fp8_moe_weights(full_fp8_method, 2, 128, 128)
        a = torch.randn(m, 128).to(torch.float8_e4m3fn)
        scales = torch.full((m,), 0.1)
        weights = torch.full((m, 1), 0.7)
        ids = (torch.arange(m, dtype=torch.int32) % 2).reshape(m, 1)
        try:
            expected = run_full_fp8_moe(layer, a, scales, weights, ids)
        except RuntimeError as error:
            if "requires CPU FP8 BRGEMM support" in str(error) or (
                "requires a PyTorch build with CPU FP8 BRGEMM support" in str(error)
            ):
                pytest.skip(str(error))
            raise
        with inductor_config.patch(cpp_wrapper=True):
            cpu_graph_runner.register_fake_ops(tp_size=1)
        for cpp_wrapper in (False, True):

            def forward(a, scales, weights, ids):
                return run_full_fp8_moe(
                    layer, a, scales, weights, ids, cpp_wrapper=cpp_wrapper
                )

            with inductor_config.patch(cpp_wrapper=cpp_wrapper):
                compiled = torch.compile(forward, backend="inductor", fullgraph=True)
                result = compiled(a, scales, weights, ids)
            assert result.shape == (m, 128)
            assert result.dtype == torch.bfloat16
            torch.testing.assert_close(result, expected, atol=0, rtol=0)


class TestFusedExperts:
    def test_unsupported_activation_is_rejected(self):
        m, n, k, e, topk = 2, 32, 32, 4, 2
        a = torch.randn((m, k), dtype=dtype) / 10
        w1 = make_bf16_weights(e, 2 * n, k)
        w2 = make_bf16_weights(e, k, n)
        topk_weights, topk_ids = make_routing(m, e, topk, dtype=dtype)
        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2

        with pytest.raises(RuntimeError, match="Unsupported activation"):
            run_fused_experts(
                a,
                packed_w1,
                packed_w2,
                topk_weights,
                topk_ids,
                quant=CPUQuantMethod.UNQUANT,
                is_vnni=prepack,
                activation="relu",
            )

    @pytest.mark.parametrize("m", [2, 114])
    @pytest.mark.parametrize("n", [32])
    @pytest.mark.parametrize("k", [32])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False, True])
    @pytest.mark.parametrize("activation", ["silu", "gelu"])
    def test_bf16_moe(self, m, n, k, e, topk, renormalize, activation):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1 = make_bf16_weights(e, 2 * n, k)
        w2 = make_bf16_weights(e, k, n)
        topk_weights, topk_ids, score = make_routing(
            m,
            e,
            topk,
            dtype=dtype,
            renormalize=renormalize,
            return_score=True,
        )
        torch_output = torch_naive_fused_moe(
            a, w1, w2, score, topk, renormalize, activation=activation
        )

        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2
        fused_output = run_fused_experts(
            a,
            packed_w1,
            packed_w2,
            topk_weights,
            topk_ids,
            quant=CPUQuantMethod.UNQUANT,
            is_vnni=prepack,
            activation=activation,
            inplace=True,
        )

        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [1, 32])
    @pytest.mark.parametrize("n", [128, 64])
    @pytest.mark.parametrize("k", [128, 64])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False])
    def test_bf16_moe_bias(self, m, n, k, e, topk, renormalize):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1, w1_b = make_bf16_weights(e, 2 * n, k, with_bias=True)
        w2, w2_b = make_bf16_weights(e, k, n, with_bias=True)
        topk_weight, topk_ids = make_routing(
            m, e, topk, dtype=dtype, renormalize=renormalize
        )
        torch_output = torch_naive_fused_moe_gptoss(
            a,
            w1,
            w2,
            w1_b,
            w2_b,
            topk_weight,
            topk_ids,
            renormalize,
            alpha,
            limit,
            e,
        )
        packed_w1 = kernel.convert_weight_packed(w1)
        packed_w2 = kernel.convert_weight_packed(w2)
        fused_output = run_fused_experts(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.UNQUANT,
            w1_bias=w1_b,
            w2_bias=w2_b,
            alpha=alpha,
            limit=limit,
            is_vnni=True,
            inplace=False,
        )
        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [1, 39])
    @pytest.mark.parametrize("N", [128])
    @pytest.mark.parametrize("K", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [3])
    def test_int8_moe(self, M, N, K, E, topk):
        # Initialize int8 quantization parameters
        int8_factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        # Input tensor
        # M * K
        a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)

        # Generate int8 weights
        w1 = make_int8_weights(E, 2 * N, K, int8_max=int8_max, int8_min=int8_min)
        w2 = make_int8_weights(E, K, N, int8_max=int8_max, int8_min=int8_min)

        # Generate scale for each column (per-column quantization)
        w1_s = torch.rand(E, 2 * N) * int8_factor_for_scale
        w2_s = torch.rand(E, K) * int8_factor_for_scale

        # Calculate routing
        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )

        inplace = True
        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2
        out = run_fused_experts(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.INT8_W8A8,
            w1_scale=w1_s,
            w2_scale=w2_s,
            is_vnni=prepack,
            inplace=inplace,
        )

        atol = rtol = precision[ref_out.dtype]
        # Increase the tolerance for large input shapes
        if M > 35:
            atol = rtol = 0.02
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [2, 121])
    @pytest.mark.parametrize("N", [352, 512])
    @pytest.mark.parametrize("K", [256, 320])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [4])
    @pytest.mark.parametrize("activation", ["silu", "gelu"])
    def test_fp8_moe(self, M, N, K, E, topk, activation):
        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1, w1s, w1_scaled = make_fp8_weights(E, 2 * N, K)
        w2, w2s, w2_scaled = make_fp8_weights(E, K, N)

        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype)

        w1 = kernel.convert_weight_packed(w1)
        w2 = kernel.convert_weight_packed(w2)

        ref_out = native_fp8_fused_moe(
            a, w1_scaled, w2_scaled, topk_weight, topk_ids, topk, activation=activation
        )
        out = run_fused_experts(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.FP8_W8A16,
            w1_scale=w1s,
            w2_scale=w2s,
            block_size=[BLOCK_N, BLOCK_K],
            is_vnni=True,
            activation=activation,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [1, 32])
    @pytest.mark.parametrize("n", [128, 64])
    @pytest.mark.parametrize("k", [128, 64])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False])
    def test_fp8_moe_bias(self, m, n, k, e, topk, renormalize):
        a = torch.randn((m, k), dtype=dtype) / 10

        w1, w1s, w1_scaled = make_fp8_weights(e, 2 * n, k)
        w2, w2s, w2_scaled = make_fp8_weights(e, k, n)
        w1_b = torch.randn((e, 2 * n), dtype=torch.float32) / 10

        w2_b = torch.randn((e, k), dtype=torch.float32) / 10

        w1_scaled = w1_scaled.to(dtype)
        w2_scaled = w2_scaled.to(dtype)

        topk_weight, topk_ids = make_routing(
            m, e, topk, dtype=dtype, renormalize=renormalize
        )

        ref_out = torch_naive_fused_moe_gptoss(
            a,
            w1_scaled,
            w2_scaled,
            w1_b,
            w2_b,
            topk_weight,
            topk_ids,
            renormalize,
            alpha,
            limit,
            e,
        )

        w1 = kernel.convert_weight_packed(w1)
        w2 = kernel.convert_weight_packed(w2)

        out = run_fused_experts(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.FP8_W8A16,
            w1_scale=w1s,
            w2_scale=w2s,
            block_size=[BLOCK_N, BLOCK_K],
            w1_bias=w1_b,
            w2_bias=w2_b,
            alpha=alpha,
            limit=limit,
            is_vnni=True,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [2, 121])
    @pytest.mark.parametrize("N", [352, 512])
    @pytest.mark.parametrize("K", [256, 320])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [4])
    @pytest.mark.parametrize("activation", ["silu", "gelu"])
    def test_mxfp4_moe(self, M, N, K, E, topk, activation):
        a = torch.randn(M, K, dtype=dtype) / 10

        w1dq, w1_packed, w1s_packed = make_mxfp4_weights(E, 2 * N, K, dtype=dtype)
        w2dq, w2_packed, w2s_packed = make_mxfp4_weights(E, K, N, dtype=dtype)

        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype)

        ref_out = native_fp8_fused_moe(
            a,
            w1dq.float(),
            w2dq.float(),
            topk_weight,
            topk_ids,
            topk,
            activation=activation,
        )
        out = run_fused_experts(
            a,
            w1_packed,
            w2_packed,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.MXFP4,
            w1_scale=w1s_packed,
            w2_scale=w2s_packed,
            is_vnni=True,
            activation=activation,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    def test_mxfp4_moe_rejects_wrong_scale_shape(self):
        """A packed MXFP4 scale that lacks a group per 32 weights must be rejected."""
        M, N, K, E, topk = 2, 352, 256, 8, 4
        a = torch.randn(M, K, dtype=dtype) / 10
        _, w1_packed, w1s_packed = make_mxfp4_weights(E, 2 * N, K, dtype=dtype)
        _, w2_packed, w2s_packed = make_mxfp4_weights(E, K, N, dtype=dtype)
        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype)

        with pytest.raises(RuntimeError, match="one scale per 32 weights"):
            run_fused_experts(
                a,
                w1_packed,
                w2_packed,
                topk_weight,
                topk_ids,
                quant=CPUQuantMethod.MXFP4,
                w1_scale=w1s_packed[..., :-1].contiguous(),
                w2_scale=w2s_packed,
                is_vnni=True,
                activation="silu",
                inplace=False,
            )

    @pytest.mark.parametrize("m", [1, 32])
    @pytest.mark.parametrize("n", [128, 64])
    @pytest.mark.parametrize("k", [128, 64])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False])
    def test_mxfp4_moe_bias(self, m, n, k, e, topk, renormalize):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1dq, w1_b, w1_packed, w1s_packed = make_mxfp4_weights(
            e, 2 * n, k, dtype=dtype, with_bias=True
        )
        w2dq, w2_b, w2_packed, w2s_packed = make_mxfp4_weights(
            e, k, n, dtype=dtype, with_bias=True
        )
        topk_weight, topk_ids = make_routing(
            m, e, topk, dtype=dtype, renormalize=renormalize
        )
        torch_output = torch_naive_fused_moe_gptoss(
            a,
            w1dq,
            w2dq,
            w1_b,
            w2_b,
            topk_weight,
            topk_ids,
            renormalize,
            alpha,
            limit,
            e,
        )

        fused_output = run_fused_experts(
            a,
            w1_packed,
            w2_packed,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.MXFP4,
            w1_scale=w1s_packed,
            w2_scale=w2s_packed,
            w1_bias=w1_b,
            w2_bias=w2_b,
            alpha=alpha,
            limit=limit,
            is_vnni=True,
            inplace=False,
        )
        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [1, 6])
    @pytest.mark.parametrize("N", [512])
    @pytest.mark.parametrize("K", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [4])
    def test_int4_moe(self, M, N, K, E, topk, group_size=128):
        a = torch.rand(M, K, dtype=dtype) / math.sqrt(K)

        awq_w13_weight = torch.randint(-127, 128, (E, K, 2 * N // 8)).to(torch.int)
        awq_w13_zero = torch.randint(0, 10, (E, K // group_size, 2 * N // 8)).to(
            torch.int
        )
        awq_w13_scales = (
            torch.rand(E, int(K // group_size), 2 * N) * factor_for_scale
        ).to(torch.bfloat16)

        awq_w2_weight = torch.randint(-127, 128, (E, N, K // 8)).to(torch.int)
        awq_w2_zero = torch.randint(0, 10, (E, N // group_size, K // 8)).to(torch.int)
        awq_w2_scales = (torch.rand(E, int(N // group_size), K) * factor_for_scale).to(
            torch.bfloat16
        )
        bf16_w13_weight = []
        bf16_w2_weight = []
        for i in range(E):
            bf16_w13_weight_i, _ = unpack_and_dequant_awq(
                awq_w13_weight[i], awq_w13_zero[i], awq_w13_scales[i], 4, 128
            )
            bf16_w2_weight_i, _ = unpack_and_dequant_awq(
                awq_w2_weight[i], awq_w2_zero[i], awq_w2_scales[i], 4, 128
            )
            bf16_w13_weight.append(bf16_w13_weight_i)
            bf16_w2_weight.append(bf16_w2_weight_i)
        bf16_w13_weight = torch.stack(bf16_w13_weight).detach()
        bf16_w2_weight = torch.stack(bf16_w2_weight).detach()

        score = torch.rand((M, E), dtype=dtype)

        ref_out = torch_naive_fused_moe(
            a, bf16_w13_weight, bf16_w2_weight, score, topk, False
        )
        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype, score=score)
        awq_w13_weight_pack, awq_w13_zero_pack, awq_w13_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w13_weight, awq_w13_zero, awq_w13_scales, 0
            )
        )
        awq_w2_weight_pack, awq_w2_zero_pack, awq_w2_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w2_weight, awq_w2_zero, awq_w2_scales, 0
            )
        )

        out = run_fused_experts(
            a,
            awq_w13_weight_pack,
            awq_w2_weight_pack,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.INT4_W4A8,
            w1_scale=awq_w13_scales_pack,
            w2_scale=awq_w2_scales_pack,
            w1_zp=awq_w13_zero_pack,
            w2_zp=awq_w2_zero_pack,
            is_vnni=True,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

        # bf16 routing weights are converted to float like in the other quant methods.
        topk_weight_bf16 = topk_weight.to(torch.bfloat16)
        outs = [
            run_fused_experts(
                a,
                awq_w13_weight_pack,
                awq_w2_weight_pack,
                weights,
                topk_ids,
                quant=CPUQuantMethod.INT4_W4A8,
                w1_scale=awq_w13_scales_pack,
                w2_scale=awq_w2_scales_pack,
                w1_zp=awq_w13_zero_pack,
                w2_zp=awq_w2_zero_pack,
                is_vnni=True,
                inplace=False,
            )
            for weights in (topk_weight_bf16, topk_weight_bf16.float())
        ]
        torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
