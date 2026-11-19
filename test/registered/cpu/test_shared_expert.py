import itertools
import math
import unittest

import pytest
import sgl_kernel  # noqa: F401
import torch
from torch._inductor import config as inductor_config

from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.model_executor import cpu_graph_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import (
    BLOCK_K,
    BLOCK_N,
    factor_for_scale,
    fp8_max,
    fp8_min,
    per_token_quant_int8,
    precision,
    scaled_weight,
    torch_naive_moe,
    torch_w8a8_per_column_moe,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="extra-test-cpu-intel")

torch.manual_seed(1234)


@pytest.fixture
def shared_fp8_mode(monkeypatch):
    monkeypatch.setattr(fp8_module, "_is_cpu", True)
    monkeypatch.setattr(fp8_module, "_is_cpu_amx_available", True)
    monkeypatch.setattr(fp8_module, "SGLANG_DEEPSEEK_FP8A8", True)
    monkeypatch.setattr(fp8_module, "SGLANG_BRGEMM_REF_FP8A8", False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        yield


def make_shared_fp8_weights(k, n):
    config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    layers, weights = [], []
    for name, rows, columns in (("gate_up_proj", 2 * n, k), ("down_proj", k, n)):
        weight = (torch.randn(rows, columns) * 8).to(torch.float8_e4m3fn)
        scales = (torch.rand(rows // 128, columns // 128) + 0.5) * 0.02
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        method = Fp8LinearMethod(config, f"model.layers.3.mlp.shared_experts.{name}")
        method.process_weights_after_loading(layer)
        assert layer.use_f8f8 and layer.use_intel_amx_backend
        assert layer.weight.ndim == 4
        layers.append(layer)
        weights.append((weight, scales))
    return layers, weights


def shared_fp8_reference_linear(x, weight, scales):
    x = x.float()
    xs = (x.abs().amax(dim=-1) / 448).clamp_min(torch.finfo(torch.float32).eps)
    xq = (
        (x * xs.reciprocal().unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .float()
    )
    weight = weight.float()
    scales = scales.repeat_interleave(128, 0)
    result = torch.zeros(x.shape[0], weight.shape[0])
    for group in range(weight.shape[1] // 128):
        begin = group * 128
        result += (
            xq[:, begin : begin + 128] @ weight[:, begin : begin + 128].T
        ) * scales[:, group]
    return (result * xs.unsqueeze(-1)).bfloat16()


def shared_fp8_reference(x, weights, routed, factor):
    gate_up = shared_fp8_reference_linear(x, *weights[0])
    gate, up = gate_up.float().chunk(2, dim=-1)
    intermediate = (torch.nn.functional.silu(gate) * up).bfloat16()
    down = shared_fp8_reference_linear(intermediate, *weights[1])
    if routed is not None:
        down = (
            down.float() + routed.float() * (factor if factor is not None else 0)
        ).bfloat16()
    return down


def run_shared_fp8(x, layers, routed=None, factor=None, inplace=False, **overrides):
    args = dict(
        hidden_states=x,
        w1=layers[0].weight,
        w2=layers[1].weight,
        fused_experts_out=routed,
        routed_scaling_factor=factor,
        inplace=inplace,
        use_int8_w8a8=False,
        use_fp8_w8a16=True,
        w1_scale=layers[0].weight_scale_inv,
        w2_scale=layers[1].weight_scale_inv,
        block_size=[128, 128],
        is_vnni=True,
    )
    args.update(overrides)
    return torch.ops.sgl_kernel.shared_expert_cpu(**args)


@pytest.mark.parametrize(
    "m,k,n,inplace,factor",
    [
        (0, 256, 128, False, None),
        (1, 128, 128, True, 2.5),
        (3, 256, 128, False, 0.0),
        (35, 256, 512, False, -1.25),
        (96, 512, 256, True, None),
        (1, 7168, 512, False, 2.5),
        (96, 7168, 512, True, 2.5),
    ],
)
def test_shared_fp8_w8a8(m, k, n, inplace, factor, shared_fp8_mode):
    layers, weights = make_shared_fp8_weights(k, n)
    x = torch.randn(m, k, dtype=torch.bfloat16) / math.sqrt(k)
    if m > 1:
        x[0].zero_()
        x[-1, 0] = 64
    before = x.clone()
    routed = torch.randn_like(x) if factor is not None else None
    routed_before = routed.clone() if routed is not None else None
    expected = shared_fp8_reference(x, weights, routed, factor)
    actual = run_shared_fp8(x, layers, routed, factor, inplace)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-2)
    assert actual.dtype == torch.bfloat16 and actual.shape == before.shape
    if inplace:
        assert actual.data_ptr() == x.data_ptr()
    else:
        torch.testing.assert_close(x, before, atol=0, rtol=0)
    if routed is not None:
        torch.testing.assert_close(routed, routed_before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "invalid",
    [
        "w1_scale",
        "w2_scale",
        "block",
        "dtype",
        "routed_dtype",
        "routed_shape",
        "mixed_layout",
        "flags",
        "unpacked",
        "missing_routed",
    ],
)
def test_shared_fp8_w8a8_invalid(invalid, shared_fp8_mode):
    layers, _ = make_shared_fp8_weights(256, 128)
    x = torch.randn(1, 256, dtype=torch.bfloat16)
    before = x.clone()
    overrides = {}
    if invalid == "w1_scale":
        overrides["w1_scale"] = None
    elif invalid == "w2_scale":
        overrides["w2_scale"] = layers[1].weight_scale_inv[..., :16].contiguous()
    elif invalid == "block":
        overrides["block_size"] = [64, 128]
    elif invalid == "dtype":
        overrides["hidden_states"] = x.float()
    elif invalid == "routed_dtype":
        overrides["fused_experts_out"] = torch.randn_like(x).float()
    elif invalid == "routed_shape":
        overrides["fused_experts_out"] = x[:, :128]
    elif invalid == "mixed_layout":
        overrides["w2"] = layers[1].weight.reshape(256, 128)
    elif invalid == "flags":
        overrides["use_fp8_w8a16"] = False
    elif invalid == "unpacked":
        overrides["is_vnni"] = False
    elif invalid == "missing_routed":
        overrides["routed_scaling_factor"] = 2.5
    with pytest.raises(RuntimeError):
        run_shared_fp8(x, layers, inplace=True, **overrides)
    torch.testing.assert_close(x, before, atol=0, rtol=0)


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("cpp_wrapper", [False, True])
def test_shared_fp8_w8a8_compile(m, inplace, cpp_wrapper, shared_fp8_mode):
    layers, _ = make_shared_fp8_weights(256, 128)
    x = torch.randn(m, 256, dtype=torch.bfloat16)
    routed = torch.randn_like(x)
    expected = run_shared_fp8(x.clone(), layers, routed, 2.5, inplace)

    def forward(x, routed):
        return run_shared_fp8(x, layers, routed, 2.5, inplace)

    with inductor_config.patch(cpp_wrapper=True):
        cpu_graph_runner.register_fake_ops(tp_size=1)
    with inductor_config.patch(cpp_wrapper=cpp_wrapper):
        compiled = torch.compile(forward, backend="inductor", fullgraph=True)
        actual = compiled(x.clone(), routed)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestSharedExpert(CustomTestCase):
    M = [2, 121]
    N = [32, 32 * 4]
    K = [32, 32 * 2]
    routed_scaling_factor = [16]
    apply_scaling_factor = [True, False]

    M_fp8 = [2, 12]
    N_fp8 = [512]
    K_fp8 = [256]

    def _bf16_shared_expert(self, m, n, k, routed_scaling_factor, apply_scaling_factor):
        dtype = torch.bfloat16

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = (
            torch.randn(m, k, dtype=dtype) / k if apply_scaling_factor else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        # bfloat16
        ref = torch_naive_moe(
            hidden_states,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            output_dtype=dtype,
        )
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            True,
            False,
            False,
            None,
            None,
            None,
            False,
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_bf16_shared_expert(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.routed_scaling_factor,
            self.apply_scaling_factor,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                routed_scaling_factor=params[3],
                apply_scaling_factor=params[4],
            ):
                self._bf16_shared_expert(*params)

    def _int8_shared_expert(self, m, n, k, routed_scaling_factor, apply_scaling_factor):
        dtype = torch.bfloat16

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = (
            torch.randn(m, k, dtype=dtype) / k if apply_scaling_factor else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        w1_q, w1_s = per_token_quant_int8(w1)
        w2_q, w2_s = per_token_quant_int8(w2)
        ref = torch_w8a8_per_column_moe(
            hidden_states,
            w1_q,
            w2_q,
            w1_s,
            w2_s,
            fused_output,
            routed_scaling_factor,
        )
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1_q,
            w2_q,
            fused_output,
            routed_scaling_factor,
            True,
            True,
            False,
            w1_s,
            w2_s,
            None,
            False,
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_int8_shared_expert(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.routed_scaling_factor,
            self.apply_scaling_factor,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                routed_scaling_factor=params[3],
                apply_scaling_factor=params[4],
            ):
                self._int8_shared_expert(*params)

    def _fp8_shared_expert(self, m, n, k, routed_scaling_factor, apply_scaling_factor):
        dtype = torch.bfloat16

        hidden_states = torch.randn(m, k, dtype=dtype) / math.sqrt(k)

        w1_fp32 = torch.randn(1, 2 * n, k)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(1, k, n)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = torch.randn(1, 2 * n // BLOCK_N, k // BLOCK_K) * factor_for_scale
        w2s = torch.randn(1, k // BLOCK_N, n // BLOCK_K) * factor_for_scale

        w1_scaled = scaled_weight(w1, w1s).view(2 * n, k)
        w2_scaled = scaled_weight(w2, w2s).view(k, n)

        # change back to 2D
        w1, w2 = w1.squeeze(0), w2.squeeze(0)
        w1s, w2s = w1s.squeeze(0), w2s.squeeze(0)
        w1_scaled, w2_scaled = w1_scaled.squeeze(0), w2_scaled.squeeze(0)

        fused_output = (
            torch.randn(m, k, dtype=dtype) / math.sqrt(k)
            if apply_scaling_factor
            else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None
        hidden_states2 = hidden_states.clone()

        # ref with bfloat16
        ref = torch_naive_moe(
            hidden_states,
            w1_scaled,
            w2_scaled,
            fused_output,
            routed_scaling_factor,
            output_dtype=dtype,
        )

        w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)  # [2N, K]
        w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)  # [K, N]
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            True,
            False,
            True,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            True,
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_fp8_shared_expert(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.routed_scaling_factor,
            self.apply_scaling_factor,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                routed_scaling_factor=params[3],
                apply_scaling_factor=params[4],
            ):
                self._fp8_shared_expert(*params)


if __name__ == "__main__":
    unittest.main()
