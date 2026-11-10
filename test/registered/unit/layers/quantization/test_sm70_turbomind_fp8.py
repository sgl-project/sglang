import pytest
import torch
from sglang.srt.layers.quantization.sm70_turbomind_fp8 import (
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES,
    _load_sm70_qpn8_ops,
    _load_sm70_turbomind_fp8_ops,
    apply_sm70_turbomind_fp8_fused_silu_and_mul,
    apply_sm70_turbomind_fp8_linear,
    prepare_sm70_turbomind_fp8_linear,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="TurboMind SM70 FP8 kernels require an NVIDIA V100",
)


class _DummyLinear(torch.nn.Module):
    def __init__(self, output_size: int, input_size: int, gated: bool, prefix: str):
        super().__init__()
        weight = (
            torch.randn((output_size, input_size), device="cuda", dtype=torch.float16)
            * 0.1
        ).to(torch.float8_e4m3fn)
        scales = (
            torch.rand(
                ((output_size + 127) // 128, (input_size + 127) // 128),
                device="cuda",
                dtype=torch.float32,
            )
            .mul_(0.05)
            .add_(0.95)
        )
        self.register_parameter("weight", torch.nn.Parameter(weight, False))
        self.register_parameter("weight_scale_inv", torch.nn.Parameter(scales, False))
        self.orig_dtype = torch.float16
        self.weight_block_size = [128, 128]
        self.output_size_per_partition = output_size
        self.input_size_per_partition = input_size
        self.logical_widths = (
            [output_size // 2, output_size // 2] if gated else [output_size]
        )
        self.prefix = prefix
        self.tp_size = 4


def _expanded_weight(layer: _DummyLinear) -> torch.Tensor:
    weight = torch.empty(
        (layer.input_size_per_partition, layer.output_size_per_partition),
        dtype=torch.float16,
        device=layer.weight.device,
    )
    torch.ops.sglang_sm70_turbomind.fp8_dequantize_out(
        weight, layer.weight, layer.weight_scale_inv, 128
    )
    return weight


@pytest.mark.parametrize(
    "gated,input_size,output_size,prefix",
    [
        (False, 1536, 5120, "model.layers.0.linear_attn.out_proj"),
        (True, 5120, 8704, "model.layers.0.mlp.gate_up_proj"),
    ],
)
def test_prefill_dispatch_and_decode_kernel(
    monkeypatch, gated, input_size, output_size, prefix
):
    monkeypatch.setenv("SGLANG_SM70_FP8_PREFILL_BACKEND", "fp16")
    monkeypatch.delenv("SGLANG_SM70_FP8_PREFILL_MIN_TOKENS", raising=False)
    torch.manual_seed(7)
    assert _load_sm70_turbomind_fp8_ops()

    layer = _DummyLinear(output_size, input_size, gated, prefix)
    prepare_sm70_turbomind_fp8_linear(layer)
    assert layer.sm70_fp8_prefill_exact_dense_workspace_ptr != 0
    weight = _expanded_weight(layer)
    assert torch.isfinite(weight).all()
    prefill_m = 3920
    prefill_input = torch.randn(
        prefill_m, input_size, device="cuda", dtype=torch.float16
    )
    compact = torch.empty(
        (prefill_m, output_size // 2 if gated else output_size),
        device="cuda",
        dtype=torch.float16,
    )
    torch.ops.sglang_sm70_turbomind.fp8_gemm(
        compact,
        prefill_input,
        layer.weight,
        layer.weight_scale_inv,
        128,
        layer.sm70_fp8_k_ld,
        layer.sm70_fp8_q_ld,
        gated,
    )
    if gated:
        actual = apply_sm70_turbomind_fp8_fused_silu_and_mul(layer, prefill_input)
        expected = compact
    else:
        bias = torch.randn(output_size, device="cuda", dtype=torch.float16)
        actual = apply_sm70_turbomind_fp8_linear(layer, prefill_input, bias)
        expected = compact.add_(bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    decode_input = torch.randn(1, input_size, device="cuda", dtype=torch.float16)
    decode = (
        apply_sm70_turbomind_fp8_fused_silu_and_mul(layer, decode_input)
        if gated
        else apply_sm70_turbomind_fp8_linear(layer, decode_input, None)
    )
    assert torch.isfinite(decode).all()
    if gated:
        gate_up = _load_sm70_qpn8_ops().qpn8_linear(
            decode_input,
            layer.sm70_fp8_qpn8_codes,
            layer.sm70_fp8_qpn8_gscales,
            output_size,
            input_size,
            16,
            3,
        )
        reference = (
            torch.nn.functional.silu(gate_up[:, : output_size // 2])
            * gate_up[:, output_size // 2 :]
        )
        torch.testing.assert_close(decode, reference, rtol=0, atol=0)


def test_prefill_exact_dense_workspace_is_bounded():
    assert _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES == 85 * 1024**2


def test_decode_workspace_is_safe_across_cuda_streams():
    torch.manual_seed(19)
    assert _load_sm70_turbomind_fp8_ops()
    size = 1024
    weight = (torch.randn((size, size), device="cuda", dtype=torch.float16) * 0.05).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones((8, 8), device="cuda", dtype=torch.float32)
    packed, packed_scales, meta = torch.ops.sglang_sm70_turbomind.fp8_prepare(
        weight, scales, 128, False
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])
    inputs = [
        torch.randn((64, size), device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    baselines = []
    for input_ in inputs:
        output = torch.empty((64, size), device="cuda", dtype=torch.float16)
        torch.ops.sglang_sm70_turbomind.fp8_gemm(
            output,
            input_,
            packed,
            packed_scales,
            128,
            k_ld,
            q_ld,
            False,
        )
        baselines.append(output)
    torch.cuda.synchronize()

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [
        [torch.empty((64, size), device="cuda", dtype=torch.float16) for _ in range(8)]
        for _ in streams
    ]
    for stream, input_, stream_outputs in zip(streams, inputs, outputs):
        with torch.cuda.stream(stream):
            for output in stream_outputs:
                torch.ops.sglang_sm70_turbomind.fp8_gemm(
                    output,
                    input_,
                    packed,
                    packed_scales,
                    128,
                    k_ld,
                    q_ld,
                    False,
                )
    torch.cuda.synchronize()

    for baseline, stream_outputs in zip(baselines, outputs):
        for output in stream_outputs:
            torch.testing.assert_close(output, baseline, rtol=0, atol=0)
