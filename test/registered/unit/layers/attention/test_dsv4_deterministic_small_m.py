from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4 import compressor
from sglang.srt.layers.quantization import fp8, fp8_utils, unquant
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


def test_compressor_deterministic_small_m_dispatch():
    calls = []

    def torch_mm(input_tensor, weight, *, out_dtype):
        calls.append(("torch", tuple(input_tensor.shape), tuple(weight.shape)))
        return torch.empty(input_tensor.shape[0], weight.shape[1], dtype=out_dtype)

    def aiter_linear(input_tensor, weight):
        calls.append(("aiter", tuple(input_tensor.shape), tuple(weight.shape)))
        return torch.empty(input_tensor.shape[0], weight.shape[0], dtype=torch.float32)

    contracts = (
        (128, 1, 512, 1024),
        (4, 2, 512, 2048),
        (4, 2, 128, 512),
    )
    with (
        mock.patch.object(compressor, "_is_hip", True),
        mock.patch.object(compressor.torch, "mm", side_effect=torch_mm),
        mock.patch.object(compressor, "linear_bf16_fp32", side_effect=aiter_linear),
    ):
        for ratio, coff, head_dim, output_size in contracts:
            marked = SimpleNamespace(
                ratio=ratio,
                coff=coff,
                head_dim=head_dim,
                _use_deterministic_small_m_projection=True,
                wkv_gate=SimpleNamespace(
                    weight=torch.empty(output_size, 7168, dtype=torch.bfloat16)
                ),
            )
            output = compressor.Compressor._compute_kv_score_projection(
                marked, torch.empty(2, 7168, dtype=torch.bfloat16)
            )
            assert output.shape == (2, output_size)
            assert calls[-1][0] == "torch"

            output = compressor.Compressor._compute_kv_score_projection(
                marked, torch.empty(33, 7168, dtype=torch.bfloat16)
            )
            assert output.shape == (33, output_size)
            assert calls[-1][0] == "aiter"

            unmarked = SimpleNamespace(
                ratio=ratio,
                coff=coff,
                head_dim=head_dim,
                wkv_gate=marked.wkv_gate,
            )
            compressor.Compressor._compute_kv_score_projection(
                unmarked, torch.empty(2, 7168, dtype=torch.bfloat16)
            )
            assert calls[-1][0] == "aiter"


def test_compressor_deterministic_small_m_rejects_bad_contract():
    marked = SimpleNamespace(
        ratio=4,
        coff=2,
        head_dim=128,
        _use_deterministic_small_m_projection=True,
        wkv_gate=SimpleNamespace(weight=torch.empty(511, 7168, dtype=torch.bfloat16)),
    )
    with mock.patch.object(compressor, "_is_hip", True):
        try:
            compressor.Compressor._compute_kv_score_projection(
                marked, torch.empty(2, 7168, dtype=torch.bfloat16)
            )
        except RuntimeError as error:
            assert "unexpected contract" in str(error)
        else:
            raise AssertionError("invalid compressor contract did not fail")


def test_unquantized_deterministic_small_m_dispatch():
    method = unquant.UnquantizedLinearMethod()
    marked = SimpleNamespace(
        weight=torch.empty(64, 7168, dtype=torch.bfloat16),
        _use_deterministic_small_m_linear=True,
    )
    unmarked = SimpleNamespace(weight=marked.weight)
    calls = []

    def torch_linear(input_tensor, weight, bias):
        calls.append(("torch", tuple(input_tensor.shape), tuple(weight.shape)))
        return torch.empty(
            input_tensor.shape[0], weight.shape[0], dtype=input_tensor.dtype
        )

    class FakeTgemm:
        @staticmethod
        def mm(input_tensor, weight, bias, otype):
            calls.append(("aiter", tuple(input_tensor.shape), tuple(weight.shape)))
            return torch.empty(input_tensor.shape[0], weight.shape[0], dtype=otype)

    with (
        mock.patch.object(unquant, "_use_aiter", True),
        mock.patch.object(unquant, "tgemm", FakeTgemm, create=True),
        mock.patch.object(unquant.F, "linear", side_effect=torch_linear),
    ):
        method.apply(marked, torch.empty(2, 7168, dtype=torch.bfloat16))
        assert calls[-1][0] == "torch"
        method.apply(marked, torch.empty(33, 7168, dtype=torch.bfloat16))
        assert calls[-1][0] == "aiter"
        method.apply(unmarked, torch.empty(2, 7168, dtype=torch.bfloat16))
        assert calls[-1][0] == "aiter"


def test_fp8_method_forwards_small_m_direct_ck_flag():
    calls = []

    def block_linear(**kwargs):
        calls.append(kwargs)
        input_tensor = kwargs["input"]
        return torch.empty(input_tensor.shape[0], 768, dtype=torch.bfloat16)

    method = SimpleNamespace(
        block_quant=True,
        use_marlin=False,
        use_mxfp8=False,
        weight_block_size=[128, 128],
        w8a8_block_fp8_linear=block_linear,
    )
    layer = SimpleNamespace(
        weight=torch.empty(768, 7168, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.empty(6, 56, dtype=torch.float32),
        _force_deterministic_small_m_bpreshuffle_ck=True,
    )

    with mock.patch.object(fp8, "use_intel_amx_backend", return_value=False):
        fp8.Fp8LinearMethod.apply(
            method,
            layer,
            torch.empty(2, 7168, dtype=torch.bfloat16),
        )
        assert calls[-1]["force_bpreshuffle_ck"] is True

        fp8.Fp8LinearMethod.apply(
            method,
            layer,
            (
                torch.empty(2, 7168, dtype=torch.float8_e4m3fn),
                torch.empty(2, 56, dtype=torch.float32),
            ),
        )
        assert calls[-1]["force_bpreshuffle_ck"] is True

        fp8.Fp8LinearMethod.apply(
            method,
            layer,
            torch.empty(33, 7168, dtype=torch.bfloat16),
        )
        assert "force_bpreshuffle_ck" not in calls[-1]


def test_fp8_direct_ck_contract_and_output_buffer():
    output_buffers = []

    def direct_ck(input_tensor, weight, input_scale, weight_scale, output):
        output_buffers.append(output)
        output.fill_(1)
        return output

    input_tensor = torch.empty(2, 7168, dtype=torch.bfloat16)
    weight = torch.empty(768, 7168, dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(6, 56, dtype=torch.float32)

    with (
        mock.patch.object(fp8_utils, "_use_aiter_bpreshuffle_gfx95", True),
        mock.patch.object(
            fp8_utils,
            "aiter_per1x128_quant",
            return_value=(
                torch.empty(2, 7168, dtype=torch.float8_e4m3fn),
                torch.empty(2, 56, dtype=torch.float32),
            ),
        ),
        mock.patch.object(
            fp8_utils,
            "view_aiter_fused_rms_transposed_fp8_scale",
            side_effect=lambda tensor: tensor,
        ),
        mock.patch(
            "aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle_ck",
            side_effect=direct_ck,
        ),
    ):
        output = fp8_utils.aiter_w8a8_block_fp8_linear(
            input_tensor,
            weight,
            [128, 128],
            weight_scale,
            force_bpreshuffle_ck=True,
        )

    assert output.shape == (2, 768)
    assert output.dtype == torch.bfloat16
    assert output_buffers[0].shape == (2, 768)

    with mock.patch.object(fp8_utils, "_use_aiter_bpreshuffle_gfx95", True):
        try:
            fp8_utils.aiter_w8a8_block_fp8_linear(
                torch.empty(33, 7168, dtype=torch.bfloat16),
                weight,
                [128, 128],
                weight_scale,
                force_bpreshuffle_ck=True,
            )
        except RuntimeError as error:
            assert "unexpected contract" in str(error)
        else:
            raise AssertionError("invalid direct CK contract did not fail")
