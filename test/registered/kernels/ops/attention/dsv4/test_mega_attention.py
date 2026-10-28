import copy
import unittest
from unittest.mock import patch

import torch
from sgl_kernel.flash_mla_mega import flash_mla_mega_decode

from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
    interleave_output_columns,
    interleave_query_rows,
    pack_query,
    quantize_wo_a,
)
from sglang.srt.models.deepseek_v4 import _fp8_wo_a_einsum
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    quantize_k_cache_v41,
    quantize_k_cache_v41_fp4,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def unpack_output(data, scale):
    tokens, groups, hidden = data.shape
    sf = scale.contiguous().view(torch.uint8).float()
    values = data.float().view(tokens, groups, hidden // 32, 32)
    values *= torch.exp2(sf - 127).unsqueeze(-1)
    return (
        values.view(tokens, groups, 16, 8, 32)
        .transpose(2, 3)
        .reshape(tokens, groups, hidden)
    )


def paged_view(raw, page_size, bytes_per_token):
    return raw[:, : page_size * bytes_per_token].view(
        raw.shape[0], page_size, 1, bytes_per_token
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "MegaAttention requires SM100/SM103",
)
class TestMegaAttention(CustomTestCase):
    def test_wq_b_permutation_matches_gemm_then_pack(self):
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        for backend in (
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", backend):
                method = Fp8LinearMethod(
                    Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        use_mxfp8=True,
                        weight_block_size=[1, 32],
                        scale_fmt="ue8m0",
                    )
                )
            with self.subTest(backend=backend):
                linear = torch.nn.Module()
                linear.quant_method = method
                linear.weight = torch.randn(64 * 512, 1280, device="cuda").to(
                    torch.float8_e4m3fn
                )
                linear.weight_scale_inv = torch.randint(
                    119, 129, (64 * 512, 40), device="cuda", dtype=torch.uint8
                )
                method._process_mxfp8_linear_weight_scale(linear)
                reference = copy.deepcopy(linear)
                for param in (linear.weight, linear.weight_scale_inv):
                    param.copy_(interleave_query_rows(param, 64))
                method._process_mxfp8_linear_weight_scale(linear)
                x = torch.randn(1536, 1280, device="cuda", dtype=torch.bfloat16)
                quantized = Mxfp8SwizzledInput(
                    *fp8_utils.flashinfer_mxfp8_quantize(
                        x[:3], is_sf_swizzled_layout=True, alignment=32
                    )
                )
                for inputs in (x, quantized):
                    torch.testing.assert_close(
                        method.apply(linear, inputs),
                        pack_query(
                            method.apply(reference, inputs).view(-1, 64, 512), 64
                        ).view(-1, 64 * 512),
                        rtol=0,
                        atol=0,
                    )

    def test_wo_a_projection_matches_reference(self):
        torch.manual_seed(456)
        positions = torch.tensor([0, 7, 127], device="cuda", dtype=torch.int32)
        angle = (
            torch.arange(256, device="cuda")[:, None]
            / torch.arange(1, 33, device="cuda")[None]
        )
        table = torch.cat((angle.cos(), angle.sin()), dim=-1).float()
        main = paged_view(
            quantize_k_cache_v41(
                torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16)
            ),
            128,
            528,
        )
        extra = paged_view(
            quantize_k_cache_v41_fp4(
                torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16)
            ),
            64,
            288,
        )
        indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(3, 1)
        extra_indices = torch.arange(64, device="cuda", dtype=torch.int32).repeat(3, 1)
        for heads in (16, 64):
            with self.subTest(heads=heads):
                q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
                data, scale, _ = flash_mla_mega_decode(
                    pack_query(q, 64),
                    main,
                    indices,
                    positions,
                    table,
                    512**-0.5,
                    attn_sink=torch.randn(64, device="cuda", dtype=torch.float32),
                    topk_length=torch.tensor(
                        [128, 96, 1], device="cuda", dtype=torch.int32
                    ),
                    extra_kv=extra,
                    extra_indices=extra_indices,
                    extra_topk_length=torch.tensor(
                        [64, 32, 0], device="cuda", dtype=torch.int32
                    ),
                )
                groups = heads // 8
                weight = torch.randn(
                    groups, 64, 4096, device="cuda", dtype=torch.bfloat16
                )
                actual = _fp8_wo_a_einsum(
                    data[:, :groups],
                    scale[:, :groups],
                    *quantize_wo_a(interleave_output_columns(weight)),
                    recipe=(1, 1, 32),
                )
                expected = torch.einsum(
                    "tgd,grd->tgr",
                    unpack_output(data, scale)[:, :groups],
                    weight.float(),
                )
                error = (actual.float() - expected).square().mean().sqrt()
                self.assertLess(
                    error.item() / expected.square().mean().sqrt().item(), 0.04
                )


if __name__ == "__main__":
    unittest.main()
