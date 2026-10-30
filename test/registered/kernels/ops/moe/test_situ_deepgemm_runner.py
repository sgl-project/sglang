"""Check SiTU quantization through two real DeepGEMM projections."""

import os
import unittest

import torch

# test_utils transitively imports the DeepGEMM wrapper, which snapshots this
# option at import. Set it before any SGLang import so standalone CI only
# compiles the shapes under test, not the full serving-time warmup range.
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"

from sglang.srt.runtime_context import (  # noqa: E402
    get_context,
    get_flags,
    get_parallel,
)
from sglang.srt.utils import get_device_sm  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_EXPERTS = 2
_ROWS_PER_EXPERT = 128
# Two packed scale words avoid the released layout validator's singleton-stride
# restriction. Equal hidden/intermediate sizes allow an identity down projection.
_WIDTH = 1024
_GROUP = 128
_BETA = 4.0
_LINEAR_BETA = 25.0


def _make_weights(fp4):
    import deep_gemm

    if fp4:
        w13 = torch.zeros(
            (_EXPERTS, 2 * _WIDTH, _WIDTH // 2), device="cuda", dtype=torch.uint8
        )
        # E2M1 codes 2 and 4 represent 1 and 2 in the low nibble (K=0).
        w13[0, :, 0] = 2
        w13[1, :, 0] = 4
        w2 = torch.zeros(
            (_EXPERTS, _WIDTH, _WIDTH // 2), device="cuda", dtype=torch.uint8
        )
        diagonal = torch.arange(_WIDTH, device="cuda")
        w2[:, diagonal, diagonal // 2] = (2 << ((diagonal % 2) * 4)).to(torch.uint8)
        # Match the production MXFP4 loader's packed storage, not uint8/float4.
        w13, w2 = w13.view(torch.int8), w2.view(torch.int8)
        s13 = torch.ones(
            (_EXPERTS, 2 * _WIDTH, _WIDTH // 32), device="cuda", dtype=torch.float32
        )
        s2 = torch.ones(
            (_EXPERTS, _WIDTH, _WIDTH // 32), device="cuda", dtype=torch.float32
        )
        s13 = deep_gemm.transform_sf_into_required_layout(
            s13, 2 * _WIDTH, _WIDTH, (1, 32), num_groups=_EXPERTS
        )
        s2 = deep_gemm.transform_sf_into_required_layout(
            s2, _WIDTH, _WIDTH, (1, 32), num_groups=_EXPERTS
        )
    else:
        w13 = torch.zeros((_EXPERTS, 2 * _WIDTH, _WIDTH), device="cuda")
        w13[0, :, 0] = 1
        w13[1, :, 0] = 2
        w13 = w13.to(torch.float8_e4m3fn)
        w2 = (
            torch.eye(_WIDTH, device="cuda")
            .expand(_EXPERTS, -1, -1)
            .contiguous()
            .to(torch.float8_e4m3fn)
        )
        s13 = torch.ones(
            (_EXPERTS, 2 * _WIDTH // _GROUP, _WIDTH // _GROUP),
            device="cuda",
            dtype=torch.float32,
        )
        s2 = torch.ones(
            (_EXPERTS, _WIDTH // _GROUP, _WIDTH // _GROUP),
            device="cuda",
            dtype=torch.float32,
        )
    return w13, w2, s13, s2


def _reference_and_bound(gate_up):
    assert gate_up.device.type == "cpu"
    gate, up = gate_up.float().chunk(2, dim=-1)
    reference = (
        _BETA
        * torch.tanh(gate / _BETA)
        * torch.sigmoid(gate)
        * _LINEAR_BETA
        * torch.tanh(up / _LINEAR_BETA)
    )
    scale = (
        reference.abs()
        .reshape(reference.shape[0], _WIDTH // _GROUP, _GROUP)
        .amax(-1)
        .clamp_min(1e-10)
        / 448.0
    )
    # Independent CPU round-up of positive normal FP32 scales to UE8M0.
    bits = scale.view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    rounded_scale = (exponent << 23).view(torch.float32)
    return reference, rounded_scale.repeat_interleave(_GROUP, dim=-1) * 17.0


class TestSituDeepGemmRunner(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("the UE8M0 SiTU runner requires Blackwell")

    def test_fp8_and_mxfp4_experts(self):
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.deep_gemm_wrapper import compile_utils

        self.assertFalse(
            compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE,
            "set SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 before importing the wrapper",
        )
        self.assertTrue(
            deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            "this regression requires the real UE8M0 DeepGEMM backend",
        )
        with (
            get_context().override_server_args(
                tp_size=1,
                attn_dp_size=1,
                enable_dp_attention=False,
                enable_symm_mem=False,
                enable_torch_compile=False,
            ),
            get_parallel().override(tp_group=None),
            get_flags().dp.override(enabled=False),
        ):
            for fp4 in (False, True):
                with self.subTest(weight_format="MXFP4" if fp4 else "FP8"):
                    self._check_runner(fp4)

    def _check_runner(self, fp4):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            DeepGemmMoeQuantInfo,
            DeepGemmRunnerCore,
            DeepGemmRunnerInput,
        )

        rows = _EXPERTS * _ROWS_PER_EXPERT
        inputs = torch.zeros((rows, _WIDTH), device="cuda", dtype=torch.bfloat16)
        inputs[:, 0] = 1
        q, scales = sglang_per_token_group_quant_fp8(
            inputs,
            _GROUP,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        layout = torch.arange(_EXPERTS, device="cuda", dtype=torch.int32)
        layout = layout.repeat_interleave(_ROWS_PER_EXPERT)
        w13, w2, s13, s2 = _make_weights(fp4)
        quant = DeepGemmMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_fp8=True,
            w13_scale=s13,
            w2_scale=s2,
            block_shape=[128, 128],
            is_fp4_experts=fp4,
        )

        # Validate the fixture with a real first GEMM. Each expert must produce
        # exactly 1 or 2, so upstream quantization cannot explain a later error.
        first = torch.empty((rows, 2 * _WIDTH), device="cuda", dtype=torch.bfloat16)
        with deep_gemm_wrapper.contiguous_layout_alignment_scope(128):
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
                (q, scales),
                (w13, s13),
                first,
                layout,
                recipe_a=(1, 128) if fp4 else None,
                recipe_b=(1, 32) if fp4 else None,
            )
        first_cpu = first.cpu().float()
        expected_first = torch.arange(1, _EXPERTS + 1, device="cpu")
        expected_first = expected_first.repeat_interleave(_ROWS_PER_EXPERT)
        self.assertTrue(
            torch.equal(first_cpu, expected_first[:, None].expand_as(first_cpu)),
            "invalid fixture: first DeepGEMM projection was not exact",
        )
        reference, bound = _reference_and_bound(first_cpu)

        config = MoeRunnerConfig(
            num_experts=_EXPERTS,
            num_local_experts=_EXPERTS,
            hidden_size=_WIDTH,
            intermediate_size_per_partition=_WIDTH,
            top_k=1,
            activation="situ",
            is_gated=True,
            gemm1_alpha=_BETA,
            gemm1_clamp_limit=_LINEAR_BETA,
        )
        runner_input = DeepGemmRunnerInput(
            hidden_states=q,
            hidden_states_scale=scales,
            use_masked_gemm=False,
            m_indices=layout,
            activation_scale_block_size=_GROUP,
        )
        state = dict(
            all_tokens=rows,
            hidden_states_shape=(rows, _WIDTH),
            hidden_states_device=inputs.device,
            hidden_states_dtype=inputs.dtype,
            contiguous_layout_alignment=128,
        )
        runner = DeepGemmRunnerCore(config)
        output = runner.run(runner_input, quant, state).hidden_states
        actual = output.cpu().float()
        error = (actual - reference).abs()
        within_bound = torch.isfinite(actual) & (error <= bound)
        self.assertTrue(
            bool(within_bound.all()),
            f"{'MXFP4' if fp4 else 'FP8'} SiTU output after two DeepGEMM projections "
            f"exceeds 17 * rounded_scale: {int((~within_bound).sum())}/{error.numel()} "
            f"elements, max error={error.max().item()}, "
            f"first reference={reference[0, 0].item()}, actual={actual[0, 0].item()}",
        )


if __name__ == "__main__":
    unittest.main()
