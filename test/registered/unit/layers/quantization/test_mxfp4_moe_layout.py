"""MXFP4 MoE weights must retain the layout consumed by the selected runner."""

import os
import unittest
from unittest.mock import patch

# Set before importing SGLang: quantization modules cache this flag at import.
os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import mxfp4 as mxfp4_module
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

E, H, INTERMEDIATE, TOPK = 8, 512, 512, 2
TOKENS = (1, 33)


def _checkpoint():
    # CPU generation makes checkpoint bytes identical across runs and backends.
    gen = torch.Generator(device="cpu").manual_seed(23)

    def packed(*shape):
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    def scales(*shape):
        # Biased E8M0 exponents 2^-9 .. 2^-6 keep the bf16 activations bounded.
        return torch.randint(118, 122, shape, generator=gen, dtype=torch.uint8)

    def bias(*shape):
        return (torch.randn(shape, generator=gen) * 0.01).bfloat16()

    weights = {
        "w13_weight": packed(E, 2 * INTERMEDIATE, H // 2),
        "w13_weight_scale": scales(E, 2 * INTERMEDIATE, H // 32),
        "w13_weight_bias": bias(E, 2 * INTERMEDIATE),
        "w2_weight": packed(E, H, INTERMEDIATE // 2),
        "w2_weight_scale": scales(E, H, INTERMEDIATE // 32),
        "w2_weight_bias": bias(E, H),
    }
    inputs = [
        (
            torch.randn(tokens, H, generator=gen).bfloat16(),
            torch.randn(tokens, E, generator=gen),
        )
        for tokens in TOKENS
    ]
    return weights, inputs


@unittest.skipUnless(is_hip(), "requires ROCm")
class TestMxfp4MoELayout(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29643)
        cls.weights, cls.inputs = _checkpoint()

    def _run(self, backend, use_aiter, w13_shape=None):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        with (
            patch.object(mxfp4_module, "_use_aiter", use_aiter),
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=backend),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
            torch.device("cuda"),
            torch.inference_mode(),
        ):
            model = torch.nn.Module()
            model.experts = FusedMoE(
                num_experts=E,
                hidden_size=H,
                intermediate_size=INTERMEDIATE,
                layer_id=0,
                top_k=TOPK,
                params_dtype=torch.bfloat16,
                quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
                prefix="model.layers.0.mlp.experts",
                with_bias=True,
                gemm1_alpha=1.702,
                gemm1_clamp_limit=10.0,
            )
            experts = model.experts
            self.assertIs(type(experts.quant_method), Mxfp4MoEMethod)
            # The MXFP4 loader copies checkpoint tensors into the leading slice.
            for name, value in self.weights.items():
                param = getattr(experts, name)
                param.data[tuple(slice(0, size) for size in value.shape)].copy_(value)
            DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
            if w13_shape is not None:
                self.assertEqual(tuple(experts.w13_weight.shape), w13_shape)

            outputs = []
            for x, logits in self.inputs:
                x, logits = x.cuda(), logits.cuda()
                topk = select_experts(
                    hidden_states=x,
                    router_logits=logits,
                    topk_config=TopKConfig(top_k=TOPK, renormalize=True),
                )
                outputs.append(experts.forward(x, topk).float())
            return experts, outputs

    def test_explicit_triton_ignores_aiter_env(self):
        # The Triton runner consumes bf16 weights upcast from the checkpoint.
        w13_shape = (E, 2 * INTERMEDIATE, H)
        _, reference = self._run(MoeRunnerBackend.TRITON, False, w13_shape)
        with self.subTest(use_aiter=True):
            experts, outputs = self._run(MoeRunnerBackend.TRITON, True, w13_shape)
            self.assertEqual(experts.quant_method.runner.runner_backend.value, "triton")
            for out, ref in zip(outputs, reference):
                torch.testing.assert_close(out, ref)

    def test_explicit_aiter_matches_triton(self):
        _, reference = self._run(MoeRunnerBackend.TRITON, use_aiter=False)
        experts, outputs = self._run(MoeRunnerBackend.AITER, use_aiter=True)
        self.assertEqual(experts.quant_method.runner.runner_backend.value, "aiter")
        for out, ref in zip(outputs, reference):
            assert_output_close(self, out, ref, cos_threshold=0.99)

    def test_auto_with_aiter_env_matches_explicit_aiter(self):
        _, reference = self._run(MoeRunnerBackend.AITER, use_aiter=True)
        experts, outputs = self._run(MoeRunnerBackend.AUTO, use_aiter=True)
        self.assertEqual(experts.quant_method.runner.runner_backend.value, "aiter")
        for out, ref in zip(outputs, reference):
            torch.testing.assert_close(out, ref)

    def test_auto_without_aiter_env_matches_explicit_triton(self):
        _, reference = self._run(MoeRunnerBackend.TRITON, use_aiter=False)
        experts, outputs = self._run(MoeRunnerBackend.AUTO, use_aiter=False)
        self.assertEqual(experts.quant_method.runner.runner_backend.value, "triton")
        for out, ref in zip(outputs, reference):
            torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    unittest.main()
