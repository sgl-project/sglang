"""CHANNEL FP8 weights must retain the layout consumed by the selected runner."""

import os
import unittest

# Set before importing SGLang: quantization modules cache this flag at import.
os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

E, H, INTERMEDIATE, TOPK = 8, 1024, 1024, 2
QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*mlp\\.experts.*"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "channel",
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
    "ignore": ["lm_head"],
}


@unittest.skipUnless(is_hip(), "requires ROCm")
class TestCompressedTensorsFp8MoELayout(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29639)

    def _check_runner(self, backend):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        with (
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
                quant_config=CompressedTensorsConfig.from_config(QUANT_CONFIG),
                prefix="model.layers.0.mlp.experts",
            )
            self.assertIs(type(model.experts.scheme), CompressedTensorsW8A8Fp8MoE)
            self.assertEqual(model.experts.scheme.runner.runner_backend, backend)

            # CPU generation makes checkpoint bytes identical across GPU backends.
            gen = torch.Generator(device="cpu").manual_seed(17)
            reference_weights = {}
            for expert in range(E):
                for shard, rows, cols in (
                    ("w1", INTERMEDIATE, H),
                    ("w3", INTERMEDIATE, H),
                    ("w2", H, INTERMEDIATE),
                ):
                    weight = torch.randn(rows, cols, generator=gen, device="cpu")
                    weight /= cols**0.5
                    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448
                    quantized = (
                        (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                    )
                    reference_weights[expert, shard] = (
                        quantized.float() * scale
                    ).cuda()
                    prefix = "w2" if shard == "w2" else "w13"
                    for suffix, value in (
                        ("weight", quantized),
                        ("weight_scale", scale),
                    ):
                        name = f"{prefix}_{suffix}"
                        model.experts.weight_loader(
                            getattr(model.experts, name),
                            value.cuda(),
                            name,
                            shard_id=shard,
                            expert_id=expert,
                        )
            DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))

            for tokens in (1, 33):
                with self.subTest(tokens=tokens, backend=backend):
                    x = (
                        torch.randn(tokens, H, generator=gen, device="cpu")
                        .cuda()
                        .bfloat16()
                    )
                    logits = torch.randn(tokens, E, generator=gen, device="cpu").cuda()
                    topk = select_experts(
                        hidden_states=x,
                        router_logits=logits,
                        topk_config=TopKConfig(top_k=TOPK, renormalize=True),
                    )
                    ref = torch.zeros(tokens, H, dtype=torch.float32)
                    for expert in range(E):
                        gate = F.linear(x.float(), reference_weights[expert, "w1"])
                        up = F.linear(x.float(), reference_weights[expert, "w3"])
                        down = F.linear(
                            F.silu(gate) * up, reference_weights[expert, "w2"]
                        )
                        routing = (topk.topk_weights * (topk.topk_ids == expert)).sum(
                            dim=-1, keepdim=True
                        )
                        ref += down * routing
                    out = model.experts.forward(x, topk)
                    assert_output_close(self, out, ref, cos_threshold=0.99)
                    relative_error = (out.float() - ref).norm() / ref.norm()
                    self.assertLess(relative_error.item(), 0.1)

    def test_explicit_triton(self):
        self._check_runner(MoeRunnerBackend.TRITON)

    @unittest.skipUnless(get_bool_env_var("SGLANG_USE_AITER"), "requires AITER enabled")
    def test_aiter(self):
        self._check_runner(MoeRunnerBackend.AITER)


if __name__ == "__main__":
    unittest.main()
