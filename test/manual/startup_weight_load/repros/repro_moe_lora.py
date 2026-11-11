"""Reproduce MoE LoRA reload limitations on one CUDA GPU.

PASS means the known loading failures were reproduced, not that overlap is
supported. This uses the real wrapper, Qwen expert loader, FusedMoE parameter
loader, and DefaultModelLoader postprocess traversal. Minimal module instances
avoid distributed setup; the unused dispatcher is mocked. These are ownership
diagnostics, not adapter-kernel or inference tests. Run this file explicitly;
it is intentionally excluded from default test discovery.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.test_utils import CustomTestCase


def _make_model(*, fp8=False):
    experts = FusedMoE.__new__(FusedMoE)
    nn.Module.__init__(experts)
    experts.num_experts = experts.num_local_experts = 2
    experts.hidden_size = experts.intermediate_size_per_partition = 128
    experts.moe_tp_rank = experts.moe_ep_rank = experts._expert_storage_rank = 0
    experts.moe_tp_size = experts.moe_ep_size = 1
    experts._num_local_routed = experts._num_global_routed = 2
    experts._has_fused_shared = False
    experts.num_fused_shared_experts = 0
    experts.scheme = None
    experts.use_flashinfer_trtllm_moe = False
    experts.use_triton_kernels = False
    experts.use_presharded_weights = False
    experts.should_fuse_routed_scaling_factor_in_topk = False
    experts.moe_runner_config = MoeRunnerConfig(
        num_experts=2,
        num_local_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=128,
        top_k=1,
        params_dtype=torch.bfloat16,
    )
    experts.dispatcher = Mock()
    experts.quant_config = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if fp8
        else None
    )
    experts.quant_method = (
        Fp8MoEMethod(experts.quant_config) if fp8 else UnquantizedFusedMoEMethod()
    )
    with torch.device("cuda"):
        experts.quant_method.create_weights(
            experts,
            num_experts=2,
            hidden_size=128,
            intermediate_size_per_partition=128,
            params_dtype=torch.bfloat16,
            weight_loader=experts.weight_loader,
        )
    if fp8:
        experts.runner = MoeRunner(MoeRunnerBackend.TRITON, experts.moe_runner_config)
    else:
        experts.quant_method.create_moe_runner(experts, experts.moe_runner_config)
        experts.runner = experts.quant_method.runner
    for parameter in experts.parameters():
        parameter.zero_()

    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=2)
    model.quant_config = experts.quant_config
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.experts = experts
    return model, experts


def _weights(value, *, fp8=False):
    for expert_id in range(2):
        for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            prefix = f"model.layers.0.mlp.experts.{expert_id}.{projection}"
            yield (
                f"{prefix}.weight",
                torch.full((128, 128), value + index + expert_id).to(
                    torch.float8_e4m3fn if fp8 else torch.bfloat16
                ),
            )
            if fp8:
                yield f"{prefix}.weight_scale_inv", torch.full((1, 1), value)


def _wrap(model, experts):
    backend = BaseLoRABackend(max_loras_per_batch=1, device=torch.device("cuda"))
    wrapper = FusedMoEWithLoRA(experts, backend)
    model.model.layers[0].mlp.experts = wrapper
    return wrapper


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestMoELoRAReloadLimitations(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(get_context().override_server_args(enable_lora=True))
        self.enterContext(get_parallel().override(tp_size=1))
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )

    @torch.no_grad()
    def test_reproduces_skipped_wrapped_expert_weights(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                model, experts = _make_model(fp8=fp8)
                model.load_weights(_weights(1.0, fp8=fp8))
                experts.quant_method.process_weights_after_loading(experts)
                initial = experts.w13_weight.float().clone()
                wrapper = _wrap(model, experts)
                cached = wrapper._quant_info
                self.assertIs(cached.w13_weight, experts.w13_weight)
                names = dict(model.named_parameters())
                self.assertNotIn("model.layers.0.mlp.experts.w13_weight", names)
                self.assertIn("model.layers.0.mlp.experts.base_layer.w13_weight", names)

                model.load_weights(_weights(7.0, fp8=fp8))
                torch.testing.assert_close(experts.w13_weight.float(), initial)

                # Diagnostic only: original names restore loading, and the cache
                # observes the in-place update. This is not a production fix.
                pointer = experts.w13_weight.data_ptr()
                model.model.layers[0].mlp.experts = experts
                try:
                    DefaultModelLoader.load_weights_and_postprocess(
                        model, _weights(7.0, fp8=fp8), torch.device("cuda")
                    )
                finally:
                    model.model.layers[0].mlp.experts = wrapper
                self.assertEqual(experts.w13_weight.data_ptr(), pointer)
                self.assertFalse(torch.equal(experts.w13_weight.float(), initial))
                self.assertIs(cached.w13_weight, experts.w13_weight)
                self.assertIs(cached.w2_weight, experts.w2_weight)
                self.assertEqual(float(cached.w13_weight[0, 0, 0]), 7.0)
                if fp8:
                    self.assertIs(cached.w13_scale, experts.w13_weight_scale_inv)
                    self.assertEqual(float(cached.w13_scale[0, 0, 0]), 7.0)
                print(
                    f"LIMITATION REPRODUCED (fp8={fp8}): wrapper hides expert names; "
                    "unwrapped loading updates the existing cache aliases"
                )

    @torch.no_grad()
    def test_reproduces_fp8_postprocess_on_wrong_owner(self):
        model, experts = _make_model(fp8=True)
        model.load_weights(_weights(1.0, fp8=True))
        experts.quant_method.process_weights_after_loading(experts)
        wrapper = _wrap(model, experts)
        owners = [
            name
            for name, module in model.named_modules()
            if getattr(module, "quant_method", None) is experts.quant_method
        ]
        self.assertEqual(
            owners,
            [
                "model.layers.0.mlp.experts",
                "model.layers.0.mlp.experts.base_layer",
            ],
        )
        self.assertIs(wrapper.quant_method, experts.quant_method)
        with self.assertRaisesRegex(AttributeError, "w13_weight") as caught:
            DefaultModelLoader.load_weights_and_postprocess(
                model, _weights(7.0, fp8=True), torch.device("cuda")
            )
        print(f"LIMITATION REPRODUCED: FP8 postprocess on wrapper: {caught.exception}")


if __name__ == "__main__":
    unittest.main()
