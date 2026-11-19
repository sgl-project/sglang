"""An online-MXFP8 MoE layer must accept a weight update that is already MXFP8."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt import runtime_context
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, I, TOPK = 8, 256, 256, 2
DEVICE = torch.device("cuda")
PARAMS = ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv")


def _checkpoint(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return {
        ("w13_weight", "w1"): torch.randn(E, I, H, device=DEVICE, generator=gen),
        ("w13_weight", "w3"): torch.randn(E, I, H, device=DEVICE, generator=gen),
        ("w2_weight", "w2"): torch.randn(E, H, I, device=DEVICE, generator=gen),
    }


def _quantize(stack):
    num_experts, rows, cols = stack.shape
    weight, scale = flashinfer_mxfp8_quantize(
        stack.to(torch.bfloat16).view(-1, cols).contiguous(), False
    )
    return (
        weight.view(num_experts, rows, cols),
        scale.view(torch.uint8).contiguous().view(num_experts, rows, cols // 32),
    )


def _load(layer, name, shard_id, stack):
    param = getattr(layer, name)
    for expert_id in range(E):
        param.weight_loader(
            param, stack[expert_id], name, shard_id=shard_id, expert_id=expert_id
        )


class TestMxfp8MoEPrequantizedUpdate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29642)
        torch.set_default_device("cuda")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device("cpu")

    @contextmanager
    def _runtime(self):
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM),
            patch.object(runtime_context, "_validate_parallel"),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        ):
            yield

    @staticmethod
    def _cold_load(checkpoint):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        model = torch.nn.Module()
        model.experts = FusedMoE(
            num_experts=E,
            hidden_size=H,
            intermediate_size=I,
            layer_id=0,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
            quant_config=Fp8Config(use_mxfp8=True),
            prefix="model.layers.0.mlp.experts",
        ).cuda()
        for (name, shard_id), stack in checkpoint.items():
            _load(model.experts, name, shard_id, stack.to(torch.bfloat16))
        DefaultModelLoader.postprocess_weights(model, DEVICE)
        return model

    def test_prequantized_update_matches_cold_load(self):
        """Updating with MXFP8 weights and scales must leave the layer as a cold
        load of the same high-precision weights does; the update used to fail."""
        updated_checkpoint = _checkpoint(seed=1)
        with self._runtime():
            reference = self._cold_load(updated_checkpoint)
            model = self._cold_load(_checkpoint(seed=0))
            for (name, shard_id), stack in updated_checkpoint.items():
                weight, scale = _quantize(stack)
                _load(model.experts, name, shard_id, weight)
                _load(model.experts, f"{name}_scale_inv", shard_id, scale)
            DefaultModelLoader.postprocess_weights(model, DEVICE)

            for name in PARAMS:
                actual = getattr(model.experts, name).data
                expected = getattr(reference.experts, name).data
                self.assertEqual(actual.dtype, expected.dtype, name)
                self.assertTrue(
                    torch.equal(
                        actual.contiguous().view(torch.uint8),
                        expected.contiguous().view(torch.uint8),
                    ),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
