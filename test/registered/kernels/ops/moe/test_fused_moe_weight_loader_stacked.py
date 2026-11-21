"""FusedMoE.weight_loader_stacked must load exactly what the per-expert loop loads."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt import runtime_context
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, I, TOPK, BLOCK = 8, 256, 384, 2, 128


def _bf16_checkpoint():
    gen = torch.Generator(device="cuda").manual_seed(0)
    return {
        ("w13_weight", "w1"): torch.randn(E, I, H, device="cuda", generator=gen),
        ("w13_weight", "w3"): torch.randn(E, I, H, device="cuda", generator=gen),
        ("w2_weight", "w2"): torch.randn(E, H, I, device="cuda", generator=gen),
    }


def _block_fp8_checkpoint():
    gen = torch.Generator(device="cuda").manual_seed(1)
    checkpoint = {}
    for prefix, shard_id, shape in (
        ("w13", "w1", (I, H)),
        ("w13", "w3", (I, H)),
        ("w2", "w2", (H, I)),
    ):
        weight = torch.randn(E, *shape, device="cuda", generator=gen)
        scale = torch.rand(
            E, shape[0] // BLOCK, shape[1] // BLOCK, device="cuda", generator=gen
        )
        checkpoint[f"{prefix}_weight", shard_id] = weight.to(torch.float8_e4m3fn)
        checkpoint[f"{prefix}_weight_scale_inv", shard_id] = scale
    return checkpoint


class TestFusedMoEWeightLoaderStacked(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29641)
        torch.set_default_device("cuda")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device("cpu")

    @contextmanager
    def _runtime(self, backend, tp_size=1, tp_rank=0, ep_size=1):
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=backend),
            patch.object(runtime_context, "_validate_parallel"),
            get_parallel().override(
                moe_ep_size=ep_size,
                moe_ep_rank=0,
                moe_tp_size=tp_size,
                moe_tp_rank=tp_rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
            ),
        ):
            yield

    @staticmethod
    def _build(quant_config):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        return FusedMoE(
            num_experts=E,
            hidden_size=H,
            intermediate_size=I,
            layer_id=0,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.experts",
        ).cuda()

    @staticmethod
    def _load_per_expert(layer, checkpoint):
        for (name, shard_id), stack in checkpoint.items():
            param = getattr(layer, name)
            for expert_id in range(stack.shape[0]):
                param.weight_loader(param, stack[expert_id], name, shard_id, expert_id)

    def _load_stacked(self, layer, checkpoint):
        for (name, shard_id), stack in checkpoint.items():
            param = getattr(layer, name)
            self.assertTrue(
                layer.weight_loader_stacked(
                    param=param,
                    loaded_weight=stack,
                    weight_name=name,
                    shard_id=shard_id,
                ),
                name,
            )

    def _assert_equivalent(self, backend, quant_config, checkpoint, tp_size):
        for tp_rank in range(tp_size):
            with self._runtime(backend, tp_size, tp_rank):
                looped, stacked = self._build(quant_config), self._build(quant_config)
                self._load_per_expert(looped, checkpoint)
                self._load_stacked(stacked, checkpoint)
                for name in {name for name, _ in checkpoint}:
                    expected = getattr(looped, name).data
                    actual = getattr(stacked, name).data
                    self.assertEqual(actual.shape, expected.shape, name)
                    self.assertTrue(
                        torch.equal(
                            actual.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8),
                        ),
                        f"{name} differs at tp_rank {tp_rank}/{tp_size}",
                    )

    def test_unquantized(self):
        checkpoint = {k: v.to(torch.bfloat16) for k, v in _bf16_checkpoint().items()}
        for backend in (MoeRunnerBackend.TRITON, MoeRunnerBackend.FLASHINFER_TRTLLM):
            for tp_size in (1, 2):
                with self.subTest(backend=backend.value, tp_size=tp_size):
                    self._assert_equivalent(backend, None, checkpoint, tp_size)

    def test_online_mxfp8(self):
        checkpoint = {k: v.to(torch.bfloat16) for k, v in _bf16_checkpoint().items()}
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                self._assert_equivalent(
                    MoeRunnerBackend.FLASHINFER_TRTLLM,
                    Fp8Config(use_mxfp8=True),
                    checkpoint,
                    tp_size,
                )

    def test_block_fp8_weights_and_scales(self):
        self._assert_equivalent(
            MoeRunnerBackend.TRITON,
            Fp8Config(
                is_checkpoint_fp8_serialized=True, weight_block_size=[BLOCK, BLOCK]
            ),
            _block_fp8_checkpoint(),
            tp_size=1,
        )

    def test_declines_under_expert_parallelism(self):
        stack = torch.randn(E, I, H, dtype=torch.bfloat16)
        with self._runtime(MoeRunnerBackend.TRITON, ep_size=2):
            layer = self._build(None)
            before = layer.w13_weight.data.clone()
            self.assertFalse(
                layer.weight_loader_stacked(
                    param=layer.w13_weight,
                    loaded_weight=stack,
                    weight_name="w13_weight",
                    shard_id="w1",
                )
            )
            self.assertTrue(torch.equal(layer.w13_weight.data, before))

    def test_declines_other_tensor_kinds(self):
        with self._runtime(MoeRunnerBackend.TRITON):
            layer = self._build(None)
            param = layer.w13_weight
            for stack, name, shard_id in (
                (torch.randn(I, H, dtype=torch.bfloat16), "w13_weight", "w1"),
                (torch.randn(E, I, H, dtype=torch.bfloat16), "w13_weight", "w13"),
                (torch.randn(E, I, H, dtype=torch.bfloat16), "w13_input_scale", "w1"),
            ):
                self.assertFalse(
                    layer.weight_loader_stacked(
                        param=param,
                        loaded_weight=stack,
                        weight_name=name,
                        shard_id=shard_id,
                    )
                )


if __name__ == "__main__":
    unittest.main()
