"""Restore, reload and finalize of a Marlin W4A16 MoE must reproduce a fresh load."""

import unittest
from contextlib import contextmanager

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

E, H, I, TOPK, M = 8, 1024, 1024, 2, 32
GROUP_SIZE, PACK_FACTOR = 128, 8
DEVICE = torch.device("cuda")
QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*mlp\\.experts.*"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": GROUP_SIZE,
            },
            "input_activations": None,
        }
    },
    "ignore": ["lm_head"],
}


def _random_checkpoint(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    shards = {}
    for expert_id in range(E):
        for shard_id, (rows, cols) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
            shards[expert_id, shard_id] = {
                # any int32 packs eight valid int4 values
                "weight_packed": torch.randint(
                    -(2**31),
                    2**31 - 1,
                    (rows, cols // PACK_FACTOR),
                    dtype=torch.int32,
                    device=DEVICE,
                    generator=gen,
                ),
                "weight_scale": (
                    torch.rand(rows, cols // GROUP_SIZE, device=DEVICE, generator=gen)
                    / 100
                    + 1e-3
                ).to(torch.bfloat16),
                "weight_shape": torch.tensor(
                    [rows, cols], dtype=torch.float32, device=DEVICE
                ),
            }
    return shards


def _load(layer, shards):
    for (expert_id, shard_id), tensors in shards.items():
        prefix = "w2" if shard_id == "w2" else "w13"
        for suffix, tensor in tensors.items():
            name = f"{prefix}_{suffix}"
            layer.weight_loader(
                getattr(layer, name),
                tensor,
                name,
                shard_id=shard_id,
                expert_id=expert_id,
            )


class TestWNA16MoEMarlinReload(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29633)
        torch.set_default_device("cuda")

    @contextmanager
    def _runtime(self):
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.MARLIN),
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

    def _fresh_model(self, shards):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        model = torch.nn.Module()
        model.experts = FusedMoE(
            num_experts=E,
            hidden_size=H,
            intermediate_size=I,
            layer_id=0,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
            quant_config=CompressedTensorsConfig.from_config(QUANT_CONFIG),
            prefix="model.layers.0.mlp.experts",
        ).cuda()
        _load(model.experts, shards)
        DefaultModelLoader.postprocess_weights(model, DEVICE)
        return model

    @staticmethod
    def _refit(model, shards):
        # the component the scheduler drives for each runner in a session
        updater = WeightUpdater(
            tp_rank=0,
            device="cuda",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=None,
            recapture_cuda_graph=None,
            get_model_runner=None,
        )
        updater.begin_weight_update()
        _load(model.experts, shards)
        updater.end_weight_update(run_post_load=False)

    def _assert_same_weights(self, model, expected):
        expected_params = dict(expected.named_parameters())
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.equal(param, expected_params[name]),
                f"{name} differs from a fresh load",
            )

    @staticmethod
    def _forward(model, x, topk_output):
        out = model.experts.forward(x, topk_output)
        if isinstance(out, torch.Tensor):
            return out
        return out[0] if isinstance(out, tuple) else out.hidden_states

    def test_refit_reproduces_a_fresh_load(self):
        """Without restore the checkpoint shards do not fit the Marlin layout; without finalize the kernel reads raw shards."""
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        with self._runtime():
            ckpt_a, ckpt_b = _random_checkpoint(1), _random_checkpoint(2)
            fresh_a, fresh_b = self._fresh_model(ckpt_a), self._fresh_model(ckpt_b)
            model = self._fresh_model(ckpt_a)
            # SM100 auto picks the Triton subclass, which cannot be refit: its loads fail on shape
            self.assertIs(type(model.experts.scheme), CompressedTensorsWNA16MoE)
            ptrs = {name: p.data_ptr() for name, p in model.named_parameters()}

            self._refit(model, ckpt_b)
            self._assert_same_weights(model, fresh_b)
            self._refit(model, ckpt_a)
            self._assert_same_weights(model, fresh_a)
            # captured CUDA graphs keep reading the original buffers
            self.assertEqual(
                {name: p.data_ptr() for name, p in model.named_parameters()}, ptrs
            )

            torch.manual_seed(0)
            x = torch.randn(M, H, dtype=torch.bfloat16) / 10
            topk_output = select_experts(
                hidden_states=x,
                router_logits=torch.randn(M, E, dtype=torch.float32),
                topk_config=TopKConfig(top_k=TOPK, renormalize=True),
            )
            out_a = self._forward(fresh_a, x, topk_output).float()
            out_b = self._forward(fresh_b, x, topk_output).float()
            assert_output_close(
                self, self._forward(model, x, topk_output), out_a, cos_threshold=0.9999
            )
            cos_ab = torch.nn.functional.cosine_similarity(
                out_a.flatten(), out_b.flatten(), dim=0
            ).item()
            self.assertLess(
                cos_ab, 0.9, "the two checkpoints must give different outputs"
            )


if __name__ == "__main__":
    unittest.main()
