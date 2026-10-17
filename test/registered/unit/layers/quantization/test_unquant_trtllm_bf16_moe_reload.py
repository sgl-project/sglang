"""Refit of flashinfer_trtllm BF16 MoE experts must reproduce a fresh load on both runners."""

import unittest
from contextlib import contextmanager

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

# The trtllm-gen block layout needs 128-aligned hidden and intermediate sizes.
E, H, I, TOPK = 4, 256, 256, 2
PREFIX = "model.layers.0.mlp.experts"
DEVICE = torch.device("cuda")
CANONICAL_SHAPES = {"w13_weight": (E, 2 * I, H), "w2_weight": (E, H, I)}
RUNNERS = (
    MoeRunnerBackend.FLASHINFER_TRTLLM,
    MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
)


def _random_checkpoint(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return {
        (expert_id, shard_id): torch.randn(
            rows, cols, dtype=torch.bfloat16, device=DEVICE, generator=gen
        )
        for expert_id in range(E)
        for shard_id, (rows, cols) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I)))
    }


def _load(layer, shards):
    for (expert_id, shard_id), tensor in shards.items():
        name = "w2_weight" if shard_id == "w2" else "w13_weight"
        # the dotted checkpoint name, as model.load_weights passes it
        layer.weight_loader(
            getattr(layer, name),
            tensor,
            f"{PREFIX}.{name}",
            shard_id=shard_id,
            expert_id=expert_id,
        )


def _shapes(layer):
    return {name: tuple(getattr(layer, name).shape) for name in CANONICAL_SHAPES}


class TestFlashinferTrtllmBf16MoeReload(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29634)
        torch.set_default_device("cuda")

    @contextmanager
    def _runtime(self, backend):
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
            prefix=PREFIX,
        ).cuda()
        _load(model.experts, shards)
        DefaultModelLoader.postprocess_weights(model, DEVICE)
        return model

    @staticmethod
    def _updater(model):
        # the component the scheduler drives for each runner in a session
        return WeightUpdater(
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

    def _refit(self, model, shards):
        updater = self._updater(model)
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

    def test_session_restores_the_load_shape_and_finalize_repacks(self):
        """begin_weight_update must hand back the checkpoint shapes on both trtllm runners;
        end_weight_update must repack them. Before the restore hook existed the
        non-routed runner left the block layout in place."""
        for backend in RUNNERS:
            with self.subTest(backend=backend.value), self._runtime(backend):
                model = self._fresh_model(_random_checkpoint(1))
                packed = _shapes(model.experts)
                self.assertNotEqual(packed, CANONICAL_SHAPES)

                updater = self._updater(model)
                updater.begin_weight_update()
                self.assertEqual(_shapes(model.experts), CANONICAL_SHAPES)
                updater.end_weight_update(run_post_load=False)
                self.assertEqual(_shapes(model.experts), packed)

    def test_refit_reproduces_a_fresh_load(self):
        """A checkpoint shard cannot be copied into the 4-D block layout; the routed runner
        restored per tensor at load time, the non-routed one raised on the copy."""
        for backend in RUNNERS:
            with self.subTest(backend=backend.value), self._runtime(backend):
                ckpt_a, ckpt_b = _random_checkpoint(1), _random_checkpoint(2)
                fresh_a, fresh_b = self._fresh_model(ckpt_a), self._fresh_model(ckpt_b)
                model = self._fresh_model(ckpt_a)
                ptrs = {name: p.data_ptr() for name, p in model.named_parameters()}

                self._refit(model, ckpt_b)
                self._assert_same_weights(model, fresh_b)
                self._refit(model, ckpt_a)
                self._assert_same_weights(model, fresh_a)
                # captured CUDA graphs keep reading the original buffers
                self.assertEqual(
                    {name: p.data_ptr() for name, p in model.named_parameters()}, ptrs
                )


if __name__ == "__main__":
    unittest.main()
