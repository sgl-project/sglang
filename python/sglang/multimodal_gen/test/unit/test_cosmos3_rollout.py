# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Cosmos3 RL rollout path: rollout scheduler grid
selection and fused-parameter shard-id plumbing in the weights updater."""

import types
import unittest

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.post_training.rollout_scheduler import (
    prepare_rollout_request_scheduler,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _load_weights_into_module,
)

NUM_STEPS = 16
NUM_TRAIN_TIMESTEPS = 1000


def _serving_scheduler() -> FlowUniPCMultistepScheduler:
    """Mirror the Cosmos3 checkpoint's serving scheduler (UniPC, flow
    prediction) with the T2I per-mode flow_shift default applied."""
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    scheduler.set_shift(3.0)
    scheduler.set_timesteps(NUM_STEPS, device="cpu")
    return scheduler


def _rollout_batch() -> types.SimpleNamespace:
    return types.SimpleNamespace(rollout=True, scheduler=None, timesteps=None)


def _prepare(serving, batch, explicit_shift):
    prepare_rollout_request_scheduler(
        batch,
        serving,
        explicit_shift=explicit_shift,
        num_inference_steps=NUM_STEPS,
        device=torch.device("cpu"),
    )


class TestPrepareRolloutRequestScheduler(unittest.TestCase):

    def test_inherits_serving_grid_without_explicit_shift(self):
        serving = _serving_scheduler()
        batch = _rollout_batch()

        _prepare(serving, batch, None)

        self.assertIsInstance(batch.scheduler, FlowMatchEulerDiscreteScheduler)
        self.assertIsNot(batch.scheduler, serving)
        # Serving grid inherited verbatim, terminal sigma re-appended as 0.
        torch.testing.assert_close(
            batch.scheduler.sigmas[:-1].float(),
            serving.sigmas[:-1].float(),
            atol=1e-6,
            rtol=0,
        )
        self.assertEqual(batch.scheduler.sigmas[-1].item(), 0.0)
        # batch.timesteps switches to the rollout grid (t = sigma * T).
        torch.testing.assert_close(
            batch.timesteps.float(),
            batch.scheduler.sigmas[:-1].float() * NUM_TRAIN_TIMESTEPS,
            atol=1e-3,
            rtol=0,
        )

    def test_explicit_flow_shift_selects_plain_shifted_grid(self):
        serving = _serving_scheduler()
        batch = _rollout_batch()
        shift = 2.0

        _prepare(serving, batch, shift)

        # Plain flow-match Euler grid under the requested shift, not the serving grid.
        base = torch.linspace(1.0, 1.0 / NUM_TRAIN_TIMESTEPS, NUM_STEPS)
        expected = shift * base / (1 + (shift - 1) * base)
        torch.testing.assert_close(
            batch.scheduler.sigmas[:-1].float(), expected.float(), atol=1e-5, rtol=0
        )
        self.assertFalse(
            torch.allclose(batch.scheduler.sigmas[:-1], serving.sigmas[:-1].float())
        )

    def test_rl_capable_serving_scheduler_passes_through(self):
        # An RL-capable serving scheduler is shared and keeps its serving grid.
        serving = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS
        )
        serving.set_shift(3.0)
        serving.set_timesteps(NUM_STEPS, device="cpu")
        expected = serving.sigmas.clone()
        batch = _rollout_batch()

        _prepare(serving, batch, 3.0)

        self.assertIs(batch.scheduler, serving)
        torch.testing.assert_close(serving.sigmas, expected)


class _FusedParamModule(torch.nn.Module):
    """Diffusers-style q/k/v weights that map into one fused to_qkv param."""

    def __init__(self):
        super().__init__()
        qkv = torch.nn.Module()
        qkv.weight = torch.nn.Parameter(torch.zeros(6, 2))
        attn = torch.nn.Module()
        attn.to_qkv = qkv
        self.attn = attn
        self.out_proj = torch.nn.Linear(2, 2, bias=False)
        self.param_names_mapping = {
            r"^attn\.q\.(weight)$": (r"attn.to_qkv.\1", 0, 3),
            r"^attn\.k\.(weight)$": (r"attn.to_qkv.\1", 1, 3),
            r"^attn\.v\.(weight)$": (r"attn.to_qkv.\1", 2, 3),
            r"^proj\.(weight)$": r"out_proj.\1",
        }


class TestWeightsUpdaterFusedParams(unittest.TestCase):

    def test_merge_index_reaches_weight_loader_as_shard_id(self):
        module = _FusedParamModule()
        calls = []

        def loader(param, weight, *args):
            calls.append((weight.clone(), args))

        module.attn.to_qkv.weight.weight_loader = loader

        _load_weights_into_module(
            module,
            [
                ("attn.q.weight", torch.full((2, 2), 1.0)),
                ("attn.k.weight", torch.full((2, 2), 2.0)),
                ("attn.v.weight", torch.full((2, 2), 3.0)),
                ("proj.weight", torch.full((2, 2), 4.0)),
            ],
        )

        # Fused q/k/v parts arrive with their merge index as shard id.
        self.assertEqual([args for _, args in calls], [(0,), (1,), (2,)])
        # Renamed non-fused weight lands via plain in-place copy.
        torch.testing.assert_close(module.out_proj.weight.data, torch.full((2, 2), 4.0))

    def test_direct_hit_keeps_two_arg_weight_loader_call(self):
        module = _FusedParamModule()
        calls = []

        def loader(param, weight, *args):
            calls.append(args)

        module.attn.to_qkv.weight.weight_loader = loader

        _load_weights_into_module(module, [("attn.to_qkv.weight", torch.zeros(6, 2))])

        self.assertEqual(calls, [()])


if __name__ == "__main__":
    unittest.main()
