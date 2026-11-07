# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import pytest
import torch

import sglang.multimodal_gen.runtime.pipelines_core.stages.vla as vla
from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
    slice_prefix_context,
)
from sglang.test.test_utils import CustomTestCase


class Model:
    device = torch.device("cpu")
    config = SimpleNamespace(action_horizon=3)

    def __init__(self):
        self.prefix_calls = self.noise_calls = self.native_calls = 0
        self.core_model = SimpleNamespace(prepare_denoise_layout=lambda *a, **kw: None)

    def _offload_action_expert_between_requests(self):
        return False

    def _can_use_action_sequence_parallel(self, *args):
        return False

    def action_parallel_info(self, prefix):
        return {"action_sequence_parallel": False}

    def encode_prefix(self, observation, **kwargs):
        self.prefix_calls += 1
        self.last_prefix_graph = kwargs["use_cuda_graph"]
        n = observation.length
        kv = torch.ones(1, 1, n, 4)
        return PrefixContext(
            VLADensePrefixCache([(kv, kv.clone(), None)]),
            torch.ones(1, n, dtype=torch.bool),
            n,
            {"full_attention": True},
        )

    def sample_noise(self, size, generator=None):
        self.noise_calls += 1
        return torch.ones(size, 3, 4)

    def denoise_step(self, pc, x, t, **kwargs):
        return 0.2 * x.sin() + t[:, None, None] * 0.1

    def sample_actions(self, observation, pc, **kwargs):
        self.native_calls += 1
        return torch.zeros(1, 3, 4)


def request(name, steps, length):
    return SimpleNamespace(
        request_id=name,
        num_inference_steps=steps,
        is_warmup=False,
        generator=None,
        seed=42,
        generator_device=None,
        prompt="task",
        metrics=None,
        extra={
            "vla": {
                "observation": {"length": length},
                "options": {"enable_cuda_graph": False},
            }
        },
    )


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(vla, "get_vla_split_group", lambda: None)
    model = Model()
    calls = []

    def preprocess(raw):
        calls.append(raw)
        return SimpleNamespace(length=raw["length"], batch_size=1, noise=None)

    args = SimpleNamespace(
        disable_conditioning_cache=True,
        pipeline_config=Pi05PipelineConfig(
            enable_segmented_actions=True,
            empty_cache_after_prefix=False,
            output_action_dim=4,
        ),
    )
    return (
        model,
        calls,
        args,
        vla.VLAObservationPreprocessStage(preprocess),
        vla.VLAPrefixEncodingStage(model, None),
        vla.VLAActionDenoisingStage(model),
        vla.VLAActionPostprocessStage(),
    )


def test_stage_resume_skips_preprocessing_prefix_and_noise(setup):
    model, calls, args, prep, prefix, action, post = setup
    a, b, c = request("a", 3, 2), request("b", 5, 5), request("c", 4, 3)

    def run(reqs, length):
        for req in reqs:
            vla.vla_state(req)["pi05_segment_length"] = length
            prep.forward(req, args)
            prefix.forward(req, args)
        action.run_grouped_requests(reqs, args)
        return [post.forward(req, args) for req in reqs]

    results = run([a, b], 3)
    assert "actions" in results[0].output[0]
    assert results[1].output == [None]
    original_prefix = vla.vla_state(b)["prefix_context"]
    results = run([b, c], 2)
    assert results[0].output[0]["parameters"]["num_inference_steps"] == 5
    assert results[1].output == [None]
    assert model.prefix_calls == model.noise_calls == len(calls) == 3
    assert model.native_calls == 0
    assert model.last_prefix_graph is False
    assert vla.vla_state(b)["prefix_context"] is original_prefix
    assert vla.vla_state(c)["pi05_flow"].remaining == 2


def test_unmarked_request_uses_existing_whole_trajectory_path(setup):
    model, calls, args, prep, prefix, action, post = setup
    args.pipeline_config.enable_segmented_actions = False
    req = request("native", 5, 3)
    prep.forward(req, args)
    prefix.forward(req, args)
    action.forward(req, args)
    result = post.forward(req, args)
    assert model.native_calls == 1 and model.noise_calls == 1
    assert "pi05_flow" not in vla.vla_state(req)
    assert result.output[0]["parameters"] == {"num_inference_steps": 5}
    assert result.output[0]["actions"] == [[0.0] * 4] * 3


def test_parked_request_releases_retired_cohort_storage(setup):
    """A survivor must not retain the original batched prefix or observations."""
    _, _, args, prep, _, action, post = setup
    a, b = request("short", 3, 5), request("long", 5, 5)
    kv = torch.ones(2, 1, 5, 4)
    group = PrefixContext(
        VLADensePrefixCache([(kv, kv.clone(), None)]),
        torch.ones(2, 5, dtype=torch.bool),
        5,
        {"full_attention": True},
    )
    for i, req in enumerate((a, b)):
        prep.forward(req, args)
        vla.vla_state(req).update(
            pi05_segment_length=3,
            prefix_context=slice_prefix_context(group, i),
            prefix_context_group=group,
            observation_group=object(),
        )
    action.run_grouped_requests([a, b], args)
    state = vla.vla_state(b)
    flow = state["pi05_flow"]
    assert flow.remaining == 2
    assert (
        not {"prefix_context_group", "observation_group", "observation_batch"}
        & state.keys()
    )
    compact = state["prefix_context"]
    assert compact is flow.prefix
    tensors = [compact.prefix_pad_masks, flow.x_t]
    tensors.extend(compact.past_key_values[0][:2])
    for tensor in tensors:
        assert (
            tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        )
    torch.testing.assert_close(compact.past_key_values[0][0], kv[1:])
    # Repeated parking is zero-copy for an already compact prefix.
    latent = flow.x_t
    flow.compact()
    assert flow.prefix is compact
    assert flow.x_t is latent
    state["pi05_segment_length"] = 2
    action.forward(b, args)
    resumed = post.forward(b, args).output[0]["actions"]
    native = request("reference", 5, 5)
    prep.forward(native, args)
    vla.vla_state(native).update(
        pi05_segment_length=5, prefix_context=slice_prefix_context(group, 1)
    )
    action.forward(native, args)
    assert resumed == post.forward(native, args).output[0]["actions"]


def test_marked_request_requires_opt_in(setup):
    _, _, args, prep, prefix, action, _ = setup
    args.pipeline_config.enable_segmented_actions = False
    req = request("a", 3, 3)
    vla.vla_state(req)["pi05_segment_length"] = 3
    prep.forward(req, args)
    prefix.forward(req, args)
    with pytest.raises(RuntimeError, match="not enabled"):
        action.forward(req, args)


def test_runtime_rejects_split_execution(setup, monkeypatch):
    _, _, args, _, _, action, _ = setup
    req = request("a", 3, 3)
    vla.vla_state(req)["pi05_segment_length"] = 3
    monkeypatch.setattr(vla, "get_vla_split_group", lambda: object())
    with pytest.raises(RuntimeError, match="split VLA"):
        action.forward(req, args)


def test_segment_disables_mutable_prefix_graph_output(setup):
    _, _, args, prep, prefix, _, _ = setup
    req = request("a", 3, 3)
    vla.vla_state(req)["options"]["enable_cuda_graph"] = True
    assert vla._cuda_graph_enabled(req) is True
    vla.vla_state(req)["pi05_segment_length"] = 3
    assert vla._cuda_graph_enabled(req) is False


class SeededModel(Model):
    def sample_noise(self, size, generator=None):
        self.noise_calls += 1
        return torch.randn(size, 3, 4, generator=generator)

    def sample_actions(self, observation, pc, **kwargs):
        self.native_calls += 1
        return kwargs["noise"]

    def denoise_step(self, pc, x, t, **kwargs):
        return torch.zeros_like(x)


class TestPi05RequestSeeds(CustomTestCase):
    def setUp(self):
        self.model = SeededModel()
        self.stage = vla.VLAActionDenoisingStage(self.model)
        self.args = SimpleNamespace(
            pipeline_config=Pi05PipelineConfig(
                enable_segmented_actions=True, output_action_dim=4
            )
        )
        kv = torch.ones(1, 1, 2, 4)
        self.prefix = PrefixContext(
            VLADensePrefixCache([(kv, kv.clone(), None)]),
            torch.ones(1, 2, dtype=torch.bool),
            2,
            {"full_attention": True},
        )

    def make_request(self, seed, noise=None):
        req = request(str(seed), 3, 2)
        req.seed = seed
        req.generator_device = "cpu"
        vla.vla_state(req).update(
            observation_batch=SimpleNamespace(batch_size=1, noise=noise),
            prefix_context=self.prefix,
        )
        return req

    def single(self, req):
        self.stage.forward(req, self.args)
        return vla.vla_state(req)["actions"]

    def test_seed_repeats_and_changes_noise_without_using_global_rng(self):
        rng = torch.random.get_rng_state()
        first = self.single(self.make_request(17))
        torch.testing.assert_close(first, self.single(self.make_request([17])))
        self.assertFalse(torch.equal(first, self.single(self.make_request(18))))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_grouped_noise_matches_single_requests_in_either_order(self):
        expected = {seed: self.single(self.make_request(seed))[0] for seed in (17, 18)}
        for seeds in ((17, 18), (18, 17)):
            reqs = [self.make_request(seed) for seed in seeds]
            group = object()
            for req in reqs:
                vla.vla_state(req).update(
                    prefix_context_group=group,
                    observation_group=SimpleNamespace(noise=None),
                )
            calls = self.model.native_calls
            self.stage.run_grouped_requests(reqs, self.args)
            self.assertEqual(self.model.native_calls, calls + 1)
            for seed, req in zip(seeds, reqs):
                torch.testing.assert_close(
                    torch.tensor(vla.vla_state(req)["actions_output"]), expected[seed]
                )
                self.assertIsNone(req.generator)

    def test_explicit_noise_overrides_seed(self):
        noise = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        for seed in (17, 18):
            torch.testing.assert_close(
                self.single(self.make_request(seed, noise)), noise
            )
        self.assertEqual(self.model.noise_calls, 0)

    def test_segment_continuation_reuses_seeded_initial_noise(self):
        expected = self.single(self.make_request(17))
        req = self.make_request(17)
        state = vla.vla_state(req)
        state["pi05_segment_length"] = 1
        self.stage.forward(req, self.args)
        calls = self.model.noise_calls
        state["pi05_segment_length"] = 2
        self.stage.forward(req, self.args)
        self.assertEqual(self.model.noise_calls, calls)
        self.assertEqual(state["pi05_flow"].remaining, 0)
        torch.testing.assert_close(state["actions"], expected)


if __name__ == "__main__":
    unittest.main()
