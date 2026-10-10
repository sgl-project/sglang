# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import sglang.multimodal_gen.runtime.pipelines_core.stages.vla as vla
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)


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
        pipeline_config=SimpleNamespace(
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
    assert model.native_calls == 1 and model.noise_calls == 0
    assert "pi05_flow" not in vla.vla_state(req)
    assert result.output[0]["parameters"] == {"num_inference_steps": 5}
    assert result.output[0]["actions"] == [[0.0] * 4] * 3


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
