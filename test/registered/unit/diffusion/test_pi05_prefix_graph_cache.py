# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAPrefixEncodingStage,
    vla_state,
)
from sglang.multimodal_gen.runtime.vla.cuda_graph import (
    VLADenoiseGraphRunner,
    VLADenoiseGraphSignature,
    VLAPrefixGraphRunner,
    VLAPrefixGraphSignature,
)
from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
    VLAPrefixCacheManager,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


@pytest.fixture(autouse=True)
def isolated_graph_pool():
    # Each case destroys all its graphs; a later case needs a live pool handle.
    previous = get_global_graph_memory_pool()
    set_global_graph_memory_pool(None)
    try:
        yield
    finally:
        set_global_graph_memory_pool(previous)


class PrefixModel:
    """A tiny encoder exercising the real prefix graph and request stages."""

    def __init__(self):
        self.runner = VLAPrefixGraphRunner(max_entries=1)

    def build_prefix_cache_key(self, observation, **kwargs):
        return str(observation)

    def encode_prefix(self, observation, *, use_cuda_graph, **kwargs):
        values = (
            observation.state[:, 0]
            if isinstance(observation, VLAObservationBatch)
            else torch.tensor([observation])
        )
        inputs = (
            values.to("cuda")[:, None, None, None].expand(-1, 1, 3, 4).contiguous(),
        )

        def encode(tensors):
            keys = tensors[0] * 2
            return PrefixContext(
                VLADensePrefixCache([(keys, keys + 1, None)]),
                keys[:, 0, :, 0] > 0,
                3,
                {"full_attention": True},
            )

        if not use_cuda_graph:
            return encode(inputs)
        signature = VLAPrefixGraphSignature(
            len(values), (tuple(inputs[0].shape),), (str(inputs[0].dtype),), (), "test"
        )
        return self.runner.capture_or_run(signature, encode, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("capacity", [0, 1])
def test_cache_misses_replay_graph_without_overwriting_pending_requests(
    capacity, monkeypatch
):
    """Cache misses must capture graphs without overwriting earlier requests' KV.
    Same-shape prefixes finish before denoising, even if the first is evicted.
    """
    args = SimpleNamespace(
        disable_conditioning_cache=False,
        conditioning_cache_max_size_mb=1024,
        pipeline_config=SimpleNamespace(
            enable_global_prefix_cache=True, empty_cache_after_prefix=False
        ),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
        args,
    )
    model = PrefixModel()
    manager = VLAPrefixCacheManager(max_entries=capacity)
    stage = VLAPrefixEncodingStage(model, manager)

    def run(value, graphs=True):
        request = SimpleNamespace(
            is_warmup=False,
            extra={
                "vla": {
                    "observation_batch": value,
                    "options": {"enable_cuda_graph": graphs},
                }
            },
        )
        stage.forward(request, args)
        return vla_state(request)

    try:
        with torch.inference_mode():
            first = run(2.0)
            second = run(-3.0)
            again = run(2.0)
            eager = run(4.0, graphs=False)
        assert model.runner.cache_info().captures == 1
        assert model.runner.cache_info().hits >= 1
        assert not first["cache"]["hit"] and not second["cache"]["hit"]
        assert not again["cache"]["hit"]
        for state, value in ((first, 2), (second, -3), (again, 2), (eager, 4)):
            context = state["prefix_context"]
            keys, values = context.past_key_values.get_prefix(0)
            torch.testing.assert_close(keys, torch.full_like(keys, value * 2))
            torch.testing.assert_close(values, torch.full_like(values, value * 2 + 1))
            assert bool(context.prefix_pad_masks.all()) == (value > 0)
            assert not context.layout.get("mutable_graph_output", False)
    finally:
        model.runner.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_denoise_syncs_prefix_once_per_observation(monkeypatch):
    """One prefix replay must not cause a full KV copy at every denoise step."""
    import sglang.multimodal_gen.runtime.vla.cuda_graph as graphs

    prefix_model = PrefixModel()
    runner = VLADenoiseGraphRunner(max_entries=1)
    signature = VLADenoiseGraphSignature(1, 3, False, 3, 4, "float32", "test")
    original_copy = graphs._copy_prefix_context_
    copies = []

    def copy(dst, src):
        copies.append(getattr(src, "revision", 0))
        return original_copy(dst, src)

    monkeypatch.setattr(graphs, "_copy_prefix_context_", copy)

    def step(context, x, t):
        return (
            x + t * context.past_key_values[0][0].sum() + context.prefix_pad_masks.sum()
        )

    try:
        with torch.inference_mode():
            x = torch.ones(1, 3, 4, device="cuda")
            for index, value in enumerate((2.0, -3.0, 5.0)):
                context = prefix_model.encode_prefix(value, use_cuda_graph=True)
                for time in (0.1, 0.5, 0.9):
                    t = torch.tensor([time], device="cuda")
                    actual = runner.capture_or_run(signature, step, context, x, t)
                    torch.testing.assert_close(
                        actual, step(context, x, t), rtol=0, atol=0
                    )
                assert len(copies) == index
    finally:
        runner.clear()
        prefix_model.runner.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("group_size", [1, 2])
def test_uncached_pending_requests_survive_later_prefix_replays(
    monkeypatch, group_size
):
    """Different request groups can share a graph despite different denoise steps."""
    args = SimpleNamespace(
        disable_conditioning_cache=True,
        enable_layerwise_nvtx_marker=False,
        comfyui_mode=True,
        pipeline_config=SimpleNamespace(empty_cache_after_prefix=False),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
        args,
    )
    model = PrefixModel()
    stage = VLAPrefixEncodingStage(model, VLAPrefixCacheManager(0))
    requests = [
        SimpleNamespace(
            is_warmup=False,
            metrics=None,
            perf_dump_path=None,
            generator=torch.Generator(),
            num_inference_steps=10 if value > 0 else 7,
            action_dim=4,
            action_horizon=3,
            extra={
                "vla": {
                    "observation_batch": VLAObservationBatch(
                        prompt=["test"],
                        images={},
                        image_masks={},
                        state=torch.tensor([[value]]),
                        noise=None,
                        tokens=torch.ones(1, 3, dtype=torch.long),
                        token_masks=torch.ones(1, 3, dtype=torch.bool),
                        batch_size=1,
                    )
                }
            },
        )
        for value in (2.0,) * group_size + (-3.0,) * group_size
    ]
    if group_size > 1:
        for request in requests:
            request.generator = None
    try:
        with torch.inference_mode():
            results = stage.run_grouped_requests(requests, args)
        assert model.runner.cache_info().captures == 1
        assert model.runner.cache_info().hits == 1
        for result, value in zip(results, (4.0,) * group_size + (-6.0,) * group_size):
            context = vla_state(result)["prefix_context"]
            keys = context.past_key_values[0][0]
            torch.testing.assert_close(
                keys, torch.full_like(keys, value), rtol=0, atol=0
            )
    finally:
        model.runner.clear()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
