# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.runtime.cache.conditioning import (
    ConditioningCache,
    cached_conditioning,
    cached_vae_encode,
    invalidate_conditioning_caches,
    prefer_conditioning_cache,
)
from sglang.multimodal_gen.runtime.models.encoders.base import (
    EncoderTensorParallelMixin,
)


class Encoder(EncoderTensorParallelMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, x, attention_mask=None, **kwargs):
        self.calls += 1
        result = x * self.weight
        if attention_mask is not None:
            result = result * attention_mask
        return BaseEncoderOutput(last_hidden_state=result, hidden_states=(result,))


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.use_tiling = False

    @cached_vae_encode
    def encode(self, x):
        self.calls += 1
        return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(x.clone()))


class VisionLanguageEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.vision_calls = 0

    @cached_conditioning
    def image_features(self, pixels):
        self.vision_calls += 1
        return pixels * 2

    def forward(self, text, pixels):
        self.calls += 1
        return self.image_features(pixels) + text


@torch.no_grad()
def test_changed_instruction_reuses_vision_but_not_joint_embedding():
    cache = ConditioningCache(4096)
    model = VisionLanguageEncoder().eval()
    pixels = torch.ones(4)
    with cache.scope():
        first = model(torch.ones(4), pixels)
        changed = model(torch.full((4,), 2.0), pixels)
        assert not torch.equal(first, changed)
        assert model.calls == 2 and model.vision_calls == 1
        torch.testing.assert_close(
            model(torch.full((4,), 2.0), pixels), changed, rtol=0, atol=0
        )
        assert model.calls == 2
        model(torch.ones(4), pixels + 1)
        assert model.vision_calls == 2


@torch.no_grad()
def test_content_masks_model_identity_and_copy_isolation():
    cache = ConditioningCache(1024 * 1024)
    model = Encoder().eval()
    x = torch.arange(1024, dtype=torch.float32).reshape(1, 4, 16, 16)
    with cache.scope():
        original = model(x)
        expected = original.last_hidden_state.clone()
        original.last_hidden_state.zero_()
        hit = model(x.clone())
        assert hit.hidden_states[0] is hit.last_hidden_state
        torch.testing.assert_close(hit.last_hidden_state, expected, rtol=0, atol=0)
        hit.hidden_states[0].zero_()
        torch.testing.assert_close(model(x).last_hidden_state, expected, rtol=0, atol=0)
        changed_alpha = x.clone()
        changed_alpha[:, 3, 0, 0] += 1
        model(changed_alpha)
        model(x, attention_mask=torch.zeros_like(x))
        other_model = Encoder().eval()
        other_model(x)
    assert model.calls == 3
    assert other_model.calls == 1
    assert cache.hits == 2


@torch.no_grad()
def test_lru_budget_and_oversized_outputs():
    cache = ConditioningCache(32)
    model = Encoder().eval()
    with cache.scope():
        for value in (1, 2, 1, 3, 2):
            model(torch.full((4,), float(value)))
        before = cache.bytes
        model(torch.ones(100))
    assert model.calls == 5  # one LRU hit, then an oversized miss
    assert cache.hits == 1
    assert cache.evictions == 2
    assert cache.bytes == before == 32
    assert cache.bypasses == 1


@torch.no_grad()
def test_warmup_executes_nested_encoders_and_seeds_serving_cache():
    cache = ConditioningCache(4096)
    model = VisionLanguageEncoder().eval()
    text, pixels = torch.ones(4), torch.ones(4)
    with cache.scope():
        expected = model(text, pixels)
        with cache.scope(refresh=True):
            model(text, pixels)
            model(text, pixels)
        assert model.calls == model.vision_calls == 3
        torch.testing.assert_close(model(text, pixels), expected, rtol=0, atol=0)
        assert model.calls == model.vision_calls == 3
        model(text + 1, pixels)
        assert model.calls == 4 and model.vision_calls == 3


@pytest.mark.parametrize("bypass", ["disabled", "inactive", "training", "grad", "ar"])
def test_bypass(bypass):
    cache = ConditioningCache(0 if bypass == "disabled" else 1024)
    model = Encoder().eval()
    if bypass == "training":
        model.train()
    kwargs = {"use_cache": True} if bypass == "ar" else {}
    with (
        torch.set_grad_enabled(bypass == "grad"),
        cache.scope(enabled=bypass != "inactive"),
    ):
        model(torch.ones(4), **kwargs)
        model(torch.ones(4), **kwargs)
    assert model.calls == 2
    assert cache.hits == 0


@torch.no_grad()
def test_posterior_cache_preserves_rng_and_sampling():
    cache = ConditioningCache(1024 * 1024)
    vae = VAE().eval()
    x = torch.linspace(-1, 1, 64).reshape(1, 4, 4, 4)
    with cache.scope():
        posterior = vae.encode(x).latent_dist
        cold = posterior.sample(generator=torch.Generator().manual_seed(42))
        posterior.mean.add_(100)  # downstream normalization must not poison storage
        generator = torch.Generator().manual_seed(42)
        state = generator.get_state().clone()
        hit = vae.encode(x).latent_dist
        assert torch.equal(generator.get_state(), state)
        warm = hit.sample(generator=generator)
        torch.testing.assert_close(cold, warm, rtol=0, atol=0)
        assert not torch.equal(
            warm, hit.sample(generator=torch.Generator().manual_seed(43))
        )
        vae.use_tiling = True
        vae.encode(x)
    assert vae.calls == 2


@torch.no_grad()
def test_weight_invalidation_and_precision():
    cache = ConditioningCache(1024)
    model = Encoder().eval()
    x = torch.ones(4)
    with cache.scope():
        model(x)
        model(x)
        invalidate_conditioning_caches()
        model.weight.fill_(2)
        assert torch.equal(model(x).last_hidden_state, x * 2)
        model.to(torch.float64)
        model(x)
    assert model.calls == 3
    assert cache.hits == 1


@torch.no_grad()
def test_scoped_invalidation_preserves_independent_encoders_and_shared_weights():
    cache = ConditioningCache(1024)
    independent = Encoder().eval()
    nested = Encoder().eval()
    alias = Encoder().eval()
    alias.weight = nested.weight
    transformer = torch.nn.ModuleList([nested])
    x = torch.ones(4)
    with cache.scope():
        for model in (independent, nested, alias):
            model(x)
        invalidate_conditioning_caches([transformer])
        nested.weight.fill_(2)
        assert torch.equal(independent(x).last_hidden_state, x)
        assert torch.equal(nested(x).last_hidden_state, x * 2)
        assert torch.equal(alias(x).last_hidden_state, x * 2)
    assert (independent.calls, nested.calls, alias.calls) == (1, 2, 2)
    assert cache.bytes <= cache.max_bytes


@torch.no_grad()
def test_negative_conditioning_survives_positive_capacity_pressure():
    model = Encoder().eval()
    # only one entry fits, reproducing two LTX embeddings in the host budget
    cache = ConditioningCache(24)
    positive, negative = torch.ones(4), torch.zeros(4)
    with cache.scope(refresh=True):
        model(positive)
        with prefer_conditioning_cache():
            model(negative)
    with cache.scope():
        model(positive + 1)
        with prefer_conditioning_cache():
            actual = model(negative)
        assert model.calls == 3
        assert cache.hits == 1
        assert cache.bytes == 16
        assert cache.evictions == 1
        torch.testing.assert_close(actual.last_hidden_state, negative)
        # preferred entries still replace each other and remain bounded
        with prefer_conditioning_cache():
            model(negative - 1)
        assert cache.bytes == 16
        assert cache.evictions == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA transfers")
@torch.no_grad()
def test_cuda_snapshot_waits_for_producing_stream_before_restore():
    cache = ConditioningCache(32 * 1024 * 1024)
    model = Encoder().eval()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        value = torch.arange(1024 * 1024, device="cuda").reshape(1024, 1024)
        output = BaseEncoderOutput(
            last_hidden_state=value,
            hidden_states=(value, value.T, value[::2, ::2]),
        )
        cache.run(model, "forward", (), {}, lambda: output)
        value.zero_()
    restored = cache.run(model, "forward", (), {}, lambda: pytest.fail("cache miss"))
    expected = torch.arange(1024 * 1024).reshape(1024, 1024)
    assert restored.last_hidden_state is restored.hidden_states[0]
    for actual, reference in zip(
        restored.hidden_states, (expected, expected.T, expected[::2, ::2]), strict=True
    ):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA transfers")
@torch.no_grad()
def test_cuda_cache_hit_does_not_wait_for_unrelated_gpu_work():
    cache = ConditioningCache(4096)
    model = Encoder().eval()
    value = torch.arange(64, device="cuda")
    cache.run(model, "forward", (), {}, lambda: value)
    torch.cuda.synchronize()
    # a negative-prompt hit must not synchronize the preceding positive encode
    torch.cuda._sleep(250_000_000)
    pending = torch.cuda.Event()
    pending.record()
    restored = cache.run(model, "forward", (), {}, lambda: pytest.fail("cache miss"))
    assert not pending.query()
    torch.testing.assert_close(restored, value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
@torch.no_grad()
def test_cuda_graph_capture_bypasses_host_cache():
    cache = ConditioningCache(1024)
    model = Encoder().cuda().eval()
    x = torch.ones(4, device="cuda")
    with cache.scope():
        model(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(x)
        x.fill_(3)
        graph.replay()
        torch.testing.assert_close(output.last_hidden_state, x, rtol=0, atol=0)
    assert model.calls == 2
    assert cache.hits == 0


def _rank_eviction(rank, init_method, disabled_rank):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        cache = ConditioningCache(0 if disabled_rank and rank == 0 else 1024)
        model = Encoder().eval()
        group = SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD)
        x = torch.ones(4)

        def compute():
            # The same ordering requirement as a TP encoder forward.
            flag = torch.ones(1)
            dist.all_reduce(flag)
            return model.forward(x)

        with torch.no_grad():
            for attempt in range(3):
                if attempt == 1 and rank == 0:
                    cache.clear()
                cache.run(model, "forward", (x,), {}, compute, group)
        assert model.calls == (3 if disabled_rank else 2)
        assert cache.hits == (0 if disabled_rank else 1)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("disabled_rank", [False, True])
def test_rank_local_eviction_forces_collective_miss(tmp_path, disabled_rank):
    mp.spawn(
        _rank_eviction,
        args=(f"file://{tmp_path / 'rendezvous'}", disabled_rank),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
