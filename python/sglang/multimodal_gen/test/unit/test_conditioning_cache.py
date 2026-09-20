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
    cached_vae_encode,
    invalidate_conditioning_caches,
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
        return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(x))


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
    cache = ConditioningCache(64)
    model = Encoder().eval()
    with cache.scope():
        for value in (1, 2, 1, 3, 2):
            model(torch.full((4,), float(value)))
        before = cache.bytes
        model(torch.ones(100))
    assert model.calls == 5  # one LRU hit, then an oversized miss
    assert cache.hits == 1
    assert cache.evictions == 2
    assert cache.bytes == before == 64
    assert cache.bypasses == 1


@pytest.mark.parametrize("bypass", ["disabled", "warmup", "training", "grad", "ar"])
def test_bypass(bypass):
    cache = ConditioningCache(0 if bypass == "disabled" else 1024)
    model = Encoder().eval()
    if bypass == "training":
        model.train()
    kwargs = {"use_cache": True} if bypass == "ar" else {}
    with (
        torch.set_grad_enabled(bypass == "grad"),
        cache.scope(enabled=bypass != "warmup"),
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


def _rank_eviction(rank, init_method):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        cache = ConditioningCache(1024)
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
        assert model.calls == 2
        assert cache.hits == 1
    finally:
        dist.destroy_process_group()


def test_rank_local_eviction_forces_collective_miss(tmp_path):
    mp.spawn(
        _rank_eviction, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
