# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import (
    MiniMaxH3AdalnCache,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

_ARCH = MiniMaxH3DiTArchConfig(
    num_layers=2,
    hidden_size=4,
    time_embed_dim=3,
)
_BLOCK_WIDTH = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * _ARCH.hidden_size
_FINAL_WIDTH = 2 * _ARCH.hidden_size


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _weights_fill(*shape: int, scale: float) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
    return ((values % 7) * 0.01 * scale).reshape(shape)


def _write_online_weights(
    path: Path,
    *,
    omit: str | None = None,
    scale: float = 0.0,
) -> None:
    # State-machine tests only need checkpoint-compatible shapes (scale 0);
    # value-equality tests pass a nonzero scale for distinguishable outputs.
    def _fill(*shape: int) -> torch.Tensor:
        return _weights_fill(*shape, scale=scale)

    tensors: dict[str, torch.Tensor] = {}
    for layer in range(_ARCH.num_layers):
        prefix = f"blocks.{layer}.adaln_proj.linear"
        tensors[f"{prefix}.weight"] = _fill(_BLOCK_WIDTH, _ARCH.time_embed_dim)
        tensors[f"{prefix}.bias"] = _fill(_BLOCK_WIDTH)
    prefix = "final_layer.adaln_proj.linear"
    tensors[f"{prefix}.weight"] = _fill(_FINAL_WIDTH, _ARCH.time_embed_dim)
    tensors[f"{prefix}.bias"] = _fill(_FINAL_WIDTH)
    if omit is not None:
        tensors.pop(omit)
    save_file(tensors, path)


def _online_cache(
    tmp_path: Path,
    *,
    max_plans: int = 2,
    max_plan_width: int = 2,
    omit: str | None = None,
    host_cache_bytes: int = 0,
    scale: float = 0.0,
) -> MiniMaxH3AdalnCache:
    _ensure_single_process_parallel_runtime()
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path, omit=omit, scale=scale)
    cache = MiniMaxH3AdalnCache(
        _ARCH,
        weight_files=[str(weight_path)],
        max_plans=max_plans,
        max_plan_width=max_plan_width,
        host_cache_bytes=host_cache_bytes,
    )
    cache.load(torch.device("cpu"))
    return cache


# One host-tier page for the tiny arch, in bytes (see MiniMaxH3AdalnHostTier).
_PAGE_BYTES = (_ARCH.num_layers * _BLOCK_WIDTH + _FINAL_WIDTH) * 2


def _embed(timesteps: torch.Tensor) -> torch.Tensor:
    return timesteps[:, None].expand(-1, _ARCH.time_embed_dim)


def test_minimax_h3_adaln_cache_matches_bf16_embedding(tmp_path):
    cache_path = tmp_path / "adaln.safetensors"
    plan_timesteps = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    plan_lengths = torch.tensor([1, 2], dtype=torch.int64)
    block_params = (
        torch.arange(2 * 2 * 2 * _BLOCK_WIDTH, dtype=torch.float32)
        .reshape(2, 2, 2, _BLOCK_WIDTH)
        .bfloat16()
    )
    final_params = (
        torch.arange(2 * 2 * _FINAL_WIDTH, dtype=torch.float32)
        .reshape(2, 2, _FINAL_WIDTH)
        .bfloat16()
    )
    save_file(
        {
            "plan_timesteps": plan_timesteps,
            "plan_lengths": plan_lengths,
            "block_params": block_params,
            "final_params": final_params,
        },
        cache_path,
        metadata={"format_version": "2", "model_variant": "fl2va"},
    )

    cache = MiniMaxH3AdalnCache(
        _ARCH,
        path=str(cache_path),
        model_variant="fl2va",
    )
    cache.load(torch.device("cpu"))

    cache_plan_index = cache.lookup(plan_timesteps[1])
    block = cache.block(1, cache_plan_index, 2)
    final = cache.final(cache_plan_index, 2)

    # block() hands the forward pass six [num_timesteps * modality, hidden]
    # chunks, while the checkpoint stores a plan as one flat
    # [num_timesteps, 6 * modality * hidden] row -- same elements, and the
    # modality axis folds into the leading one rather than staying separate.
    assert torch.equal(
        torch.cat(block, dim=-1).reshape(block_params[1, :, 1].shape),
        block_params[1, :, 1],
    )
    assert torch.equal(torch.cat(final, dim=-1), final_params[1])


def test_sidecar_resolve_slots_and_block_all_match_per_step_paths(tmp_path):
    """Host-resolved slots and the batched gather must mirror lookup/block."""
    cache_path = tmp_path / "adaln.safetensors"
    plan_timesteps = torch.tensor([[0.5, 0.0], [1.0, 2.0]])
    plan_lengths = torch.tensor([1, 2], dtype=torch.int64)
    block_params = (
        torch.arange(2 * 2 * 2 * _BLOCK_WIDTH, dtype=torch.float32)
        .reshape(2, 2, 2, _BLOCK_WIDTH)
        .bfloat16()
    )
    final_params = torch.zeros(2, 2, _FINAL_WIDTH, dtype=torch.bfloat16)
    save_file(
        {
            "plan_timesteps": plan_timesteps,
            "plan_lengths": plan_lengths,
            "block_params": block_params,
            "final_params": final_params,
        },
        cache_path,
        metadata={"format_version": "2", "model_variant": "fl2va"},
    )
    cache = MiniMaxH3AdalnCache(_ARCH, path=str(cache_path), model_variant="fl2va")
    cache.load(torch.device("cpu"))

    slots = cache.resolve_slots([torch.tensor([0.5]), torch.tensor([1.0, 2.0])])
    assert slots.dtype == torch.int64
    assert slots.tolist() == [0, 1]
    assert int(cache.lookup(torch.tensor([1.0, 2.0]))) == int(slots[1])

    stacked = cache.block_all(slots[1], 2)
    assert len(stacked) == _ARCH.num_layers
    for index in range(_ARCH.num_layers):
        expected = cache.block(index, slots[1], 2)
        for got, want in zip(stacked[index], expected):
            assert torch.equal(got, want)
            assert got.stride() == want.stride()

    with pytest.raises(ValueError, match="does not cover"):
        cache.resolve_slots([torch.tensor([9.0])])


def test_online_cache_resolve_slots_after_build(tmp_path):
    cache = _online_cache(tmp_path, max_plan_width=2)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0, 3.0])

    cache.build([plan_a, plan_b, plan_a], embed=_embed)
    slots = cache.resolve_slots([plan_a, plan_b, plan_a])
    assert slots.tolist()[0] == slots.tolist()[2]
    assert int(cache.lookup(plan_b)) == int(slots[1])


def test_slab_buffers_stay_out_of_state_dict(tmp_path):
    cache = _online_cache(tmp_path)
    assert not any("params" in key or "plan" in key for key in cache.state_dict())


def test_host_tier_swap_in_restores_evicted_plans_bit_exactly(tmp_path):
    """A GPU-evicted plan set must return from the host tier byte-identical,
    without another checkpoint pass."""
    cache = _online_cache(
        tmp_path,
        max_plans=2,
        max_plan_width=1,
        host_cache_bytes=64 * _PAGE_BYTES,
        scale=1.0,
    )
    set_a = [torch.tensor([1.0]), torch.tensor([2.0])]
    set_b = [torch.tensor([3.0]), torch.tensor([4.0])]

    cache.build(set_a, embed=_embed)
    slots_a = cache.resolve_slots(set_a)
    snapshot = [
        (
            [t.clone() for t in cache.block(0, slots_a[i], 1)],
            [t.clone() for t in cache.final(slots_a[i], 1)],
        )
        for i in range(2)
    ]
    assert cache.stats.built_plans == 2

    cache.build(set_b, embed=_embed)  # evicts set_a from the GPU slab
    passes = cache.rebuilds
    cache.build(set_a, embed=_embed)  # swaps back in from the host tier
    assert cache.rebuilds == passes
    assert cache.stats.host_hit_plans == 2

    slots_a = cache.resolve_slots(set_a)
    for i in range(2):
        blocks, finals = snapshot[i]
        for got, want in zip(cache.block(0, slots_a[i], 1), blocks):
            assert torch.equal(got, want)
        for got, want in zip(cache.final(slots_a[i], 1), finals):
            assert torch.equal(got, want)
        assert float(cache.plan_timesteps[slots_a[i], 0]) == float(set_a[i][0])


def test_host_tier_over_capacity_group_recomputes(tmp_path):
    """A group that cannot fit is skipped (never raises) and rebuilt later."""
    cache = _online_cache(
        tmp_path,
        max_plans=2,
        max_plan_width=1,
        host_cache_bytes=1 * _PAGE_BYTES,  # one page: a 2-plan group never fits
    )
    set_a = [torch.tensor([1.0]), torch.tensor([2.0])]
    set_b = [torch.tensor([3.0]), torch.tensor([4.0])]

    cache.build(set_a, embed=_embed)
    assert cache.stats.host_pressure_skips == 1
    cache.build(set_b, embed=_embed)
    passes = cache.rebuilds
    cache.build(set_a, embed=_embed)  # host tier empty: full rebuild again
    assert cache.rebuilds == passes + 1


def test_host_tier_lru_eviction_and_shared_plan_refcount(tmp_path):
    cache = _online_cache(
        tmp_path,
        max_plans=4,
        max_plan_width=1,
        host_cache_bytes=3 * _PAGE_BYTES,
    )
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    plan_c = torch.tensor([3.0])

    cache.build([plan_a, plan_b], embed=_embed)  # group 1 {a, b}: 2 pages
    cache.build([plan_a, plan_c], embed=_embed)  # group 2 {a, c}: +1 page (a shared)
    tier = cache._host_tier
    assert tier is not None
    assert len(tier._plans) == 3 and len(tier._free_pages) == 0

    # A third group needs a page; group 1 is LRU. Its shared plan_a must
    # survive because group 2 still references it.
    plan_d = torch.tensor([4.0])
    cache.build([plan_a, plan_d], embed=_embed)
    assert cache.stats.host_evicted_groups == 1
    keys = {tuple(tier.timesteps(key).tolist()) for key in tier._plans}
    assert keys == {(1.0,), (3.0,), (4.0,)}


def test_precision_fp32_projects_in_fp32_then_stores_bf16(tmp_path):
    _ensure_single_process_parallel_runtime()
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path, scale=1.0)
    cache = MiniMaxH3AdalnCache(
        _ARCH,
        weight_files=[str(weight_path)],
        max_plans=2,
        max_plan_width=1,
        precision="fp32",
    )
    cache.load(torch.device("cpu"))
    plan = torch.tensor([0.31])

    def bf16_embed(timesteps: torch.Tensor) -> torch.Tensor:
        # The production embed stays fp32 in this mode; a bf16 input here
        # proves the cache upcasts before the projection (a 'match'-style
        # bf16 x fp32 GEMM would fail on dtype mismatch).
        return _embed(timesteps).bfloat16()

    cache.build([plan], embed=bf16_embed)
    slot = cache.resolve_slots([plan])[0]

    adaln_input = bf16_embed(plan).float()
    weight = _weights_fill(_BLOCK_WIDTH, _ARCH.time_embed_dim, scale=1.0)
    bias = _weights_fill(_BLOCK_WIDTH, scale=1.0)
    for layer in range(_ARCH.num_layers):
        expected = torch.nn.functional.linear(adaln_input, weight, bias).bfloat16()
        got = torch.cat(cache.block(layer, slot, 1), dim=-1).reshape(1, _BLOCK_WIDTH)
        assert torch.equal(got, expected)

    with pytest.raises(ValueError, match="precision"):
        MiniMaxH3AdalnCache(_ARCH, weight_files=[str(weight_path)], precision="fp64")


def test_lora_guard_rejects_adaln_keys_in_cache_mode():
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3DiTModel,
    )

    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model._adaln_precomputed = True
    adapter = {"blocks.0.adaln_proj.linear.lora_A": torch.zeros(1)}
    with pytest.raises(ValueError, match="adaln_proj"):
        MiniMaxH3DiTModel.prepare_lora_adapter(model, adapter)


def test_invalidate_drops_all_tiers_and_allows_rebuild(tmp_path):
    cache = _online_cache(
        tmp_path,
        max_plans=2,
        max_plan_width=1,
        host_cache_bytes=64 * _PAGE_BYTES,
    )
    plan_a = torch.tensor([1.0])
    cache.build([plan_a], embed=_embed)

    cache.invalidate()
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan_a)
    with pytest.raises(ValueError, match="does not cover"):
        cache.resolve_slots([plan_a])

    passes = cache.rebuilds
    cache.build([plan_a], embed=_embed)  # host tier was cleared too
    assert cache.rebuilds == passes + 1
    cache.lookup(plan_a)


def test_online_cache_eviction_preserves_in_flight_request_plans(tmp_path):
    """Capacity eviction must never drop plans reused by the current request."""
    cache = _online_cache(tmp_path, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    plan_c = torch.tensor([3.0])

    cache.build([plan_a, plan_b], embed=_embed)
    cache.build([plan_a, plan_c], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_c)
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan_b)


def test_online_cache_lru_keeps_alternating_plan_sets_resident(tmp_path):
    """Two alternating schedules must both stay resident once built.

    The pre-LRU slab did a full reset whenever slots overflowed, so two
    alternating plan sets re-read the whole checkpoint on every request.
    """
    cache = _online_cache(tmp_path, max_plans=4, max_plan_width=1)
    set_a = [torch.tensor([1.0]), torch.tensor([2.0])]
    set_b = [torch.tensor([3.0]), torch.tensor([4.0])]

    cache.build(set_a, embed=_embed)
    cache.build(set_b, embed=_embed)
    passes = cache.rebuilds
    cache.build(set_a, embed=_embed)
    cache.build(set_b, embed=_embed)
    assert cache.rebuilds == passes

    slots_a = cache.resolve_slots(set_a)
    slots_b = cache.resolve_slots(set_b)
    assert sorted(slots_a.tolist() + slots_b.tolist()) == [0, 1, 2, 3]


def test_online_cache_evicts_least_recently_used_plan_first(tmp_path):
    cache = _online_cache(tmp_path, max_plans=2, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    plan_c = torch.tensor([3.0])

    cache.build([plan_a], embed=_embed)
    cache.build([plan_b], embed=_embed)
    # Touch plan_a so plan_b becomes the LRU entry, then overflow with plan_c.
    cache.build([plan_a, plan_c], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_c)
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan_b)
    with pytest.raises(ValueError, match="does not cover"):
        cache.resolve_slots([plan_b])


def test_online_cache_failed_rebuild_can_be_retried(tmp_path):
    """A failed rebuild must not publish a cache hit that blocks its retry."""
    missing_name = "final_layer.adaln_proj.linear.bias"
    cache = _online_cache(tmp_path, omit=missing_name)
    plan_a = torch.tensor([1.0])

    with pytest.raises(KeyError, match=missing_name):
        cache.build([plan_a], embed=_embed)

    _write_online_weights(tmp_path / "model.safetensors")
    cache.build([plan_a], embed=_embed)
    cache.lookup(plan_a)


def test_online_cache_width_rejection_preserves_resident_plans(tmp_path):
    """Rejecting an over-width plan must not evict usable resident plans."""
    cache = _online_cache(tmp_path, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    wide_plan = torch.tensor([3.0, 4.0])

    cache.build([plan_a, plan_b], embed=_embed)
    with pytest.raises(ValueError, match="--minimax-h3-adaln-plan-width"):
        cache.build([wide_plan], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_b)
