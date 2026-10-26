"""CPU-only tests: dense fused LoRA modules (qkv, gate_up, MLA qkv_a, Inkling qkvr) load each adapter
projection into its own slot of the fused buffer.

A complete adapter must land byte-identical to the previous concatenate-then-slice loader, and any
subset of a fused module's projections must load with the absent projections left zero."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.layers import (
    InklingQKVRLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
)
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HIDDEN = 4
_RANK = 2
_PREFIX = "base_model.model.model.layers.0."


# Reference fused-B slicing, including independent Q/K and V head widths.
# The Inkling and merged references follow the previous concatenate-then-slice loader.


def _reference_qkv_slice(bl, B):
    kv_shard_id = bl.tp_rank // bl.num_kv_head_replicas
    q_size = bl.total_num_heads * bl.head_size
    k_size = bl.total_num_kv_heads * bl.head_size
    q_start = bl.tp_rank * bl.q_proj_shard_size
    k_start = q_size + kv_shard_id * bl.kv_proj_shard_size
    v_start = q_size + k_size + kv_shard_id * bl.v_proj_shard_size
    return torch.concat(
        (
            B[q_start : q_start + bl.q_proj_shard_size],
            B[k_start : k_start + bl.kv_proj_shard_size],
            B[v_start : v_start + bl.v_proj_shard_size],
        )
    )


def _reference_inkling_slice(bl, B):
    hd, nkv, nh, dr, tp, tp_rank = (
        bl.inkling_head_dim,
        bl.inkling_num_kv_heads,
        bl.inkling_num_heads,
        bl.inkling_d_rel,
        bl.inkling_tp_size,
        bl.tp_rank,
    )
    q_size, kv_size, r_size = hd * nh, hd * nkv, dr * nh
    k_off, v_off, r_off = q_size, q_size + kv_size, q_size + 2 * kv_size
    q_per, r_per = q_size // tp, r_size // tp
    q = B[tp_rank * q_per : (tp_rank + 1) * q_per]
    r = B[r_off + tp_rank * r_per : r_off + (tp_rank + 1) * r_per]
    if tp > nkv:
        kv_idx = tp_rank // (tp // nkv)
        k = B[k_off + kv_idx * hd : k_off + (kv_idx + 1) * hd]
        v = B[v_off + kv_idx * hd : v_off + (kv_idx + 1) * hd]
    else:
        kv_per = kv_size // tp
        k = B[k_off + tp_rank * kv_per : k_off + (tp_rank + 1) * kv_per]
        v = B[v_off + tp_rank * kv_per : v_off + (tp_rank + 1) * kv_per]
    return torch.concat([q, k, v, r])


def _reference_merged_slice(bl, B):
    slices, offset = [], 0
    for full, part in zip(bl.output_sizes, bl.output_partition_sizes):
        slices.append(B[offset + bl.tp_rank * part : offset + bl.tp_rank * part + part])
        offset += full
    return torch.concat(slices)


# --- one fused layer of each kind ---------------------------------------------------------------


def _wrap(cls, base_layer, **attrs):
    wrapper = cls.__new__(cls)
    object.__setattr__(wrapper, "base_layer", base_layer)
    for name, value in attrs.items():
        object.__setattr__(wrapper, name, value)
    return wrapper


def _group(tp_rank, tp_size):
    # The wrappers read rank and size from the base layer's process group.
    return SimpleNamespace(rank_in_group=tp_rank, world_size=tp_size)


def _qkv(tp_rank, tp_size=4, head_size=2, v_head_size=2):
    group = _group(tp_rank, tp_size)
    bl = SimpleNamespace(
        tp_rank=tp_rank,
        tp_group=group,
        kv_tp_group=group,
        total_num_heads=4,
        total_num_kv_heads=2,
        head_size=head_size,
        v_head_size=v_head_size,
        q_proj_shard_size=4 * head_size // tp_size,
        kv_proj_shard_size=max(1, 2 // tp_size) * head_size,
        v_proj_shard_size=max(1, 2 // tp_size) * v_head_size,
        num_kv_head_replicas=max(1, tp_size // 2),
    )
    return (
        "self_attn.qkv_proj",
        ("q_proj", "k_proj", "v_proj"),
        (4 * head_size, 2 * head_size, 2 * v_head_size),
        _wrap(QKVParallelLinearWithLoRA, bl),
        lambda B: _reference_qkv_slice(bl, B),
    )


def _gate_up(tp_rank, tp_size=2):
    bl = SimpleNamespace(
        tp_rank=tp_rank,
        tp_group=_group(tp_rank, tp_size),
        output_sizes=[6, 6],
        output_partition_sizes=[6 // tp_size] * 2,
    )
    return (
        "mlp.gate_up_proj",
        ("gate_proj", "up_proj"),
        (6, 6),
        _wrap(MergedColumnParallelLinearWithLoRA, bl),
        lambda B: _reference_merged_slice(bl, B),
    )


def _qkvr(tp_rank, tp_size=4):
    bl = SimpleNamespace(
        tp_rank=tp_rank,
        tp_group=_group(tp_rank, tp_size),
        inkling_head_dim=2,
        inkling_num_heads=4,
        inkling_num_kv_heads=2,
        inkling_d_rel=1,
        inkling_tp_size=tp_size,
    )
    return (
        "attn.qkvr",
        ("wq_du", "wk_dv", "wv_dv", "wr_du"),
        (8, 4, 4, 4),
        _wrap(InklingQKVRLinearWithLoRA, bl),
        lambda B: _reference_inkling_slice(bl, B),
    )


def _mla_qkv_a(tp_rank, tp_size=1):
    del tp_rank, tp_size
    wrapper = _wrap(
        ReplicatedLinearWithLoRA, SimpleNamespace(), first_output_dim=3, output_size=5
    )
    return (
        "self_attn.fused_qkv_a_proj_with_mqa",
        ("q_a_proj", "kv_a_proj_with_mqa"),
        (3, 2),
        wrapper,
        lambda B: B,
    )


_LAYERS = {
    "qkv": (_qkv, 4),
    "narrow_v_qkv": (lambda tp_rank: _qkv(tp_rank, head_size=3, v_head_size=2), 4),
    "gate_up": (_gate_up, 2),
    "qkvr": (_qkvr, 4),
    "mla_qkv_a": (_mla_qkv_a, 1),
}


# --- the memory pool under test -------------------------------------------------------------------


def _pool(module_path, wrapper, full_rows, max_rank):
    fused = module_path.rsplit(".", 1)[-1]
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.num_layer = 1
    pool.tp_rank = (
        wrapper.base_layer.tp_rank if hasattr(wrapper.base_layer, "tp_rank") else 0
    )
    pool.max_lora_rank = max_rank
    pool.target_modules = {fused}
    pool.strict_loading = True
    pool.pin_memory_available = False
    pool.enable_lora_overlap_loading = False
    pool.experts_shared_outer_loras = False
    pool.base_model = torch.nn.Identity()
    pool.lora_added_tokens_size = 0
    pool.embedding_A_buffer = {}
    pool.embedding_B_buffer = {}
    pool.lm_head_A_buffer = {}
    pool.lm_head_B_buffer = {}
    pool.new_embeddings_buffer = {}
    local_rows = sum(
        wrapper.slice_lora_b_member(i, torch.zeros(rows, 1)).shape[0]
        for i, rows in enumerate(full_rows)
    )
    pool.A_buffer = {fused: [torch.full((1, max_rank * len(full_rows), _HIDDEN), -9.0)]}
    pool.B_buffer = {fused: [torch.full((1, local_rows, max_rank), -9.0)]}
    return pool


def _load(pool, module_path, wrapper, weights):
    adapter = SimpleNamespace(
        config=SimpleNamespace(r=_RANK),
        scaling=1.0,
        layers=[SimpleNamespace(weights=weights, pinned_weights={})],
        embedding_layers={},
        added_tokens_embeddings={},
    )
    pool.load_lora_weight_to_buffer(
        uid="adapter",
        buffer_id=0,
        lora_adapter=adapter,
        lora_modules=[{f"model.layers.0.{module_path}": wrapper}],
        lora_embed_tokens_module=None,
        lora_lm_head_module=None,
    )
    fused = module_path.rsplit(".", 1)[-1]
    return pool.A_buffer[fused][0][0], pool.B_buffer[fused][0][0]


def _member_weights(members, full_rows, present):
    torch.manual_seed(0)
    a = {m: torch.randn(_RANK, _HIDDEN) for m in members}
    b = {m: torch.randn(rows, _RANK) for m, rows in zip(members, full_rows)}
    block = _PREFIX + (
        "attn."
        if members[0] == "wq_du"
        else "mlp."
        if members[0] == "gate_proj"
        else "self_attn."
    )
    weights = {}
    for m in present:
        weights[f"{block}{m}.lora_A.weight"] = a[m]
        weights[f"{block}{m}.lora_B.weight"] = b[m]
    return weights, a, b


def _expected(members, full_rows, a, b, present, reference_slice):
    """The previous loader's buffer for the adapter with every absent member written as zeros."""
    fused_a = torch.cat(
        [a[m] if m in present else torch.zeros_like(a[m]) for m in members]
    )
    fused_b = torch.cat(
        [
            b[m] if m in present else torch.zeros(rows, _RANK)
            for m, rows in zip(members, full_rows)
        ]
    )
    return fused_a, reference_slice(fused_b)


def _cases():
    for kind, (factory, tp_size) in _LAYERS.items():
        for tp_rank in range(tp_size):
            yield pytest.param(kind, tp_rank, id=f"{kind}-tp{tp_rank}")


@pytest.mark.parametrize("kind, tp_rank", list(_cases()))
@pytest.mark.parametrize("max_rank", [_RANK, _RANK + 1], ids=["max_r=r", "max_r>r"])
def test_a_complete_adapter_loads_like_the_concatenating_loader(
    kind, tp_rank, max_rank
):
    module_path, members, full_rows, wrapper, reference_slice = _LAYERS[kind][0](
        tp_rank
    )
    pool = _pool(module_path, wrapper, full_rows, max_rank)
    weights, a, b = _member_weights(members, full_rows, members)

    a_buffer, b_buffer = _load(pool, module_path, wrapper, weights)

    expected_a, expected_b = _expected(
        members, full_rows, a, b, members, reference_slice
    )
    assert torch.equal(a_buffer[: len(members) * _RANK], expected_a)
    assert torch.equal(b_buffer[:, :_RANK], expected_b)


@pytest.mark.parametrize("kind, tp_rank", list(_cases()))
def test_any_member_subset_loads_with_the_absent_members_zero(kind, tp_rank):
    """A q-only adapter raised KeyError, a k/v-only or up-only adapter raised ValueError, and a
    q_a-only MLA adapter got kv_a rows of q_a's width."""
    module_path, members, full_rows, wrapper, reference_slice = _LAYERS[kind][0](
        tp_rank
    )
    for size in range(1, len(members)):
        for present in itertools.combinations(members, size):
            pool = _pool(module_path, wrapper, full_rows, _RANK)
            weights, a, b = _member_weights(members, full_rows, present)

            a_buffer, b_buffer = _load(pool, module_path, wrapper, weights)

            expected_a, expected_b = _expected(
                members, full_rows, a, b, present, reference_slice
            )
            assert torch.equal(a_buffer[: len(members) * _RANK], expected_a), present
            assert torch.equal(b_buffer[:, :_RANK], expected_b), present


def test_zero_element_placeholders_load_as_absent_members():
    """PEFT can ship an untrained up_proj as zero-element tensors; they must load as zeros."""
    module_path, members, full_rows, wrapper, reference_slice = _gate_up(tp_rank=0)
    weights, a, b = _member_weights(members, full_rows, ["gate_proj"])
    weights[f"{_PREFIX}mlp.up_proj.lora_A.weight"] = torch.empty(0)
    weights[f"{_PREFIX}mlp.up_proj.lora_B.weight"] = torch.empty(0)

    a_buffer, b_buffer = _load(
        _pool(module_path, wrapper, full_rows, _RANK), module_path, wrapper, weights
    )

    expected_a, expected_b = _expected(
        members, full_rows, a, b, ["gate_proj"], reference_slice
    )
    assert torch.equal(a_buffer[: len(members) * _RANK], expected_a)
    assert torch.equal(b_buffer[:, :_RANK], expected_b)


@pytest.mark.parametrize("kind", ["qkv", "narrow_v_qkv"])
@pytest.mark.parametrize("stacked_a", [False, True], ids=["shared-A", "stacked-A"])
def test_an_already_fused_qkv_adapter_matches_its_members(kind, stacked_a):
    module_path, members, full_rows, wrapper, reference_slice = _LAYERS[kind][0](
        tp_rank=3
    )
    torch.manual_seed(1)
    fused_a = torch.randn(_RANK * (3 if stacked_a else 1), _HIDDEN)
    fused_b = torch.randn(sum(full_rows), _RANK)
    weights = {
        f"{_PREFIX}self_attn.qkv_proj.lora_A.weight": fused_a,
        f"{_PREFIX}self_attn.qkv_proj.lora_B.weight": fused_b,
    }

    a_buffer, b_buffer = _load(
        _pool(module_path, wrapper, full_rows, _RANK), module_path, wrapper, weights
    )

    expected_a = fused_a if stacked_a else fused_a.repeat(3, 1)
    assert torch.equal(a_buffer[: 3 * _RANK], expected_a)
    assert torch.equal(b_buffer[:, :_RANK], reference_slice(fused_b))


def test_an_adapter_with_fused_and_separate_projections_is_rejected():
    module_path, members, full_rows, wrapper, _ = _qkv(tp_rank=0)
    weights, _, _ = _member_weights(members, full_rows, ("q_proj",))
    weights[f"{_PREFIX}self_attn.qkv_proj.lora_A.weight"] = torch.randn(_RANK, _HIDDEN)
    weights[f"{_PREFIX}self_attn.qkv_proj.lora_B.weight"] = torch.randn(
        sum(full_rows), _RANK
    )

    with pytest.raises(AssertionError, match="both fused and as separate projections"):
        _load(
            _pool(module_path, wrapper, full_rows, _RANK), module_path, wrapper, weights
        )


def test_the_adapter_keeps_dense_projections_per_member():
    """Normalization no longer stacks dense projections; routed-expert gate/up still are."""
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter.base_hf_config = SimpleNamespace()
    object.__setattr__(adapter, "base_model", None)
    object.__setattr__(adapter, "_moe_is_gated_by_layer", {})
    names = [
        f"{_PREFIX}self_attn.q_proj.lora_A.weight",
        f"{_PREFIX}mlp.gate_proj.lora_A.weight",
        f"{_PREFIX}self_attn.q_a_proj.lora_A.weight",
        f"{_PREFIX}mlp.experts.0.gate_proj.lora_A.weight",
        f"{_PREFIX}mlp.experts.0.up_proj.lora_A.weight",
    ]
    object.__setattr__(
        adapter,
        "layers",
        [SimpleNamespace(weights={n: torch.ones(_RANK, _HIDDEN) for n in names})],
    )

    adapter._normalize_weights()

    assert sorted(adapter.layers[0].weights) == sorted(
        names[:3] + [f"{_PREFIX}mlp.experts.0.gate_up_proj.lora_A.weight"]
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
