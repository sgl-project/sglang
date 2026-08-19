"""CPU-only regression tests for Inkling's linearized shared-sink LoRA path:
in-place refresh of the derived decode operands on an adapter swap, and the
shared-sink factor layout from checkpoint to one moe-TP shard's operands."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.srt.lora import lora_manager as lora_manager_module
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_registry import LoRARef
from sglang.test.ci.ci_register import register_cpu_ci

# Pure layout / bookkeeping math: no CUDA, no distributed groups, no kernels.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HIDDEN = 3
_ADAPTER_RANK = 2
_NUM_SHARED = 2
# Per-expert intermediate width before moe-TP sharding, and this rank's shard.
_INTERMEDIATE = 4
_MOE_TP_SIZE = 2
_SHARD = _INTERMEDIATE // _MOE_TP_SIZE
_MOE_TP_RANK = 1

_PREFIX = "model.layers.0.mlp.shared_experts"
_GATE_UP_A = f"{_PREFIX}.gate_up_proj.lora_A.weight"
_GATE_UP_B = f"{_PREFIX}.gate_up_proj.lora_B.weight"
_DOWN_A = f"{_PREFIX}.down_proj.lora_A.weight"
_DOWN_B = f"{_PREFIX}.down_proj.lora_B.weight"


class _FakeSharedSink(nn.Module):
    """Stand-in for ``InklingBatchDenseMLP``: only the attributes
    ``InklingBatchDenseMLPWithLoRA`` reads."""

    def __init__(
        self,
        *,
        num_shared: int = _NUM_SHARED,
        intermediate: int = _SHARD,
        moe_tp_size: int = 1,
    ):
        super().__init__()
        self.n_shared_experts = num_shared
        self.intermediate_size_per_partition = intermediate
        self.moe_tp_size = moe_tp_size
        self.moe_tp_rank = 0
        self._linearized_bf16_enabled = True


def _load_sink_lora_class():
    """Import the real Inkling LoRA layer against the fake dense-sink base."""
    stub = types.ModuleType("sglang.srt.models.inkling_common.dense_mlp")
    stub.InklingBatchDenseMLP = _FakeSharedSink
    patched = dict(sys.modules)
    patched["sglang.srt.models.inkling_common.dense_mlp"] = stub
    # Drop any cached copy so the real module body re-runs against the stub,
    # and let the temporary sys.modules state be discarded on exit.
    patched.pop("sglang.srt.models.inkling_common.lora", None)
    with mock.patch.dict(sys.modules, patched, clear=True):
        module = importlib.import_module("sglang.srt.models.inkling_common.lora")
    return module.InklingBatchDenseMLPWithLoRA


InklingSharedSinkWithLoRA = _load_sink_lora_class()


def _new_sink(*, slots: int = 1, moe_tp_size: int = 1, intermediate: int = _SHARD):
    layer = InklingSharedSinkWithLoRA(
        moe_tp_size=moe_tp_size, intermediate=intermediate
    )
    # initialize_lora() reads only these two backend fields (and flips
    # is_moe_lora on the backend it is handed).
    layer.initialize_lora(
        SimpleNamespace(
            name="triton" if slots > 1 else "torch-test",
            max_loras_per_batch=slots,
            is_moe_lora=False,
        )
    )
    return layer


def _adapter(config) -> LoRAAdapter:
    """Bind the real normalizer methods to a bare ``LoRAAdapter``."""
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter.base_hf_config = config
    return adapter


def _hf_config(*, architectures, model_type):
    return SimpleNamespace(
        architectures=architectures,
        model_type=model_type,
        n_shared_experts=_NUM_SHARED,
    )


def _inkling_config():
    return _hf_config(
        architectures=["InklingForConditionalGeneration"], model_type="llama"
    )


def _normalize(config, weights):
    """Run the adapter-side shared-sink reshape + gate/up stacking."""
    adapter = _adapter(config)
    adapter._normalize_shared_expert_moe(weights)
    adapter.normalize_gate_up_proj(list(weights), weights)
    return weights


def _checkpoint_factors():
    """The four flat 2D factors an Inkling shared-sink adapter ships."""

    def ramp(*shape):
        numel = 1
        for dim in shape:
            numel *= dim
        # Exact binary fractions keep every downstream matmul exact.
        return (torch.arange(numel, dtype=torch.float32) / 8.0).reshape(*shape)

    return {
        _GATE_UP_A: ramp(_ADAPTER_RANK, _HIDDEN),
        _GATE_UP_B: ramp(_NUM_SHARED * 2 * _INTERMEDIATE, _ADAPTER_RANK),
        _DOWN_A: ramp(_ADAPTER_RANK, _NUM_SHARED * _INTERMEDIATE),
        _DOWN_B: ramp(_HIDDEN, _ADAPTER_RANK),
    }


def _shard_factors(layer, normalized):
    """Slice the normalized factors down to this moe-TP rank."""
    return (
        layer.slice_moe_lora_a_weights(
            normalized[_GATE_UP_A], _MOE_TP_RANK, "gate_up_proj_moe"
        ),
        layer.slice_moe_lora_b_weights(
            normalized[_GATE_UP_B], _MOE_TP_RANK, "gate_up_proj_moe"
        ),
        layer.slice_moe_lora_a_weights(
            normalized[_DOWN_A], _MOE_TP_RANK, "down_proj_moe"
        ),
        layer.slice_moe_lora_b_weights(
            normalized[_DOWN_B], _MOE_TP_RANK, "down_proj_moe"
        ),
    )


def _slot_shapes(*, slots: int, max_rank: int):
    return (
        (slots, 1, 2 * max_rank, _HIDDEN),
        (slots, _NUM_SHARED, 2 * _SHARD, max_rank),
        (slots, _NUM_SHARED, max_rank, _SHARD),
        (slots, 1, _HIDDEN, max_rank),
    )


def _empty_pool(*, slots: int, max_rank: int):
    return tuple(
        torch.zeros(shape) for shape in _slot_shapes(slots=slots, max_rank=max_rank)
    )


def _pool_from_shards(shards, *, max_rank: int, slots: int = 1):
    """Lay one adapter's shard out as ``LoRAMemoryPool.load_lora_weight_to_buffer``
    does: stacked gate/up LoRA-A halves at ``max_rank`` offsets, rank tails zero."""
    a_gate_up, b_gate_up, a_down, b_down = shards
    rank = b_gate_up.shape[-1]
    buffers = _empty_pool(slots=slots, max_rank=max_rank)
    with torch.no_grad():
        for slot in range(slots):
            for half in range(2):
                buffers[0][slot, 0, half * max_rank : half * max_rank + rank] = (
                    a_gate_up[0, half * rank : (half + 1) * rank]
                )
            buffers[1][slot, :, :, :rank] = b_gate_up
            buffers[2][slot, :, :rank, :] = a_down
            buffers[3][slot, 0, :, :rank] = b_down[0]
    return buffers


def _sink_with_shard(shards, *, max_rank: int):
    layer = _new_sink(moe_tp_size=_MOE_TP_SIZE)
    layer.set_lora_info(*_pool_from_shards(shards, max_rank=max_rank))
    return layer


def _consumer_delta(layer, x, act):
    """The two GEMMs the single-slot shared-outer path runs on the operands,
    mirroring ``forward_with_lora`` in ``lora/trtllm_lora_temp/inkling_dense.py``."""
    gate_up_shrink = x @ layer.gate_up_lora_a_weights[0, 0].T
    down_shrink = act @ layer._a_cat[0].T
    return (
        gate_up_shrink @ layer._w1_delta[0].T,
        down_shrink @ layer.down_lora_b_weights[0, 0].T,
    )


def _reference_delta(shards, x, act):
    """The same delta straight from the sharded adapter factors; the sink's w13 is
    gate/up *interleaved*, so gate lands at ``[..., 0::2]`` and up at
    ``[..., 1::2]`` of each expert's block."""
    a_gate_up, b_gate_up, a_down, b_down = shards
    rank = b_gate_up.shape[-1]
    experts, f = b_gate_up.shape[0], b_gate_up.shape[1] // 2
    gate_shrink = x @ a_gate_up[0, :rank].T
    up_shrink = x @ a_gate_up[0, rank:].T
    y = x.new_zeros(x.shape[0], experts, 2 * f)
    for expert in range(experts):
        y[:, expert, 0::2] = gate_shrink @ b_gate_up[expert, :f].T
        y[:, expert, 1::2] = up_shrink @ b_gate_up[expert, f:].T
    down_shrink = torch.einsum("tef,ekf->tk", act.view(-1, experts, f), a_down)
    return y.reshape(x.shape[0], -1), down_shrink @ b_down[0].T


def _slot_factors(*, gate_up_b: float, down_a: float, max_rank: int):
    """One slot's pool contents, each factor a flat constant with a zero rank tail,
    so the largest magnitude in a derived operand identifies its source adapter."""
    rank = _ADAPTER_RANK
    a_gate_up, b_gate_up, a_down, b_down = _empty_pool(slots=1, max_rank=max_rank)
    a_gate_up[0, 0, :rank] = 1.0
    a_gate_up[0, 0, max_rank : max_rank + rank] = 1.0
    b_gate_up[0, ..., :rank] = gate_up_b
    a_down[0, :, :rank, :] = down_a
    b_down[0, ..., :rank] = 1.0
    return (a_gate_up[0], b_gate_up[0], a_down[0], b_down[0])


def _fake_adapter(*, gate_up_b: float, down_a: float, max_rank: int):
    return SimpleNamespace(
        factors=_slot_factors(gate_up_b=gate_up_b, down_a=down_a, max_rank=max_rank)
    )


def _derived_signature(layer):
    """Which adapter the derived operands currently hold."""
    return (
        layer._w1_delta.abs().max().item(),
        layer._a_cat.abs().max().item(),
    )


class _FakeMemoryPool:
    """Minimal ``LoRAMemoryPool``: a new uid takes a free (or evicted) slot and its
    factors are copied into the shared buffers there, removing a uid frees its
    slot, and a base-model uid (``None``) zeroes the slot."""

    def __init__(self, buffers, *, max_loras_per_batch: int, events: list):
        self.buffers = buffers
        self.max_loras_per_batch = max_loras_per_batch
        self.uid_to_buffer_id: dict = {}
        self.events = events

    def prepare_lora_batch(self, *, cur_uids, lora_adapters, **kwargs):
        self.events.append(("pool", set(cur_uids)))
        for uid in sorted(cur_uids, key=str):
            if uid in self.uid_to_buffer_id:
                continue
            slot = self._take_slot()
            self.uid_to_buffer_id[uid] = slot
            self._load(slot, lora_adapters.get(uid))

    def remove_lora(self, uid):
        self.events.append(("remove", uid))
        slot = self.uid_to_buffer_id.pop(uid, None)
        if slot is not None:
            self._load(slot, None)
        return slot

    def _take_slot(self) -> int:
        used = set(self.uid_to_buffer_id.values())
        for slot in range(self.max_loras_per_batch):
            if slot not in used:
                return slot
        evicted = next(iter(self.uid_to_buffer_id))
        return self.uid_to_buffer_id.pop(evicted)

    def _load(self, slot: int, adapter) -> None:
        with torch.no_grad():
            for index, buffer in enumerate(self.buffers):
                if adapter is None:
                    buffer[slot].zero_()
                else:
                    buffer[slot].copy_(adapter.factors[index])


def _new_manager(pool, layer, *, uids):
    """A ``LoRAManager`` carrying only the fields the tested methods touch."""
    manager = lora_manager_module.LoRAManager.__new__(lora_manager_module.LoRAManager)
    manager.device = torch.device("cpu")
    manager.max_loras_per_batch = pool.max_loras_per_batch
    manager.memory_pool = pool
    manager.lora_modules = [{_PREFIX: layer}]
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    manager.num_pinned_loras = 0
    manager.loras = {}
    manager.configs = {uid: object() for uid in uids}
    manager.lora_refs = {
        uid: LoRARef(lora_id=uid, lora_name=uid, lora_path=f"/lora/{uid}", pinned=False)
        for uid in uids
    }
    return manager


def _record_refreshes(layer, events: list) -> None:
    """Log every slot-update notification, forwarding to the real handler."""
    handler = layer.on_lora_slots_updated

    def record(slot_ids):
        events.append(("refresh", None if slot_ids is None else set(slot_ids)))
        handler(slot_ids)

    layer.on_lora_slots_updated = record


def test_derived_operands_refresh_in_place_after_slot_copy(monkeypatch):
    """Guards adapter swaps landing in the pre-allocated derived operands; reds when
    a swap rebinds instead of copying in place, notifies before the pool copies the
    new adapter, or skips the refresh on an unload, a reload, or a base-only batch."""
    max_rank = _ADAPTER_RANK
    buffers = _empty_pool(slots=1, max_rank=max_rank)
    layer = _new_sink(slots=1)
    layer.set_lora_info(*buffers)
    assert layer.experts_shared_outer_loras is True

    events: list = []
    _record_refreshes(layer, events)
    pool = _FakeMemoryPool(buffers, max_loras_per_batch=1, events=events)
    manager = _new_manager(pool, layer, uids=("uid-a", "uid-b"))
    manager.loras = {
        "uid-a": _fake_adapter(gate_up_b=1.0, down_a=0.5, max_rank=max_rank),
        "uid-b": _fake_adapter(gate_up_b=2.0, down_a=0.25, max_rank=max_rank),
    }
    pointers = (layer._w1_delta.data_ptr(), layer._a_cat.data_ptr())

    # The pool copies the adapter into its slot first; only then is the slot's
    # derived operand rebuilt. Refreshing first would derive it from zeros.
    manager.fetch_new_loras({"uid-a"})
    assert events == [("pool", {"uid-a"}), ("refresh", {0})]
    assert _derived_signature(layer) == (1.0, 0.5)

    # No uid took a new slot: nothing to re-derive.
    events.clear()
    manager.fetch_new_loras({"uid-a"})
    assert events == [("pool", {"uid-a"})]

    # Hot swap: uid-b evicts uid-a from slot 0. Refreshing before the pool copy
    # would leave uid-a's delta in place for every later replay.
    events.clear()
    manager.fetch_new_loras({"uid-b"})
    assert events == [("pool", {"uid-b"}), ("refresh", {0})]
    assert _derived_signature(layer) == (2.0, 0.25)

    # A base-only batch takes the slot too, and must clear the delta.
    events.clear()
    manager.fetch_new_loras({None})
    assert events == [("pool", {None}), ("refresh", {0})]
    assert _derived_signature(layer) == (0.0, 0.0)

    # Unloading frees the slot, and the freed slot must be re-derived.
    events.clear()
    manager.fetch_new_loras({"uid-a"})
    assert _derived_signature(layer) == (1.0, 0.5)
    # The unload info logs sample GPU memory; there is no GPU on this runner.
    monkeypatch.setattr(
        lora_manager_module, "get_available_gpu_memory", lambda *args, **kwargs: 0.0
    )
    events.clear()
    assert manager.unload_lora_adapter(manager.lora_refs["uid-a"]).success
    assert events == [("remove", "uid-a"), ("refresh", {0})]
    assert _derived_signature(layer) == (0.0, 0.0)

    # Reloading the same uid with different weights re-derives, even though the
    # uid and the slot are unchanged. (Re-seeded the way _load_lora_adapter
    # would, without touching a checkpoint.)
    events.clear()
    manager.configs["uid-a"] = object()
    manager.lora_refs["uid-a"] = LoRARef(
        lora_id="uid-a", lora_name="uid-a", lora_path="/lora/uid-a-v2", pinned=False
    )
    manager.loras["uid-a"] = _fake_adapter(
        gate_up_b=3.0, down_a=0.75, max_rank=max_rank
    )
    manager.fetch_new_loras({"uid-a"})
    assert events == [("pool", {"uid-a"}), ("refresh", {0})]
    assert _derived_signature(layer) == (3.0, 0.75)

    # Every refresh wrote through the buffers allocated at set_lora_info() time.
    assert (layer._w1_delta.data_ptr(), layer._a_cat.data_ptr()) == pointers


def test_shared_sink_factor_layout_roundtrip():
    """Guards the shared-sink factor layout from checkpoint to one moe-TP shard's
    decode operands; reds on a lost ``down_proj`` transpose, gate-major gate/up
    LoRA-B, a collapsed gate/up shard, or a rank offset from the wrong rank."""
    rank, hidden = _ADAPTER_RANK, _HIDDEN
    experts, full = _NUM_SHARED, _INTERMEDIATE
    flat = _checkpoint_factors()
    gate_up_a, gate_up_b = flat[_GATE_UP_A], flat[_GATE_UP_B]
    down_a, down_b = flat[_DOWN_A], flat[_DOWN_B]

    # -- adapter side -----------------------------------------------------
    # Both gates into the Inkling reshape: the architecture list and the
    # model type.
    for config in (
        _inkling_config(),
        _hf_config(architectures=None, model_type="inkling_text"),
    ):
        normalized = _normalize(config, dict(flat))
        # gate/up LoRA-A is shared across experts and stacked (gate, up).
        torch.testing.assert_close(
            normalized[_GATE_UP_A], gate_up_a.unsqueeze(0).repeat(1, 2, 1)
        )
        # gate/up LoRA-B is expert-major, gate rows before up rows.
        torch.testing.assert_close(
            normalized[_GATE_UP_B], gate_up_b.reshape(experts, 2 * full, rank)
        )
        # down LoRA-A arrives rank-major and must come out expert-major.
        torch.testing.assert_close(
            normalized[_DOWN_A],
            down_a.reshape(rank, experts, full).transpose(0, 1).contiguous(),
        )
        # down LoRA-B is shared across experts.
        torch.testing.assert_close(normalized[_DOWN_B], down_b.unsqueeze(0))

    # A non-Inkling base model keeps the stock 2D shared-expert path.
    stock = dict(flat)
    _adapter(
        _hf_config(architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe")
    )._normalize_shared_expert_moe(stock)
    assert stock[_GATE_UP_B].dim() == 2
    assert stock[_DOWN_A].dim() == 2

    # A *named* per-expert factor is not a shared-outer factor: leaving it 2D is
    # what lets adapter validation reject it later.
    named_per_expert = f"{_PREFIX}.1.gate_up_proj.lora_A.weight"
    per_expert = {named_per_expert: gate_up_a.clone()}
    _adapter(_inkling_config())._normalize_shared_expert_moe(per_expert)
    torch.testing.assert_close(per_expert[named_per_expert], gate_up_a)

    # -- layer side: this moe-TP rank's shard -----------------------------
    normalized = _normalize(_inkling_config(), dict(flat))
    layer = _new_sink(moe_tp_size=_MOE_TP_SIZE)
    shards = _shard_factors(layer, normalized)
    start = _MOE_TP_RANK * _SHARD
    # The gate half and the up half are sharded independently and re-paired, so
    # this rank's gate rows meet this rank's up rows.
    expected_gate_up_b = torch.stack(
        [
            torch.cat(
                [
                    expert_b[start : start + _SHARD],
                    expert_b[full + start : full + start + _SHARD],
                ]
            )
            for expert_b in gate_up_b.reshape(experts, 2 * full, rank)
        ]
    )
    torch.testing.assert_close(shards[1], expected_gate_up_b)
    # down LoRA-A is sharded on the intermediate axis it contracts over.
    expected_down_a = down_a.reshape(rank, experts, full).transpose(0, 1)[
        ..., start : start + _SHARD
    ]
    torch.testing.assert_close(shards[2], expected_down_a)
    # The shrink-side gate/up LoRA-A and the expand-side down LoRA-B are not
    # sharded by moe-TP.
    torch.testing.assert_close(shards[0], gate_up_a.unsqueeze(0).repeat(1, 2, 1))
    torch.testing.assert_close(shards[3], down_b.unsqueeze(0))

    # The layer also accepts the checkpoint's flat 2D factors, and must produce
    # the same shard as the normalized 3D ones.
    torch.testing.assert_close(
        layer.slice_moe_lora_b_weights(gate_up_b, _MOE_TP_RANK, "gate_up_proj_moe"),
        expected_gate_up_b,
    )
    torch.testing.assert_close(
        layer.slice_moe_lora_a_weights(down_a, _MOE_TP_RANK, "down_proj_moe"),
        expected_down_a,
    )

    # -- pool layout + derived operands -----------------------------------
    x = torch.tensor([[0.5, -1.0, 0.25], [1.25, 0.75, -0.5]])
    act = torch.tensor([[0.5, -0.25, 1.0, 0.75], [-1.0, 0.25, 0.5, -0.75]])
    assert x.shape[-1] == hidden and act.shape[-1] == experts * _SHARD
    expected_gate_up_delta, expected_down_delta = _reference_delta(shards, x, act)

    # A pool padded past the adapter's rank must produce the same delta as an
    # exactly-sized one: the up half's column offset comes from the padded
    # max_rank, matching where the pool stacks the up half of LoRA-A.
    for max_rank in (rank, rank + 1):
        sink = _sink_with_shard(shards, max_rank=max_rank)
        assert sink.experts_shared_outer_loras is True
        gate_up_delta, down_delta = _consumer_delta(sink, x, act)
        torch.testing.assert_close(gate_up_delta, expected_gate_up_delta)
        torch.testing.assert_close(down_delta, expected_down_delta)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
