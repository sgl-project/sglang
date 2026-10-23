"""Unit tests for LoRAMemoryPool's MoE expert-parallel (EP) handling.

Covers the behaviour whose failure would be silent: deterministic slot/LRU
state across ranks, in-place slot clearing, global->local expert-id remapping
of routed weights, replicated shared-expert loads, and the expert-id
convention shared with `StandardDispatcher`. Per-rank buffer-shape layouts
live in `test_mem_pool_layouts.py` (weekly).

`LoRAMemoryPool` is instantiated via `__new__` with only the fields the code
under test reads, so the tests stay hermetic (CPU-only, no distributed groups).

Usage:
    python3 test/registered/unit/lora/test_mem_pool_ep_unit.py
"""

from sglang.test.ci.ci_register import register_cpu_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import types
import unittest
import unittest.mock as mock

import torch

from sglang.srt.lora.eviction_policy import get_eviction_policy


class _FakeBaseLayerWithLoRA:
    pass


class _FakeFusedMoEWithLoRA(_FakeBaseLayerWithLoRA):
    pass


_LORA_LAYERS_STUB = types.ModuleType("sglang.srt.lora.layers")
_LORA_LAYERS_STUB.BaseLayerWithLoRA = _FakeBaseLayerWithLoRA
_LORA_LAYERS_STUB.FusedMoEWithLoRA = _FakeFusedMoEWithLoRA
_LORA_LAYERS_STUB.unwrap_lora_layer = lambda module: (
    module.base_layer if isinstance(module, _FakeBaseLayerWithLoRA) else module
)
_LORA_ADAPTER_STUB = types.ModuleType("sglang.srt.lora.lora")
_LORA_ADAPTER_STUB.LoRAAdapter = object
with mock.patch.dict(
    "sys.modules",
    {
        "sglang.srt.lora.layers": _LORA_LAYERS_STUB,
        "sglang.srt.lora.lora": _LORA_ADAPTER_STUB,
    },
):
    import sglang.srt.lora.mem_pool as mem_pool_module
    from sglang.srt.lora.mem_pool import (
        EMPTY_SLOT,
        LoRAMemoryPool,
        _moe_runner_keeps_global_expert_ids,
    )

import sglang.srt.layers.moe.utils as moe_utils
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import standard as standard_dispatcher
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_parallel


class _IdentityMoeSlices:
    def slice_moe_lora_a_weights(self, weights, _rank, _target):
        return weights

    def slice_moe_lora_b_weights(self, weights, _rank, _target):
        return weights


class _FakeSharedMoeLayer(_IdentityMoeSlices):
    is_shared_fused_moe = True

    def __init__(self, tp_rank: int = 0):
        self.moe_tp_rank = tp_rank


class _FakeRoutedMoeLayer(_FakeFusedMoEWithLoRA, _IdentityMoeSlices):
    def __init__(self, tp_rank: int = 0):
        object.__setattr__(
            self,
            "base_layer",
            types.SimpleNamespace(
                moe_tp_rank=tp_rank,
                is_shared_fused_moe=False,
            ),
        )

    def slice_moe_lora_a_weights(self, weights, _rank, _target):
        return weights

    def slice_moe_lora_b_weights(self, weights, _rank, _target):
        return weights


class _FakeDenseLayer:
    def slice_lora_a_weights(self, weights):
        return weights

    def slice_lora_b_weights(self, weights):
        return weights


def _load_lora_weight_to_buffer(pool, **kwargs):
    with mock.patch.dict("sys.modules", {"sglang.srt.lora.layers": _LORA_LAYERS_STUB}):
        return pool.load_lora_weight_to_buffer(**kwargs)


def _make_pool(
    *,
    num_experts_global: int,
    moe_ep_size: int,
    moe_ep_rank: int,
    moe_use_local_expert_ids: bool,
) -> LoRAMemoryPool:
    """Construct a minimal LoRAMemoryPool for helper-level tests.

    Bypasses `__init__` (which requires a real base model, HF config, and
    device allocations) and sets only the fields consulted by the EP
    helpers under test.
    """
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.moe_ep_size = moe_ep_size
    pool.moe_ep_rank = moe_ep_rank
    pool.moe_use_local_expert_ids = moe_use_local_expert_ids
    # Helpers under test in this module don't consult moe_tp_size, but set
    # defaults so accidental reads don't AttributeError.
    pool.moe_tp_size = 1
    pool.moe_tp_rank = 0
    if moe_use_local_expert_ids and num_experts_global % moe_ep_size == 0:
        pool._num_experts_local = num_experts_global // moe_ep_size
    else:
        pool._num_experts_local = num_experts_global
    return pool


class _IterationOrderedSet(set):
    def __init__(self, values, iteration_order):
        super().__init__(values)
        self.iteration_order = iteration_order

    def __iter__(self):
        return iter(self.iteration_order)


class TestDeterministicPoolSlots(unittest.TestCase):
    @staticmethod
    def _prepare(iteration_order):
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.max_loras_per_batch = 4
        pool.uid_to_buffer_id = {}
        pool.buffer_id_to_uid = [EMPTY_SLOT] * 4
        pool.eviction_policy = get_eviction_policy("lru")
        loaded = []
        pool.load_lora_weight_to_buffer = lambda uid, slot, *_args: loaded.append(
            (uid, slot)
        )
        uids = _IterationOrderedSet(
            {None, "adapter-a", "adapter-b", "adapter-c"}, iteration_order
        )

        pool.prepare_lora_batch(
            cur_uids=uids,
            lora_adapters={},
            lora_modules=[],
            lora_refs={},
            lora_embed_tokens_module=None,
            lora_lm_head_module=None,
        )
        return loaded, list(pool.eviction_policy.access_order)

    def test_uid_iteration_order_does_not_change_slots_or_lru(self):
        forward = [None, "adapter-a", "adapter-b", "adapter-c"]
        reverse = list(reversed(forward))
        expected = (
            [
                (None, 0),
                ("adapter-a", 1),
                ("adapter-b", 2),
                ("adapter-c", 3),
            ],
            ["adapter-a", "adapter-b", "adapter-c"],
        )

        self.assertEqual(self._prepare(forward), expected)
        self.assertEqual(self._prepare(reverse), expected)

    @staticmethod
    def _evict_after_touch(iteration_order):
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.max_loras_per_batch = 4
        pool.uid_to_buffer_id = {
            None: 0,
            "adapter-a": 1,
            "adapter-b": 2,
            "adapter-c": 3,
        }
        pool.buffer_id_to_uid = [None, "adapter-a", "adapter-b", "adapter-c"]
        pool.eviction_policy = get_eviction_policy("lru")
        for uid in ("adapter-a", "adapter-b", "adapter-c"):
            pool.eviction_policy.mark_used(uid)
        pool.load_lora_weight_to_buffer = lambda *_args: None
        common = dict(
            lora_adapters={},
            lora_modules=[],
            lora_refs={},
            lora_embed_tokens_module=None,
            lora_lm_head_module=None,
        )

        pool.prepare_lora_batch(
            cur_uids=_IterationOrderedSet(
                {"adapter-a", "adapter-b", "adapter-c"}, iteration_order
            ),
            **common,
        )
        pool.prepare_lora_batch(cur_uids={"adapter-d"}, **common)
        return pool.uid_to_buffer_id, list(pool.eviction_policy.access_order)

    def test_uid_iteration_order_does_not_change_next_lru_victim(self):
        forward = ["adapter-a", "adapter-b", "adapter-c"]
        reverse = list(reversed(forward))
        expected = (
            {None: 0, "adapter-d": 1, "adapter-b": 2, "adapter-c": 3},
            ["adapter-b", "adapter-c", "adapter-d"],
        )

        self.assertEqual(self._evict_after_touch(forward), expected)
        self.assertEqual(self._evict_after_touch(reverse), expected)


class TestBufferSlotClearing(unittest.TestCase):
    @staticmethod
    def _buffers(pool):
        layered = [
            tensor
            for buffers in (*pool.A_buffer.values(), *pool.B_buffer.values())
            for tensor in buffers
        ]
        direct = [
            tensor
            for buffers in (
                pool.embedding_A_buffer,
                pool.embedding_B_buffer,
                pool.lm_head_A_buffer,
                pool.lm_head_B_buffer,
                pool.new_embeddings_buffer,
            )
            for tensor in buffers.values()
        ]
        return layered + direct

    @classmethod
    def _make_pool(cls):
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.num_layer = 2
        pool.A_buffer = {"a": [torch.ones(2, 1, 2) for _ in range(2)]}
        pool.B_buffer = {"b": [torch.full((2, 1, 2), float("nan")) for _ in range(2)]}
        pool.embedding_A_buffer = {"a": torch.ones(2, 1, 2)}
        pool.embedding_B_buffer = {"b": torch.full((2, 1, 2), float("nan"))}
        pool.lm_head_A_buffer = {"a": torch.ones(2, 1, 2)}
        pool.lm_head_B_buffer = {"b": torch.full((2, 1, 2), float("inf"))}
        pool.new_embeddings_buffer = {"embeddings": torch.full((2, 1, 2), float("nan"))}
        for tensor in cls._buffers(pool):
            tensor[1].fill_(7)

        pool.uid_to_buffer_id = {"adapter": 0}
        pool.buffer_id_to_uid = ["adapter", "other"]
        pool.eviction_policy = get_eviction_policy("lru")
        pool.eviction_policy.mark_used("adapter")
        return pool

    def test_remove_lora_clears_slot_in_place_and_residency(self):
        pool = self._make_pool()
        tensors = self._buffers(pool)
        pointers = [tensor.data_ptr() for tensor in tensors]

        self.assertEqual(pool.remove_lora("adapter"), 0)

        for tensor, pointer in zip(tensors, pointers):
            torch.testing.assert_close(tensor[0], torch.zeros_like(tensor[0]))
            torch.testing.assert_close(tensor[1], torch.full_like(tensor[1], 7))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertEqual(tensor.data_ptr(), pointer)
        self.assertNotIn("adapter", pool.uid_to_buffer_id)
        self.assertIs(pool.buffer_id_to_uid[0], EMPTY_SLOT)
        self.assertNotIn("adapter", pool.eviction_policy.access_order)


class TestGlobalToLocalExpertId(unittest.TestCase):
    """`_global_to_local_expert_id` — the per-rank filter + remap."""

    def test_rank2_of_ep4_owns_third_quarter(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=2,
            moe_use_local_expert_ids=True,
        )
        # Owned globals 4, 5 -> local 0, 1
        self.assertEqual(pool._global_to_local_expert_id(4), 0)
        self.assertEqual(pool._global_to_local_expert_id(5), 1)
        for gid in (0, 1, 2, 3, 6, 7):
            self.assertIsNone(pool._global_to_local_expert_id(gid))


class TestIterLocalExpertWeightsDict(unittest.TestCase):
    """`_iter_local_expert_weights` with dict input (the common case)."""

    def test_passthrough_without_ep(self):
        pool = _make_pool(
            num_experts_global=4,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        weights = {gid: torch.full((2,), float(gid)) for gid in range(4)}
        cache_keys = {gid: f"expert.{gid}" for gid in weights}
        got = {
            lid: w.tolist()
            for lid, w, _ in pool._iter_local_expert_weights(weights, cache_keys)
        }
        self.assertEqual(
            got, {0: [0.0, 0.0], 1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}
        )

    def test_sparse_dict_only_yields_owned_experts(self):
        """Adapters may only target a subset of experts. The iterator must
        still correctly filter and remap whatever subset is provided.
        """
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=4,
            moe_ep_rank=2,
            moe_use_local_expert_ids=True,
        )
        # Only globals 1, 4, 5, 7 present in adapter.
        weights = {
            1: torch.full((2,), 1.0),
            4: torch.full((2,), 4.0),
            5: torch.full((2,), 5.0),
            7: torch.full((2,), 7.0),
        }
        cache_keys = {gid: f"expert.{gid}" for gid in weights}
        # Rank 2 owns globals 4, 5 -> locals 0, 1.
        got = {
            lid: w.tolist()
            for lid, w, _ in pool._iter_local_expert_weights(weights, cache_keys)
        }
        self.assertEqual(got, {0: [4.0, 4.0], 1: [5.0, 5.0]})


class TestIterLocalExpertWeightsTensor(unittest.TestCase):
    """`_iter_local_expert_weights` with 3D tensor input (shared-outer and
    packed MoE-LoRA formats)."""

    def test_rank1_of_ep2_sees_upper_half(self):
        pool = _make_pool(
            num_experts_global=4,
            moe_ep_size=2,
            moe_ep_rank=1,
            moe_use_local_expert_ids=True,
        )
        weights = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        got = [
            (lid, w.clone(), cache_key)
            for lid, w, cache_key in pool._iter_local_expert_weights(weights, "weights")
        ]
        # Rank 1 of EP=2 with 4 experts owns globals 2, 3 -> locals 0, 1.
        self.assertEqual([lid for lid, _, _ in got], [0, 1])
        self.assertTrue(torch.equal(got[0][1], weights[2]))
        self.assertTrue(torch.equal(got[1][1], weights[3]))
        self.assertEqual(
            [cache_key for _, _, cache_key in got],
            ["weights#expert2", "weights#expert3"],
        )


class TestSharedMoeProductionLoad(unittest.TestCase):
    def test_unmarked_2d_shared_expert_uses_dense_buffers(self):
        pool = _make_pool(
            num_experts_global=8,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        pool.num_layer = 1
        pool.tp_rank = 0
        pool.max_lora_rank = 2
        pool.target_modules = {"down_proj"}
        pool.experts_shared_outer_loras = False
        pool.strict_loading = True
        pool.lora_added_tokens_size = 0
        pool.pin_memory_available = False
        pool.enable_lora_overlap_loading = False
        pool.base_model = object()
        pool.A_buffer = {
            "down_proj": [torch.full((1, 2, 3), -1.0)],
            "down_proj_shared_moe": [torch.full((1, 2, 2, 3), -7.0)],
        }
        pool.B_buffer = {
            "down_proj": [torch.full((1, 5, 2), -1.0)],
            "down_proj_shared_moe": [torch.full((1, 1, 5, 2), -7.0)],
        }
        pool.embedding_A_buffer = {}
        pool.embedding_B_buffer = {}
        pool.lm_head_A_buffer = {}
        pool.lm_head_B_buffer = {}
        pool.new_embeddings_buffer = {}

        down_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        down_b = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        adapter = types.SimpleNamespace(
            config=types.SimpleNamespace(r=2),
            scaling=2.5,
            layers=[
                types.SimpleNamespace(
                    weights={
                        "model.layers.0.mlp.shared_experts.down_proj.lora_A.weight": down_a,
                        "model.layers.0.mlp.shared_experts.down_proj.lora_B.weight": down_b,
                    },
                    pinned_weights={},
                )
            ],
            embedding_layers={},
            added_tokens_embeddings={},
        )

        _load_lora_weight_to_buffer(
            pool,
            uid="dense-shared",
            buffer_id=0,
            lora_adapter=adapter,
            lora_modules=[
                {"model.layers.0.mlp.shared_experts.down_proj": _FakeDenseLayer()}
            ],
            lora_embed_tokens_module=None,
            lora_lm_head_module=None,
        )

        torch.testing.assert_close(pool.A_buffer["down_proj"][0][0], down_a)
        torch.testing.assert_close(pool.B_buffer["down_proj"][0][0], down_b)
        self.assertTrue(torch.all(pool.A_buffer["down_proj_shared_moe"][0] == -7))
        self.assertTrue(torch.all(pool.B_buffer["down_proj_shared_moe"][0] == -7))

    def test_rank3_loads_shared_gate_b_and_down_a(self):
        """Shared-sink weights stay replicated even on a nonzero EP rank."""
        pool = _make_pool(
            num_experts_global=256,
            moe_ep_size=4,
            moe_ep_rank=3,
            moe_use_local_expert_ids=True,
        )
        pool.num_layer = 1
        pool.max_lora_rank = 2
        pool.target_modules = {"gate_up_proj", "down_proj"}
        pool.experts_shared_outer_loras = True
        pool.strict_loading = True
        pool.lora_added_tokens_size = 0
        pool.pin_memory_available = False
        pool.enable_lora_overlap_loading = False
        pool.base_model = object()
        pool.A_buffer = {
            "gate_up_proj": [torch.full((1, 4, 5), -7.0)],
            "down_proj": [torch.full((1, 2, 3), -7.0)],
            "gate_up_proj_shared_moe": [torch.full((1, 1, 4, 5), -1.0)],
            "down_proj_shared_moe": [torch.full((1, 2, 2, 3), -1.0)],
        }
        pool.B_buffer = {
            "gate_up_proj": [torch.full((1, 6, 2), -7.0)],
            "down_proj": [torch.full((1, 5, 2), -7.0)],
            "gate_up_proj_shared_moe": [torch.full((1, 2, 6, 2), -1.0)],
            "down_proj_shared_moe": [torch.full((1, 1, 5, 2), -1.0)],
        }
        pool.embedding_A_buffer = {}
        pool.embedding_B_buffer = {}
        pool.lm_head_A_buffer = {}
        pool.lm_head_B_buffer = {}
        pool.new_embeddings_buffer = {}

        gate_b = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
        down_a = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
        adapter = types.SimpleNamespace(
            config=types.SimpleNamespace(r=2),
            scaling=2.5,
            layers=[
                types.SimpleNamespace(
                    weights={
                        "model.layers.0.mlp.shared_experts.gate_up_proj.lora_B.weight": gate_b,
                        "model.layers.0.mlp.shared_experts.down_proj.lora_A.weight": down_a,
                    },
                    pinned_weights={},
                )
            ],
            embedding_layers={},
            added_tokens_embeddings={},
        )

        shared_sink = _FakeSharedMoeLayer(tp_rank=1)

        _load_lora_weight_to_buffer(
            pool,
            uid="shared",
            buffer_id=0,
            lora_adapter=adapter,
            lora_modules=[{"shared_experts": shared_sink}],
            lora_embed_tokens_module=None,
            lora_lm_head_module=None,
        )

        torch.testing.assert_close(pool.A_buffer["down_proj_shared_moe"][0][0], down_a)
        torch.testing.assert_close(
            pool.B_buffer["gate_up_proj_shared_moe"][0][0], gate_b * adapter.scaling
        )
        self.assertTrue(torch.all(pool.A_buffer["down_proj"][0] == -7))
        self.assertTrue(torch.all(pool.B_buffer["gate_up_proj"][0] == -7))

        pool.A_buffer["down_proj_shared_moe"][0].fill_(-1)
        pool.B_buffer["gate_up_proj_shared_moe"][0].fill_(-1)
        adapter.layers[0].weights = {
            **{
                f"model.layers.0.mlp.shared_experts.{i}.gate_up_proj.lora_B.weight": gate_b[
                    i
                ]
                for i in range(2)
            },
            **{
                f"model.layers.0.mlp.shared_experts.{i}.down_proj.lora_A.weight": down_a[
                    i
                ]
                for i in range(2)
            },
        }
        _load_lora_weight_to_buffer(
            pool,
            uid="shared-named",
            buffer_id=0,
            lora_adapter=adapter,
            lora_modules=[{"shared_experts": shared_sink}],
            lora_embed_tokens_module=None,
            lora_lm_head_module=None,
        )
        torch.testing.assert_close(pool.A_buffer["down_proj_shared_moe"][0][0], down_a)
        torch.testing.assert_close(
            pool.B_buffer["gate_up_proj_shared_moe"][0][0], gate_b * adapter.scaling
        )

    def test_missing_inner_factors_clear_reused_shared_and_routed_slots(self):
        for suffix, expert_path in (
            ("_shared_moe", "shared_experts"),
            ("_moe", "experts"),
        ):
            with self.subTest(suffix=suffix):
                pool = _make_pool(
                    num_experts_global=2,
                    moe_ep_size=1,
                    moe_ep_rank=0,
                    moe_use_local_expert_ids=False,
                )
                pool.num_layer = 1
                pool.max_lora_rank = 2
                pool.target_modules = {"gate_up_proj", "down_proj"}
                pool.experts_shared_outer_loras = True
                pool.strict_loading = True
                pool.lora_added_tokens_size = 0
                pool.pin_memory_available = False
                pool.enable_lora_overlap_loading = False
                pool.base_model = object()
                pool.A_buffer = {
                    f"gate_up_proj{suffix}": [torch.full((1, 1, 4, 5), -1.0)],
                    f"down_proj{suffix}": [torch.full((1, 2, 2, 3), -1.0)],
                }
                pool.B_buffer = {
                    f"gate_up_proj{suffix}": [torch.full((1, 2, 6, 2), -1.0)],
                    f"down_proj{suffix}": [torch.full((1, 1, 5, 2), -1.0)],
                }
                pool.embedding_A_buffer = {}
                pool.embedding_B_buffer = {}
                pool.lm_head_A_buffer = {}
                pool.lm_head_B_buffer = {}
                pool.new_embeddings_buffer = {}

                adapter = types.SimpleNamespace(
                    config=types.SimpleNamespace(r=2),
                    scaling=2.5,
                    layers=[
                        types.SimpleNamespace(
                            weights={
                                f"model.layers.0.mlp.{expert_path}.gate_up_proj."
                                "lora_A.weight": torch.ones(1, 4, 5),
                                f"model.layers.0.mlp.{expert_path}.down_proj."
                                "lora_B.weight": torch.ones(1, 5, 2),
                            },
                            pinned_weights={},
                        )
                    ],
                    embedding_layers={},
                    added_tokens_embeddings={},
                )
                module = (
                    _FakeSharedMoeLayer()
                    if suffix == "_shared_moe"
                    else _FakeRoutedMoeLayer()
                )

                _load_lora_weight_to_buffer(
                    pool,
                    uid="adapter",
                    buffer_id=0,
                    lora_adapter=adapter,
                    lora_modules=[{"experts": module}],
                    lora_embed_tokens_module=None,
                    lora_lm_head_module=None,
                )

                self.assertEqual(
                    torch.count_nonzero(pool.A_buffer[f"down_proj{suffix}"][0]), 0
                )
                self.assertEqual(
                    torch.count_nonzero(pool.B_buffer[f"gate_up_proj{suffix}"][0]), 0
                )


class TestMoeSlotRankReuse(unittest.TestCase):
    def test_smaller_rank_reuse_zeroes_the_previous_tails(self):
        """MoE LoRA kernels read every physical-rank row and column, so a rank-2
        adapter reusing a rank-4 slot must leave zeros past its rank: in A after
        both the gate block and the up block (each at its max-rank offset), and
        in B."""
        hidden, inter, max_rank, experts = 3, 5, 4, 2
        pool = _make_pool(
            num_experts_global=experts,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_use_local_expert_ids=False,
        )
        pool.num_layer = 1
        pool.max_lora_rank = max_rank
        pool.target_modules = {"gate_up_proj", "down_proj"}
        pool.experts_shared_outer_loras = False
        pool.strict_loading = True
        pool.lora_added_tokens_size = 0
        pool.pin_memory_available = False
        pool.enable_lora_overlap_loading = False
        pool.base_model = object()
        pool.A_buffer = {
            "gate_up_proj_moe": [torch.zeros(1, experts, 2 * max_rank, hidden)],
            "down_proj_moe": [torch.zeros(1, experts, max_rank, inter)],
        }
        pool.B_buffer = {
            "gate_up_proj_moe": [torch.zeros(1, experts, 2 * inter, max_rank)],
            "down_proj_moe": [torch.zeros(1, experts, hidden, max_rank)],
        }
        pool.embedding_A_buffer = {}
        pool.embedding_B_buffer = {}
        pool.lm_head_A_buffer = {}
        pool.lm_head_B_buffer = {}
        pool.new_embeddings_buffer = {}

        def adapter(rank, start):
            def ramp(*shape):
                return (
                    torch.arange(start, start + torch.Size(shape).numel())
                    .reshape(shape)
                    .float()
                )

            prefix = "model.layers.0.mlp.experts"
            weights = {}
            for expert in range(experts):
                for name, shape in (
                    ("gate_up_proj.lora_A", (2 * rank, hidden)),
                    ("gate_up_proj.lora_B", (2 * inter, rank)),
                    ("down_proj.lora_A", (rank, inter)),
                    ("down_proj.lora_B", (hidden, rank)),
                ):
                    weights[f"{prefix}.{expert}.{name}.weight"] = ramp(*shape)
            return types.SimpleNamespace(
                config=types.SimpleNamespace(r=rank),
                scaling=2.0,
                layers=[types.SimpleNamespace(weights=weights, pinned_weights={})],
                embedding_layers={},
                added_tokens_embeddings={},
            )

        small = adapter(rank=2, start=1000)
        for uid, lora in (("rank4", adapter(rank=4, start=1)), ("rank2", small)):
            _load_lora_weight_to_buffer(
                pool,
                uid=uid,
                buffer_id=0,
                lora_adapter=lora,
                lora_modules=[{"experts": _FakeRoutedMoeLayer()}],
                lora_embed_tokens_module=None,
                lora_lm_head_module=None,
            )

        weights = small.layers[0].weights
        gate_up_a = pool.A_buffer["gate_up_proj_moe"][0][0]
        down_a = pool.A_buffer["down_proj_moe"][0][0]
        gate_up_b = pool.B_buffer["gate_up_proj_moe"][0][0]
        down_b = pool.B_buffer["down_proj_moe"][0][0]
        for expert in range(experts):
            with self.subTest(expert=expert):
                prefix = f"model.layers.0.mlp.experts.{expert}"
                stacked_a = weights[f"{prefix}.gate_up_proj.lora_A.weight"]
                for block in range(2):  # gate, then up, each at block * max_rank
                    rows = gate_up_a[expert, block * max_rank : (block + 1) * max_rank]
                    torch.testing.assert_close(
                        rows[:2], stacked_a[block * 2 : (block + 1) * 2]
                    )
                    self.assertEqual(torch.count_nonzero(rows[2:]), 0)
                torch.testing.assert_close(
                    down_a[expert, :2], weights[f"{prefix}.down_proj.lora_A.weight"]
                )
                self.assertEqual(torch.count_nonzero(down_a[expert, 2:]), 0)
                for buffer, name in (
                    (gate_up_b, "gate_up_proj"),
                    (down_b, "down_proj"),
                ):
                    torch.testing.assert_close(
                        buffer[expert, :, :2],
                        weights[f"{prefix}.{name}.lora_B.weight"] * small.scaling,
                    )
                    self.assertEqual(torch.count_nonzero(buffer[expert, :, 2:]), 0)


class TestModuleLevelHelpers(unittest.TestCase):
    """The pool's `_moe_runner_keeps_global_expert_ids` must agree with
    `StandardDispatcher` on which runners keep global expert ids: a runner on
    one list but not the other keys the LoRA buffers by the wrong expert ids
    under EP, with no error."""

    def test_real_backend_predicates_match_supported_id_contracts(self):
        dispatcher_global_ids = {
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.FLASHINFER_MXFP4,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
            MoeRunnerBackend.FLASHINFER_MEGAMOE,
            MoeRunnerBackend.HPC_OPS,
        }
        # MegaMoE keeps global ids in the dispatcher but has no LoRA runner.
        lora_global_ids = dispatcher_global_ids - {MoeRunnerBackend.FLASHINFER_MEGAMOE}
        for backend in MoeRunnerBackend:
            config = MoeRunnerConfig(
                num_experts=8,
                num_local_experts=2,
                num_fused_shared_experts=0,
                top_k=2,
                runner_backend=backend,
            )
            with (
                self.subTest(backend=backend.value),
                mock.patch.object(
                    standard_dispatcher, "get_moe_runner_backend", return_value=backend
                ),
                mock.patch.object(
                    moe_utils, "get_moe_runner_backend", return_value=backend
                ),
                # EP rank 1 of 4 under `--tp 4 --ep 4`, stated as a whole
                # layout so the topology check sees a consistent one.
                get_parallel().override(
                    tp_size=4,
                    tp_rank=1,
                    attn_tp_size=4,
                    attn_tp_rank=1,
                    attn_dp_size=1,
                    attn_dp_rank=0,
                    attn_cp_size=1,
                    attn_cp_rank=0,
                    moe_tp_size=1,
                    moe_ep_size=4,
                    moe_ep_rank=1,
                    moe_dp_size=1,
                ),
            ):
                dispatcher = standard_dispatcher.StandardDispatcher(config)
                self.assertEqual(
                    dispatcher.skip_local_expert_mapping,
                    backend in dispatcher_global_ids,
                )
                self.assertEqual(
                    _moe_runner_keeps_global_expert_ids(),
                    backend in lora_global_ids,
                )


class TestPoolInitPicksUpEpContext(unittest.TestCase):
    """`LoRAMemoryPool.__init__` keys per-expert buffers by the EP-local slice
    unless the MoE runner keeps global expert ids."""

    @staticmethod
    def _new_pool_on_ep4_rank2(keeps_global: bool) -> LoRAMemoryPool:
        """Run `__init__` (minus `init_buffers`) as EP rank 2 of 4."""
        with (
            mock.patch.object(
                mem_pool_module, "_get_moe_ep_context", return_value=(4, 2)
            ),
            mock.patch.object(
                mem_pool_module, "_get_moe_tp_context", return_value=(1, 0)
            ),
            mock.patch.object(
                mem_pool_module,
                "_moe_runner_keeps_global_expert_ids",
                return_value=keeps_global,
            ),
            mock.patch.object(LoRAMemoryPool, "init_buffers", lambda self, _m: None),
            # The pool reads its TP placement from the parallel context; state
            # a whole layout so the topology check sees a consistent one.
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_tp_size=1,
                moe_ep_size=1,
                moe_dp_size=1,
            ),
        ):
            hf_cfg = types.SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=8,
                vocab_size=32,
                num_experts=8,
            )
            base_model = torch.nn.Linear(8, 8, bias=False)
            base_model.config = hf_cfg
            return LoRAMemoryPool(
                base_hf_config=hf_cfg,
                max_loras_per_batch=1,
                dtype=torch.bfloat16,
                max_lora_rank=8,
                target_modules={"qkv_proj"},
                base_model=base_model,
                eviction_policy="lru",
                lora_added_tokens_size=0,
            )

    def test_ep4_keys_buffers_locally_unless_runner_keeps_global_ids(self):
        # Triton/DeepGEMM runners take local ids from the dispatcher; the
        # FlashInfer runners keep global topk_ids, so buffers stay global.
        for keeps_global in (False, True):
            with self.subTest(keeps_global=keeps_global):
                pool = self._new_pool_on_ep4_rank2(keeps_global)
                self.assertEqual((pool.moe_ep_size, pool.moe_ep_rank), (4, 2))
                self.assertEqual(pool.moe_use_local_expert_ids, not keeps_global)


if __name__ == "__main__":
    unittest.main()
