"""Which physical slots the MoE weight loaders fill; no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import contextlib
import random
import types
import unittest

import torch

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    _compute_logical_to_all_physical_map,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization import moe_wna16
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

# EP 2, 4 logical experts + 2 redundant: rank 0 holds slots 0-2, rank 1 slots 3-5.
# Logical expert 3 has both of its replicas on rank 1; rank 0 has none.
SAME_RANK_REPLICAS = [[0, 1, 2, 3, 3, 0]] * 3

# EP 4 on 2 nodes, 2 slots per rank. Expert 0 has two replicas on rank 0 and one on
# rank 2, so rank 1 (no replica, node 0) narrows to a same-node replica.
SAME_NODE_NARROWING = [[0, 0, 1, 2, 3, 0, 1, 3]] * 2


def _metadata(
    layout,
    *,
    ep_size,
    nnodes,
    moe_ep_rank,
    moe_a2a_backend="deepep",
    ep_join_mode=None,
):
    """What ExpertLocationMetadata.init_by_mapping builds on one EP rank."""
    physical_to_logical_map = torch.tensor(layout)
    num_logical_experts = int(physical_to_logical_map.max()) + 1
    with get_context().override_server_args(
        ep_size=ep_size,
        nnodes=nnodes,
        moe_a2a_backend=moe_a2a_backend,
        ep_join_mode=ep_join_mode,
        ep_dispatch_algorithm="static" if moe_a2a_backend != "none" else "dynamic",
    ):
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=num_logical_experts,
            ep_size=ep_size,
            moe_ep_rank=moe_ep_rank,
        )
        return ExpertLocationMetadata._init_raw(
            ep_size=ep_size,
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
            moe_ep_rank=moe_ep_rank,
        )


def _random_layout(rng, num_layers, num_logical_experts, num_physical_experts):
    """Every logical expert at least once, the remaining slots random replicas."""
    layout = []
    for _ in range(num_layers):
        row = list(range(num_logical_experts)) + [
            rng.randrange(num_logical_experts)
            for _ in range(num_physical_experts - num_logical_experts)
        ]
        rng.shuffle(row)
        layout.append(row)
    return layout


def _stand_in_layer(
    *,
    layer_id,
    ep_size,
    rank,
    num_local_routed,
    num_global_routed,
    num_shared=0,
):
    """The FusedMoE fields and methods that the expert weight loaders read."""
    layer = types.SimpleNamespace(
        quant_config=None,
        quant_method=None,
        uses_expert_location=True,
        layer_id=layer_id,
        moe_ep_size=ep_size,
        num_local_experts=num_local_routed + num_shared,
        num_fused_shared_experts=num_shared,
        _has_fused_shared=num_shared > 0,
        _expert_storage_rank=rank,
        _num_local_routed=num_local_routed,
        _num_global_routed=num_global_routed,
    )
    for name in (
        "_map_global_expert_id_to_local_expert_id",
        "_local_routed_expert_ids",
        "_weight_loader_physical",
    ):
        setattr(layer, name, types.MethodType(getattr(FusedMoE, name), layer))
    return layer


@contextlib.contextmanager
def _published(metadata):
    previous = get_global_expert_location_metadata()
    set_global_expert_location_metadata(metadata, allow_overwrite=True)
    try:
        yield
    finally:
        set_global_expert_location_metadata(previous, allow_overwrite=True)


def _load_with_fused_moe(
    metadata,
    *,
    layer_id,
    rank,
    num_local_routed,
    num_global_routed,
    num_shared=0,
    require_global_experts=False,
):
    """Run FusedMoE.weight_loader for every checkpoint expert on a stand-in layer and
    return {slot: checkpoint expert written there}; global slots for a global param."""
    layer = _stand_in_layer(
        layer_id=layer_id,
        ep_size=metadata.ep_size,
        rank=rank,
        num_local_routed=num_local_routed,
        num_global_routed=num_global_routed,
        num_shared=num_shared,
    )
    written = {}

    def _weight_loader_impl(*, param, loaded_weight, weight_name, shard_id, expert_id):
        written[expert_id] = int(loaded_weight)

    layer._weight_loader_impl = _weight_loader_impl
    param = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    if require_global_experts:
        param._sglang_require_global_experts = True
    # Shared experts first, so that a routed expert written over a shared slot shows.
    checkpoint_experts = [
        metadata.num_logical_experts + i for i in range(num_shared)
    ] + list(range(metadata.num_logical_experts))
    with _published(metadata):
        for expert_id in checkpoint_experts:
            FusedMoE.weight_loader(
                layer, param, torch.tensor(expert_id), "w13_weight", "w1", expert_id
            )
    return written


def _scale_up_layout(num_logical, per_rank):
    """EP 2 -> 4 scale-up: the new ranks' slots repeat experts 0..num_logical-1."""
    return [list(range(num_logical)) + [i % num_logical for i in range(2 * per_rank)]]


class TestExpertLocationWeightLoading(CustomTestCase):
    def test_lookup_returns_exactly_the_slots_of_each_expert(self):
        rng = random.Random(0)
        cases = [
            ("same_rank_replicas", SAME_RANK_REPLICAS, 2, 1),
            ("same_node_narrowing", SAME_NODE_NARROWING, 4, 2),
            ("random_ep4", _random_layout(rng, 4, 8, 16), 4, 1),
        ]
        for name, layout, ep_size, nnodes in cases:
            for moe_a2a_backend in ["deepep", "none"]:
                for rank in range(ep_size):
                    with self.subTest(name, backend=moe_a2a_backend, rank=rank):
                        metadata = _metadata(
                            layout,
                            ep_size=ep_size,
                            nnodes=nnodes,
                            moe_ep_rank=rank,
                            moe_a2a_backend=moe_a2a_backend,
                        )
                        for layer_id, row in enumerate(layout):
                            for logical_expert_id in range(
                                metadata.num_logical_experts
                            ):
                                self.assertEqual(
                                    metadata.logical_to_all_physical(
                                        layer_id, logical_expert_id
                                    ),
                                    [
                                        slot
                                        for slot, expert in enumerate(row)
                                        if expert == logical_expert_id
                                    ],
                                )

    def test_loader_fills_every_local_slot_with_its_expert(self):
        """Every slot a rank stores gets the expert the layout puts there,
        including a second replica of an expert on the same rank."""
        rng = random.Random(1)
        cases = [
            ("same_rank_replicas", SAME_RANK_REPLICAS, 2, 1),
            ("same_node_narrowing", SAME_NODE_NARROWING, 4, 2),
            ("random_ep2", _random_layout(rng, 3, 8, 16), 2, 1),
            ("random_ep4_2nodes", _random_layout(rng, 3, 8, 16), 4, 2),
        ]
        for name, layout, ep_size, nnodes in cases:
            per_rank = len(layout[0]) // ep_size
            for rank in range(ep_size):
                with self.subTest(name, rank=rank):
                    metadata = _metadata(
                        layout, ep_size=ep_size, nnodes=nnodes, moe_ep_rank=rank
                    )
                    for layer_id, row in enumerate(layout):
                        written = _load_with_fused_moe(
                            metadata,
                            layer_id=layer_id,
                            rank=rank,
                            num_local_routed=per_rank,
                            num_global_routed=len(row),
                        )
                        own = row[rank * per_rank : (rank + 1) * per_rank]
                        self.assertEqual(written, dict(enumerate(own)))

    def test_static_dispatch_only_targets_loaded_slots(self):
        """No rank's static dispatch table sends an expert's tokens to a slot that
        its owner left unloaded or loaded with another expert."""
        all_metadata = [
            _metadata(SAME_RANK_REPLICAS, ep_size=2, nnodes=1, moe_ep_rank=rank)
            for rank in range(2)
        ]
        per_rank = all_metadata[0].num_local_physical_experts
        for rank, metadata in enumerate(all_metadata):
            table = metadata.logical_to_rank_dispatch_physical_map.tolist()
            for layer_id, row in enumerate(table):
                for logical_expert_id, slot in enumerate(row):
                    owner = slot // per_rank
                    written = _load_with_fused_moe(
                        all_metadata[owner],
                        layer_id=layer_id,
                        rank=owner,
                        num_local_routed=per_rank,
                        num_global_routed=per_rank * 2,
                    )
                    self.assertEqual(
                        written.get(slot - owner * per_rank),
                        logical_expert_id,
                        f"{rank=} {layer_id=} {logical_expert_id=}",
                    )

    def test_scale_up_keeps_fused_shared_slots(self):
        """After an EP 2 -> 4 scale-up, loading routed experts on a starting rank
        leaves its fused shared slot to the shared expert."""
        # The new ranks' slots 256-511 lie past the starting ranks' _num_global_routed.
        num_logical, per_rank = 256, 128
        layout = _scale_up_layout(num_logical, per_rank)
        for rank in range(2):
            with self.subTest(rank=rank):
                metadata = _metadata(
                    layout,
                    ep_size=4,
                    nnodes=1,
                    moe_ep_rank=rank,
                    ep_join_mode="scale",
                )
                written = _load_with_fused_moe(
                    metadata,
                    layer_id=0,
                    rank=rank,
                    num_local_routed=per_rank,
                    num_global_routed=num_logical,
                    num_shared=1,
                )
                expected = dict(
                    enumerate(layout[0][rank * per_rank : (rank + 1) * per_rank])
                )
                expected[per_rank] = num_logical  # the shared expert's checkpoint id
                self.assertEqual(written, expected)

    def test_global_params_skip_the_local_filter(self):
        """A param with _sglang_require_global_experts (global input scales) still
        gets every slot of every expert, on every rank."""
        for rank in range(2):
            with self.subTest(rank=rank):
                metadata = _metadata(
                    SAME_RANK_REPLICAS, ep_size=2, nnodes=1, moe_ep_rank=rank
                )
                written = _load_with_fused_moe(
                    metadata,
                    layer_id=0,
                    rank=rank,
                    num_local_routed=3,
                    num_global_routed=6,
                    require_global_experts=True,
                )
                # Global params use slots e, e + 4, ... below the 6 physical slots.
                self.assertEqual(written, {0: 0, 4: 0, 1: 1, 5: 1, 2: 2, 3: 3})

    def test_wna16_qzeros_use_the_same_slots(self):
        # moe_wna16 writes qzeros outside the layer's loader, so it picks the
        # slots itself and must agree with the loader.
        scale_up = _scale_up_layout(256, 128)
        cases = [
            # name, layout, ep_size, ep_join_mode, rank, layer_id,
            # (local routed slots, global routed slots at startup),
            # fused shared experts, checkpoint expert, expected local slots
            ("second_replica", SAME_RANK_REPLICAS, 2, None, 1, 0, (3, 6), 0, 3, [0, 1]),
            ("scale_up_rank0", scale_up, 4, "scale", 0, 0, (128, 256), 1, 0, [0]),
            ("scale_up_rank1", scale_up, 4, "scale", 1, 0, (128, 256), 1, 0, []),
            # A draft layer past the expert map keeps the fused shared expert.
            ("draft_shared", [[0, 1, 2, 3]] * 3, 1, None, 0, 3, (4, 4), 1, 4, [4]),
        ]
        for (
            name,
            layout,
            ep_size,
            ep_join_mode,
            rank,
            layer_id,
            (num_local_routed, num_global_routed),
            num_shared,
            expert_id,
            expected,
        ) in cases:
            with self.subTest(name):
                metadata = _metadata(
                    layout,
                    ep_size=ep_size,
                    nnodes=1,
                    moe_ep_rank=rank,
                    ep_join_mode=ep_join_mode,
                )
                layer = _stand_in_layer(
                    layer_id=layer_id,
                    ep_size=ep_size,
                    rank=rank,
                    num_local_routed=num_local_routed,
                    num_global_routed=num_global_routed,
                    num_shared=num_shared,
                )
                with _published(metadata):
                    self.assertEqual(
                        moe_wna16._local_expert_ids(layer, expert_id), expected
                    )

    def test_dispatch_candidates_are_unchanged(self):
        # The per-rank nearest-replica pick for dispatch stays as it was.
        metadata = _metadata(SAME_RANK_REPLICAS, ep_size=2, nnodes=1, moe_ep_rank=1)
        self.assertEqual(int(metadata.logical_to_all_physical_map_num_valid[0, 3]), 1)
        self.assertEqual(int(metadata.logical_to_all_physical_map[0, 3, 0]), 3)

    def test_fallbacks_and_bounds(self):
        metadata = _metadata(SAME_RANK_REPLICAS, ep_size=2, nnodes=1, moe_ep_rank=0)
        num_layers = len(SAME_RANK_REPLICAS)
        # A draft layer beyond the expert map loads the identity slot.
        self.assertEqual(metadata.logical_to_all_physical(num_layers, 3), [3])
        self.assertEqual(
            metadata.logical_to_all_physical(0, 1, require_global_experts=True),
            [1, 5],
        )
        for bad_id in (metadata.num_logical_experts, -1):
            with self.subTest(bad_id=bad_id), self.assertRaises(IndexError):
                metadata.logical_to_all_physical(0, bad_id)


if __name__ == "__main__":
    unittest.main()
