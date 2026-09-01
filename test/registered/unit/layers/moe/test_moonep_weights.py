import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.deep_gemm import _moonep_m_indices
from sglang.srt.layers.moe.token_dispatcher import moonep_weights
from sglang.srt.layers.moe.token_dispatcher.moonep_weights import (
    MoonEPWeightPool,
    _num_moe_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

EP_RANK = 1
EP_SIZE = 4
NUM_LOCAL_EXPERTS = 4
NUM_PREFETCH_SLOTS = 2
NUM_LAYERS = 2
SPECS = {
    moonep_weights.W13_WEIGHT: ((8, 16), torch.uint8),
    moonep_weights.W2_WEIGHT: ((4, 32), torch.uint8),
}
CHUNK_ROWS = 12


def _local_first_enabled():
    return envs.SGLANG_ENABLE_MOONEP_LOCAL_FIRST.override(True)


def _fake_moonep(rotation=None):
    moonep = types.ModuleType("moonep")
    buffer = types.ModuleType("moonep.buffer")
    prefetch = types.ModuleType("moonep.prefetch")

    def create_nvl_dist_tensor(
        chunk_shape, dtype, local_rank, world_size, group=None, local_first=False
    ):
        return torch.zeros(world_size * chunk_shape[0], *chunk_shape[1:], dtype=dtype)

    buffer.local_first_chunk_index = rotation or (
        lambda owner_rank, local_rank, world_size: (owner_rank - local_rank)
        % world_size
    )
    buffer.create_nvl_dist_tensor = create_nvl_dist_tensor
    buffer.pad_dim0_for_alignment = lambda shape, dtype: 1
    prefetch.prefetch_retile_nbytes = lambda nbytes: -(-nbytes // 128) * 128

    moonep.buffer = buffer
    moonep.prefetch = prefetch
    return {"moonep": moonep, "moonep.buffer": buffer, "moonep.prefetch": prefetch}


class TestMoonEPPoolRows(CustomTestCase):
    def setUp(self):
        with patch.dict(sys.modules, _fake_moonep()), _local_first_enabled():
            self.pool = MoonEPWeightPool(
                num_layers=NUM_LAYERS,
                num_local_experts=NUM_LOCAL_EXPERTS,
                num_prefetch_slots=NUM_PREFETCH_SLOTS,
                ep_rank=EP_RANK,
                ep_size=EP_SIZE,
                group=None,
                specs=SPECS,
            )
        moonep_weights._pool = self.pool

    def tearDown(self):
        moonep_weights._pool = None

    def test_chunk_rows_cover_every_layer_block(self):
        self.assertEqual(self.pool.chunk_rows, CHUNK_ROWS)

    def test_expert_rows_place_each_owner_in_its_rotated_chunk(self):
        rows = moonep_weights.expert_rows(
            layer_id=7, expert_ids=torch.tensor([4, 5, 0, 8, 12])
        )
        self.assertEqual(
            rows.tolist(),
            [0, 1, 3 * CHUNK_ROWS, 1 * CHUNK_ROWS, 2 * CHUNK_ROWS],
        )
        self.assertEqual(rows.dtype, torch.int32)

    def test_row_mapping_and_loader_views_agree_on_every_owner(self):
        first_expert = torch.tensor(
            [owner * NUM_LOCAL_EXPERTS for owner in range(EP_SIZE)]
        )
        rows = moonep_weights.expert_rows(layer_id=7, expert_ids=first_expert)
        self.assertEqual(
            rows.tolist(),
            [self.pool.chunk_start(owner) for owner in range(EP_SIZE)],
        )

    def test_expert_rows_pass_negative_ids_through(self):
        rows = moonep_weights.expert_rows(
            layer_id=7, expert_ids=torch.tensor([-1, 4, -1])
        )
        self.assertEqual(rows.tolist(), [-1, 0, -1])

    def test_expert_rows_offset_by_the_layers_block(self):
        first = moonep_weights.expert_rows(layer_id=7, expert_ids=torch.tensor([4]))
        second = moonep_weights.expert_rows(layer_id=9, expert_ids=torch.tensor([4]))
        block = NUM_LOCAL_EXPERTS + NUM_PREFETCH_SLOTS
        self.assertEqual(first.tolist(), [0])
        self.assertEqual(second.tolist(), [block])

    def test_pool_refuses_more_layers_than_it_was_sized_for(self):
        for layer_id in range(NUM_LAYERS):
            self.pool.layer_offset(layer_id)
        with self.assertRaisesRegex(RuntimeError, "sized for 2 layers"):
            self.pool.layer_offset(NUM_LAYERS)

    def test_refuses_to_build_without_the_local_first_declaration(self):
        with (
            patch.dict(sys.modules, _fake_moonep()),
            envs.SGLANG_ENABLE_MOONEP_LOCAL_FIRST.override(False),
        ):
            with self.assertRaises(RuntimeError) as caught:
                MoonEPWeightPool(
                    num_layers=NUM_LAYERS,
                    num_local_experts=NUM_LOCAL_EXPERTS,
                    num_prefetch_slots=NUM_PREFETCH_SLOTS,
                    ep_rank=EP_RANK,
                    ep_size=EP_SIZE,
                    group=None,
                    specs=SPECS,
                )
        message = str(caught.exception)
        self.assertIn("SGLANG_ENABLE_MOONEP_LOCAL_FIRST", message)
        self.assertIn("local_first_chunk_index", message)
        self.assertIn("bytedance-iaas/MoonEP", message)

    def test_rejects_a_moonep_whose_rotation_expert_rows_does_not_assume(self):
        with (
            patch.dict(
                sys.modules,
                _fake_moonep(
                    rotation=lambda owner_rank, local_rank, world_size: (
                        local_rank - owner_rank
                    )
                    % world_size
                ),
            ),
            _local_first_enabled(),
        ):
            with self.assertRaisesRegex(RuntimeError, "local-first rotation"):
                MoonEPWeightPool(
                    num_layers=NUM_LAYERS,
                    num_local_experts=NUM_LOCAL_EXPERTS,
                    num_prefetch_slots=NUM_PREFETCH_SLOTS,
                    ep_rank=EP_RANK,
                    ep_size=EP_SIZE,
                    group=None,
                    specs=SPECS,
                )

    def test_rejects_a_shape_the_prefetch_copy_cannot_tile(self):
        with patch.dict(sys.modules, _fake_moonep()), _local_first_enabled():
            with self.assertRaisesRegex(ValueError, "w2_weight.*multiple of 128"):
                MoonEPWeightPool(
                    num_layers=NUM_LAYERS,
                    num_local_experts=NUM_LOCAL_EXPERTS,
                    num_prefetch_slots=NUM_PREFETCH_SLOTS,
                    ep_rank=EP_RANK,
                    ep_size=EP_SIZE,
                    group=None,
                    specs={**SPECS, moonep_weights.W2_WEIGHT: ((4, 16), torch.uint8)},
                )

    def test_group_rows_keep_home_groups_and_remap_the_prefetch_tail(self):
        num_global_experts = EP_SIZE * NUM_LOCAL_EXPERTS
        expert_ids = torch.cat(
            [torch.arange(num_global_experts), torch.tensor([12, -1])]
        )
        rows = moonep_weights.group_rows(7, expert_ids, num_global_experts)

        home = moonep_weights.expert_rows(7, torch.arange(num_global_experts))
        self.assertEqual(rows[:num_global_experts].tolist(), home.tolist())
        # slot_base = this rank's chunk + this layer's block + its own experts.
        self.assertEqual(rows[num_global_experts].item(), NUM_LOCAL_EXPERTS)
        self.assertEqual(rows[num_global_experts + 1].item(), -1)


class TestMoonEPMIndices(CustomTestCase):
    def test_empty_groups_are_skipped_and_the_tail_is_minus_one(self):
        m = _moonep_m_indices(
            cu_seqlens=torch.tensor([2, 2, 2, 5]),
            expert_ids=torch.tensor([5, 9, 11, 3]),
            all_tokens=7,
        )
        self.assertEqual(m.tolist(), [5, 5, 3, 3, 3, -1, -1])
        self.assertEqual(m.dtype, torch.int32)

    def test_rows_take_the_expert_id_not_the_group_index(self):
        m = _moonep_m_indices(
            cu_seqlens=torch.tensor([1, 2]),
            expert_ids=torch.tensor([40, 41]),
            all_tokens=2,
        )
        self.assertEqual(m.tolist(), [40, 41])

    def test_unfilled_prefetch_slots_stay_minus_one(self):
        m = _moonep_m_indices(
            cu_seqlens=torch.tensor([2, 4]),
            expert_ids=torch.tensor([5, -1]),
            all_tokens=5,
        )
        self.assertEqual(m.tolist(), [5, 5, -1, -1, -1])


class TestMoonEPLayerCount(CustomTestCase):
    def _count(self, *, pp_rank=0, pp_size=1, hf_text_config=None):
        model_config = SimpleNamespace(
            num_hidden_layers=8,
            first_k_dense_replace=2,
            hf_text_config=hf_text_config or SimpleNamespace(),
        )
        with (
            patch(
                "sglang.srt.runtime_context.process_model_config",
                return_value=model_config,
            ),
            patch(
                "sglang.srt.distributed.get_pp_group",
                return_value=SimpleNamespace(rank_in_group=pp_rank, world_size=pp_size),
            ),
        ):
            return _num_moe_layers()

    def test_counts_layers_after_the_dense_prefix(self):
        self.assertEqual(self._count(), 6)

    def test_counts_only_this_pipeline_stage(self):
        self.assertEqual(self._count(pp_rank=0, pp_size=2), 2)
        self.assertEqual(self._count(pp_rank=1, pp_size=2), 4)

    def test_refuses_a_model_whose_moe_layers_are_strided(self):
        with self.assertRaisesRegex(NotImplementedError, "moe_layer_freq"):
            self._count(hf_text_config=SimpleNamespace(moe_layer_freq=2))


if __name__ == "__main__":
    unittest.main()
