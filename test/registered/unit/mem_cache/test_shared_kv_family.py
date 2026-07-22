import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.shared_kv import family as family_module
from sglang.srt.mem_cache.shared_kv.family import (
    OwnerShardedFamily,
    OwnerShardedFamilySpec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fake_slab(
    *,
    num_layers: int,
    layer_rows: int,
    rank_stride_rows: int,
    aligned_bytes_per_rank: int,
):
    return SimpleNamespace(
        allocation=SimpleNamespace(
            aligned_bytes_per_rank=aligned_bytes_per_rank,
        ),
        layer_rows=layer_rows,
        rank_stride_rows=rank_stride_rows,
        global_views=[torch.empty((1,), dtype=torch.uint8) for _ in range(num_layers)],
        rank_local_views=[
            torch.empty((1,), dtype=torch.uint8) for _ in range(num_layers)
        ],
        local_views=[torch.empty((1,), dtype=torch.uint8) for _ in range(num_layers)],
        close=MagicMock(),
    )


class TestOwnerShardedFamily(CustomTestCase):
    def test_family_uses_aligned_slab_stride_for_physical_owner_mapping(self):
        spec = OwnerShardedFamilySpec(
            name="packed-c4",
            num_layers=2,
            logical_rows_per_layer=18,
            ownership_granule=4,
            storage_rows_per_granule=1,
            row_shape=(16,),
            dtype=torch.uint8,
        )
        slab = _fake_slab(
            num_layers=2,
            layer_rows=3,
            rank_stride_rows=16,
            aligned_bytes_per_rank=4096,
        )

        with (
            patch(
                "sglang.srt.mem_cache.shared_kv.family.create_rank_major_shared_slab",
                return_value=slab,
            ) as create_slab,
            patch.object(family_module.dist, "get_world_size", return_value=2),
        ):
            family = OwnerShardedFamily.create(
                spec=spec,
                cp_size=2,
                cpu_group="cp-cpu",
                zero_initialize=False,
            )

        create_slab.assert_called_once_with(
            (3, 16),
            layer_num=2,
            dtype=torch.uint8,
            cpu_group="cp-cpu",
            first_dim_multiple=1,
            map_rank_local=True,
        )
        self.assertEqual(family.layout.minimum_blocks_per_rank, 3)
        self.assertEqual(family.layout.blocks_per_rank, 16)
        self.assertEqual(
            family.layout.physical_rows(torch.tensor([0, 4])).tolist(),
            [0, 64],
        )

    def test_family_exposes_checked_layer_views_and_idempotent_close(self):
        spec = OwnerShardedFamilySpec(
            name="c4-state",
            num_layers=2,
            logical_rows_per_layer=8,
            ownership_granule=4,
            storage_rows_per_granule=4,
            row_shape=(32,),
            dtype=torch.float32,
        )
        slab = _fake_slab(
            num_layers=2,
            layer_rows=4,
            rank_stride_rows=16,
            aligned_bytes_per_rank=4096,
        )

        with (
            patch(
                "sglang.srt.mem_cache.shared_kv.family.create_rank_major_shared_slab",
                return_value=slab,
            ),
            patch.object(family_module.dist, "get_world_size", return_value=2),
        ):
            family = OwnerShardedFamily.create(
                spec=spec,
                cp_size=2,
                cpu_group="cp-cpu",
                zero_initialize=False,
            )

        self.assertIs(family.layer_global(1), slab.global_views[1])
        self.assertIs(family.layer_rank_relative(1), slab.rank_local_views[1])
        self.assertIs(family.layer_owner_local(1), slab.local_views[1])
        with self.assertRaisesRegex(IndexError, "layer"):
            family.layer_global(2)

        family.close()
        family.close()
        slab.close.assert_called_once_with()

    def test_accounting_separates_logical_tail_and_vmm_alignment(self):
        spec = OwnerShardedFamilySpec(
            name="packed-c4",
            num_layers=2,
            logical_rows_per_layer=18,
            ownership_granule=4,
            storage_rows_per_granule=1,
            row_shape=(16,),
            dtype=torch.uint8,
        )
        slab = _fake_slab(
            num_layers=2,
            layer_rows=3,
            rank_stride_rows=16,
            aligned_bytes_per_rank=4096,
        )

        with (
            patch(
                "sglang.srt.mem_cache.shared_kv.family.create_rank_major_shared_slab",
                return_value=slab,
            ),
            patch.object(family_module.dist, "get_world_size", return_value=2),
        ):
            family = OwnerShardedFamily.create(
                spec=spec,
                cp_size=2,
                cpu_group="cp-cpu",
                zero_initialize=False,
            )

        accounting = family.accounting()
        self.assertEqual(accounting.logical_blocks_per_layer, 5)
        self.assertEqual(accounting.minimum_blocks_per_rank, 3)
        self.assertEqual(accounting.logical_storage_bytes, 160)
        self.assertEqual(accounting.minimum_physical_bytes_per_rank, 96)
        self.assertEqual(accounting.mapped_bytes_per_rank, 4096)
        self.assertEqual(accounting.alignment_overhead_bytes_per_rank, 4000)

    def test_large_capacity_products_do_not_overflow_int32(self):
        spec = OwnerShardedFamilySpec(
            name="large-packed-kv",
            num_layers=43,
            logical_rows_per_layer=2**31,
            ownership_granule=256,
            storage_rows_per_granule=1,
            row_shape=(584,),
            dtype=torch.uint8,
        )
        minimum_blocks = (spec.logical_rows_per_layer // 256 + 7) // 8
        local_rows = minimum_blocks
        minimum_bytes = 43 * local_rows * 584
        slab = _fake_slab(
            num_layers=43,
            layer_rows=local_rows,
            rank_stride_rows=43 * local_rows,
            aligned_bytes_per_rank=minimum_bytes + 65536,
        )

        with (
            patch(
                "sglang.srt.mem_cache.shared_kv.family.create_rank_major_shared_slab",
                return_value=slab,
            ),
            patch.object(family_module.dist, "get_world_size", return_value=8),
        ):
            family = OwnerShardedFamily.create(
                spec=spec,
                cp_size=8,
                cpu_group="cp-cpu",
                zero_initialize=False,
            )

        self.assertGreater(family.accounting().logical_storage_bytes, 2**31)
        self.assertGreater(family.accounting().mapped_bytes_per_rank, 2**31)

    def test_family_rejects_cp_size_that_differs_from_process_group(self):
        spec = OwnerShardedFamilySpec(
            name="mismatched-group",
            num_layers=1,
            logical_rows_per_layer=8,
            ownership_granule=4,
            storage_rows_per_granule=1,
            row_shape=(16,),
            dtype=torch.uint8,
        )

        with (
            patch.object(family_module.dist, "get_world_size", return_value=4),
            patch.object(family_module, "create_rank_major_shared_slab") as create_slab,
            self.assertRaisesRegex(ValueError, "cp_size=8, group_size=4"),
        ):
            OwnerShardedFamily.create(
                spec=spec,
                cp_size=8,
                cpu_group="cp-cpu",
                zero_initialize=False,
            )

        create_slab.assert_not_called()

    def test_zero_initialization_failure_is_published_before_cleanup(self):
        spec = OwnerShardedFamilySpec(
            name="zero-init-failure",
            num_layers=1,
            logical_rows_per_layer=8,
            ownership_granule=4,
            storage_rows_per_granule=1,
            row_shape=(16,),
            dtype=torch.uint8,
        )
        slab = _fake_slab(
            num_layers=1,
            layer_rows=1,
            rank_stride_rows=1,
            aligned_bytes_per_rank=4096,
        )
        slab.allocation.local_view = MagicMock()
        local_error = RuntimeError("CUDA zero initialization failed")

        with (
            patch.object(family_module.dist, "get_world_size", return_value=2),
            patch.object(family_module.dist, "get_rank", return_value=1),
            patch.object(
                family_module, "create_rank_major_shared_slab", return_value=slab
            ),
            patch.object(
                family_module.torch.cuda, "synchronize", side_effect=local_error
            ),
            patch.object(
                family_module,
                "_synchronize_vmm_stage",
                side_effect=RuntimeError("symmetric zero-init failure"),
                create=True,
            ) as synchronize,
            self.assertRaisesRegex(RuntimeError, "symmetric zero-init failure"),
        ):
            OwnerShardedFamily.create(
                spec=spec,
                cp_size=2,
                cpu_group="cp-cpu",
                zero_initialize=True,
            )

        synchronize.assert_called_once_with(
            "cp-cpu", 1, "family zero initialization", local_error
        )
        slab.close.assert_called_once_with()

    def test_invalid_family_specs_are_rejected(self):
        valid = dict(
            name="family",
            num_layers=1,
            logical_rows_per_layer=8,
            ownership_granule=4,
            storage_rows_per_granule=4,
            row_shape=(16,),
            dtype=torch.uint8,
        )
        for field, value in (
            ("name", ""),
            ("num_layers", 0),
            ("logical_rows_per_layer", -1),
            ("ownership_granule", 0),
            ("storage_rows_per_granule", 0),
            ("row_shape", ()),
        ):
            kwargs = dict(valid)
            kwargs[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    OwnerShardedFamilySpec(**kwargs)


if __name__ == "__main__":
    unittest.main()
