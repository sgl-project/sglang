"""Canonical rank ownership and owner-local snapshot assembly invariants."""

import unittest
from unittest.mock import MagicMock, call

import msgspec
import torch
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.protocol import (
    ContractError,
    canonical_bytes,
    digest_bytes,
    validate_manifest,
)
from sglang.srt.training_capture.snapshot import (
    PreparedSnapshotPartition,
    SnapshotMetadata,
    assemble_snapshot,
)
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_partition import make_partitioned_snapshot
from sglang.test.training_capture_utils import Registrar, make_kv_spec

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def heterogeneous_kv():
    kv = make_kv_spec()
    return msgspec.structs.replace(
        kv,
        layers=[
            msgspec.structs.replace(layer, num_kv_heads=8 if layer.layer_id == 1 else 2)
            for layer in kv.layers
        ],
    )


class TestCaptureTopology(CustomTestCase):
    def test_canonical_heads_match_real_qkv_loader_replication(self):
        from sglang.srt.layers.linear import QKVParallelLinear

        kv = heterogeneous_kv()
        for tp_size in (1, 2, 4, 8):
            layout = plan_capture_layout(kv, tp_size=tp_size, pp_layer_ranges=[(0, 4)])
            for geometry in kv.layers:
                with self.subTest(tp_size=tp_size, heads=geometry.num_kv_heads):
                    seen = set()
                    for rank in range(tp_size):
                        linear = QKVParallelLinear(
                            2,
                            head_size=4,
                            total_num_heads=8,
                            total_num_kv_heads=geometry.num_kv_heads,
                            bias=False,
                            params_dtype=torch.float32,
                            tp_size=tp_size,
                            tp_rank=rank,
                        )
                        weights = (
                            torch.arange(geometry.num_kv_heads)
                            .repeat_interleave(4)
                            .float()[:, None]
                            .expand(-1, 2)
                            .contiguous()
                        )
                        linear.weight_loader(linear.weight, weights, "k")
                        offset = linear.num_heads * 4
                        loaded = (
                            linear.weight[offset : offset + linear.num_kv_heads * 4]
                            .reshape(-1, 4, 2)[:, 0, 0]
                            .tolist()
                        )
                        canonical = [int(head) for head in loaded if head not in seen]
                        part = layout.partition(f"dp0-pp0-tp{rank}")
                        planned = [
                            head
                            for region in part.heads
                            if region.layer_id == geometry.layer_id
                            for head in range(region.start, region.end)
                        ]
                        self.assertEqual(planned, canonical)
                        seen.update(loaded)
                    self.assertEqual(seen, set(range(geometry.num_kv_heads)))

    def test_pp_local_budgets_inactive_replicas_and_aux_only_stage(self):
        kv = heterogeneous_kv()
        layout = plan_capture_layout(
            kv, tp_size=4, pp_layer_ranges=[(0, 2), (2, 4), (4, 6)], aux_tp_rank=1
        )
        self.assertEqual(
            set(layout.topology.owners),
            {
                "dp0-pp0-tp0",
                "dp0-pp0-tp1",
                "dp0-pp0-tp2",
                "dp0-pp0-tp3",
                "dp0-pp1-tp0",
                "dp0-pp1-tp2",
                "dp0-pp2-tp1",
            },
        )
        registrar = Registrar()
        args = {
            "kv": kv,
            "max_tokens": 6,
            "slots": 1,
            "max_bytes": 256,
            "registrar": registrar,
            "pin_memory": False,
            "partition": layout.partition("dp0-pp0-tp0"),
            "device": torch.device("cpu"),
            "kv_d2h_batch_tokens": 2,
            "max_device_bytes": 96,
        }
        pool = HostBufferPool(**args)
        try:
            slot = pool.acquire()
            self.assertEqual(set(slot.tensors), {"target_k.1", "target_v.1"})
            self.assertEqual(slot.tensors["target_k.1"].shape, (6, 2, 4))
            self.assertEqual(slot.manifest_buffer.numel(), 0)
            self.assertEqual(pool.allocated_bytes, 256)
            self.assertEqual(pool.device_allocated_bytes, 96)
            pool.release(slot, transfer_complete=True)
        finally:
            pool.close()
        for change in (
            {"max_bytes": 255},
            {"max_device_bytes": 95},
            {"partition": layout.partition("dp0-pp1-tp1")},
        ):
            with (
                self.subTest(change=change),
                self.assertRaises((ValueError, ContractError)),
            ):
                HostBufferPool(**(args | change))
            self.assertFalse(registrar.buffers)
        pool = HostBufferPool(
            **(
                args
                | {
                    "partition": layout.partition("dp0-pp2-tp1"),
                    "max_bytes": 2 << 20,
                    "device": None,
                    "max_device_bytes": 0,
                }
            )
        )
        try:
            slot = pool.acquire()
            self.assertEqual(len(slot.tensors), 8)
            self.assertFalse(any(name.startswith("target_") for name in slot.tensors))
            self.assertIsNone(slot.device_storage)
            self.assertGreater(slot.manifest_buffer.numel(), 0)
            pool.release(slot, transfer_complete=True)
        finally:
            pool.close()

    def test_bad_layouts_and_foreign_contracts_fail_before_allocation(self):
        kv = heterogeneous_kv()
        for args in (
            {"tp_size": 3, "pp_layer_ranges": [(0, 4)]},
            {"tp_size": 0, "pp_layer_ranges": [(0, 4)]},
            {"tp_size": 4, "pp_layer_ranges": [(0, 2), (3, 4)]},
            {"tp_size": 4, "pp_layer_ranges": [(0, 3), (2, 4)]},
            {"tp_size": 4, "pp_layer_ranges": [(0, 2)]},
            {"tp_size": 4, "pp_layer_ranges": [(0, 4)], "aux_tp_rank": 4},
        ):
            with self.subTest(args=args), self.assertRaises(ContractError):
                plan_capture_layout(kv, **args)
        layout = plan_capture_layout(kv, tp_size=4, pp_layer_ranges=[(0, 4)])
        registrar = Registrar()
        with self.assertRaisesRegex(ContractError, "different KV contract"):
            HostBufferPool(
                kv=msgspec.structs.replace(kv, storage_chunk_tokens=7),
                max_tokens=8,
                slots=1,
                max_bytes=2 << 20,
                registrar=registrar,
                pin_memory=False,
                partition=layout.partition("dp0-pp0-tp0"),
            )
        self.assertFalse(registrar.buffers)

    def test_exporter_reads_only_local_pp_layers_and_local_heads(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        kv = heterogeneous_kv()
        layout = plan_capture_layout(kv, tp_size=4, pp_layer_ranges=[(0, 2), (2, 4)])
        partition = layout.partition("dp0-pp1-tp2")
        sources = {
            "target_k.3": torch.arange(40).reshape(10, 1, 4).bfloat16(),
            "target_v.3": torch.arange(40, 80).reshape(10, 1, 4).bfloat16(),
        }
        pool = MagicMock(spec=MHATokenToKVPool)
        pool.is_quantized_kv_cache = pool.use_hnd = False
        pool.page_size, pool.start_layer, pool.layer_num = kv.source_page_size, 2, 2
        pool.get_key_buffer.side_effect = lambda layer: sources[f"target_k.{layer}"]
        pool.get_value_buffer.side_effect = lambda layer: sources[f"target_v.{layer}"]
        exporter = SelectedLayerKVExporter.from_pool(kv, pool, partition=partition)
        self.assertEqual(pool.get_key_buffer.call_args_list, [call(3)])
        self.assertEqual(pool.get_value_buffer.call_args_list, [call(3)])
        targets = {name: torch.empty(3, 1, 4, dtype=torch.bfloat16) for name in sources}
        indices = torch.tensor([7, 1, 4], dtype=torch.int32)
        expected = {name: value[indices].clone() for name, value in sources.items()}
        exporter.export(indices, targets, 0, 3)
        for name, source in sources.items():
            source.zero_()
            torch.testing.assert_close(targets[name], expected[name], rtol=0, atol=0)


class TestPartitionAssembly(CustomTestCase):
    def setUp(self):
        self.manifest, _ = make_partitioned_snapshot()
        self.metadata = SnapshotMetadata(
            **{
                name: getattr(self.manifest, name)
                for name in SnapshotMetadata.__struct_fields__
            }
        )
        self.layout = plan_capture_layout(
            self.manifest.kv, tp_size=2, pp_layer_ranges=[(0, 4)], aux_tp_rank=1
        )
        self.prepared = [
            PreparedSnapshotPartition(
                owner_id=owner,
                metadata_sha256=digest_bytes(canonical_bytes(self.metadata)),
                token_ids_sha256=next(
                    obj.sha256
                    for obj in self.manifest.objects
                    if obj.name == "token_ids"
                ),
                valid_kv_tokens=5,
                objects=tuple(
                    obj for obj in self.manifest.objects if obj.owner_id == owner
                ),
            )
            for owner in self.manifest.topology.owners
        ]

    def test_metadata_mismatches_missing_owners_and_validity_disagree(self):
        for parts in (
            self.prepared[:1],
            self.prepared + self.prepared[:1],
            [
                msgspec.structs.replace(self.prepared[0], metadata_sha256="0" * 64),
                self.prepared[1],
            ],
            [
                msgspec.structs.replace(self.prepared[0], valid_kv_tokens=6),
                self.prepared[1],
            ],
            [
                msgspec.structs.replace(part, valid_kv_tokens=6)
                for part in self.prepared
            ],
        ):
            with self.subTest(parts=parts), self.assertRaises(ContractError):
                assemble_snapshot(self.metadata, parts, layout=self.layout)

    def test_equal_lengths_cannot_hide_different_committed_tokens(self):
        for parts in (
            [
                msgspec.structs.replace(self.prepared[0], token_ids_sha256="0" * 64),
                self.prepared[1],
            ],
            [
                msgspec.structs.replace(part, token_ids_sha256="0" * 64)
                for part in self.prepared
            ],
        ):
            with (
                self.subTest(parts=parts),
                self.assertRaisesRegex(ContractError, "token sequence"),
            ):
                assemble_snapshot(self.metadata, parts, layout=self.layout)

    def test_full_coverage_cannot_hide_swapped_owner_head_identity(self):
        swapped = [
            msgspec.structs.replace(
                obj, head_range=(1 - obj.head_range[0], 2 - obj.head_range[0])
            )
            if obj.kind == "kv"
            else obj
            for obj in self.manifest.objects
        ]
        self.assertEqual(
            validate_manifest(msgspec.structs.replace(self.manifest, objects=swapped)),
            5,
        )
        parts = [
            msgspec.structs.replace(
                part,
                objects=tuple(obj for obj in swapped if obj.owner_id == part.owner_id),
            )
            for part in self.prepared
        ]
        with self.assertRaisesRegex(ContractError, "canonical owner"):
            assemble_snapshot(self.metadata, parts, layout=self.layout)


if __name__ == "__main__":
    unittest.main()
