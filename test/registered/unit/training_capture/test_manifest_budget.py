"""Admission sizing against independently built complete snapshot manifests."""

import itertools
import random
import unittest
from unittest.mock import patch

import msgspec
import torch
from sglang.srt.training_capture.protocol import (
    DTYPES,
    SequenceInfo,
    aux_specs,
    canonical_bytes,
)
from sglang.srt.training_capture.snapshot import (
    SnapshotMetadata,
    assemble_snapshot,
    manifest_size_bound,
    prepare_snapshot_partition,
)
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import make_snapshot

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestManifestBudget(CustomTestCase):
    def setUp(self):
        base, _ = make_snapshot()
        self.metadata = SnapshotMetadata(
            **{name: getattr(base, name) for name in SnapshotMetadata.__struct_fields__}
        )

    @staticmethod
    def actual_manifest(metadata, layout, valid):
        n, r = metadata.sequence.total_length, metadata.sequence.response_length
        parts = []
        for partition in layout.partitions:
            if not partition.active:
                continue
            buffers = (
                {
                    name: torch.zeros(shape, dtype=DTYPES[dtype])
                    for name, (dtype, shape) in aux_specs(n, r).items()
                }
                if partition.include_aux
                else {}
            )
            for layer in partition.local_layers(metadata.kv):
                for component, dim in (
                    ("k", layer.key_head_dim),
                    ("v", layer.value_head_dim),
                ):
                    buffers[f"target_{component}.{layer.layer_id}"] = torch.zeros(
                        valid, layer.num_kv_heads, dim, dtype=DTYPES[metadata.kv.dtype]
                    )
            prepared, _ = prepare_snapshot_partition(
                metadata,
                buffers,
                valid_kv_tokens=valid,
                partition=partition,
                token_ids=[0] * n,
            )
            parts.append(prepared)
        return assemble_snapshot(metadata, parts, layout=layout)

    def test_bound_covers_full_partial_and_shortened_sharded_manifests(self):
        rng = random.Random(483)
        # Include digit-width boundaries and TP replicas, plus an aux-only PP stage.
        for n, tp in itertools.product((2, 9, 10, 99, 100, 101, 1001), (1, 2, 4)):
            kv = msgspec.structs.replace(
                self.metadata.kv,
                dtype="float16" if n % 2 else "bfloat16",
                codec=(
                    "dense_fp16_post_rope_v1" if n % 2 else "dense_bf16_post_rope_v1"
                ),
                storage_chunk_tokens=rng.choice((1, 7, 64, 1000)),
                layers=[
                    msgspec.structs.replace(
                        layer,
                        num_kv_heads=8 if layer.layer_id == 1 else 2,
                        key_head_dim=5,
                        value_head_dim=7,
                    )
                    for layer in self.metadata.kv.layers
                ],
            )
            layout = plan_capture_layout(
                kv, tp_size=tp, pp_layer_ranges=[(0, 2), (2, 4), (4, 6)]
            )
            prompt = max(1, n // 3)
            metadata = msgspec.structs.replace(
                self.metadata,
                sample_id="s" * 160,
                generation_id="g" * 160,
                kv=kv,
                topology=layout.topology,
                sequence=SequenceInfo(
                    prompt_length=prompt,
                    response_length=n - prompt,
                    total_length=n,
                    stop_reason="length",
                ),
                provenance=msgspec.structs.replace(
                    self.metadata.provenance,
                    sampling_config={"stop": ['\u4e2d"\\\n' * 31]},
                ),
            )
            bound = manifest_size_bound(metadata, layout=layout)
            for actual_n in {n, prompt + 1}:
                for tail, reason in itertools.product(
                    (0, 1), ("length", "eos", "stop_token", "stop_string")
                ):
                    actual_metadata = msgspec.structs.replace(
                        metadata,
                        sequence=msgspec.structs.replace(
                            metadata.sequence,
                            response_length=actual_n - prompt,
                            total_length=actual_n,
                            stop_reason=reason,
                        ),
                    )
                    with self.subTest(n=n, tp=tp, actual=actual_n, tail=tail):
                        actual = self.actual_manifest(
                            actual_metadata, layout, actual_n - tail
                        )
                        self.assertLessEqual(len(canonical_bytes(actual)), bound)

    def test_bound_work_is_independent_of_token_chunk_count(self):
        kv = msgspec.structs.replace(self.metadata.kv, storage_chunk_tokens=1)
        layout = plan_capture_layout(kv, tp_size=2, pp_layer_ranges=[(0, 4)])
        metadata = msgspec.structs.replace(
            self.metadata,
            kv=kv,
            topology=layout.topology,
            sequence=SequenceInfo(
                prompt_length=1,
                response_length=2147483646,
                total_length=2147483647,
                stop_reason="length",
            ),
        )
        with (
            patch(
                "sglang.srt.training_capture.snapshot.canonical_bytes",
                wraps=canonical_bytes,
            ) as encode,
            patch("torch.empty", side_effect=AssertionError("payload allocation")),
        ):
            self.assertGreater(manifest_size_bound(metadata, layout=layout), 1 << 40)
        self.assertLessEqual(encode.call_count, 32)


if __name__ == "__main__":
    unittest.main()
