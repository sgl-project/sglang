# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from tools.expert_pack.kimi_ggml import (
    PACK_ALIGNMENT,
    PACK_ENTRY,
    PACK_HEADER,
    PACK_MAGIC,
    KimiK3Spec,
    TensorRecord,
    validate_ggml_moe_pack,
    write_ggml_moe_pack,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _spec() -> KimiK3Spec:
    return KimiK3Spec(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        num_experts=2,
        top_k=1,
        num_shared_experts=1,
        hidden_size=8,
        routed_expert_hidden_size=4,
        moe_intermediate_size=3,
        hidden_act="situ",
        active_moe_layer_ids=(1,),
    )


def _tensor_records(source: Path) -> dict[tuple[int, str], TensorRecord]:
    result = {}
    for index, role in enumerate(("up", "gate", "down")):
        result[(1, role)] = TensorRecord(
            name=f"blk.1.ffn_{role}_exps.weight",
            shape=(4, 3, 2) if role != "down" else (3, 4, 2),
            dtype="Q2_K" if role != "down" else "Q3_K",
            dtype_id=10 if role != "down" else 11,
            shard_index=0,
            shard_path=str(source),
            data_offset=index * 2 * PACK_ALIGNMENT,
            nbytes=2 * PACK_ALIGNMENT,
        )
    return result


def _write_pack(path: Path, source: Path, *, bad_expert: bool = False) -> None:
    tensors = _tensor_records(source)
    entries = []
    data_start = PACK_ALIGNMENT
    offset = data_start
    payloads = []
    for expert in range(2):
        for role in ("up", "gate", "down"):
            tensor = tensors[(1, role)]
            payload_offset = tensor.data_offset + expert * PACK_ALIGNMENT
            payload = source.read_bytes()[
                payload_offset : payload_offset + PACK_ALIGNMENT
            ]
            name = tensor.name.encode("utf-8").ljust(128, b"\0")
            entries.append(
                PACK_ENTRY.pack(
                    name,
                    7 if bad_expert and len(entries) == 0 else expert,
                    0,
                    offset,
                    len(payload),
                )
            )
            payloads.append(payload)
            offset += len(payload)
    with path.open("wb") as stream:
        stream.write(
            PACK_HEADER.pack(PACK_MAGIC, 1, PACK_HEADER.size, len(entries), data_start)
        )
        stream.write(b"".join(entries))
        stream.write(bytes(data_start - stream.tell()))
        stream.write(b"".join(payloads))


class TestKimiGGMLPack(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_validates_expert_major_pack_and_payload_samples(self):
        source = self.root / "source.gguf"
        source.write_bytes(
            b"".join(bytes([value]) * PACK_ALIGNMENT for value in range(6))
        )
        pack = self.root / "experts.pack"
        _write_pack(pack, source)

        result = validate_ggml_moe_pack(
            pack, _tensor_records(source), _spec(), payload_samples=6
        )

        self.assertEqual(result["index_count"], 6)
        self.assertEqual(result["physical_role_order"], ["up", "gate", "down"])
        self.assertEqual(result["object_bytes"], 3 * PACK_ALIGNMENT)
        self.assertEqual(result["payload_samples_verified"], 6)
        self.assertEqual(result["roles"]["down"]["dtype"], "Q3_K")

    def test_builds_expert_major_pack_from_source_tensor_slices(self):
        source = self.root / "source.gguf"
        source.write_bytes(
            b"".join(bytes([value]) * PACK_ALIGNMENT for value in range(6))
        )
        pack = self.root / "experts.pack"

        size = write_ggml_moe_pack(pack, _tensor_records(source), _spec())
        result = validate_ggml_moe_pack(
            pack, _tensor_records(source), _spec(), payload_samples=6
        )

        self.assertEqual(pack.stat().st_size, size)
        self.assertEqual(result["payload_samples_verified"], 6)
        self.assertFalse(pack.with_name(pack.name + ".partial").exists())

    def test_rejects_pack_that_changes_expert_identity(self):
        source = self.root / "source.gguf"
        source.write_bytes(bytes(6 * PACK_ALIGNMENT))
        pack = self.root / "experts.pack"
        _write_pack(pack, source, bad_expert=True)

        with self.assertRaisesRegex(ValueError, "complete expert-major"):
            validate_ggml_moe_pack(
                pack, _tensor_records(source), _spec(), payload_samples=0
            )

    def test_rejects_unaligned_pack_offset(self):
        source = self.root / "source.gguf"
        source.write_bytes(bytes(6 * PACK_ALIGNMENT))
        pack = self.root / "experts.pack"
        _write_pack(pack, source)
        with pack.open("r+b") as stream:
            offset_field = PACK_HEADER.size + 128 + 4 + 4
            stream.seek(offset_field)
            stream.write(struct.pack("<Q", PACK_ALIGNMENT + 1))

        with self.assertRaisesRegex(ValueError, "not 4 KiB aligned"):
            validate_ggml_moe_pack(
                pack, _tensor_records(source), _spec(), payload_samples=0
            )


if __name__ == "__main__":
    unittest.main()
