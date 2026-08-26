# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "expert_pack"
sys.path.insert(0, str(TOOLS))

from format import (  # noqa: E402
    ENTRY_STRUCT,
    FLAG_IDENTITY_PAYLOAD,
    FLAG_TRIPLET_OBJECTS,
    HEADER_STRUCT,
    IndexEntry,
    PackHeader,
    align_up,
    read_header,
    read_index,
)

from sglang.srt.layers.moe.expert_pack import (  # noqa: E402
    ExpertPackStore,
    _CacheSlot,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_pack(directory: Path) -> tuple[Path, Path, dict[str, str]]:
    layers, experts, top_k = 1, 2, 1
    role_bytes = 17
    object_stride = 4096
    index_count = layers * experts * 3
    data_start = align_up(HEADER_STRUCT.size + index_count * ENTRY_STRUCT.size, 4096)
    digests = {
        "model_identity": hashlib.sha256(b"model-identity").hexdigest(),
        "source": hashlib.sha256(b"source").hexdigest(),
        "config": hashlib.sha256(b"config").hexdigest(),
    }
    header = PackHeader(
        flags=FLAG_IDENTITY_PAYLOAD | FLAG_TRIPLET_OBJECTS,
        index_count=index_count,
        data_start=data_start,
        alignment=4096,
        num_layers=layers,
        num_experts=experts,
        top_k=top_k,
        role_count=3,
        model_identity_sha256=digests["model_identity"],
        source_blob_sha256=digests["source"],
        config_sha256=digests["config"],
    )
    entries = []
    payloads = []
    for expert in range(experts):
        object_offset = data_start + expert * object_stride
        generation = expert + 100
        for role_id, role in enumerate(("gate", "up", "down")):
            payload = bytes([expert * 3 + role_id]) * role_bytes
            payload_hash = hashlib.sha256(payload).hexdigest()
            entries.append(
                IndexEntry(
                    layer=0,
                    expert=expert,
                    role=role,
                    dtype_id=39,
                    dtype="MXFP4",
                    tensor_name=f"blk.0.ffn_{role}_exps.weight",
                    source_tensor_offset=0,
                    source_tensor_nbytes=role_bytes * experts,
                    source_slice_offset=expert * role_bytes,
                    source_slice_nbytes=role_bytes,
                    pack_offset=object_offset + role_id * role_bytes,
                    pack_nbytes=role_bytes,
                    source_tensor_sha256=payload_hash,
                    source_slice_sha256=payload_hash,
                    checksum=payload_hash,
                    shape=(32, 1),
                    quant_scheme="MXFP4",
                    transform_id="identity-v1",
                    block_size=32,
                    generation=generation,
                )
            )
            payloads.append((object_offset + role_id * role_bytes, payload))

    pack = directory / "runtime.expert-pack"
    with pack.open("w+b") as stream:
        stream.write(header.pack())
        for entry in entries:
            stream.write(entry.pack())
        stream.truncate(data_start + experts * object_stride)
        for offset, payload in payloads:
            stream.seek(offset)
            stream.write(payload)
    manifest = directory / "runtime.expert-pack.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "object_stride": object_stride,
                "pack_sha256": _sha256(pack),
            }
        ),
        encoding="utf-8",
    )
    return pack, manifest, digests


def _refresh_manifest_hash(pack: Path, manifest: Path) -> None:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["pack_sha256"] = _sha256(pack)
    manifest.write_text(json.dumps(value), encoding="utf-8")


class TestExpertPackRuntime(unittest.TestCase):
    def test_victim_prefers_low_frequency_then_lru(self):
        store = object.__new__(ExpertPackStore)
        keys = [(0, 0), (0, 1), (0, 2)]
        store._cache_slots = [
            _CacheSlot(key=keys[0], frequency=5),
            _CacheSlot(key=keys[1], frequency=1),
            _CacheSlot(key=keys[2], frequency=1),
        ]
        store._key_to_slot = {key: index for index, key in enumerate(keys)}
        store._lru = dict.fromkeys(keys)

        self.assertEqual(store._victim_slot(set()), 1)
        self.assertEqual(store._victim_slot({keys[1]}), 2)
        self.assertEqual(store._victim_slot(set(), preserve_oldest=True), 2)
        with self.assertRaisesRegex(RuntimeError, "active top-k"):
            store._victim_slot(set(keys))

    def test_zero_staging_slots_are_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            with self.assertRaisesRegex(ValueError, "staging budgets"):
                ExpertPackStore(
                    pack,
                    manifest_path=manifest,
                    expected_layers=1,
                    expected_experts=2,
                    expected_top_k=1,
                    expected_source_sha256=digests["source"],
                    expected_model_identity_sha256=digests["model_identity"],
                    expected_config_sha256=digests["config"],
                    cache_vram_mib=1,
                    stage_slots=0,
                )

    def test_full_pack_verification_is_opt_in(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            store = ExpertPackStore(
                pack,
                manifest_path=manifest,
                expected_layers=1,
                expected_experts=2,
                expected_top_k=1,
                expected_source_sha256=digests["source"],
                expected_model_identity_sha256=digests["model_identity"],
                expected_config_sha256=digests["config"],
                cache_vram_mib=1,
                stage_slots=1,
            )
            self.assertEqual(len(store.entries), 6)
            self.assertEqual(store.object_payload_bytes, 51)
            staging = torch.empty(51, dtype=torch.uint8)
            read_bytes, elapsed_ns = store._read_object(0, 1, staging)
            self.assertEqual(read_bytes, 51)
            self.assertGreaterEqual(elapsed_ns, 0)
            self.assertEqual(staging.tolist(), [3] * 17 + [4] * 17 + [5] * 17)
            store.close()

            with pack.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x01")

            store = ExpertPackStore(
                pack,
                manifest_path=manifest,
                expected_layers=1,
                expected_experts=2,
                expected_top_k=1,
                expected_source_sha256=digests["source"],
                expected_model_identity_sha256=digests["model_identity"],
                expected_config_sha256=digests["config"],
                cache_vram_mib=1,
                stage_slots=1,
            )
            store.close()

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ExpertPackStore(
                    pack,
                    manifest_path=manifest,
                    expected_layers=1,
                    expected_experts=2,
                    expected_top_k=1,
                    expected_source_sha256=digests["source"],
                    expected_model_identity_sha256=digests["model_identity"],
                    expected_config_sha256=digests["config"],
                    cache_vram_mib=1,
                    stage_slots=1,
                    verify_pack_sha256=True,
                )

    def test_split_read_ranges_reconstruct_exact_object(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            store = ExpertPackStore(
                pack,
                manifest_path=manifest,
                expected_layers=1,
                expected_experts=2,
                expected_top_k=1,
                expected_source_sha256=digests["source"],
                expected_model_identity_sha256=digests["model_identity"],
                expected_config_sha256=digests["config"],
                cache_vram_mib=1,
                stage_slots=1,
            )
            ranges = store._object_read_ranges()
            self.assertEqual(len(ranges), 4)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(sum(length for _, length in ranges), 51)
            self.assertTrue(all(length > 0 for _, length in ranges))
            self.assertTrue(
                all(
                    ranges[index][0] + ranges[index][1] == ranges[index + 1][0]
                    for index in range(len(ranges) - 1)
                )
            )

            staging = torch.empty(51, dtype=torch.uint8)
            for start, length in ranges:
                read_bytes, elapsed_ns = store._read_object_range(
                    0, 1, staging, start=start, length=length
                )
                self.assertEqual(read_bytes, length)
                self.assertGreaterEqual(elapsed_ns, 0)
            self.assertEqual(staging.tolist(), [3] * 17 + [4] * 17 + [5] * 17)
            store.close()

    def test_read_splits_follow_runtime_configuration(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            store = ExpertPackStore(
                pack,
                manifest_path=manifest,
                expected_layers=1,
                expected_experts=2,
                expected_top_k=1,
                expected_source_sha256=digests["source"],
                expected_model_identity_sha256=digests["model_identity"],
                expected_config_sha256=digests["config"],
                cache_vram_mib=1,
                stage_slots=1,
                read_splits=2,
                stats_flush_interval=7,
            )
            ranges = store._object_read_ranges()
            self.assertEqual(store.stats["read_splits"], 2)
            self.assertEqual(store.stats_flush_interval, 7)
            self.assertEqual(len(ranges), 2)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(sum(length for _, length in ranges), 51)
            store.close()

    def test_short_object_read_raises(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            store = ExpertPackStore(
                pack,
                manifest_path=manifest,
                expected_layers=1,
                expected_experts=2,
                expected_top_k=1,
                expected_source_sha256=digests["source"],
                expected_model_identity_sha256=digests["model_identity"],
                expected_config_sha256=digests["config"],
                cache_vram_mib=1,
                stage_slots=1,
            )
            second_object = store.object_offsets[(0, 1)]
            with pack.open("r+b", buffering=0) as stream:
                stream.truncate(second_object + 10)
            staging = torch.empty(store.object_payload_bytes, dtype=torch.uint8)
            with self.assertRaisesRegex(OSError, "short expert-pack read"):
                store._read_object(0, 1, staging)
            store.close()

    def test_duplicate_entry_is_rejected_even_with_valid_pack_hash(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            with pack.open("r+b", buffering=0) as stream:
                header = read_header(stream)
                entries = read_index(stream, header)
                duplicate = replace(
                    entries[1],
                    layer=entries[0].layer,
                    expert=entries[0].expert,
                    role=entries[0].role,
                )
                stream.seek(HEADER_STRUCT.size + ENTRY_STRUCT.size)
                stream.write(duplicate.pack())
            _refresh_manifest_hash(pack, manifest)
            with self.assertRaisesRegex(ValueError, "duplicate expert-pack entry"):
                ExpertPackStore(
                    pack,
                    manifest_path=manifest,
                    expected_layers=1,
                    expected_experts=2,
                    expected_top_k=1,
                    expected_source_sha256=digests["source"],
                    expected_model_identity_sha256=digests["model_identity"],
                    expected_config_sha256=digests["config"],
                    cache_vram_mib=1,
                    stage_slots=1,
                )

    def test_out_of_range_entry_is_rejected_even_with_valid_pack_hash(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            with pack.open("r+b", buffering=0) as stream:
                header = read_header(stream)
                entries = read_index(stream, header)
                invalid = replace(
                    entries[0], pack_offset=pack.stat().st_size + header.alignment
                )
                stream.seek(HEADER_STRUCT.size)
                stream.write(invalid.pack())
            _refresh_manifest_hash(pack, manifest)
            with self.assertRaisesRegex(ValueError, "object layout mismatch"):
                ExpertPackStore(
                    pack,
                    manifest_path=manifest,
                    expected_layers=1,
                    expected_experts=2,
                    expected_top_k=1,
                    expected_source_sha256=digests["source"],
                    expected_model_identity_sha256=digests["model_identity"],
                    expected_config_sha256=digests["config"],
                    cache_vram_mib=1,
                    stage_slots=1,
                )

    def test_source_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            with self.assertRaisesRegex(ValueError, "source_blob_sha256"):
                ExpertPackStore(
                    pack,
                    manifest_path=manifest,
                    expected_layers=1,
                    expected_experts=2,
                    expected_top_k=1,
                    expected_source_sha256="0" * 64,
                    expected_model_identity_sha256=digests["model_identity"],
                    expected_config_sha256=digests["config"],
                    cache_vram_mib=1,
                    stage_slots=1,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA pinned memory")
    def test_direct_io_rejects_unaligned_object_ranges(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            store = ExpertPackStore(
                pack,
                manifest_path=manifest,
                expected_layers=1,
                expected_experts=2,
                expected_top_k=1,
                expected_source_sha256=digests["source"],
                expected_model_identity_sha256=digests["model_identity"],
                expected_config_sha256=digests["config"],
                cache_vram_mib=1,
                cache_vram_reserve_mib=1,
                stage_slots=1,
                direct_io=True,
            )
            with self.assertRaisesRegex(ValueError, "aligned read ranges"):
                store.initialize_device_cache("cuda")
            store.close()

    def test_close_flushes_stats_atomically(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pack, manifest, digests = _make_pack(root)
            stats_path = root / "stats.json"
            store = ExpertPackStore(
                pack,
                manifest_path=manifest,
                expected_layers=1,
                expected_experts=2,
                expected_top_k=1,
                expected_source_sha256=digests["source"],
                expected_model_identity_sha256=digests["model_identity"],
                expected_config_sha256=digests["config"],
                cache_vram_mib=1,
                stage_slots=1,
                stats_path=stats_path,
            )
            store._route_calls_by_layer[0] = 2
            store._route_tokens_by_layer[0] = 3
            store.stats["pack_reads"] = 4
            store.stats["pack_read_bytes"] = 204
            store.close()
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["pack_reads"], 4)
            self.assertEqual(stats["pack_read_bytes"], 204)
            self.assertEqual(stats["route_calls_by_layer"], [2])
            self.assertEqual(stats["route_tokens_by_layer"], [3])
            self.assertFalse(list(root.glob("stats.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
