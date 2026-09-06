"""CPU tests for host-local weight materialization and recovery."""

import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import zstandard

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_module():
    if "local_checkpoint_under_test" in sys.modules:
        return sys.modules["local_checkpoint_under_test"]
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/weight_sync/local_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("local_checkpoint_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_checkpoint_under_test"] = module
    spec.loader.exec_module(module)
    return module


local_checkpoint = _load_module()


def _write_safetensors(path, tensors, metadata=None):
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for data in tensors.values():
            f.write(data)


def _adler32_hex(data) -> str:
    return f"{zlib.adler32(bytes(data), 1):08x}"


class _Publisher:
    SHARD = "model-00001-of-00001.safetensors"

    def __init__(self, root):
        self.base_dir = os.path.join(root, "base")
        self.source_dir = os.path.join(root, "published")
        os.makedirs(self.base_dir)
        os.makedirs(self.source_dir)
        rng = np.random.default_rng(7)
        self.state = {
            "layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes(),
            "layer.b": rng.integers(0, 256, 2048, dtype=np.uint8).tobytes(),
        }
        self.versions = {0: dict(self.state)}
        _write_safetensors(os.path.join(self.base_dir, self.SHARD), self.state)

    def publish_delta(self, version, changed):
        version_dir = os.path.join(self.source_dir, f"weight_v{version:06d}")
        os.makedirs(version_dir)
        payloads = {}
        checksums = {}
        for name, new in changed.items():
            old = self.state[name]
            diff = (
                np.frombuffer(new, dtype=np.uint8) ^ np.frombuffer(old, dtype=np.uint8)
            ).tobytes()
            payloads[name] = zstandard.ZstdCompressor().compress(diff)
            checksums[name] = _adler32_hex(new)
            self.state[name] = new
        self.versions[version] = dict(self.state)
        _write_safetensors(
            os.path.join(version_dir, self.SHARD), payloads, metadata=checksums
        )
        with open(os.path.join(version_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "metadata": {
                        "version": f"{version:06d}",
                        "base_version": f"{version - 1:06d}",
                        "delta_encoding": "xor",
                        "compression_format": "zstd",
                        "checksum_format": "adler32",
                    },
                    "weight_map": {name: self.SHARD for name in payloads},
                },
                f,
            )


def _read_local(local_dir):
    path = os.path.join(local_dir, _Publisher.SHARD)
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        body = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        out[name] = body[begin:end]
    return out


class TestPull(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.publisher = _Publisher(root)
        self.local_dir = os.path.join(root, "local")
        rng = np.random.default_rng(11)
        self.publisher.publish_delta(
            1, {"layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()}
        )
        self.publisher.publish_delta(
            2, {"layer.b": rng.integers(0, 256, 2048, dtype=np.uint8).tobytes()}
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _pull(self, target):
        local_checkpoint.pull(
            self.local_dir,
            self.publisher.base_dir,
            self.publisher.source_dir,
            target,
        )

    def _assert_at_version(self, version):
        self.assertEqual(_read_local(self.local_dir), self.publisher.versions[version])
        self.assertEqual(
            local_checkpoint._read_applied_version(self.local_dir), version
        )

    def test_seed_and_chain(self):
        self._pull(2)
        self._assert_at_version(2)

    def test_incremental_pulls_are_idempotent(self):
        self._pull(1)
        self._assert_at_version(1)
        self._pull(1)
        self._assert_at_version(1)
        self._pull(2)
        self._assert_at_version(2)

    def test_torn_apply_reseeds_instead_of_repatching(self):
        self._pull(1)
        shard = os.path.join(self.local_dir, _Publisher.SHARD)
        locations = local_checkpoint._tensor_locations(self.local_dir)
        _, offset, nbytes = locations["layer.b"]
        with open(shard, "r+b") as f:
            f.seek(offset)
            f.write(self.publisher.versions[2]["layer.b"][: nbytes // 2])
        self._pull(2)
        self._assert_at_version(2)

    def test_corrupt_local_state_recovers_via_reseed(self):
        self._pull(1)
        shard = os.path.join(self.local_dir, _Publisher.SHARD)
        locations = local_checkpoint._tensor_locations(self.local_dir)
        _, offset, _ = locations["layer.b"]
        with open(shard, "r+b") as f:
            f.seek(offset)
            f.write(bytes(16))
        self._pull(2)
        self._assert_at_version(2)

    def test_bad_published_delta_fails_loud(self):
        import shutil

        version_dir = os.path.join(self.publisher.source_dir, "weight_v000001")
        shard = os.path.join(version_dir, _Publisher.SHARD)
        with open(shard, "rb") as f:
            data = bytearray(f.read())
        data[-1] ^= 0xFF
        with open(shard, "wb") as f:
            f.write(bytes(data))
        with self.assertRaises((RuntimeError, zstandard.ZstdError)):
            self._pull(1)
        shutil.rmtree(version_dir)
        self.publisher.state = dict(self.publisher.versions[0])
        rng = np.random.default_rng(11)
        self.publisher.publish_delta(
            1, {"layer.a": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()}
        )
        self._pull(1)
        self._assert_at_version(1)

    def test_missing_source_version_fails_fast_without_reseed(self):
        self._pull(1)
        self._assert_at_version(1)
        reset_calls = []
        original_reset = local_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return original_reset(*args, **kwargs)

        local_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaises(FileNotFoundError):
                self._pull(3)
        finally:
            local_checkpoint._reset_checkpoint = original_reset
        self.assertEqual(reset_calls, [], "must not reseed on a missing source version")
        self._pull(2)
        self._assert_at_version(2)

    def test_incomplete_source_version_fails_fast_then_recovers(self):
        self._pull(2)
        self._assert_at_version(2)
        self.publisher.publish_delta(
            3,
            {
                "layer.a": np.random.default_rng(3)
                .integers(0, 256, 4096, dtype=np.uint8)
                .tobytes()
            },
        )
        shard = os.path.join(
            self.publisher.source_dir, "weight_v000003", _Publisher.SHARD
        )
        with open(shard, "rb") as f:
            blob = f.read()
        os.remove(shard)

        reset_calls = []
        original_reset = local_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return original_reset(*args, **kwargs)

        local_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaises(FileNotFoundError):
                self._pull(3)
        finally:
            local_checkpoint._reset_checkpoint = original_reset
        self.assertEqual(
            reset_calls, [], "must not reseed on an incomplete source version"
        )
        self._assert_at_version(2)

        with open(shard, "wb") as f:
            f.write(blob)
        self._pull(3)
        self._assert_at_version(3)

    def test_truncated_source_blob_fails_fast_then_recovers(self):
        self._pull(2)
        self._assert_at_version(2)
        self.publisher.publish_delta(
            3,
            {
                "layer.a": np.random.default_rng(5)
                .integers(0, 256, 4096, dtype=np.uint8)
                .tobytes()
            },
        )
        shard = os.path.join(
            self.publisher.source_dir, "weight_v000003", _Publisher.SHARD
        )
        with open(shard, "rb") as f:
            complete_blob = f.read()
        with open(shard, "wb") as f:
            f.write(complete_blob[:-256])

        reset_calls = []
        original_reset = local_checkpoint._reset_checkpoint

        def _spy(*args, **kwargs):
            reset_calls.append(args)
            return original_reset(*args, **kwargs)

        local_checkpoint._reset_checkpoint = _spy
        try:
            with self.assertRaises(FileNotFoundError):
                self._pull(3)
        finally:
            local_checkpoint._reset_checkpoint = original_reset
        self.assertEqual(
            reset_calls, [], "must not reseed on a truncated (not-ready) blob"
        )
        self._assert_at_version(2)

        with open(shard, "wb") as f:
            f.write(complete_blob)
        self._pull(3)
        self._assert_at_version(3)


if __name__ == "__main__":
    unittest.main()
