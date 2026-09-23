import json
import os
import shutil
import struct
import tempfile
import unittest
import zlib

import numpy as np
import zstandard

from sglang.srt.weight_sync import local_checkpoint
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _adler32(data: bytes) -> str:
    return f"{zlib.adler32(data, 1):08x}"


def _write_safetensors(path: str, tensors: dict, metadata: dict = None) -> None:
    header, blobs, offset = {}, [], 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def _read_tensors(ckpt_dir: str) -> dict:
    out = {}
    for name, (path, offset, nbytes) in local_checkpoint._tensor_locations(
        ckpt_dir
    ).items():
        with open(path, "rb") as f:
            f.seek(offset)
            out[name] = f.read(nbytes)
    return out


def _write_full_version(version_dir: str, tensors: dict) -> None:
    os.makedirs(version_dir)
    _write_safetensors(os.path.join(version_dir, "model.safetensors"), tensors)


def _write_delta_version(
    version_dir: str, version: int, old: dict, new: dict, encoding: str
) -> None:
    """Publish `new` as a delta against `old` the way the trainer side does."""
    os.makedirs(version_dir)
    compressor = zstandard.ZstdCompressor()
    payloads, checksums = {}, {}
    for name in new:
        o = np.frombuffer(old[name], dtype=np.uint8)
        n = np.frombuffer(new[name], dtype=np.uint8)
        if encoding == "xor":
            raw = (o ^ n).tobytes()
        else:
            positions = np.nonzero(o != n)[0].astype("<u4")
            raw = (
                len(positions).to_bytes(4, "little")
                + positions.tobytes()
                + n[positions].tobytes()
            )
        payloads[name] = compressor.compress(raw)
        checksums[name] = _adler32(new[name])
    _write_safetensors(
        os.path.join(version_dir, "model.safetensors"), payloads, checksums
    )
    with open(os.path.join(version_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {
                "metadata": {
                    "delta_encoding": encoding,
                    "version": str(version),
                    "base_version": str(version - 1),
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {name: "model.safetensors" for name in new},
            },
            f,
        )


class TestLocalCheckpointPull(unittest.TestCase):
    V0 = {"a": bytes(range(16)), "b": b"\x10" * 8}
    V1 = {"a": bytes(range(16, 32)), "b": b"\x10" * 8}
    V2 = {"a": bytes(range(16, 32)), "b": b"\x10\x11\x10\x10\x10\x10\x22\x10"}

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.base = os.path.join(self.root, "base")
        self.source = os.path.join(self.root, "published")
        self.local = os.path.join(self.root, "local")
        os.makedirs(self.base)
        os.makedirs(self.source)
        _write_safetensors(os.path.join(self.base, "model.safetensors"), self.V0)
        with open(os.path.join(self.base, "config.json"), "w") as f:
            f.write("{}")

    def _vdir(self, version: int) -> str:
        return local_checkpoint._version_dir(self.source, version)

    def _pull(self, target: int) -> None:
        local_checkpoint.pull(
            local_checkpoint_dir=self.local,
            base_dir=self.base,
            source_dir=self.source,
            target_version=target,
        )

    def test_seeds_from_base_then_applies_xor_delta(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")

        self._pull(1)

        self.assertEqual(_read_tensors(self.local), self.V1)
        self.assertTrue(os.path.exists(os.path.join(self.local, "config.json")))
        self.assertEqual(local_checkpoint._read_applied_version(self.local), 1)

    def test_chains_overwrite_delta_and_repeated_pull_is_a_noop(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        _write_delta_version(self._vdir(2), 2, self.V1, self.V2, "overwrite")

        self._pull(1)
        self._pull(2)
        self._pull(2)

        self.assertEqual(_read_tensors(self.local), self.V2)
        self.assertEqual(local_checkpoint._read_applied_version(self.local), 2)

    def test_fresh_host_seeds_from_newest_full_version_and_prunes_stale_files(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        _write_full_version(self._vdir(2), self.V2)
        os.makedirs(self.local)
        with open(os.path.join(self.local, "stale.safetensors"), "w") as f:
            f.write("x")

        self._pull(2)

        self.assertEqual(_read_tensors(self.local), self.V2)
        self.assertFalse(os.path.exists(os.path.join(self.local, "stale.safetensors")))
        # seeded from v2 directly, so base-only files never came along
        self.assertFalse(os.path.exists(os.path.join(self.local, "config.json")))

    def test_checksum_mismatch_raises_and_does_not_advance_the_version(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        # publish a v1 whose payload was computed against the wrong base
        wrong_base = {"a": bytes(16), "b": self.V0["b"]}
        shutil.rmtree(self._vdir(1))
        _write_delta_version(self._vdir(1), 1, wrong_base, self.V1, "xor")

        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self._pull(1)
        self.assertEqual(local_checkpoint._read_applied_version(self.local), 0)

    def test_out_of_order_delta_raises(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        _write_delta_version(self._vdir(2), 2, self.V1, self.V2, "xor")
        self._pull(1)
        # v2 now claims to build on a version the host never applied
        index = os.path.join(self._vdir(2), "model.safetensors.index.json")
        with open(index) as f:
            meta = json.load(f)
        meta["metadata"]["base_version"] = "5"
        with open(index, "w") as f:
            json.dump(meta, f)

        with self.assertRaisesRegex(RuntimeError, "out-of-order delta"):
            self._pull(2)


if __name__ == "__main__":
    unittest.main()
