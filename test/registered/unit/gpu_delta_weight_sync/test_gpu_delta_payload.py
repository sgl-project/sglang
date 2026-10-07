"""Outer CPU Zstd transport tests; no CUDA allocation or model imports."""

import unittest

import zstandard as zstd

from sglang.srt.weight_sync.gpu_delta.payload import (
    HostPayloadPool,
    validate_codec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestOuterZstd(unittest.TestCase):
    def test_manifest_codec_and_frame_geometry_admission(self):
        admitted = dict(codec="snappy-zstd", frame_bytes=1 << 20)
        for codec in ("snappy-zstd", "lz4-zstd", "lz4"):
            for size in (1 << 16, 192 << 10, 1 << 19, 1 << 20, 2 << 20, 4 << 20):
                with self.subTest(codec=codec, size=size):
                    validate_codec(admitted | {"codec": codec, "frame_bytes": size})
        for patch_value in (
            {"codec": "zstd"},
            {"codec": None},
            {"frame_bytes": 0},
            {"frame_bytes": -1},
            {"frame_bytes": (4 << 20) + 1},
            {"frame_bytes": 1.5},
            {"frame_bytes": True},
        ):
            with (
                self.subTest(patch=patch_value),
                self.assertRaisesRegex(ValueError, "codec"),
            ):
                validate_codec(admitted | patch_value)

    def test_gpu_outer_chunks_decode_directly_to_one_destination(self):
        values = [bytes(range(256)) * 4096, bytes(range(19))]
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        for known_size in (False, True):
            payload, chunks = bytearray(), []
            for index, value in enumerate(values):
                payload.extend(bytes((-len(payload)) % 16))
                compressed = zstd.ZstdCompressor(
                    level=1, write_content_size=known_size
                ).compress(value)
                chunks.append(
                    dict(
                        encoded_offset=len(payload),
                        encoded_bytes=len(compressed),
                        decoded_offset=index * (1 << 20),
                        decoded_bytes=len(value),
                    )
                )
                payload.extend(compressed)
            target = bytearray(sum(map(len, values)))
            pool.executor.submit(
                pool.decode_zstd, memoryview(payload), chunks, memoryview(target)
            ).result()
            self.assertEqual(target, b"".join(values))

    def test_invalid_frames_fail_without_publishing_a_decoded_result(self):
        raw = bytes(range(64)) * 128
        encoded = bytearray(zstd.ZstdCompressor(write_checksum=True).compress(raw))
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        chunk = dict(
            encoded_offset=0,
            encoded_bytes=len(encoded),
            decoded_offset=0,
            decoded_bytes=len(raw) + 1,
        )
        with self.assertRaisesRegex(ValueError, "truncated"):
            pool.executor.submit(
                pool.decode_zstd,
                memoryview(encoded),
                [chunk],
                memoryview(bytearray(len(raw) + 1)),
            ).result()
        chunk["decoded_bytes"] = len(raw)
        encoded[-1] ^= 1
        with self.assertRaises(zstd.ZstdError):
            pool.executor.submit(
                pool.decode_zstd,
                memoryview(encoded),
                [chunk],
                memoryview(bytearray(len(raw))),
            ).result()


if __name__ == "__main__":
    unittest.main()
