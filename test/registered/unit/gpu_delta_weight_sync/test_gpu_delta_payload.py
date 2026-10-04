"""Outer CPU Zstd transport tests; no CUDA allocation or model imports."""

import copy
import unittest
from unittest.mock import patch

import zstandard as zstd

from sglang.srt.weight_sync.gpu_delta_payload import (
    OuterZstdPool,
    configured_codec,
    validate_codec,
    validate_outer_entries,
    validate_zstd_frame,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def entry(payload, name="weight", size=36):
    return {
        "name": name,
        "nbytes": (1 << 16) + 16,
        "encoding": "xor_bytes",
        "outer": {
            "file": "owner.bin",
            "encoded_offset": 0,
            "encoded_bytes": len(payload),
            "decoded_bytes": size,
            "frames": [
                dict(
                    encoded_offset=0,
                    encoded_bytes=len(payload),
                    decoded_offset=0,
                    decoded_bytes=size,
                )
            ],
        },
        "frames": [
            {
                "encoded_offset": 0,
                "encoded_bytes": 8,
                "decoded_offset": 0,
                "decoded_bytes": 1 << 16,
            },
            {
                "encoded_offset": 16,
                "encoded_bytes": 20,
                "decoded_offset": 1 << 16,
                "decoded_bytes": 16,
            },
        ],
    }


class TestOuterZstd(unittest.TestCase):
    def test_codec_is_admitted_once_and_rejects_legacy_profiles(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(configured_codec(), "snappy-zstd")
        admitted = dict(protocol_version=4, codec="snappy-zstd", frame_bytes=1 << 20)
        for size in (1 << 16, 1 << 20):
            validate_codec(admitted | {"frame_bytes": size}, "snappy-zstd")
        for patch_value in (
            {"protocol_version": 2},
            {"protocol_version": 3},
            {"protocol_version": 4.0},
            {"codec": "zstd"},
            {"codec": None},
            {"codec_profile": "snappy-independent-1mib-gpu-zstd-v1"},
            {"frame_bytes": 1 << 21},
            {"frame_bytes": True},
        ):
            with (
                self.subTest(patch=patch_value),
                self.assertRaisesRegex(ValueError, "codec"),
            ):
                validate_codec(admitted | patch_value, "snappy-zstd")
        # A later environment change cannot override a frozen admitted contract.
        with patch.dict("os.environ", {"WEIGHT_DELTA_CODEC": "invalid"}):
            validate_codec(admitted, "snappy-zstd")

    def test_gpu_outer_chunks_decode_directly_to_one_destination(self):
        values = [bytes(range(256)) * 4096, bytes(range(19))]
        pool = OuterZstdPool(2)
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
                pool.decode, memoryview(payload), chunks, memoryview(target)
            ).result()
            self.assertEqual(target, b"".join(values))

    def test_exact_frame_extent_rejects_every_truncation_and_trailing_data(self):
        raw = bytes(range(64)) * 128
        for checksum in (False, True):
            encoded = zstd.ZstdCompressor(level=1, write_checksum=checksum).compress(
                raw
            )
            validate_zstd_frame(encoded, len(raw))
            for stop in range(len(encoded)):
                with (
                    self.subTest(checksum=checksum, stop=stop),
                    self.assertRaises((ValueError, zstd.ZstdError)),
                ):
                    validate_zstd_frame(encoded[:stop], len(raw))
            for suffix in (b"\0", encoded):
                with self.assertRaisesRegex(ValueError, "trailing or truncated"):
                    validate_zstd_frame(encoded + suffix, len(raw))

    def test_invalid_frames_fail_without_publishing_a_decoded_result(self):
        raw = bytes(range(64)) * 128
        encoded = bytearray(zstd.ZstdCompressor(write_checksum=True).compress(raw))
        pool = OuterZstdPool(2)
        self.addCleanup(pool.close)
        chunk = dict(
            encoded_offset=0,
            encoded_bytes=len(encoded),
            decoded_offset=0,
            decoded_bytes=len(raw) + 1,
        )
        with self.assertRaisesRegex(ValueError, "content size"):
            pool.executor.submit(
                pool.decode,
                memoryview(encoded),
                [chunk],
                memoryview(bytearray(len(raw) + 1)),
            ).result()
        chunk["decoded_bytes"] = len(raw)
        encoded[-1] ^= 1
        with self.assertRaises(zstd.ZstdError):
            pool.executor.submit(
                pool.decode,
                memoryview(encoded),
                [chunk],
                memoryview(bytearray(len(raw))),
            ).result()

    def test_relative_span_accepts_expanded_snappy_without_raw_fallback(self):
        record = entry(bytes(50))
        validate_outer_entries([record], {"owner.bin": 50}, 1 << 16)
        mutations = (
            lambda r: r["outer"].update(decoded_bytes=37),
            lambda r: r["outer"].update(encoded_bytes=51),
            lambda r: r["outer"].update(codec="zstd"),
            lambda r: r["frames"][0].update(encoded_offset=16),
            lambda r: r["frames"][1].update(encoded_offset=15),
            lambda r: r["frames"][1].update(codec="snappy"),
            lambda r: r["frames"][1].update(file="owner.bin"),
            lambda r: r["frames"][1].update(encoded_bytes=51),
            lambda r: r["frames"][0].update(decoded_bytes=8),
            lambda r: r["frames"][1].update(decoded_offset=1),
        )
        for mutate in mutations:
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_outer_entries([bad], {"owner.bin": 50}, 1 << 16)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_outer_entries(
                [record, copy.deepcopy(record)], {"owner.bin": 50}, 1 << 16
            )
        empty = {"encoding": "xor_bytes", "frames": []}
        validate_outer_entries([empty], {"owner.bin": 50}, 1 << 16)
        with self.assertRaises(ValueError):
            validate_outer_entries(
                [empty | {"encoding": "replace_bytes"}], {"owner.bin": 50}, 1 << 16
            )
        with self.assertRaisesRegex(ValueError, "must omit"):
            validate_outer_entries(
                [empty | {"outer": None}], {"owner.bin": 50}, 1 << 16
            )

    def test_outer_chunks_exactly_cover_natural_tensor(self):
        record = entry(bytes(50))
        validate_outer_entries([record], {"owner.bin": 50}, 1 << 16)
        mutations = (
            lambda r: r["outer"].pop("frames"),
            lambda r: r["outer"].update(frames=[]),
            lambda r: r["outer"]["frames"][0].update(encoded_offset=16),
            lambda r: r["outer"]["frames"][0].update(encoded_bytes=49),
            lambda r: r["outer"]["frames"][0].update(encoded_bytes=True),
            lambda r: r["outer"]["frames"][0].update(decoded_offset=1),
            lambda r: r["outer"]["frames"][0].update(decoded_bytes=35),
            lambda r: r["outer"]["frames"].append(dict(r["outer"]["frames"][0])),
        )
        for mutate in mutations:
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_outer_entries([bad], {"owner.bin": 50}, 1 << 16)
        # Independent outer chunks cap each decoder output, even when Snappy
        # expansion makes the reconstructed tensor arena exceed one MiB.
        record["nbytes"] = 1 << 20
        record["frames"] = [
            dict(
                encoded_offset=0,
                encoded_bytes=(1 << 20) + 16,
                decoded_offset=0,
                decoded_bytes=1 << 20,
            )
        ]
        record["outer"].update(
            decoded_bytes=(1 << 20) + 16,
            frames=[
                dict(
                    encoded_offset=0,
                    encoded_bytes=8,
                    decoded_offset=0,
                    decoded_bytes=1 << 20,
                ),
                dict(
                    encoded_offset=16,
                    encoded_bytes=34,
                    decoded_offset=1 << 20,
                    decoded_bytes=16,
                ),
            ],
        )
        validate_outer_entries([record], {"owner.bin": 50}, 1 << 20)
        record["outer"]["frames"][0]["decoded_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "chunk range"):
            validate_outer_entries([record], {"owner.bin": 50}, 1 << 20)


class TestRawPayload(unittest.TestCase):
    def test_complete_raw_ranges_and_unchanged_omission(self):
        raw = {
            "encoding": "raw_bytes",
            "nbytes": 4,
            "changed_bytes": 1,
            "frames": [],
            "raw": {"file": "owner.bin", "encoded_offset": 0, "encoded_bytes": 4},
        }
        unchanged = {
            "encoding": "raw_bytes",
            "nbytes": 4,
            "changed_bytes": 0,
            "frames": [],
        }
        validate_outer_entries([raw, unchanged], {"owner.bin": 8}, 1 << 16)
        for mutate in (
            lambda e: e.update(outer={}),
            lambda e: e.update(frames=[{}]),
            lambda e: e.update(changed_bytes=-1),
            lambda e: e.update(changed_bytes=5),
            lambda e: e.update(changed_bytes=True),
            lambda e: e.update(changed_bytes=0),
            lambda e: e.pop("raw"),
            lambda e: e["raw"].update(encoded_bytes=3),
            lambda e: e["raw"].update(encoded_offset=5),
            lambda e: e["raw"].update(codec="none"),
        ):
            bad = copy.deepcopy(raw)
            mutate(bad)
            with (
                self.subTest(mutate=mutate),
                self.assertRaises(ValueError),
            ):
                validate_outer_entries([bad], {"owner.bin": 8}, 1 << 16)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_outer_entries([raw, copy.deepcopy(raw)], {"owner.bin": 8}, 1 << 16)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_outer_entries([raw, entry(bytes(8))], {"owner.bin": 32}, 1 << 16)


if __name__ == "__main__":
    unittest.main()
