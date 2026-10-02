"""Outer CPU Zstd transport tests; no CUDA allocation or model imports."""

import copy
import unittest

import zstandard as zstd

from sglang.srt.weight_sync.gpu_delta_payload import (
    OuterZstdReader,
    outer_zstd_profile,
    validate_outer_entries,
    validate_plain_entries,
    validate_zstd_frame,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def entry(payload, *, name="weight", size=36):
    return {
        "name": name,
        "nbytes": 24,
        "encoding": "xor_bytes",
        "outer": {
            "codec": "zstd",
            "file": "owner.bin",
            "encoded_offset": 0,
            "encoded_bytes": len(payload),
            "decoded_bytes": size,
        },
        "frames": [
            {
                "file": "owner.bin",
                "encoded_offset": 0,
                "encoded_bytes": 8,
                "decoded_offset": 0,
                "decoded_bytes": 8,
                "codec": "none",
            },
            {
                "file": "owner.bin",
                "encoded_offset": 16,
                "encoded_bytes": 20,
                "decoded_offset": 8,
                "decoded_bytes": 16,
                "codec": "snappy",
            },
        ],
    }


class TestOuterZstd(unittest.TestCase):
    def test_snappy_requires_envelope_and_zstd_retains_protocol_two(self):
        for frame in ("64kib", "1mib", "2mib"):
            with self.subTest(frame=frame):
                self.assertFalse(
                    outer_zstd_profile(
                        {
                            "protocol_version": 2,
                            "codec_profile": f"zstd-independent-{frame}-v1",
                        }
                    )
                )
                # CPU and GPU producers share this manifest contract. The
                # receiver never selects admission using producer provenance.
                self.assertEqual(
                    outer_zstd_profile(
                        {
                            "protocol_version": 3,
                            "codec_profile": f"snappy-independent-{frame}-zstd-v1",
                        }
                    ),
                    "cpu",
                )
                self.assertEqual(
                    outer_zstd_profile(
                        {
                            "protocol_version": 4,
                            "codec_profile": f"snappy-independent-{frame}-gpu-zstd-v1",
                        }
                    ),
                    "gpu",
                )
        for version, profile in (
            (2, "snappy-independent-1mib-v1"),
            (2, "snappy-independent-64kib-v1"),
            (2, "snappy-independent-1mib-zstd-v1"),
            (3, "snappy-independent-1mib-v1"),
            (3, "zstd-independent-1mib-zstd-v1"),
            (3, "zstd-independent-1mib-v1"),
            (4, "snappy-independent-1mib-zstd-v1"),
            (3, "snappy-independent-1mib-gpu-zstd-v1"),
            (4.0, "snappy-independent-1mib-gpu-zstd-v1"),
            (2.0, "zstd-independent-1mib-v1"),
            (3.0, "snappy-independent-1mib-zstd-v1"),
            (2, "zstd-unknown"),
            (2, "none-independent-1mib-v1"),
        ):
            with (
                self.subTest(version=version, profile=profile),
                self.assertRaises(ValueError),
            ):
                outer_zstd_profile(
                    {"protocol_version": version, "codec_profile": profile}
                )

    def test_reconstructed_bytes_are_written_once_into_final_storage(self):
        raw = bytes(range(256)) * 2048 + bytes(17)
        allocations, metrics = [], {}

        def allocate(size):
            target = bytearray(size)
            allocations.append(target)
            return target, memoryview(target)

        encoded = zstd.ZstdCompressor(level=1, write_content_size=True).compress(raw)
        reader = OuterZstdReader({"owner.bin": encoded}, allocate, metrics)
        record = entry(encoded, size=len(raw))
        output = reader.get(record)
        self.assertEqual(output, raw)
        self.assertIs(output, allocations[0])
        self.assertIs(output, reader.get(record))
        self.assertEqual(len(allocations), 1)
        self.assertEqual(metrics["host_outer_zstd_encoded_bytes"], len(encoded))
        self.assertEqual(metrics["host_outer_zstd_decoded_bytes"], len(raw))
        self.assertEqual(metrics["host_outer_zstd_tensors"], 1)

    def test_gpu_outer_chunks_decode_directly_to_one_pinned_tensor_arena(self):
        values = [bytes(range(256)) * 4096, bytes(range(19))]
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
            allocations, metrics = [], {}

            def allocate(size):
                target = bytearray(size)
                allocations.append(target)
                return target, memoryview(target)

            record = entry(payload, size=sum(map(len, values)))
            record["outer"]["frames"] = chunks
            reader = OuterZstdReader({"owner.bin": payload}, allocate, metrics)
            output = reader.get(record)
            self.assertEqual(output, b"".join(values))
            self.assertIs(output, allocations[0])
            self.assertIs(reader.get(record), output)
            self.assertEqual(len(allocations), 1)
            self.assertEqual(metrics["host_outer_zstd_frames"], 2)
            self.assertEqual(metrics["host_outer_zstd_tensors"], 1)
            # Exact block extent is validated before the final arena allocation,
            # including the optional unknown-content-size GPU frame form.
            truncated = copy.deepcopy(record)
            truncated["outer"]["frames"][-1]["encoded_bytes"] -= 1
            with self.assertRaisesRegex(ValueError, "truncated"):
                OuterZstdReader(
                    {"owner.bin": payload},
                    lambda _: self.fail("truncated chunk allocated a tensor arena"),
                    {},
                ).get(truncated)

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

    def test_size_and_unknown_content_size_fail_before_output_allocation(self):
        raw = b"abcd" * 100
        for known in (True, False):
            encoded = zstd.ZstdCompressor(write_content_size=known).compress(raw)
            reader = OuterZstdReader(
                {"owner.bin": encoded},
                lambda _: self.fail("invalid frame allocated a pinned destination"),
                {},
            )
            with self.assertRaisesRegex(ValueError, "content size"):
                reader.get(entry(encoded, size=len(raw) + (1 if known else 0)))

    def test_block_corruption_never_caches_a_prepared_buffer(self):
        raw = bytes(range(64)) * 128
        encoded = bytearray(zstd.ZstdCompressor(write_checksum=True).compress(raw))
        encoded[-1] ^= 1  # Complete envelope, invalid content checksum.
        targets = []

        def allocate(size):
            target = bytearray(size)
            targets.append(target)
            return target, memoryview(target)

        reader = OuterZstdReader({"owner.bin": encoded}, allocate, {})
        with self.assertRaises(zstd.ZstdError):
            reader.get(entry(encoded, size=len(raw)))
        self.assertFalse(reader.cache)

    def test_relative_span_and_raw_fallback_schema(self):
        # Valid span: 8 raw bytes, 8 alignment bytes, 12 Snappy bytes.
        record = entry(bytes(50), size=28)
        record["frames"][1]["encoded_bytes"] = 12
        validate_outer_entries([record], {"owner.bin": 50})
        mutations = (
            lambda r: r["outer"].update(decoded_bytes=29),
            lambda r: r["outer"].update(encoded_bytes=51),
            lambda r: r["outer"].update(encoded_sha256="obsolete"),
            lambda r: r["frames"][0].update(encoded_offset=16),
            lambda r: r["frames"][1].update(encoded_offset=15),
            lambda r: r["frames"][1].update(codec="zstd"),
            lambda r: r["frames"][1].update(encoded_bytes=17),
            lambda r: r["frames"][0].update(codec="none", encoded_bytes=7),
        )
        for mutate in mutations:
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_outer_entries([bad], {"owner.bin": 50})
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_outer_entries([record, copy.deepcopy(record)], {"owner.bin": 50})
        empty = {"encoding": "xor_bytes", "frames": []}
        validate_outer_entries([empty], {"owner.bin": 50})
        with self.assertRaises(ValueError):
            validate_outer_entries(
                [empty | {"encoding": "replace_bytes", "nbytes": 1}], {"owner.bin": 50}
            )
        with self.assertRaisesRegex(ValueError, "must omit"):
            validate_outer_entries([empty | {"outer": None}], {"owner.bin": 50})

    def test_gpu_outer_chunks_exactly_cover_natural_tensor(self):
        record = entry(bytes(50), size=28)
        record["frames"][1]["encoded_bytes"] = 12
        record["outer"]["frames"] = [
            dict(encoded_offset=0, encoded_bytes=50, decoded_offset=0, decoded_bytes=28)
        ]
        validate_outer_entries([record], {"owner.bin": 50}, gpu=True)
        with self.assertRaisesRegex(ValueError, "descriptor"):
            validate_outer_entries([record], {"owner.bin": 50})
        mutations = (
            lambda r: r["outer"].pop("frames"),
            lambda r: r["outer"].update(frames=[]),
            lambda r: r["outer"]["frames"][0].update(encoded_offset=16),
            lambda r: r["outer"]["frames"][0].update(encoded_bytes=49),
            lambda r: r["outer"]["frames"][0].update(encoded_bytes=True),
            lambda r: r["outer"]["frames"][0].update(decoded_offset=1),
            lambda r: r["outer"]["frames"][0].update(decoded_bytes=27),
            lambda r: r["outer"]["frames"].append(dict(r["outer"]["frames"][0])),
        )
        for mutate in mutations:
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_outer_entries([bad], {"owner.bin": 50}, gpu=True)

        # Large natural tensors remain separate, while independent GPU Zstd
        # chunks cap decoder output and give every chunk an exact destination.
        record["nbytes"] = (1 << 20) + 16
        record["frames"] = [
            dict(
                file="owner.bin",
                encoded_offset=0,
                encoded_bytes=1 << 20,
                decoded_offset=0,
                decoded_bytes=1 << 20,
                codec="none",
            ),
            dict(
                file="owner.bin",
                encoded_offset=1 << 20,
                encoded_bytes=16,
                decoded_offset=1 << 20,
                decoded_bytes=16,
                codec="none",
            ),
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
        validate_outer_entries([record], {"owner.bin": 50}, gpu=True)
        record["outer"]["frames"][0]["decoded_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "chunk range"):
            validate_outer_entries([record], {"owner.bin": 50}, gpu=True)


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
        for validate in (validate_plain_entries, validate_outer_entries):
            validate([raw, unchanged], {"owner.bin": 8})
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
                    self.subTest(validate=validate, mutate=mutate),
                    self.assertRaises(ValueError),
                ):
                    validate([bad], {"owner.bin": 8})
            with self.assertRaisesRegex(ValueError, "overlapping"):
                validate([raw, copy.deepcopy(raw)], {"owner.bin": 8})

    def test_raw_ranges_cannot_overlap_compressed_tensor_payloads(self):
        raw = {
            "encoding": "raw_bytes",
            "nbytes": 4,
            "changed_bytes": 1,
            "frames": [],
            "raw": {"file": "owner.bin", "encoded_offset": 4, "encoded_bytes": 4},
        }
        plain = {
            "encoding": "xor_bytes",
            "frames": [{"file": "owner.bin", "encoded_offset": 0, "encoded_bytes": 8}],
        }
        wrapped = entry(bytes(8), size=28)
        wrapped["frames"][1]["encoded_bytes"] = 12
        for validate, other in (
            (validate_plain_entries, plain),
            (validate_outer_entries, wrapped),
        ):
            with self.assertRaisesRegex(ValueError, "overlapping"):
                validate([raw, other], {"owner.bin": 32})


if __name__ == "__main__":
    unittest.main()
