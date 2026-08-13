#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_volcengine_image_tag import build_tag, validate_suffix


class GetVolcengineImageTagTest(unittest.TestCase):
    def test_version_tag_without_suffixes(self) -> None:
        self.assertEqual(
            build_tag(
                mode="version",
                version="0.5.17",
                timestamp="202608121200",
                tag_value="glm5-2",
            ),
            "v0.5.17.byted.glm5-2.202608121200",
        )

    def test_manual_and_nightly_prefixes(self) -> None:
        self.assertEqual(
            build_tag(mode="manual", version="0.5.17", timestamp="202608121200"),
            "v0.5.17.iaas.dev.202608121200",
        )
        self.assertEqual(
            build_tag(mode="nightly", version="0.5.17", timestamp="202608121200"),
            "v0.5.17.iaas.nightly.202608121200",
        )

    def test_format_suffix_trails_cuda_suffix(self) -> None:
        self.assertEqual(
            build_tag(
                mode="version",
                version="0.5.17",
                timestamp="202608121200",
                tag_value="glm5-2",
                cuda_suffix="cu130",
                format_suffix="zstd",
            ),
            "v0.5.17.byted.glm5-2.202608121200-cu130-zstd",
        )
        self.assertEqual(
            build_tag(
                mode="version",
                version="0.5.17",
                timestamp="202608121200",
                tag_value="glm5-2",
                cuda_suffix="cu130",
                format_suffix="nydus",
            ),
            "v0.5.17.byted.glm5-2.202608121200-cu130-nydus",
        )

    def test_full_suffix_order_variant_cuda_format(self) -> None:
        self.assertEqual(
            build_tag(
                mode="version",
                version="0.5.17",
                timestamp="202608121200",
                tag_value="glm5-2",
                variant_suffix="w4a8",
                cuda_suffix="cu130",
                format_suffix="zstd",
            ),
            "v0.5.17.byted.glm5-2.202608121200-w4a8-cu130-zstd",
        )

    def test_format_suffix_without_cuda_suffix(self) -> None:
        self.assertEqual(
            build_tag(
                mode="version",
                version="0.5.17",
                timestamp="202608121200",
                tag_value="glm5-2",
                format_suffix="zstd",
            ),
            "v0.5.17.byted.glm5-2.202608121200-zstd",
        )

    def test_version_mode_requires_tag_value(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--tag-value is required"):
            build_tag(mode="version", version="0.5.17", timestamp="202608121200")

    def test_validate_suffix_accepts_safe_values(self) -> None:
        for value in ("zstd", "nydus", "cu130", "w4a8", "deepseek-v4", ""):
            validate_suffix("format-suffix", value)

    def test_validate_suffix_rejects_unsafe_values(self) -> None:
        for value in ("-zstd", "zstd/", "zs td", "-", "z$td"):
            with self.assertRaisesRegex(SystemExit, "must be a Docker tag-safe suffix"):
                validate_suffix("format-suffix", value)


if __name__ == "__main__":
    unittest.main()
