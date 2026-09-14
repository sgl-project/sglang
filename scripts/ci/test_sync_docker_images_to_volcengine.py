#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import date
from zoneinfo import ZoneInfo

from sync_docker_images_to_volcengine import (
    DockerTag,
    SyncItem,
    build_imagetools_command,
    build_sync_plan,
    parse_tags,
    resolve_tag_specs,
    resolve_vllm_tag_specs,
    tag_updated_date,
)


class SyncDockerImagesToVolcengineTest(unittest.TestCase):
    def test_builds_default_sglang_and_vllm_latest_plan(self) -> None:
        plan = build_sync_plan(
            registry="iaas-gpu-cn-beijing.cr.volces.com",
            namespace="serving",
            sglang_source="docker.io/lmsysorg/sglang",
            sglang_repository="sglang",
            sglang_tags=["latest"],
            vllm_source="docker.io/vllm/vllm-openai",
            vllm_repository="vllm",
            vllm_tags=["latest"],
        )

        self.assertEqual(
            [(item.source, item.destination) for item in plan],
            [
                (
                    "docker.io/lmsysorg/sglang:latest",
                    "iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:latest",
                ),
                (
                    "docker.io/vllm/vllm-openai:latest",
                    "iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:latest",
                ),
            ],
        )

    def test_parse_tags_accepts_commas_and_newlines(self) -> None:
        self.assertEqual(
            parse_tags("latest, nightly\nlatest-cu130"),
            ["latest", "nightly", "latest-cu130"],
        )

    def test_resolves_default_latest_version_tags_only(self) -> None:
        tags = [
            DockerTag("nightly-dev-20260616-abcdef1", "2026-06-16T01:53:18Z"),
            DockerTag("latest", "2026-06-15T08:39:02Z"),
            DockerTag("v0.5.13.post1", "2026-06-15T08:39:01Z"),
            DockerTag("latest-cu130", "2026-06-15T08:39:06Z"),
            DockerTag("v0.5.13.post1-cu130", "2026-06-15T08:39:04Z"),
            DockerTag("v0.5.12", "2026-05-10T08:00:00Z"),
            DockerTag("latest-runtime", "2026-06-15T09:25:15Z"),
            DockerTag("v0.5.13.post1-runtime", "2026-06-15T09:25:13Z"),
        ]

        self.assertEqual(
            resolve_tag_specs(["version"], tags, today=date(2026, 6, 16)),
            [
                "latest",
                "v0.5.13.post1",
            ],
        )

    def test_resolves_default_today_nightly_tags_only(self) -> None:
        tags = [
            DockerTag("nightly-dev-cu13-20260616-abcdef1", "2026-06-16T01:53:20Z"),
            DockerTag("nightly-dev-20260616-abcdef1", "2026-06-16T01:53:18Z"),
            DockerTag("dev-cu13", "2026-06-16T01:53:17Z"),
            DockerTag("dev", "2026-06-16T01:53:15Z"),
            DockerTag("nightly-dev-cu12-20260615-old", "2026-06-15T01:53:12Z"),
            DockerTag("latest", "2026-06-15T08:39:02Z"),
        ]

        self.assertEqual(
            resolve_tag_specs(
                ["today-nightly"],
                tags,
                today=date(2026, 6, 16),
                daily_aliases={"dev"},
            ),
            [
                "dev",
                "nightly-dev-20260616-abcdef1",
            ],
        )

    def test_resolves_default_vllm_today_nightly_tags_only(self) -> None:
        tags = [
            DockerTag("cu129-nightly-abcdef1", "2026-06-16T06:28:48Z"),
            DockerTag("cu129-nightly", "2026-06-16T06:28:46Z"),
            DockerTag("nightly-abcdef1", "2026-06-16T06:15:25Z"),
            DockerTag("nightly", "2026-06-16T06:15:24Z"),
            DockerTag("nightly-aarch64", "2026-06-16T06:15:21Z"),
            DockerTag("nightly-x86_64", "2026-06-16T06:05:25Z"),
        ]

        self.assertEqual(
            resolve_tag_specs(["today-nightly"], tags, today=date(2026, 6, 16)),
            [
                "nightly",
                "nightly-abcdef1",
            ],
        )

    def test_resolves_vllm_ubuntu2404_version_tags_only(self) -> None:
        tags = [
            DockerTag("latest-ubuntu2404", "2026-06-13T01:49:50Z"),
            DockerTag("v0.23.0-ubuntu2404", "2026-06-13T01:49:52Z"),
            DockerTag("latest", "2026-06-13T00:36:44Z"),
            DockerTag("v0.23.0", "2026-06-13T00:36:45Z"),
            DockerTag("latest-x86_64-ubuntu2404", "2026-06-13T01:39:39Z"),
            DockerTag("v0.23.0-x86_64-ubuntu2404", "2026-06-13T01:39:41Z"),
            DockerTag("latest-cu129-ubuntu2404", "2026-06-13T02:30:42Z"),
            DockerTag("v0.23.0-cu129-ubuntu2404", "2026-06-13T02:30:43Z"),
            DockerTag("v0.22.1-ubuntu2404", "2026-06-05T08:34:55Z"),
            DockerTag("nightly", "2026-06-17T06:16:46Z"),
        ]

        self.assertEqual(
            resolve_vllm_tag_specs(["version"], tags, today=date(2026, 6, 17)),
            [
                "latest-ubuntu2404",
                "v0.23.0-ubuntu2404",
            ],
        )

    def test_resolves_vllm_default_version_tags_when_ubuntu2404_is_unavailable(
        self,
    ) -> None:
        tags = [
            DockerTag("latest", "2026-06-13T00:36:44Z"),
            DockerTag("v0.23.0", "2026-06-13T00:36:45Z"),
            DockerTag("latest-x86_64", "2026-06-13T01:39:39Z"),
            DockerTag("v0.23.0-x86_64", "2026-06-13T01:39:41Z"),
            DockerTag("latest-cu129", "2026-06-13T02:30:42Z"),
            DockerTag("v0.23.0-cu129", "2026-06-13T02:30:43Z"),
            DockerTag("v0.22.1", "2026-06-05T08:34:55Z"),
        ]

        self.assertEqual(
            resolve_vllm_tag_specs(["version"], tags, today=date(2026, 6, 17)),
            [
                "latest",
                "v0.23.0",
            ],
        )

    def test_resolves_vllm_ubuntu2404_version_and_default_today_nightly_tags(
        self,
    ) -> None:
        tags = [
            DockerTag("latest-ubuntu2404", "2026-06-13T01:49:50Z"),
            DockerTag("v0.23.0-ubuntu2404", "2026-06-13T01:49:52Z"),
            DockerTag("latest", "2026-06-13T00:36:44Z"),
            DockerTag("v0.23.0", "2026-06-13T00:36:45Z"),
            DockerTag("cu129-nightly-abcdef1", "2026-06-17T06:28:48Z"),
            DockerTag("cu129-nightly", "2026-06-17T06:28:46Z"),
            DockerTag("nightly-abcdef1", "2026-06-17T06:15:25Z"),
            DockerTag("nightly", "2026-06-17T06:15:24Z"),
            DockerTag("nightly-aarch64", "2026-06-17T06:15:21Z"),
            DockerTag("nightly-x86_64", "2026-06-17T06:05:25Z"),
        ]

        self.assertEqual(
            resolve_vllm_tag_specs(
                ["version", "today-nightly"], tags, today=date(2026, 6, 17)
            ),
            [
                "latest-ubuntu2404",
                "v0.23.0-ubuntu2404",
                "nightly",
                "nightly-abcdef1",
            ],
        )

    def test_tag_updated_date_accepts_non_six_digit_fraction(self) -> None:
        self.assertEqual(
            tag_updated_date(
                DockerTag("dev-cu13", "2026-06-16T01:53:17.12614Z"),
                timezone=ZoneInfo("Asia/Shanghai"),
            ),
            date(2026, 6, 16),
        )

    def test_imagetools_command_filters_platform_when_requested(self) -> None:
        self.assertEqual(
            build_imagetools_command(
                SyncItem(
                    source="docker.io/lmsysorg/sglang:latest",
                    destination="iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:latest",
                ),
                platform="linux/amd64",
            ),
            [
                "docker",
                "buildx",
                "imagetools",
                "create",
                "--platform",
                "linux/amd64",
                "-t",
                "iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:latest",
                "docker.io/lmsysorg/sglang:latest",
            ],
        )

    def test_rejects_missing_registry_or_namespace(self) -> None:
        with self.assertRaisesRegex(SystemExit, "registry is required"):
            build_sync_plan(
                registry="",
                namespace="serving",
                sglang_source="docker.io/lmsysorg/sglang",
                sglang_repository="sglang",
                sglang_tags=["latest"],
                vllm_source="docker.io/vllm/vllm-openai",
                vllm_repository="vllm",
                vllm_tags=["latest"],
            )

        with self.assertRaisesRegex(SystemExit, "namespace is required"):
            build_sync_plan(
                registry="iaas-gpu-cn-beijing.cr.volces.com",
                namespace="",
                sglang_source="docker.io/lmsysorg/sglang",
                sglang_repository="sglang",
                sglang_tags=["latest"],
                vllm_source="docker.io/vllm/vllm-openai",
                vllm_repository="vllm",
                vllm_tags=["latest"],
            )


if __name__ == "__main__":
    unittest.main()
