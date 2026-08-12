#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable
from urllib.parse import quote
from urllib.request import urlopen
from zoneinfo import ZoneInfo

TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
VERSION_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?([A-Za-z0-9_.-]*)?$")
VLLM_UBUNTU2404_VERSION_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)-ubuntu2404$")
VLLM_UBUNTU2404_NIGHTLY_TAG_RE = re.compile(
    r"^(nightly|nightly-[0-9a-f]{7,64})-ubuntu2404$"
)
AUTO_TAG_SPECS = {"version", "today-nightly"}
DEFAULT_VARIANT_RE = re.compile(
    r"^(latest|dev|nightly|nightly-[0-9a-f]{7,64}|nightly-dev-[0-9]{8}-[0-9a-f]{7,64}|v\d+\.\d+\.\d+(?:\.post\d+)?)$"
)


@dataclass(frozen=True)
class SyncItem:
    source: str
    destination: str


@dataclass(frozen=True)
class DockerTag:
    name: str
    last_updated: str


def parse_tags(value: str) -> list[str]:
    tags = [tag.strip() for tag in re.split(r"[,\n]+", value) if tag.strip()]
    if not tags:
        raise SystemExit("at least one image tag is required")
    for tag in tags:
        if tag not in AUTO_TAG_SPECS and not TAG_RE.fullmatch(tag):
            raise SystemExit(f"invalid Docker tag: {tag}")
    return tags


def docker_hub_repository(image_name: str) -> str:
    image_name = normalize_image_name(image_name, "source image")
    if image_name.startswith("docker.io/"):
        image_name = image_name.removeprefix("docker.io/")
    parts = image_name.split("/")
    if len(parts) == 1:
        return f"library/{parts[0]}"
    return "/".join(parts)


def fetch_docker_hub_tags(image_name: str, *, pages: int = 5) -> list[DockerTag]:
    repository = docker_hub_repository(image_name)
    page_url = (
        "https://hub.docker.com/v2/repositories/"
        f"{quote(repository, safe='/')}/tags?page_size=100&ordering=last_updated"
    )
    tags: list[DockerTag] = []
    for _ in range(pages):
        with urlopen(page_url, timeout=30) as response:
            payload = json.load(response)
        for item in payload.get("results", []):
            name = item.get("name")
            last_updated = item.get("last_updated")
            if name and last_updated:
                tags.append(DockerTag(name=name, last_updated=last_updated))
        page_url = payload.get("next")
        if not page_url:
            break
    return tags


def version_key(tag_name: str) -> tuple[int, int, int, int] | None:
    match = VERSION_TAG_RE.fullmatch(tag_name)
    if not match:
        return None
    major, minor, patch, post, suffix = match.groups()
    if suffix and not suffix.startswith("-"):
        return None
    return (int(major), int(minor), int(patch), int(post or 0))


def latest_version_tags(tags: Iterable[DockerTag]) -> list[str]:
    names = {tag.name for tag in tags}
    keyed_versions = [
        (key, name)
        for name in names
        if DEFAULT_VARIANT_RE.fullmatch(name)
        and (key := version_key(name.removesuffix(""))) is not None
    ]
    if not keyed_versions:
        raise SystemExit("no version image tags found in source repository")

    latest_key = max(key for key, _ in keyed_versions)
    version_names = sorted(name for key, name in keyed_versions if key == latest_key)
    latest_aliases = ["latest"] if "latest" in names else []
    return sorted(set(latest_aliases + version_names))


def tag_updated_date(tag: DockerTag, timezone: ZoneInfo) -> date:
    value = tag.last_updated
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if "." in value:
        prefix, suffix = value.split(".", 1)
        match = re.fullmatch(r"(\d+)(.*)", suffix)
        if not match:
            raise SystemExit(
                f"invalid Docker Hub last_updated timestamp: {tag.last_updated}"
            )
        fraction, offset = match.groups()
        fraction = fraction[:6]
        fraction = fraction.ljust(6, "0")
        value = f"{prefix}.{fraction}{offset}"
    updated = datetime.fromisoformat(value)
    return updated.astimezone(timezone).date()


def today_nightly_tags(
    tags: Iterable[DockerTag],
    *,
    today: date,
    daily_aliases: set[str] | None = None,
    timezone: ZoneInfo | None = None,
) -> list[str]:
    timezone = timezone or ZoneInfo("Asia/Shanghai")
    daily_aliases = daily_aliases or set()
    selected = {
        tag.name
        for tag in tags
        if tag_updated_date(tag, timezone) == today
        and ("nightly" in tag.name or tag.name in daily_aliases)
        and DEFAULT_VARIANT_RE.fullmatch(tag.name)
    }
    if not selected:
        raise SystemExit(f"no today-nightly image tags found for {today.isoformat()}")
    return sorted(selected)


def resolve_tag_specs(
    specs: Iterable[str],
    docker_tags: Iterable[DockerTag],
    *,
    today: date,
    daily_aliases: set[str] | None = None,
    timezone: ZoneInfo | None = None,
) -> list[str]:
    docker_tags = list(docker_tags)
    resolved: list[str] = []
    for spec in specs:
        if spec == "version":
            resolved.extend(latest_version_tags(docker_tags))
        elif spec == "today-nightly":
            resolved.extend(
                today_nightly_tags(
                    docker_tags,
                    today=today,
                    daily_aliases=daily_aliases,
                    timezone=timezone,
                )
            )
        else:
            resolved.append(spec)

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in resolved:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


def vllm_ubuntu2404_version_key(tag_name: str) -> tuple[int, int, int] | None:
    match = VLLM_UBUNTU2404_VERSION_TAG_RE.fullmatch(tag_name)
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def latest_vllm_version_tags(tags: Iterable[DockerTag]) -> list[str]:
    names = {tag.name for tag in tags}
    keyed_versions = [
        (key, name)
        for name in names
        if (key := vllm_ubuntu2404_version_key(name)) is not None
    ]
    if not keyed_versions:
        return latest_version_tags(tags)

    latest_key = max(key for key, _ in keyed_versions)
    version_names = sorted(name for key, name in keyed_versions if key == latest_key)
    latest_aliases = ["latest-ubuntu2404"] if "latest-ubuntu2404" in names else []
    return sorted(set(latest_aliases + version_names))


def today_vllm_nightly_tags(
    tags: Iterable[DockerTag],
    *,
    today: date,
    timezone: ZoneInfo | None = None,
) -> list[str]:
    tags = list(tags)
    timezone = timezone or ZoneInfo("Asia/Shanghai")
    ubuntu2404_tags = sorted(
        {
            tag.name
            for tag in tags
            if tag_updated_date(tag, timezone) == today
            and VLLM_UBUNTU2404_NIGHTLY_TAG_RE.fullmatch(tag.name)
        }
    )
    if ubuntu2404_tags:
        return ubuntu2404_tags
    return today_nightly_tags(tags, today=today, timezone=timezone)


def resolve_vllm_tag_specs(
    specs: Iterable[str],
    docker_tags: Iterable[DockerTag],
    *,
    today: date,
    timezone: ZoneInfo | None = None,
) -> list[str]:
    docker_tags = list(docker_tags)
    resolved: list[str] = []
    for spec in specs:
        if spec == "version":
            resolved.extend(latest_vllm_version_tags(docker_tags))
        elif spec == "today-nightly":
            resolved.extend(
                today_vllm_nightly_tags(
                    docker_tags,
                    today=today,
                    timezone=timezone,
                )
            )
        else:
            resolved.append(spec)

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in resolved:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


def normalize_image_name(value: str, field_name: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise SystemExit(f"{field_name} is required")
    if ":" in value.rsplit("/", 1)[-1]:
        raise SystemExit(f"{field_name} must not include a tag: {value}")
    if value.startswith("/") or "//" in value:
        raise SystemExit(f"invalid {field_name}: {value}")
    return value


def normalize_repository(value: str, field_name: str) -> str:
    value = value.strip().strip("/")
    if not value:
        raise SystemExit(f"{field_name} is required")
    if ":" in value or "//" in value:
        raise SystemExit(f"invalid {field_name}: {value}")
    return value


def build_sync_plan(
    *,
    registry: str,
    namespace: str,
    sglang_source: str,
    sglang_repository: str,
    sglang_tags: Iterable[str],
    vllm_source: str,
    vllm_repository: str,
    vllm_tags: Iterable[str],
) -> list[SyncItem]:
    registry = normalize_repository(registry, "registry")
    namespace = normalize_repository(namespace, "namespace")
    sglang_source = normalize_image_name(sglang_source, "sglang source image")
    sglang_repository = normalize_repository(
        sglang_repository, "sglang destination repository"
    )
    vllm_source = normalize_image_name(vllm_source, "vllm source image")
    vllm_repository = normalize_repository(
        vllm_repository, "vllm destination repository"
    )

    plan: list[SyncItem] = []
    for source_image, destination_repo, tags in (
        (sglang_source, sglang_repository, sglang_tags),
        (vllm_source, vllm_repository, vllm_tags),
    ):
        for tag in tags:
            if not TAG_RE.fullmatch(tag):
                raise SystemExit(f"invalid Docker tag: {tag}")
            plan.append(
                SyncItem(
                    source=f"{source_image}:{tag}",
                    destination=f"{registry}/{namespace}/{destination_repo}:{tag}",
                )
            )
    return plan


def build_imagetools_command(item: SyncItem, *, platform: str = "") -> list[str]:
    command = ["docker", "buildx", "imagetools", "create"]
    if platform:
        command.extend(["--platform", platform])
    command.extend(["-t", item.destination, item.source])
    return command


def sync_images(plan: Iterable[SyncItem], *, execute: bool, platform: str = "") -> None:
    for item in plan:
        command = build_imagetools_command(item, platform=platform)
        print(f"{item.source} -> {item.destination}", flush=True)
        if execute:
            subprocess.run(command, check=True)
        else:
            print("+ " + " ".join(command), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync public SGLang and vLLM Docker image tags to Volcengine CR."
    )
    parser.add_argument("--registry", required=True, help="Volcengine CR registry host")
    parser.add_argument("--namespace", required=True, help="Volcengine CR namespace")
    parser.add_argument(
        "--sglang-source",
        default="docker.io/lmsysorg/sglang",
        help="Source SGLang image repository without tag",
    )
    parser.add_argument(
        "--sglang-repository",
        default="sglang",
        help="Destination SGLang repository inside the Volcengine CR namespace",
    )
    parser.add_argument(
        "--sglang-tags",
        default="version,today-nightly",
        help=(
            "Comma or newline separated SGLang tags to sync. Special values: "
            "version, today-nightly."
        ),
    )
    parser.add_argument(
        "--vllm-source",
        default="docker.io/vllm/vllm-openai",
        help="Source vLLM image repository without tag",
    )
    parser.add_argument(
        "--vllm-repository",
        default="vllm",
        help="Destination vLLM repository inside the Volcengine CR namespace",
    )
    parser.add_argument(
        "--vllm-tags",
        default="version,today-nightly",
        help=(
            "Comma or newline separated vLLM tags to sync. Special values: "
            "version, today-nightly. The automatic vLLM selector prefers "
            "ubuntu2404 tags and falls back to unsuffixed tags when no "
            "ubuntu2404 tag exists for that image family."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run docker buildx imagetools create. Without this, print a dry run.",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="Timezone used to decide which Docker Hub tags were updated today.",
    )
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Optional platform filter passed to docker buildx imagetools create.",
    )
    args = parser.parse_args()

    timezone = ZoneInfo(args.timezone)
    today = datetime.now(timezone).date()
    sglang_tag_specs = parse_tags(args.sglang_tags)
    vllm_tag_specs = parse_tags(args.vllm_tags)
    sglang_docker_tags = (
        fetch_docker_hub_tags(args.sglang_source)
        if AUTO_TAG_SPECS.intersection(sglang_tag_specs)
        else []
    )
    vllm_docker_tags = (
        fetch_docker_hub_tags(args.vllm_source)
        if AUTO_TAG_SPECS.intersection(vllm_tag_specs)
        else []
    )

    plan = build_sync_plan(
        registry=args.registry,
        namespace=args.namespace,
        sglang_source=args.sglang_source,
        sglang_repository=args.sglang_repository,
        sglang_tags=resolve_tag_specs(
            sglang_tag_specs,
            sglang_docker_tags,
            today=today,
            daily_aliases={"dev", "dev-cu12", "dev-cu13"},
            timezone=timezone,
        ),
        vllm_source=args.vllm_source,
        vllm_repository=args.vllm_repository,
        vllm_tags=resolve_vllm_tag_specs(
            vllm_tag_specs,
            vllm_docker_tags,
            today=today,
            timezone=timezone,
        ),
    )
    sync_images(plan, execute=args.execute, platform=args.platform)


if __name__ == "__main__":
    main()
