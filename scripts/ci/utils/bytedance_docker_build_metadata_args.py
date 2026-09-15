import argparse
import datetime
import json
import re
import sys

MOVING_TAG_RE = re.compile(
    r"(?:dev(?:-cu(?:12|13|129|130))?|latest(?:-[A-Za-z0-9_.-]+)?|nightly(?:-cu(?:12|13|129|130))?)"
)
TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
REPOSITORY_COMPONENT_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
HOST_RE = re.compile(
    r"(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[1-9][0-9]{0,4})?"
)
PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")


def validate_scalar(name: str, value: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if any(char in value for char in ("\r", "\n", "\0")):
        raise ValueError(f"{name} contains a forbidden control character")
    if PLACEHOLDER_RE.search(value):
        raise ValueError(f"{name} contains an unresolved placeholder: {value}")


def validate_image_repo(value: str) -> None:
    validate_scalar("image repository", value)
    if value != value.strip() or "://" in value or "@" in value:
        raise ValueError(f"invalid image repository: {value}")
    parts = value.split("/")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError(f"invalid image repository: {value}")
    start = 0
    if "." in parts[0] or ":" in parts[0] or parts[0] == "localhost":
        if not HOST_RE.fullmatch(parts[0]) or ".." in parts[0]:
            raise ValueError(f"invalid image registry host: {parts[0]}")
        start = 1
    if any(not REPOSITORY_COMPONENT_RE.fullmatch(part) for part in parts[start:]):
        raise ValueError(f"invalid image repository: {value}")


def render_tag_template(tag: str, version: str, date: str, short_sha: str) -> str:
    return (
        tag.replace("{version}", version)
        .replace("{date}", date)
        .replace("{short_sha}", short_sha)
    )


def is_moving_tag(tag: str) -> bool:
    return MOVING_TAG_RE.fullmatch(tag) is not None


def select_tag(
    tag_config: str, cuda: str, version: str, date: str, short_sha: str
) -> str:
    entries = json.loads(tag_config)
    if not isinstance(entries, list):
        raise ValueError("tag_config must be a JSON list")
    seen_cuda: set[str] = set()
    validated_entries: list[tuple[str, list[str]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each tag_config entry must be an object")
        entry_cuda = entry.get("cuda")
        if not isinstance(entry_cuda, str) or not entry_cuda:
            raise ValueError("each tag_config entry must have a non-empty cuda string")
        if entry_cuda in seen_cuda:
            raise ValueError(f"duplicate CUDA entry: {entry_cuda}")
        seen_cuda.add(entry_cuda)
        raw_tags = entry.get("tags")
        if (
            not isinstance(raw_tags, list)
            or not raw_tags
            or not all(isinstance(tag, str) and tag for tag in raw_tags)
        ):
            raise ValueError(
                f"tags for CUDA variant {entry_cuda} must be a non-empty list of strings"
            )
        validated_entries.append((entry_cuda, raw_tags))

    for entry_cuda, raw_tags in validated_entries:
        if entry_cuda != cuda:
            continue

        tags = [render_tag_template(tag, version, date, short_sha) for tag in raw_tags]
        for tag in tags:
            validate_scalar("rendered image tag", tag)
            if not TAG_RE.fullmatch(tag):
                raise ValueError(f"invalid rendered image tag: {tag}")

        for tag in tags:
            if not is_moving_tag(tag):
                return tag
        return tags[0]

    raise ValueError(f"CUDA variant {cuda} not found in tag_config")


def build_arg_tokens(
    *,
    cuda: str,
    tag_config: str,
    image_repo: str,
    version: str,
    build_commit: str,
    build_tree: str,
    python_manifest_sha256: str,
    build_source: str,
    build_url: str,
    date: str,
    short_sha: str,
) -> list[str]:
    validate_image_repo(image_repo)
    for name, value, allow_empty in (
        ("cuda", cuda, False),
        ("version", version, True),
        ("build commit", build_commit, False),
        ("build tree", build_tree, False),
        ("Python manifest digest", python_manifest_sha256, False),
        ("build source", build_source, False),
        ("build URL", build_url, True),
        ("date", date, False),
        ("short SHA", short_sha, False),
    ):
        validate_scalar(name, value, allow_empty=allow_empty)
    image_tag = select_tag(tag_config, cuda, version, date, short_sha)
    build_args = {
        "SGLANG_BUILD_COMMIT": build_commit,
        "SGLANG_BUILD_TREE": build_tree,
        "SGLANG_PYTHON_MANIFEST_SHA256": python_manifest_sha256,
        "SGLANG_BUILD_SOURCE": build_source,
        "SGLANG_BUILD_URL": build_url,
        "SGLANG_IMAGE_TAG": f"{image_repo}:{image_tag}",
    }

    tokens = []
    for key, value in build_args.items():
        tokens.extend(["--build-arg", f"{key}={value}"])
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit docker build arguments for SGLang image metadata."
    )
    parser.add_argument("--cuda", required=True, help="CUDA variant from tag_config.")
    parser.add_argument("--tag-config", required=True, help="Docker tag JSON config.")
    parser.add_argument("--image-repo", required=True, help="Docker image repository.")
    parser.add_argument("--sgl-version", default="", help="SGLang release version.")
    parser.add_argument(
        "--build-commit",
        required=True,
        help="Commit checked out for the Docker build.",
    )
    parser.add_argument(
        "--build-tree", required=True, help="Git tree checked out for the Docker build."
    )
    parser.add_argument(
        "--python-manifest-sha256",
        required=True,
        help="Deterministic digest of the tracked Python source manifest.",
    )
    parser.add_argument(
        "--build-source",
        required=True,
        help="Repository URL for the checked-out source.",
    )
    parser.add_argument("--build-url", default="", help="CI run URL.")
    parser.add_argument(
        "--date",
        default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"),
        help="Date used for {date} tag templates.",
    )
    parser.add_argument(
        "--short-sha",
        default="",
        help="Short SHA used for {short_sha}; defaults to build commit prefix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    short_sha = args.short_sha or args.build_commit[:8]

    try:
        tokens = build_arg_tokens(
            cuda=args.cuda,
            tag_config=args.tag_config,
            image_repo=args.image_repo,
            version=args.sgl_version,
            build_commit=args.build_commit,
            build_tree=args.build_tree,
            python_manifest_sha256=args.python_manifest_sha256,
            build_source=args.build_source,
            build_url=args.build_url,
            date=args.date,
            short_sha=short_sha,
        )
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\n".join(tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
