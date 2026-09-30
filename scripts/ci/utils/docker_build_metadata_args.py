import argparse
import datetime
import json
import sys

MOVING_TAGS = {"dev", "dev-cu13", "latest"}


def render_tag_template(tag: str, version: str, date: str, short_sha: str) -> str:
    return (
        tag.replace("{version}", version)
        .replace("{date}", date)
        .replace("{short_sha}", short_sha)
    )


def is_moving_tag(tag: str) -> bool:
    return tag in MOVING_TAGS or tag.startswith("latest-")


def select_tag(
    tag_config: str, cuda: str, version: str, date: str, short_sha: str
) -> str:
    entries = json.loads(tag_config)
    for entry in entries:
        if entry.get("cuda") != cuda:
            continue

        tags = [
            render_tag_template(tag, version, date, short_sha)
            for tag in entry.get("tags", [])
        ]
        if not tags:
            raise ValueError(f"No tags configured for CUDA variant {cuda}")

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
    build_url: str,
    date: str,
    short_sha: str,
) -> list[str]:
    image_tag = select_tag(tag_config, cuda, version, date, short_sha)
    build_args = {
        "SGLANG_BUILD_COMMIT": build_commit,
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
            build_url=args.build_url,
            date=args.date,
            short_sha=short_sha,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\n".join(tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
