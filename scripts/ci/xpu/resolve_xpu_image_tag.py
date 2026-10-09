#!/usr/bin/env python3
"""Resolve the newest intel/sglang-dev image tag published for a given git ref.

.github/workflows/release-docker-intel-xpu-nightly.yml tags its images
``nightly-dev-xpu-bmg-{date}-{sha}`` for main and
``nightly-dev-xpu-bmg-{ref_slug}-{date}-{sha}`` for any other ref. Neither form
is stable, so a downstream job that wants to test "the image built for
release/v0.5.20" has to look up the newest matching tag in the registry.

Usage:
    resolve_xpu_image_tag.py --ref release/v0.5.20
    resolve_xpu_image_tag.py --ref main --explicit-tag nightly-dev-xpu-bmg-20260916-abc1234
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

DEFAULT_REPO = "intel/sglang-dev"
# Keep in lockstep with the nightly workflow's tag construction.
TAG_PREFIX = "nightly-dev-xpu-bmg"
# date_tag is `date +%Y%m%d`; commit_hash is `git rev-parse --short=7`. Accept
# longer hashes too -- tags published before the --short=7 pin carry 9 chars.
_SUFFIX = r"[0-9]{8}-[0-9a-f]{7,}"
# Cap pagination so a pathological registry response can't hang the job.
MAX_PAGES = 10
PAGE_SIZE = 100


def slugify(ref: str) -> str:
    """Match the nightly workflow's `sed 's#[^a-zA-Z0-9._-]#-#g'` slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "-", ref)


def tag_pattern(ref: str) -> "re.Pattern[str]":
    """Build the regex for tags the nightly publishes for ``ref``.

    main has no slug segment, so it needs an anchored pattern -- a plain prefix
    match would also catch every branch-slugged tag.
    """
    if ref == "main":
        return re.compile(rf"^{re.escape(TAG_PREFIX)}-{_SUFFIX}$")
    return re.compile(rf"^{re.escape(TAG_PREFIX)}-{re.escape(slugify(ref))}-{_SUFFIX}$")


def select_tag(names, ref):
    """Return the first tag in ``names`` published for ``ref``, else None.

    ``names`` must already be ordered newest-first -- see fetch_tag_names.
    """
    pattern = tag_pattern(ref)
    for name in names:
        if pattern.match(name):
            return name
    return None


def fetch_tag_names(repo: str) -> list:
    """List tag names for ``repo`` from Docker Hub, newest-pushed first.

    Docker Hub ignores the documented ``ordering=-last_updated`` on this
    endpoint and answers oldest-first, so sort client-side rather than trusting
    the response order.
    """
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/?page_size={PAGE_SIZE}"
    results = []
    for _ in range(MAX_PAGES):
        if not url:
            break
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        results.extend(payload.get("results", []))
        url = payload.get("next")
    # Missing last_updated sorts oldest so a malformed entry never wins.
    results.sort(key=lambda r: r.get("last_updated") or "", reverse=True)
    return [r["name"] for r in results]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", required=True, help="Git ref the image was built from (e.g. main)."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Docker Hub repository.")
    parser.add_argument(
        "--explicit-tag",
        default="",
        help="Skip resolution and use this tag verbatim.",
    )
    args = parser.parse_args()

    if args.explicit_tag:
        print(f"{args.repo}:{args.explicit_tag}")
        return 0

    try:
        names = fetch_tag_names(args.repo)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"::error::Could not list tags for {args.repo}: {exc}", file=sys.stderr)
        return 1

    tag = select_tag(names, args.ref)
    if tag is None:
        print(
            f"::error::No {args.repo} tag found for ref '{args.ref}'. Expected a tag "
            f"matching {tag_pattern(args.ref).pattern!r}. Run "
            f"release-docker-intel-xpu-nightly.yml for this ref first, or pass "
            f"image_tag explicitly.",
            file=sys.stderr,
        )
        return 1

    print(f"{args.repo}:{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
