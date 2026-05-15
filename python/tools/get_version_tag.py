#!/usr/bin/env python3
"""Resolve the correct version tag for setuptools-scm.

Called by setuptools-scm via git_describe_command in pyproject.toml.
Outputs either a bare tag (e.g., "v0.5.10") for exact-match commits,
or a `git describe --long` string (e.g., "v0.5.10-2-gabcdef0") for
untagged commits. Both formats are accepted by setuptools-scm.

This two-step approach avoids a strverscmp bug where
`git tag --sort=-version:refname` sorts v0.5.10rc0 above v0.5.10,
which would cause CI to build the wrong version.

Strategy:
1. If the current commit has an exact version tag, use it directly.
   This handles CI release builds (both stable and rc).
2. Otherwise, find the best tag to describe relative to:
   a. Prefer ancestor tags (reachable from HEAD) so checking out a
      commit between v0.5.10 and v0.5.11 reports 0.5.11.devN correctly.
   b. Fall back to the globally highest tag via merge-base distance
      when no ancestor tags exist (e.g., on main where release tags
      live on separate release branches).
3. When the chosen tag has a .postN suffix, strip it back to the base
   release (e.g., v0.5.10.post1 -> v0.5.10). setuptools-scm's
   guess-next-dev scheme bumps the last integer, so v0.5.10 becomes
   0.5.11.devN (correct), whereas v0.5.10.post1 would become
   0.5.10.post2.devN (wrong).
"""

import re
import subprocess
import sys


def parse_version_tuple(tag: str) -> tuple:
    """Parse a version tag into a sortable tuple using PEP 440 ordering.

    Returns a tuple where:
    - Base version parts are integers: (major, minor, patch)
    - Pre-release suffix gets a lower sort key than bare version:
      v0.5.10rc0  -> (0, 5, 10, 0, 0)   # pre-release
      v0.5.10     -> (0, 5, 10, 1, 0)   # stable (sorts higher)
      v0.5.10.post1 -> (0, 5, 10, 2, 1)  # post-release (sorts highest)
    """
    v = tag.lstrip("v")
    # Split base version from suffix
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\.?(rc|post)(\d+))?$", v)
    if not m:
        return (0, 0, 0, 0, 0)
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    suffix_type = m.group(4)
    suffix_num = int(m.group(5)) if m.group(5) else 0
    if suffix_type == "rc":
        return (major, minor, patch, 0, suffix_num)
    elif suffix_type == "post":
        return (major, minor, patch, 2, suffix_num)
    else:
        return (major, minor, patch, 1, 0)


def run_git(*args: str, allow_failure: bool = False) -> str:
    """Run a git command and return stripped stdout.

    Args:
        allow_failure: If True, return "" on non-zero exit (expected for
            commands like --exact-match that legitimately fail).
            If False, log stderr on failure before returning "".
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"ERROR: Failed to run 'git {' '.join(args)}': {exc}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        if not allow_failure:
            stderr_msg = result.stderr.strip()
            print(
                f"WARNING: git {' '.join(args)} failed "
                f"(exit {result.returncode}): {stderr_msg}",
                file=sys.stderr,
            )
        return ""

    return result.stdout.strip()


def get_exact_version_tag() -> str:
    """Return the version tag name if HEAD has an exact version tag, or empty string."""
    return run_git(
        "describe", "--tags", "--exact-match", "--match", "v*", allow_failure=True
    )


def strip_post_suffix(tag: str) -> str:
    """Strip .postN suffix from a tag so setuptools-scm bumps correctly.

    setuptools-scm's guess-next-dev increments the last integer:
      v0.5.10       -> 0.5.11.devN   (correct)
      v0.5.10.post1 -> 0.5.10.post2.devN  (wrong)
    By stripping .postN we ensure the micro version gets bumped instead.
    """
    return re.sub(r"\.post\d+$", "", tag)


def get_latest_ancestor_version_tag() -> str:
    """Return the highest version tag that is an ancestor of HEAD.

    Uses ``git tag --merged HEAD`` to restrict to reachable tags, then
    sorts by PEP 440 ordering.  Returns empty string when no ancestor
    tags exist (e.g., main diverged before any release tag was created).
    """
    tags_raw = run_git("tag", "--merged", "HEAD", "--list", "v*.*.*")
    if not tags_raw:
        return ""
    tag_list = sorted(tags_raw.splitlines(), key=parse_version_tuple, reverse=True)
    return tag_list[0] if tag_list else ""


def _describe_from_tag(tag: str) -> str:
    """Build a describe string relative to *tag*, stripping .postN.

    If the tag is an ancestor of HEAD, uses ``git describe`` directly.
    Otherwise computes the distance from the merge-base manually.

    The .postN suffix is stripped so that setuptools-scm bumps the micro
    version (0.5.10 -> 0.5.11.devN) instead of the post number.
    """
    base_tag = strip_post_suffix(tag)

    # Fast path: base tag is an ancestor of HEAD
    result = run_git(
        "describe",
        "--tags",
        "--long",
        "--match",
        base_tag,
        "HEAD",
        allow_failure=True,
    )
    if result:
        return result

    # Tag is not an ancestor (release branch scenario).
    # Build describe string manually: {tag}-{distance}-g{hash}
    merge_base = run_git("merge-base", tag, "HEAD", allow_failure=True)
    if not merge_base:
        return ""
    distance = run_git("rev-list", "--count", f"{merge_base}..HEAD")
    short_hash = run_git("rev-parse", "--short", "HEAD")
    return f"{base_tag}-{distance}-g{short_hash}"


def get_latest_version_tag_describe() -> str:
    """Find the best version tag and build a describe string relative to it.

    Uses PEP 440 version ordering so that stable releases sort above
    pre-release tags (e.g., v0.5.10 > v0.5.10rc0).

    Compares the highest ancestor tag with the globally highest tag:
    - If the ancestor IS the global highest (or equivalent after stripping
      .postN), use the ancestor — we're on or past the latest release.
    - If the global tag is newer, use it via merge-base distance — this is
      the main-branch case where release tags live on release branches and
      we want ``pip install -e`` to show the upcoming release version.
    """
    global_tag = get_latest_version_tag()
    if not global_tag:
        print("WARNING: No version tags (v*.*.*) found in repo", file=sys.stderr)
        return ""

    ancestor_tag = get_latest_ancestor_version_tag()

    # Pick the tag to describe relative to.
    # Use the ancestor tag only when it represents the latest release
    # (i.e., no newer global tag exists). Otherwise use the global tag
    # so main shows the next dev version correctly.
    if ancestor_tag and parse_version_tuple(ancestor_tag) >= parse_version_tuple(
        global_tag
    ):
        tag = ancestor_tag
    else:
        tag = global_tag

    result = _describe_from_tag(tag)
    if result:
        return result

    print(
        f"WARNING: No common ancestor between {tag} and HEAD. "
        f"Is this a shallow clone? Try: git fetch --unshallow --tags",
        file=sys.stderr,
    )
    return ""


def get_version_describe() -> str:
    """Main entry point: resolve the version describe string."""
    # Prefer exact match — correct for both stable and pre-release tags
    exact = get_exact_version_tag()
    if exact:
        return exact

    # Fallback for untagged commits (e.g., dev install from main)
    return get_latest_version_tag_describe()


def get_latest_version_tag() -> str:
    """Return just the highest version tag (PEP 440 ordered), or empty string."""
    tags_raw = run_git("tag", "--list", "v*.*.*")
    if not tags_raw:
        return ""
    tag_list = sorted(tags_raw.splitlines(), key=parse_version_tuple, reverse=True)
    return tag_list[0] if tag_list else ""


def main() -> None:
    # --tag-only: print just the latest version tag (for CI scripts)
    tag_only = "--tag-only" in sys.argv
    if tag_only:
        result = get_latest_version_tag()
    else:
        result = get_version_describe()
    if not result:
        print(
            "ERROR: Could not determine version from git tags.\n"
            "Possible causes:\n"
            "  - No version tags (v*.*.*) exist: run 'git fetch --tags'\n"
            "  - Shallow clone without tags: run 'git fetch --unshallow --tags'\n"
            "  - Git safe.directory issue: run 'git config --global --add safe.directory <repo>'\n"
            "  - Not inside a git repository\n"
            "setuptools-scm will fall back to version 0.0.0.dev0",
            file=sys.stderr,
        )
        sys.exit(1)
    print(result)


if __name__ == "__main__":
    main()
