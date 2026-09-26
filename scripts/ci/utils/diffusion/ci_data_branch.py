"""
Resolve which sgl-project/ci-data-diffusion branch a diffusion GT-gen run publishes to.

Consistency CI reads all GT from the one ci-data SHA pinned in
``python/sglang/multimodal_gen/test/test_utils.py``, so ci-data ``main`` must only
receive GT that every later pin bump from ``main`` may pick up. A run from sglang
``main`` publishes there. A run from any other ref must choose explicitly:

- ``main``: shared, linear history; use for routine new-model GT.
- ``isolated``: ``gt/<ref>`` forked from ``main``; use for experiments such as a
  torch upgrade, reachable only through the pin of the PR that produced it.
- ``gt/<name>``: that isolated branch, named explicitly.

Kept free of third-party imports so a CPU-only job can resolve the branch before
any GPU work starts.
"""

import argparse
import os
import re
import sys

MAIN_BRANCH = "main"
ISOLATED_CHOICE = "isolated"
ISOLATED_BRANCH_PREFIX = "gt/"


def _strip_ref(ref):
    ref = (ref or "").strip()
    return ref.removeprefix("refs/heads/")


def _sanitize_branch_name(name):
    """Rewrite ``name`` into a legal git branch name; may return ""."""
    name = re.sub(r"\.{2,}", ".", re.sub(r"[^A-Za-z0-9._/-]+", "-", name))
    components = []
    for component in name.split("/"):
        # git check-ref-format: no component starts with "." or ends with ".lock".
        component = component.lstrip(".")
        if component.endswith(".lock"):
            component = component.removesuffix(".lock") + "-lock"
        if component:
            components.append(component)
    return "/".join(components).strip(".-")


def _isolated_branch(source_ref):
    name = _sanitize_branch_name(source_ref.removeprefix("refs/"))
    if not name:
        raise SystemExit(f"cannot derive an isolated branch name from {source_ref!r}")
    return f"{ISOLATED_BRANCH_PREFIX}{name}"


def _is_legal_isolated_branch(choice):
    if not choice.startswith(ISOLATED_BRANCH_PREFIX):
        return False
    name = choice.removeprefix(ISOLATED_BRANCH_PREFIX)
    # One definition of a legal name: whatever the sanitizer leaves unchanged.
    return bool(name) and _sanitize_branch_name(name) == name


def resolve_publish_branch(choice, source_ref):
    """Return the ci-data branch to publish to, or exit if the run did not choose one."""
    choice = _strip_ref(choice)
    source_ref = _strip_ref(source_ref)
    if choice == ISOLATED_CHOICE:
        if source_ref in ("", MAIN_BRANCH):
            raise SystemExit(
                f"ci_data_branch={ISOLATED_CHOICE} needs a non-main source ref, "
                f"got {source_ref!r}"
            )
        return _isolated_branch(source_ref)
    if choice == MAIN_BRANCH or _is_legal_isolated_branch(choice):
        return choice
    if choice:
        # Rejects typos such as "isloated", which would otherwise fork a new branch,
        # and illegal names, which would only fail at publish after all GPU work.
        raise SystemExit(
            f"ci_data_branch must be {MAIN_BRANCH}, {ISOLATED_CHOICE}, or "
            f"{ISOLATED_BRANCH_PREFIX}<name> with a legal git branch name; got {choice!r}"
        )
    if source_ref == MAIN_BRANCH:
        return MAIN_BRANCH
    if not source_ref:
        raise SystemExit(
            "cannot tell which sglang ref this GT came from; set GT_SOURCE_REF "
            "or pass an explicit ci-data branch"
        )
    raise SystemExit(
        f"GT is generated from {source_ref!r}, not sglang {MAIN_BRANCH}, so the "
        "ci-data-diffusion branch must be chosen explicitly: "
        f"ci_data_branch={MAIN_BRANCH} (GT reaches every later pin bump from "
        f"{MAIN_BRANCH}; routine new-model GT) or ci_data_branch={ISOLATED_CHOICE} "
        f"(publishes to {_isolated_branch(source_ref)}, reachable only via this "
        "PR's pin; experiments such as a torch upgrade)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--choice", default=os.environ.get("CI_DATA_BRANCH", ""))
    parser.add_argument("--source-ref", default=os.environ.get("GT_SOURCE_REF", ""))
    args = parser.parse_args()

    branch = resolve_publish_branch(args.choice, args.source_ref)
    print(f"ci-data-diffusion branch: {branch}", file=sys.stderr)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"ci-data-branch={branch}\n")
    else:
        print(branch)


if __name__ == "__main__":
    main()
