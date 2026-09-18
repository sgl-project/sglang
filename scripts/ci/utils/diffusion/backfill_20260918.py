"""One-time recovery of original nightly artifacts; not intended for main."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from generate_diffusion_dashboard import MAX_HISTORY_RUNS, generate_dashboard
from publish_comparison_results import (
    BRANCH,
    REPO_NAME,
    REPO_OWNER,
    STORAGE_PREFIX,
    _collect_chart_files,
    create_blobs,
    create_commit,
    create_tree,
    get_branch_sha,
    get_tree_sha,
    make_github_request,
    update_branch_ref,
    verify_token_permissions,
)

RUNS = (
    ("34761767207", "2026-09-13", "7f1f8c706ac000b7a84ea0bda05135fc4177c6ca"),
    ("34979805212", "2026-09-15", "832ec39cc0324cb0e7823dc8385e27a30c356bdd"),
    ("35231805157", "2026-09-17", "7ccbf5fd04f7ee23095fc38e49e749d58dc18282"),
)


def main():
    token = os.environ["GH_PAT_FOR_NIGHTLY_CI_DATA"]
    if verify_token_permissions(REPO_OWNER, REPO_NAME, token) is not True:
        raise RuntimeError("New nightly token failed repository access verification")

    runs = {}
    result_bytes = {}
    for run_id, date, sha in RUNS:
        body = Path(f"backfill/{run_id}/comparison-results.json").read_bytes()
        result = json.loads(body)
        if (
            str(result["run_id"]) != run_id
            or result["commit_sha"] != sha
            or result["timestamp"][:10] != date
            or len(result["results"]) != 12
        ):
            raise ValueError(
                f"Unexpected artifact provenance or result count: {run_id}"
            )
        name = f"{date}_{run_id}.json"
        runs[name] = result
        result_bytes[name] = body
        print(f"Original artifact: {name}, sha256={hashlib.sha256(body).hexdigest()}")

    base_sha = get_branch_sha(REPO_OWNER, REPO_NAME, BRANCH, token)
    api = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
    listing = json.loads(
        make_github_request(f"{api}/contents/{STORAGE_PREFIX}?ref={base_sha}", token)
    )
    existing = {
        entry["name"]: entry
        for entry in listing
        if entry["type"] == "file"
        and entry["name"].endswith(".json")
        and entry["name"] != "index.json"
    }
    for name, body in result_bytes.items():
        if name in existing:
            blob_sha = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            if existing[name]["sha"] != blob_sha:
                raise ValueError(f"Refusing to overwrite a different result: {name}")

    names = sorted(existing.keys() | runs.keys(), reverse=True)
    # Keep a newer dashboard if another nightly has published in the meantime.
    for name in names[: MAX_HISTORY_RUNS + 1]:
        if name not in runs:
            raw_url = (
                f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}"
                f"/{base_sha}/{STORAGE_PREFIX}/{name}"
            )
            runs[name] = json.loads(make_github_request(raw_url, token))
    current = runs[names[0]]
    history = [runs[name] for name in names[1 : MAX_HISTORY_RUNS + 1]]
    markdown, alerts = generate_dashboard(
        current, history, charts_dir="backfill/charts"
    )
    if not list(Path("backfill/charts").glob("*.png")):
        raise RuntimeError("Dashboard charts were not generated")

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prefix": STORAGE_PREFIX,
        "runs": names[:90],
    }
    files = [(f"{STORAGE_PREFIX}/{name}", body) for name, body in result_bytes.items()]
    files.extend(
        [
            (
                f"{STORAGE_PREFIX}/index.json",
                (json.dumps(index, indent=2) + "\n").encode(),
            ),
            (f"{STORAGE_PREFIX}/dashboard.md", markdown.encode()),
        ]
    )
    files.extend(_collect_chart_files("backfill/charts"))
    blobs = create_blobs(REPO_OWNER, REPO_NAME, files, token)
    tree = create_tree(
        REPO_OWNER,
        REPO_NAME,
        get_tree_sha(REPO_OWNER, REPO_NAME, base_sha, token),
        blobs,
        token,
    )
    commit = create_commit(
        REPO_OWNER,
        REPO_NAME,
        tree,
        base_sha,
        "Backfill diffusion nightly results for 2026-09-13, 15, and 17",
        token,
    )
    update_branch_ref(REPO_OWNER, REPO_NAME, BRANCH, commit, token)

    for path, body in files:
        published = json.loads(
            make_github_request(f"{api}/contents/{path}?ref={commit}", token)
        )
        expected_sha = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        if published["sha"] != expected_sha:
            raise RuntimeError(f"Published bytes differ: {path}")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(f"Restored original artifacts in commit `{commit}`.\n\n")
        summary.write(markdown)
    print(f"Verified {len(files)} files at commit {commit}")
    print(f"Dashboard alerts: {len(alerts)}; no retrospective issues created")


if __name__ == "__main__":
    main()
