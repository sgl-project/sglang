"""
Update the CI permissions configuration file.

This script updates the `CI_PERMISSIONS.json` file, which defines the CI permissions granted to each user.

The format of `CI_PERMISSIONS.json` is as follows:

{
    "username1": {
        "can_tag_run_ci_label": true,
        "can_rerun_failed_ci": true,
        "cooldown_interval_minutes": 0,
        "reason": "top contributor"
    },
    "username2": {
        "can_tag_run_ci_label": true,
        "can_rerun_failed_ci": true,
        "cooldown_interval_minutes": 60,
        "reason": "custom override"
    }
}

Permissions are assigned according to the following rules:

1. Users with write, maintain, or admin access are never listed: write access
   grants every CI permission (see slash_command_handler.py).
2. Nobody is added automatically; maintainers add entries by hand.
3. Existing entries:
   - If the reason is anything other than "top contributor" or "custom override",
     the entry is a manual exception: keep it unchanged.
   - Otherwise, if the user has no commit on main in the last 90 days, drop it.
   - Otherwise keep it unchanged.
4. All other users receive no permissions and a 120-minute cooldown (they are omitted from the file).

`cooldown_interval_minutes` is not only a rate limit: 0 also authorizes
`/rerun-test` and `/rerun-group`, which never pass through `pr-gate.yml`.

Usage:
    export GH_TOKEN="your_github_token"
    python3 update_ci_permission.py

    # Sort-only mode (no network calls, no GH_TOKEN required)
    python3 update_ci_permission.py --sort-only

    # Report what the rules would change, without writing
    python3 update_ci_permission.py --check
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    requests = None  # Only needed for non-sort-only runs

# Configuration
REPO_OWNER = "sgl-project"
REPO_NAME = "sglang"
FILE_NAME = os.path.join(os.path.dirname(__file__), "CI_PERMISSIONS.json")
HEADERS = {}
WINDOW_DAYS = 90
WRITE_PERMISSIONS = ("admin", "maintain", "write")
# Reasons the old automatic rules wrote; an entry with any other reason is a manual
# exception and is kept as is.
GENERATED_REASONS = ("top contributor", "custom override")


def github_api_get(endpoint, params=None, allow_missing=False):
    """Helper to make paginated GitHub API requests; None on 404 if allow_missing."""
    if requests is None:
        raise RuntimeError(
            "The requests package is required. Install it or use --sort-only."
        )
    if not HEADERS:
        raise RuntimeError(
            "GitHub headers not initialized. Set GH_TOKEN or use --sort-only."
        )

    results = []
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/{endpoint}"

    while url:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 404 and allow_missing:
            return None
        if response.status_code != 200:
            # A partial commit list would mark active users inactive; fail instead.
            raise RuntimeError(f"GET {url}: {response.status_code} {response.text}")

        data = response.json()
        if isinstance(data, list):
            results.extend(data)
        else:
            return data  # Non-list response (not paginated usually)

        # Handle pagination
        url = None
        if "link" in response.headers:
            links = response.headers["link"].split(", ")
            for link in links:
                if 'rel="next"' in link:
                    url = link[link.find("<") + 1 : link.find(">")]
                    params = None  # Params are included in the next link
                    break
    return results


def has_write_access(user):
    """Whether the user has write, maintain, or admin access to the repo."""
    data = github_api_get(f"collaborators/{user}/permission", allow_missing=True)
    return isinstance(data, dict) and data.get("permission") in WRITE_PERMISSIONS


def get_active_authors():
    """Lowercased logins of commit authors on the default branch in the last WINDOW_DAYS days."""
    print(f"Fetching commits from the last {WINDOW_DAYS} days...")
    since_date = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    commits = github_api_get("commits", params={"since": since_date, "per_page": 100})
    # commit['author'] is the GitHub user object (None if not linked)
    authors = {
        commit["author"]["login"].lower()
        for commit in commits
        if commit.get("author") and "login" in commit["author"]
    }
    print(f"Found {len(authors)} commit authors in the last {WINDOW_DAYS} days.")
    return authors


def load_existing_permissions():
    if os.path.exists(FILE_NAME):
        try:
            with open(FILE_NAME, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {FILE_NAME} is invalid JSON. Starting fresh.")
    return {}


def save_permissions(permissions):
    with open(FILE_NAME, "w") as f:
        json.dump(dict(sorted(permissions.items())), f, indent=4)
        f.write("\n")


def sort_permissions_file():
    """Sort the existing CI permissions file alphabetically and exit."""
    if not os.path.exists(FILE_NAME):
        print(f"{FILE_NAME} not found. Nothing to sort.")
        return
    permissions = load_existing_permissions()
    save_permissions(permissions)
    print(f"Sorted {FILE_NAME}. Total users: {len(permissions)}")


def compute_permissions(old_permissions):
    """Apply the rules in the module docstring.

    Returns the new permissions and one note per dropped entry.
    """
    active = get_active_authors()
    new_permissions = {}
    notes = []

    for user, config in old_permissions.items():
        reason = config.get("reason", "")
        if has_write_access(user):
            notes.append(
                (user, "dropped: has write access, which grants every CI permission")
            )
        elif reason not in GENERATED_REASONS:
            new_permissions[user] = config
        elif user.lower() not in active:
            notes.append(
                (user, f"dropped: no commit on main in the last {WINDOW_DAYS} days")
            )
        else:
            new_permissions[user] = config

    return new_permissions, notes


def main():
    parser = argparse.ArgumentParser(description="Update or sort CI permissions.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sort-only",
        action="store_true",
        help="Only sort CI_PERMISSIONS.json alphabetically without fetching data.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Print a warning for each entry the rules would change; never writes.",
    )
    args = parser.parse_args()

    if args.sort_only:
        sort_permissions_file()
        return

    gh_token = os.getenv("GH_TOKEN")
    if not gh_token:
        raise ValueError("Error: GH_TOKEN environment variable is not set.")

    global HEADERS
    HEADERS = {
        "Authorization": f"Bearer {gh_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    new_permissions, notes = compute_permissions(load_existing_permissions())

    if args.check:
        for user, note in notes:
            print(f"::warning file={os.path.relpath(FILE_NAME)}::{user}: {note}")
        print(f"{len(notes)} entries would change; run this script to apply.")
        return

    for user, note in notes:
        print(f"{user}: {note}")
    save_permissions(new_permissions)
    print(f"Successfully updated {FILE_NAME}. Total users: {len(new_permissions)}")


if __name__ == "__main__":
    main()
