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

1. Add the top 50 contributors from the last 90 days with full permissions, no cooldown, and the reason "top contributor".
2. Load all users from the existing `CI_PERMISSIONS.json` file and update their entries as follows:
   - If a user is already covered by rule 1, skip that user.
   - If the old reason of a user is "top contributor" but they are not in the current top contributors list, change their configuration to:
       {
           "can_tag_run_ci_label": true,
           "can_rerun_failed_ci": true,
           "cooldown_interval_minutes": 60,
           "reason": "custom override"
       }
    - For all other cases, preserve the original configuration unchanged.
3. All other users receive no permissions and a 120-minute cooldown (they are omitted from the file).

Usage:
    export GH_TOKEN="your_github_token"
    python3 update_ci_permission.py

    # Sort-only mode (no network calls, no GH_TOKEN required)
    python3 update_ci_permission.py --sort-only
"""

import argparse
import json
import os
from collections import Counter
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


def github_api_get(endpoint, params=None):
    """Helper to make paginated GitHub API requests."""
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
        if response.status_code != 200:
            print(f"Error fetching {url}: {response.status_code} {response.text}")
            # If we fail to fetch, strictly return what we have or empty to avoid crashing logic
            break

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


def get_write_access_users():
    """Fetches users with push (write) or admin access."""
    print("Fetching collaborators with write access...")
    # Note: This endpoint usually requires admin rights on the token.
    collaborators = github_api_get("collaborators", params={"per_page": 100})

    writers = set()
    for col in collaborators:
        perms = col.get("permissions", {})
        # Check for admin, maintain, or push rights
        if perms.get("admin") or perms.get("maintain") or perms.get("push"):
            writers.add(col["login"])

    print(f"Found {len(writers)} users with write access.")
    return writers


def get_top_contributors(days=90, limit=50):
    """Fetches top contributors based on commit count in the last N days."""
    print(f"Fetching commits from the last {days} days...")
    since_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Fetch commits
    commits = github_api_get("commits", params={"since": since_date, "per_page": 100})

    author_counts = Counter()
    for commit in commits:
        # commit['author'] contains the GitHub user object (can be None if not linked)
        if commit.get("author") and "login" in commit["author"]:
            author_counts[commit["author"]["login"]] += 1

    top_users = [user for user, _ in author_counts.most_common(limit)]
    print(f"Found {len(top_users)} active contributors in the last {days} days.")
    return set(top_users)


def load_existing_permissions():
    if os.path.exists(FILE_NAME):
        try:
            with open(FILE_NAME, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {FILE_NAME} is invalid JSON. Starting fresh.")
    return {}


def sort_permissions_file():
    """Sort the existing CI permissions file alphabetically and exit."""
    if not os.path.exists(FILE_NAME):
        print(f"{FILE_NAME} not found. Nothing to sort.")
        return

    old_permissions = load_existing_permissions()
    sorted_permissions = dict(sorted(old_permissions.items()))

    with open(FILE_NAME, "w") as f:
        json.dump(sorted_permissions, f, indent=4)
        f.write("\n")

    print(f"Sorted {FILE_NAME}. Total users: {len(sorted_permissions)}")


def main():
    parser = argparse.ArgumentParser(description="Update or sort CI permissions.")
    parser.add_argument(
        "--sort-only",
        action="store_true",
        help="Only sort CI_PERMISSIONS.json alphabetically without fetching data.",
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

    # Gather Data
    try:
        write_access_users = get_write_access_users()
    except Exception as e:
        print(f"Warning: Could not fetch collaborators (check token scope). Error: {e}")
        write_access_users = set()

    top_contributors = get_top_contributors(days=90, limit=50)
    old_permissions = load_existing_permissions()

    new_permissions = {}

    # Rule 1: Add Top 50 Contributors
    for user in top_contributors:
        new_permissions[user] = {
            "can_tag_run_ci_label": True,
            "can_rerun_failed_ci": True,
            "cooldown_interval_minutes": 0,
            "reason": "top contributor",
        }

    # Rule 2: Process Existing Users (Merge Logic)
    for user, config in old_permissions.items():
        if user in new_permissions:
            # Already handled by Rule 1 or 2
            continue

        old_reason = config.get("reason", "")

        # If they fell off the top contributor list
        if old_reason in ["top contributor"]:
            new_permissions[user] = {
                "can_tag_run_ci_label": True,
                "can_rerun_failed_ci": True,
                "cooldown_interval_minutes": 60,
                "reason": "custom override",
            }
        else:
            # Preserve custom overrides
            new_permissions[user] = config

    # Save and Sort
    # Sorting keys for cleaner diffs
    sorted_permissions = dict(sorted(new_permissions.items()))

    with open(FILE_NAME, "w") as f:
        json.dump(sorted_permissions, f, indent=4)
        f.write("\n")  # Add trailing newline

    print(f"Successfully updated {FILE_NAME}. Total users: {len(sorted_permissions)}")


if __name__ == "__main__":
    main()
