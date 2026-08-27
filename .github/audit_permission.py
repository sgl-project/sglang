"""
Audit GitHub repository collaborators with elevated access.

This script will:
1. Fetch all collaborators with write permission to this repo.
2. Show their github username, Nickname and the role (e.g., admin, maintain,
   custom org role, write, triage).
3. Show their last activity related to this repo (last commit, last issue,
   last pull request). Put the data in YYYY-MM-DD format. Add a column "last activity date" to the CSV, before the above three breakdown columns.
4. Show activity on other repos: repos touched via public events in the last 90 days (Push, PR, Issues, etc.). Sort the repos by the number of activities.
5. Write results to a CSV sorted by the roles (admin, maintain, custom org role, write, triage) and the last activity date (most recent first).

Usage:
    export GH_TOKEN="your_github_token"
    python3 audit_permission.py [--output path] [--repo owner/name]

Requires: requests, and a token with permission to list collaborators (push+
access to the repo).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore

DEFAULT_OWNER = "sgl-project"
DEFAULT_NAME = "sglang"

HEADERS: dict[str, str] = {}


def _request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> requests.Response:
    if requests is None:
        raise RuntimeError("Install the requests package: pip install requests")
    for attempt in range(max_retries):
        r = requests.request(method, url, headers=HEADERS, params=params, timeout=60)
        if r.status_code == 403 and "rate limit" in (r.text or "").lower():
            reset = r.headers.get("X-RateLimit-Reset")
            wait = 60
            if reset:
                try:
                    wait = max(1, int(reset) - int(time.time()) + 2)
                except ValueError:
                    pass
            print(f"Rate limited; sleeping {wait}s...", file=sys.stderr)
            time.sleep(min(wait, 3600))
            continue
        return r
    return r


def paginate_list(url: str, params: dict[str, Any] | None = None) -> list[Any]:
    out: list[Any] = []
    next_url: str | None = url
    next_params = params
    while next_url:
        r = _request("GET", next_url, params=next_params)
        next_params = None
        if r.status_code != 200:
            print(
                f"Error {r.status_code} GET {next_url}: {r.text[:500]}",
                file=sys.stderr,
            )
            break
        data = r.json()
        if isinstance(data, list):
            out.extend(data)
        else:
            break
        next_url = None
        link = r.headers.get("Link", "")
        for part in link.split(", "):
            if 'rel="next"' in part:
                start = part.find("<") + 1
                end = part.find(">")
                if start > 0 and end > start:
                    next_url = part[start:end]
                break
    return out


def collaborator_role(collab: dict[str, Any]) -> str:
    role_name = collab.get("role_name")
    if isinstance(role_name, str) and role_name.strip():
        return role_name.strip()
    perms = collab.get("permissions") or {}
    if perms.get("admin"):
        return "admin"
    if perms.get("maintain"):
        return "maintain"
    if perms.get("push"):
        return "write"
    if perms.get("triage"):
        return "triage"
    return "read"


def has_write_plus(collab: dict[str, Any]) -> bool:
    perms = collab.get("permissions") or {}
    return bool(
        perms.get("admin")
        or perms.get("maintain")
        or perms.get("push")
        or perms.get("triage")
    )


def role_sort_tier(collab: dict[str, Any]) -> int:
    """Sort order: admin (0), maintain (1), custom org role (2), write (3), triage (4)."""
    rn = collab.get("role_name")
    if isinstance(rn, str) and rn.strip():
        k = rn.strip().lower()
        if k == "admin":
            return 0
        if k == "maintain":
            return 1
        if k == "write":
            return 3
        if k == "triage":
            return 4
        if k == "read":
            return 5
        return 2
    perms = collab.get("permissions") or {}
    if perms.get("admin"):
        return 0
    if perms.get("maintain"):
        return 1
    if perms.get("push"):
        return 3
    if perms.get("triage"):
        return 4
    return 5


def fetch_display_name(login: str) -> str:
    url = f"https://api.github.com/users/{login}"
    r = _request("GET", url)
    if r.status_code != 200:
        return ""
    data = r.json()
    if not isinstance(data, dict):
        return ""
    n = data.get("name")
    return n.strip() if isinstance(n, str) else ""


def parse_github_ts(s: str) -> datetime | None:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def iso_timestamp_to_ymd(iso: str | None) -> str:
    if not iso:
        return ""
    p = parse_github_ts(iso)
    if not p:
        return ""
    return p.date().isoformat()


def max_date_ymd(*iso_dates: str | None) -> str:
    best: datetime | None = None
    for d in iso_dates:
        p = parse_github_ts(d or "")
        if p and (best is None or p > best):
            best = p
    return best.date().isoformat() if best else ""


def parse_ymd(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def last_commit_date(owner: str, repo: str, login: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    r = _request("GET", url, params={"author": login, "per_page": 1})
    if r.status_code != 200:
        return None
    data = r.json()
    if not isinstance(data, list) or not data:
        return None
    commit = data[0].get("commit") or {}
    c = commit.get("committer") or commit.get("author") or {}
    d = c.get("date")
    return d if isinstance(d, str) else None


def search_repo_item(
    owner: str, repo: str, login: str, kind: str
) -> dict[str, Any] | None:
    q = f"repo:{owner}/{repo} is:{kind} author:{login}"
    url = "https://api.github.com/search/issues"
    r = _request(
        "GET",
        url,
        params={"q": q, "sort": "updated", "order": "desc", "per_page": 1},
    )
    if r.status_code != 200:
        return None
    payload = r.json()
    items = payload.get("items")
    if not items:
        return None
    return items[0] if isinstance(items[0], dict) else None


def last_issue_pr_dates(
    owner: str, repo: str, login: str
) -> tuple[str | None, str | None]:
    issue = search_repo_item(owner, repo, login, "issue")
    pr = search_repo_item(owner, repo, login, "pr")
    issue_dt = None
    pr_dt = None
    if issue:
        issue_dt = issue.get("updated_at") or issue.get("created_at")
        if not isinstance(issue_dt, str):
            issue_dt = None
    if pr:
        pr_dt = pr.get("updated_at") or pr.get("created_at")
        if not isinstance(pr_dt, str):
            pr_dt = None
    return issue_dt, pr_dt


def other_repos_activity_column(
    login: str, owner: str, repo: str, days: int = 90
) -> str:
    """Repos other than this one touched in the window, sorted by event count (desc)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    full = f"{owner}/{repo}"
    counts: Counter[str] = Counter()
    url: str | None = f"https://api.github.com/users/{login}/events/public"
    params: dict[str, Any] = {"per_page": 100}

    while url:
        r = _request("GET", url, params=params)
        params = {}
        if r.status_code != 200:
            break
        events = r.json()
        if not isinstance(events, list):
            break
        oldest_in_page: datetime | None = None
        for ev in events:
            if not isinstance(ev, dict):
                continue
            created = parse_github_ts(ev.get("created_at") or "")
            if created:
                if oldest_in_page is None or created < oldest_in_page:
                    oldest_in_page = created
            if created and created < cutoff:
                continue
            rinfo = ev.get("repo")
            name = None
            if isinstance(rinfo, dict):
                name = rinfo.get("name")
            if isinstance(name, str) and name and name != full:
                counts[name] += 1
        next_url = None
        link = r.headers.get("Link", "")
        for part in link.split(", "):
            if 'rel="next"' in part:
                s, e = part.find("<") + 1, part.find(">")
                if s > 0 and e > s:
                    next_url = part[s:e]
                break
        if oldest_in_page and oldest_in_page < cutoff:
            break
        url = next_url
        if not events:
            break

    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return ";".join(f"{n}:{c}" for n, c in ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit repo collaborator permissions.")
    parser.add_argument(
        "--repo",
        default=f"{DEFAULT_OWNER}/{DEFAULT_NAME}",
        help=f"owner/name (default: {DEFAULT_OWNER}/{DEFAULT_NAME})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(os.path.dirname(__file__), "permission_audit.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--events-days",
        type=int,
        default=90,
        help="Window for other-repo activity via public events",
    )
    args = parser.parse_args()

    if "/" not in args.repo:
        print("Error: --repo must be owner/name", file=sys.stderr)
        sys.exit(1)
    owner, name = args.repo.split("/", 1)

    gh_token = os.getenv("GH_TOKEN")
    if not gh_token:
        print("Error: GH_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    global HEADERS
    HEADERS = {
        "Authorization": f"Bearer {gh_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    collab_url = f"https://api.github.com/repos/{owner}/{name}/collaborators"
    print(f"Fetching collaborators for {owner}/{name}...", file=sys.stderr)
    collaborators = paginate_list(
        collab_url, params={"per_page": 100, "affiliation": "all"}
    )

    rows: list[dict[str, Any]] = []
    elevated = [c for c in collaborators if isinstance(c, dict) and has_write_plus(c)]
    print(
        f"Found {len(elevated)} collaborators with admin/maintain/write/triage.",
        file=sys.stderr,
    )

    for i, col in enumerate(elevated, start=1):
        login = col.get("login")
        if not isinstance(login, str):
            continue
        print(f"  [{i}/{len(elevated)}] {login}", file=sys.stderr)

        role = collaborator_role(col)
        nickname = fetch_display_name(login)
        cd = last_commit_date(owner, name, login)
        issue_dt, pr_dt = last_issue_pr_dates(owner, name, login)
        last_act_ymd = max_date_ymd(cd, issue_dt, pr_dt)
        others = other_repos_activity_column(login, owner, name, days=args.events_days)
        rows.append(
            {
                "_role_tier": role_sort_tier(col),
                "github_username": login,
                "nickname": nickname,
                "role": role,
                "last_activity_date": last_act_ymd,
                "last_commit_date": iso_timestamp_to_ymd(cd),
                "last_issue_date": iso_timestamp_to_ymd(issue_dt),
                "last_pr_date": iso_timestamp_to_ymd(pr_dt),
                "other_repos_90d": others,
            }
        )

    def sort_key(r: dict[str, Any]) -> tuple[int, float]:
        tier = r["_role_tier"]
        act = parse_ymd(r.get("last_activity_date") or "")
        ts = act.timestamp() if act else 0.0
        return (tier, -ts)

    rows.sort(key=sort_key)

    fieldnames = [
        "github_username",
        "nickname",
        "role",
        "last_activity_date",
        "last_commit_date",
        "last_issue_date",
        "last_pr_date",
        "other_repos_90d",
    ]
    for r in rows:
        del r["_role_tier"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
