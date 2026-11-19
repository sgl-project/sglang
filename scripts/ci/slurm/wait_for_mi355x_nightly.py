#!/usr/bin/env python3
"""Wait until the MI355X disagg nightly leaves the amd-sglang nodes free.

The PR workflow (pr-test-amd-mi355x-disagg.yml) shares the four pit2
`amd-sglang` nodes and the `linux-sglang-mi35x-di` runners with the nightly
(nightly-amd-mi355x-disagg.yml), which starts at 04:00 UTC and runs for
11-15 hours. PR legs only use the nodes while the nightly does not, so this
script blocks (on a GitHub-hosted runner, not the DI runners) until:

  * no nightly run is queued or in progress, and
  * the next nightly start is more than GUARD_MINUTES away, so a PR leg
    (capped at one hour of node time) is done before the nightly needs the
    nodes, and
  * if the nightly is due but GitHub has not created today's scheduled run
    yet (scheduled runs start several minutes late), we wait for it.

Writes `clear=true` to $GITHUB_OUTPUT once the nodes are free, or
`clear=false` when --budget-minutes runs out first, so the workflow can chain
several waits past the 6-hour limit of a hosted job.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

NIGHTLY_WORKFLOW = "nightly-amd-mi355x-disagg.yml"
# Keep in sync with the `schedule` cron in nightly-amd-mi355x-disagg.yml.
NIGHTLY_START_UTC = dt.time(4, 0)
# PR legs are capped at TIME_LIMIT=01:00:00 of node time; add slack for the
# runner picking the job up and salloc returning.
GUARD_MINUTES = 75
# How long after NIGHTLY_START_UTC a scheduled run may still be created.
SCHEDULE_DELAY_MINUTES = 45
POLL_SECONDS = 300


def nightly_runs(repo: str) -> list[dict]:
    out = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            NIGHTLY_WORKFLOW,
            "--limit",
            "20",
            "--json",
            "databaseId,status,createdAt,event",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def busy_reason(repo: str, now: dt.datetime) -> str | None:
    runs = nightly_runs(repo)
    active = [r for r in runs if r["status"] != "completed"]
    if active:
        ids = ", ".join(str(r["databaseId"]) for r in active)
        return f"nightly run(s) {ids} still queued or running"

    today_start = dt.datetime.combine(now.date(), NIGHTLY_START_UTC, dt.timezone.utc)
    minutes_to_start = (today_start - now).total_seconds() / 60
    if 0 < minutes_to_start <= GUARD_MINUTES:
        return f"nightly starts in {minutes_to_start:.0f} min"

    minutes_since_start = -minutes_to_start
    if 0 <= minutes_since_start < SCHEDULE_DELAY_MINUTES:
        started = any(
            r["event"] == "schedule"
            and dt.datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
            >= today_start
            for r in runs
        )
        if not started:
            return "today's nightly is due but not created yet"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--budget-minutes", type=float, required=True)
    args = parser.parse_args()

    deadline = time.monotonic() + args.budget_minutes * 60
    clear = False
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        try:
            reason = busy_reason(args.repo, now)
        except subprocess.CalledProcessError as e:
            # A transient API error must not let a PR leg take the nodes.
            reason = f"could not read nightly status ({e.stderr.strip()[:200]})"
        if reason is None:
            print(f"{now:%H:%M} UTC: MI355X disagg nodes are free of the nightly")
            clear = True
            break
        if time.monotonic() + POLL_SECONDS > deadline:
            print(f"{now:%H:%M} UTC: still waiting ({reason}); budget used up")
            break
        print(f"{now:%H:%M} UTC: waiting, {reason}", flush=True)
        time.sleep(POLL_SECONDS)

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"clear={'true' if clear else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
