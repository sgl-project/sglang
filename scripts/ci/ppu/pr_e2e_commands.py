"""Trusted, persistent PPU command receipts and non-preempting CI reruns.

This module is stdlib-only. The write-capable entry point runs exclusively from
trusted default-branch code; PR jobs use only its read-only receipt resolution.
"""

import argparse
import hashlib
import io
import json
import os
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

FAMILIES = {"qwen", "deepseek", "glm", "kimi", "minimax"}
FORMATS = {
    "test-w4a8": {"w4a8-int8"},
    "test-w8a8": {"w8a8-int8"},
    "test-bf16": {"bf16"},
    "test-fp8": {"fp8", "fp8-channelwise"},
    "test-mxfp4": {"mxfp4-fp8"},
}
SELECTORS = {"test-" + family for family in FAMILIES} | set(FORMATS) | {"test-all"}
MODIFIERS = {"test-accuracy-full", "test-perf", "test-multinode", "test-pd"}
TOKENS = SELECTORS | MODIFIERS | {"test-functional"}
HANDLER = ".github/workflows/ppu-e2e-commands.yml"
PIPELINE = ".github/workflows/ci.yaml"
MARKER = "<!-- ppu-e2e-receipt:v1 -->\n"
USAGE = (
    "Use a new, standalone PR comment: "
    "`/run-ppu-e2e test-qwen test-w4a8 [test-perf ...]` or `/clear-ppu-e2e`. "
    "Each command replaces the persistent selection. `run-ci` remains required "
    "for hardware; active CI is never canceled by this command."
)
MAX_DOWNLOAD = 2 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def body_digest(body):
    return hashlib.sha256(body.encode()).hexdigest()


def parse_command(body):
    """Return canonical selectors, [] for clear, or None for unrelated text."""
    require(isinstance(body, str) and len(body) <= 65536, "Invalid comment body")
    text = body.strip()
    if not any(command in text for command in ("/run-ppu-e2e", "/clear-ppu-e2e")):
        return None
    require("\n" not in text and "\r" not in text, USAGE)
    parts = text.split()
    require(parts and parts[0] in {"/run-ppu-e2e", "/clear-ppu-e2e"}, USAGE)
    if parts[0] == "/clear-ppu-e2e":
        require(len(parts) == 1, "Clear accepts no arguments. " + USAGE)
        return []
    require(len(parts) > 1, "At least one selector is required. " + USAGE)
    require(set(parts[1:]) <= TOKENS, "Unknown E2E argument. " + USAGE)
    require(
        bool(set(parts[1:]) & (SELECTORS | {"test-functional"})),
        "Modifiers require a model/format selector. " + USAGE,
    )
    return sorted(set(parts[1:]))


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(urllib.parse.urlsplit(newurl).scheme == "https", "Unsafe API redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            urllib.parse.urlsplit(req.full_url).netloc
            != urllib.parse.urlsplit(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    def __init__(self):
        self.repository = os.environ["GITHUB_REPOSITORY"]
        self.base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        self.token = os.environ["GH_TOKEN"]
        self._runs = {}
        self._metadata = None

    def request(self, path, data=None, binary=False):
        require(
            not path.startswith("/") and ".." not in path.split("/"), "Unsafe API path"
        )
        request = urllib.request.Request(
            self.base + "/repos/" + self.repository + "/" + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.build_opener(SafeRedirect()).open(
            request, timeout=30
        ) as response:
            raw = response.read(MAX_DOWNLOAD + 1)
        require(len(raw) <= MAX_DOWNLOAD, "API response exceeds size limit")
        return raw if binary else (json.loads(raw) if raw else None)

    def get(self, path):
        return self.request(path)

    def post(self, path, data):
        return self.request(path, data)

    def pages(self, path, key=None):
        result = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            response = self.get(f"{path}{separator}per_page=100&page={page}")
            batch = response[key] if key else response
            require(isinstance(batch, list), "Invalid paginated API response")
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise ValueError("GitHub history exceeds bounded pagination limit")

    def events(self, number):
        return self.pages(f"issues/{number}/events")

    def comments(self, number):
        return self.pages(f"issues/{number}/comments")

    def metadata(self):
        if self._metadata is None:
            self._metadata = self.get("")
        return self._metadata

    def handler_run(self, run_id, attempt):
        key = (run_id, attempt)
        if key not in self._runs:
            self._runs[key] = self.get(f"actions/runs/{run_id}/attempts/{attempt}")
        return self._runs[key]


def maintainer(api, user):
    if (
        user.get("type") != "User"
        or not user.get("login")
        or type(user.get("id")) is not int
    ):
        return False
    permission = api.get(
        "collaborators/" + urllib.parse.quote(user["login"], safe="") + "/permission"
    )["permission"]
    return permission in {"write", "maintain", "admin"}


def command_request(api, comment):
    selectors = parse_command(comment["body"])
    require(selectors is not None, "Source comment is not an E2E command")
    require(
        comment["created_at"] == comment["updated_at"],
        "Post a new command instead of editing a comment",
    )
    require(
        maintainer(api, comment["user"]),
        "E2E commands require a human repository maintainer",
    )
    return {
        "comment_id": comment["id"],
        "author": {k: comment["user"][k] for k in ("id", "login", "type")},
        "body_digest": body_digest(comment["body"]),
        "created_at": comment["created_at"],
        "updated_at": comment["updated_at"],
        "selectors": selectors,
    }


def receipts(api, number, comments=None):
    comments = api.comments(number) if comments is None else comments
    result = []
    for comment in comments:
        if (
            comment.get("user", {}).get("login") != "github-actions[bot]"
            or comment["user"].get("type") != "Bot"
        ):
            continue
        body = comment.get("body", "")
        if not body.startswith(MARKER):
            continue
        require(len(body) < 32768, "Oversized E2E receipt")
        receipt = json.loads(body[len(MARKER) :])
        require(
            receipt.get("schema_version") == 1
            and receipt.get("repository") == api.repository
            and receipt.get("pr") == number
            and receipt.get("kind") in {"request", "rerun", "rerun-result"},
            "Malformed E2E receipt",
        )
        for key in ("handler_run_id", "handler_attempt"):
            require(
                type(receipt.get(key)) is int and receipt[key] > 0,
                "Invalid handler identity",
            )
        run = api.handler_run(receipt["handler_run_id"], receipt["handler_attempt"])
        require(
            run["path"] == HANDLER
            and run["event"] in {"issue_comment", "workflow_run"}
            and run["head_branch"] == api.metadata()["default_branch"]
            and run["head_repository"]["full_name"] == api.repository
            and run["run_attempt"] == receipt["handler_attempt"],
            "Receipt did not originate from the trusted default-branch handler",
        )
        # GitHub owns these timestamps; a forged reference to a past handler run
        # cannot authenticate a receipt created outside that attempt's lifetime.
        created = datetime.fromisoformat(comment["created_at"].replace("Z", "+00:00"))
        started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
        require(created >= started, "Receipt predates its handler attempt")
        if run["status"] == "completed":
            ended = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
            require(created <= ended, "Receipt postdates its handler attempt")
        require(comment["created_at"] == comment["updated_at"], "Edited E2E receipt")
        result.append({**receipt, "receipt_id": comment["id"]})
    return result


def resolve_request(api, number, request=None):
    """Resolve latest accepted request, or validate an in-flight pinned request."""
    comments = api.comments(number)
    accepted = [r for r in receipts(api, number, comments) if r["kind"] == "request"]
    if request is None:
        if not accepted:
            return None
        latest_id = max(r["request"]["comment_id"] for r in accepted)
        matches = [r for r in accepted if r["request"]["comment_id"] == latest_id]
    else:
        matches = [r for r in accepted if r["receipt_id"] == request["receipt_id"]]
    require(bool(matches), "Missing accepted E2E request")
    require(
        all(r["request"] == matches[0]["request"] for r in matches),
        "Conflicting E2E receipts",
    )
    chosen = min(matches, key=lambda r: r["receipt_id"])
    source = next(
        (c for c in comments if c["id"] == chosen["request"]["comment_id"]), None
    )
    require(
        source is not None, "Accepted command was deleted; post a new run/clear command"
    )
    expected = command_request(api, source)
    require(
        expected == chosen["request"],
        "Accepted command changed; post a new run/clear command",
    )
    resolved = {**expected, "receipt_id": chosen["receipt_id"]}
    if request is not None:
        require(
            resolved == request, "Pinned E2E request differs from its accepted receipt"
        )
    return resolved


def write_receipt(api, number, kind, **fields):
    envelope = {
        "schema_version": 1,
        "repository": api.repository,
        "pr": number,
        "handler_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "handler_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "kind": kind,
        **fields,
    }
    return api.post(
        f"issues/{number}/comments",
        {"body": MARKER + json.dumps(envelope, sort_keys=True)},
    )


def notice(api, number, text):
    api.post(f"issues/{number}/comments", {"body": "PPU E2E: " + text})


def associated_run(api, run, pr):
    if not (
        run.get("path") == PIPELINE
        and run.get("event") == "pull_request"
        and run.get("head_sha") == pr["head"]["sha"]
        and run.get("head_branch") == pr["head"]["ref"]
        and pr["base"]["repo"]["full_name"] == api.repository
        and run.get("head_repository", {}).get("full_name")
        == pr["head"]["repo"]["full_name"]
    ):
        return False
    candidates = run.get("pull_requests", [])
    if not candidates:
        candidates = api.pages(f"commits/{run['head_sha']}/pulls")
    return any(p["number"] == pr["number"] for p in candidates)


def attempt_plan(api, run):
    name = f"ppu-e2e-plan-{run['id']}-{run['run_attempt']}"
    artifacts = api.pages(
        f"actions/runs/{run['id']}/artifacts?name={name}", "artifacts"
    )
    matches = [a for a in artifacts if a["name"] == name and not a["expired"]]
    if not matches:
        return None
    require(len(matches) == 1, "Ambiguous selection artifact")
    raw = api.request(f"actions/artifacts/{matches[0]['id']}/zip", binary=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(
            archive.namelist() == ["selection.json"],
            "Unexpected selection artifact inventory",
        )
        require(
            archive.getinfo("selection.json").file_size <= MAX_DOWNLOAD,
            "Oversized selection plan",
        )
        plan = json.loads(archive.read("selection.json"))
    require(
        str(plan.get("run_id")) == str(run["id"])
        and str(plan.get("attempt")) == str(run["run_attempt"]),
        "Selection artifact belongs to a different attempt",
    )
    return plan


def schedule(api, pr, request):
    """At most one rerun submission per accepted request and current head."""
    number = pr["number"]
    if pr["state"] != "open" or pr["draft"]:
        return "Selection saved; an open, non-draft PR is required."
    if request["selectors"] and "run-ci" not in {
        label["name"] for label in pr["labels"]
    }:
        return "Selection saved; waiting for maintainer-applied `run-ci` authorization."
    runs = api.pages(
        f"actions/workflows/ci.yaml/runs?event=pull_request&head_sha={pr['head']['sha']}",
        "workflow_runs",
    )
    runs = sorted(
        (r for r in runs if associated_run(api, r, pr)),
        key=lambda r: r["id"],
        reverse=True,
    )
    if any(r["status"] != "completed" for r in runs):
        return "Selection saved; active/queued CI will finish before the latest request is rerun."
    if runs and runs[0].get("conclusion") == "action_required":
        return "Selection saved; GitHub Actions approval is still required."
    previous = receipts(api, number)
    if any(
        r["kind"] == "rerun"
        and r["request_id"] == request["comment_id"]
        and r["head_sha"] == pr["head"]["sha"]
        for r in previous
    ):
        return "This request already has a rerun submission; post a new command to retry a failed submission."
    candidate = None
    for run in runs:
        if run.get("conclusion") == "action_required":
            continue
        plan = attempt_plan(api, run)
        if plan is None:
            continue
        if (
            plan.get("pr") != number
            or plan.get("head_sha") != pr["head"]["sha"]
            or plan.get("base_sha") != pr["base"]["sha"]
        ):
            continue
        if (
            plan.get("request") == request
            and plan.get("state") == "open"
            and plan.get("draft") is False
            and (not request["selectors"] or "run-ci" in plan.get("labels", []))
        ):
            return (
                "Current-revision CI already consumed this request; no duplicate rerun."
            )
        if candidate is None:
            candidate = run
    if candidate is None:
        return "Selection saved; no eligible current-revision CI run exists. Push a new commit to start PR CI."
    created = datetime.fromisoformat(candidate["created_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) - created >= timedelta(days=30):
        return "Selection saved; the CI run is too old to rerun. Push a new commit."
    live = api.get(f"pulls/{number}")
    require(
        live["head"]["sha"] == pr["head"]["sha"]
        and live["base"]["sha"] == pr["base"]["sha"]
        and live["state"] == "open"
        and not live["draft"],
        "PR changed while scheduling; wait for its current-revision pipeline",
    )
    require(
        not request["selectors"]
        or "run-ci" in {label["name"] for label in live["labels"]},
        "run-ci authorization was removed while scheduling",
    )
    require(
        resolve_request(api, number) == request, "A newer command superseded scheduling"
    )
    current = api.get(f"actions/runs/{candidate['id']}")
    require(
        associated_run(api, current, live)
        and current["status"] == "completed"
        and current.get("conclusion") != "action_required"
        and current["run_attempt"] == candidate["run_attempt"],
        "CI changed or is already being rerun",
    )
    latest_runs = api.pages(
        f"actions/workflows/ci.yaml/runs?event=pull_request&head_sha={live['head']['sha']}",
        "workflow_runs",
    )
    require(
        not any(
            associated_run(api, run, live)
            and (
                run["status"] != "completed" or run["id"] not in {r["id"] for r in runs}
            )
            for run in latest_runs
        ),
        "A newer CI run appeared while scheduling; wait for its completion",
    )
    fields = {
        "request_id": request["comment_id"],
        "head_sha": pr["head"]["sha"],
        "base_sha": pr["base"]["sha"],
        "target_run_id": candidate["id"],
        "target_attempt": candidate["run_attempt"] + 1,
    }
    # Persist intent BEFORE the API call. An ambiguous network failure must not
    # automatically submit again and form a workflow_run rerun loop.
    write_receipt(api, number, "rerun", **fields)
    try:
        api.post(f"actions/runs/{candidate['id']}/rerun", {})
    except Exception:
        write_receipt(api, number, "rerun-result", **fields, accepted=False)
        raise
    write_receipt(api, number, "rerun-result", **fields, accepted=True)
    return f"Requested **rerun all jobs** for CI run {candidate['id']} (attempt {fields['target_attempt']})."


def reconcile(api, number, source_id=None):
    pr = api.get(f"pulls/{number}")
    require(
        pr["base"]["repo"]["full_name"] == api.repository,
        "PR belongs to a different repository",
    )
    comments = api.comments(number)
    accepted = [r for r in receipts(api, number, comments) if r["kind"] == "request"]
    cursor = max((r["request"]["comment_id"] for r in accepted), default=0)
    changed = False
    for comment in sorted(comments, key=lambda c: c["id"]):
        if comment["id"] <= cursor or comment.get("user", {}).get("type") != "User":
            continue
        try:
            tokens = parse_command(comment.get("body", ""))
            if tokens is None:
                continue
            request = command_request(api, comment)
        except ValueError as error:
            if comment["id"] == source_id:
                notice(api, number, str(error))
            continue
        write_receipt(api, number, "request", request=request)
        changed = True
    request = resolve_request(api, number)
    if request is None:
        return
    message = schedule(api, pr, request)
    print(message)
    if changed or source_id is not None or message.startswith("Requested **"):
        notice(api, number, message)


def resolve_event(api, event, event_name):
    require(
        event.get("repository", {}).get("full_name") == api.repository,
        "Wrong event repository",
    )
    if event_name == "issue_comment":
        require(
            event.get("action") == "created" and event["issue"].get("pull_request"),
            "Only new PR comments are accepted",
        )
        return event["issue"]["number"]
    require(
        event_name == "workflow_run" and event.get("action") == "completed",
        "Unexpected handler event",
    )
    run = api.get(f"actions/runs/{int(event['workflow_run']['id'])}")
    require(
        run["path"] == PIPELINE and run["event"] == "pull_request",
        "Not a PR CI completion",
    )
    candidates = run.get("pull_requests", [])
    if not candidates:
        candidates = api.pages(f"commits/{run['head_sha']}/pulls")
    numbers = []
    for candidate in candidates:
        pr = api.get(f"pulls/{candidate['number']}")
        if (
            pr["state"] == "open"
            and pr["head"]["sha"] == run["head_sha"]
            and pr["base"]["repo"]["full_name"] == api.repository
        ):
            if (
                pr["head"]["repo"]["full_name"] == run["head_repository"]["full_name"]
                and pr["head"]["ref"] == run["head_branch"]
            ):
                numbers.append(pr["number"])
    require(len(set(numbers)) <= 1, "Ambiguous PR CI completion")
    return numbers[0] if numbers else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["resolve", "reconcile"])
    args = parser.parse_args()
    api = GitHub()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    number = resolve_event(api, event, os.environ["GITHUB_EVENT_NAME"])
    if args.mode == "resolve":
        if number is not None:
            with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
                stream.write(f"pr={int(number)}\n")
        return
    require(
        number == int(os.environ["PR_NUMBER"]),
        "PR changed between resolve and scheduling",
    )
    source_id = (
        event["comment"]["id"]
        if os.environ["GITHUB_EVENT_NAME"] == "issue_comment"
        else None
    )
    reconcile(api, number, source_id)


if __name__ == "__main__":
    main()
