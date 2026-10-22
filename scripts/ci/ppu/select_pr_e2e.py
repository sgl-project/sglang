"""Deterministic, hardware-free PPU E2E selection and live PR authorization."""

import argparse
import ast
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "scripts/ci/ppu/pr_e2e_catalog.json"
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
LABELS = SELECTORS | MODIFIERS | {"test-functional"}
RELEVANT_LABELS = LABELS | {"run-ci"}
ENVIRONMENT = {
    "SGLANG_DEFAULT_THINKING": {"0", "1"},
    "SGLANG_DSV4_REASONING_EFFORT": {"high"},
    "SGLANG_DSV4_FP4_EXPERTS": {"0", "1"},
    "SGLANG_OPT_USE_TOPK_V2": {"0", "1"},
    "SGLANG_OPT_FLASHMLA_SPARSE_PREFILL": {"0", "1"},
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def repo_path(value, root=ROOT):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid repository-relative path: {value}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError(f"Missing/escaping repository file: {value}")
    return result


def registered_files(root=ROOT):
    result = {}
    for path in (root / "test/registered").rglob("*.py"):
        if path.name in {"__init__.py", "conftest.py"}:
            continue
        for statement in ast.parse(path.read_text()).body:
            node = statement.value if isinstance(statement, ast.Expr) else None
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_ppu_ci"
            ):
                continue
            values = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            if values.get("nightly") is True and not values.get("disabled"):
                result.setdefault(values.get("suite"), []).append(
                    str(path.relative_to(root))
                )
    return {key: sorted(value) for key, value in result.items()}


def load_catalog(path=CATALOG, root=ROOT):
    catalog = read_json(path)
    if catalog.get("schema_version") != 1 or not catalog.get("entries"):
        raise ValueError("Invalid/empty E2E catalog")
    registrations = registered_files(root)
    seen = set()
    identities = set()
    by_id = {}
    for entry in catalog["entries"]:
        name = entry["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,119}", name) or name in seen:
            raise ValueError(f"Invalid/duplicate entry ID: {name}")
        seen.add(name)
        by_id[name] = entry
        if entry["family"] not in FAMILIES or entry["format"] not in set().union(
            *FORMATS.values()
        ):
            raise ValueError(f"Unknown family/format: {name}")
        if entry["category"] not in {"answer", "accuracy", "perf", "pd"}:
            raise ValueError(f"Unknown adapter: {name}")
        modes = (
            {"smoke", "full"}
            if entry["category"] == "accuracy"
            else {entry["category"]}
        )
        if entry["mode"] not in modes or entry["board"] != "ZW-M890P":
            raise ValueError(f"Invalid mode/board: {name}")
        if entry["nodes"] not in (1, 2, 4) or entry["cards_per_node"] != 8:
            raise ValueError(f"Invalid board allocation: {name}")
        if entry["category"] == "pd" and entry["nodes"] != 2:
            raise ValueError(f"PD requires a two-node gang: {name}")
        budget = [
            entry[k] for k in ("timeout_seconds", "worker_minutes", "job_minutes")
        ]
        if any(type(v) is not int or v <= 0 for v in budget):
            raise ValueError(f"Invalid budget: {name}")
        if budget[0] + 600 > budget[1] * 60 or budget[1] + 5 > budget[2]:
            raise ValueError(f"No setup/cleanup budget: {name}")
        for key, value in entry["environment"].items():
            if value not in ENVIRONMENT.get(key, set()):
                raise ValueError(f"Unsupported environment: {name}: {key}")
        if not entry["files"] or entry["files"] != registrations.get(entry["suite"]):
            raise ValueError(f"Nonempty registration/file mismatch: {name}")
        config = read_json(repo_path(entry["config"], root))
        if config["test_id"] != name:
            raise ValueError(f"Config identity mismatch: {name}")
        hardware = config["hardware"]
        nodes = hardware.get("nnodes", 1)
        if entry["category"] == "pd":
            nodes = sum(
                config["disaggregation"][key]
                for key in ("prefill_nodes", "decode_nodes")
            )
        if nodes != entry["nodes"]:
            raise ValueError(f"Config topology mismatch: {name}")
        visible = hardware["visible_devices"]
        if (
            not visible
            or len(visible) != len(set(visible))
            or any(type(d) is not int or not 0 <= d < 8 for d in visible)
        ):
            raise ValueError(f"Invalid visible devices: {name}")
        if not isinstance(entry["blocked"], str):
            raise ValueError(f"Missing readiness policy: {name}")
        model = config["model"]
        if not entry["blocked"] and (
            not Path(model["path"]).is_absolute()
            or Path(model["path"]).name != model["checkpoint_name"]
        ):
            raise ValueError(f"Invalid absolute checkpoint path: {name}")
        identity = (
            entry["suite"],
            entry["config"],
            entry["nodes"],
            digest(entry["environment"]),
        )
        if identity in identities:
            raise ValueError(f"Duplicate execution identity: {name}")
        identities.add(identity)
    for entry in catalog["entries"]:
        dependencies = entry["dependencies"]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("Duplicate dependencies")
        full = entry["category"] == "accuracy" and entry["mode"] == "full"
        if len(dependencies) != (1 if full else 0):
            raise ValueError(f"Invalid smoke dependency: {entry['id']}")
        for dependency in dependencies:
            smoke = by_id[dependency]
            config = read_json(root / entry["config"])
            smoke_config = read_json(root / smoke["config"])
            if (
                smoke["category"] != "accuracy"
                or smoke["mode"] != "smoke"
                or smoke_config["model"] != config["model"]
            ):
                raise ValueError(f"Wrong checkpoint smoke dependency: {entry['id']}")
    return catalog


def phase(entry):
    if entry["nodes"] > 1:
        return "distributed"
    if entry["mode"] == "full" or entry["category"] == "perf":
        return "extended"
    return "default"


def select(catalog, labels):
    labels = set(labels) & LABELS
    selectors = labels & SELECTORS
    errors = []
    if labels & MODIFIERS and not selectors:
        errors.append(
            "Coverage modifiers require a model/format selector (or test-all)."
        )
    if "test-functional" in labels:
        errors.append(catalog["functional_blocker"])
    candidates = [
        entry
        for entry in catalog["entries"]
        if (
            "test-all" in selectors
            or "test-" + entry["family"] in selectors
            or any(entry["format"] in FORMATS.get(label, set()) for label in selectors)
        )
    ]
    additions = {
        "test-accuracy-full": lambda e: e["category"] == "accuracy"
        and e["mode"] == "full",
        "test-perf": lambda e: e["category"] == "perf"
        and (e["nodes"] == 1 or "test-multinode" in labels),
        "test-multinode": lambda e: e["nodes"] > 1
        and (
            e["category"] == "answer"
            or e["category"] == "perf"
            and "test-perf" in labels
        ),
        "test-pd": lambda e: e["category"] == "pd",
    }
    selected = {
        e["id"]: e
        for e in candidates
        if e["nodes"] == 1 and (e["category"] == "answer" or e["mode"] == "smoke")
    }
    for modifier in sorted(labels & MODIFIERS):
        matches = [e for e in candidates if additions[modifier](e)]
        if not matches:
            errors.append(
                f"{modifier} matches no execution in the selected model/format union."
            )
        selected.update((e["id"], e) for e in matches)
    by_id = {e["id"]: e for e in catalog["entries"]}
    for entry in list(selected.values()):
        selected.update((name, by_id[name]) for name in entry["dependencies"])
    if selectors and not selected:
        errors.append("Selectors matched no execution.")
    for entry in selected.values():
        if entry["blocked"]:
            errors.append(f"{entry['id']}: {entry['blocked']}")
    return sorted(selected.values(), key=lambda e: e["id"]), sorted(errors)


class GitHub:
    def __init__(self):
        self.repository = os.environ["GITHUB_REPOSITORY"]
        self.base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        self.token = os.environ["GH_TOKEN"]

    def get(self, path):
        request = urllib.request.Request(
            self.base + "/repos/" + self.repository + "/" + path,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def events(self, number):
        result = []
        for page in range(1, 101):
            batch = self.get(f"issues/{number}/events?per_page=100&page={page}")
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise ValueError("Label history exceeds the bounded pagination limit")


def snapshot(pr):
    return {
        "pr": pr["number"],
        "head_sha": pr["head"]["sha"],
        "base_sha": pr["base"]["sha"],
        "labels": sorted({label["name"] for label in pr["labels"]} & RELEVANT_LABELS),
        "draft": pr["draft"],
        "state": pr["state"],
    }


def label_authorization(api, state):
    if not set(state["labels"]) & LABELS:
        return []
    active = {}
    for event in sorted(api.events(state["pr"]), key=lambda e: e["id"]):
        label = event.get("label", {}).get("name")
        if event.get("event") == "labeled":
            active[label] = event.get("actor")
        elif event.get("event") == "unlabeled":
            active.pop(label, None)
    errors = []
    permissions = {}
    for label in state["labels"]:
        actor = active.get(label) or {}
        login = actor.get("login")
        if actor.get("type") != "User" or not login:
            errors.append(
                f"{label}: active label must be applied manually by a maintainer."
            )
            continue
        if login not in permissions:
            permissions[login] = api.get(
                "collaborators/" + urllib.parse.quote(login, safe="") + "/permission"
            )["permission"]
        if permissions[login] not in {"write", "maintain", "admin"}:
            errors.append(
                f"{label}: labeler lacks repository write/maintain/admin permission."
            )
    return errors


def make_plan(catalog, state, tested_sha, run_id, attempt, root=ROOT):
    entries, errors = select(catalog, state["labels"])
    plan = {
        "schema_version": 1,
        **state,
        "tested_sha": tested_sha,
        "run_id": str(run_id),
        "attempt": str(attempt),
        "catalog_digest": digest(catalog),
        "errors": errors,
        "entries": [
            {**e, "config_digest": digest(read_json(root / e["config"]))}
            for e in entries
        ],
    }
    return seal(plan)


def seal(plan):
    plan = {k: v for k, v in plan.items() if k != "plan_hash"}
    return {**plan, "plan_hash": digest(plan)}


def validate_plan(plan, root=ROOT):
    if seal(plan) != plan or plan.get("errors"):
        raise ValueError("Invalid or blocked selection plan")
    expected = make_plan(
        load_catalog(root / "scripts/ci/ppu/pr_e2e_catalog.json", root),
        {
            k: plan[k]
            for k in ("pr", "head_sha", "base_sha", "labels", "draft", "state")
        },
        plan["tested_sha"],
        plan["run_id"],
        plan["attempt"],
        root,
    )
    if expected != plan:
        raise ValueError("Plan does not match the tested checkout/catalog/configs")
    for key, env in (
        ("tested_sha", "GITHUB_SHA"),
        ("run_id", "GITHUB_RUN_ID"),
        ("attempt", "GITHUB_RUN_ATTEMPT"),
    ):
        if os.environ.get(env) and str(plan[key]) != os.environ[env]:
            raise ValueError(f"Stale selection identity: {key}; use Re-run all jobs")


def check_live(plan, api, require_authorized=True):
    validate_plan(plan)
    current = snapshot(api.get(f"pulls/{plan['pr']}"))
    if current != {key: plan[key] for key in current}:
        raise ValueError(
            "PR revision, draft state, or selected labels changed; this plan is stale"
        )
    errors = label_authorization(api, current)
    if errors:
        raise ValueError("; ".join(errors))
    if require_authorized and (
        current["draft"]
        or current["state"] != "open"
        or "run-ci" not in current["labels"]
    ):
        raise ValueError("Non-draft open PR with maintainer-applied run-ci is required")


def output(name, value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, separators=(",", ":"))
    if "\n" in str(value):
        raise ValueError("Invalid multiline workflow output")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"{name}={value}\n")


def cpu_identity(plan):
    return (
        f"{plan['run_id']}:{plan['attempt']}:{plan['tested_sha']}:{plan['plan_hash']}"
    )


def check_cpu(plan, evidence_ok, identity):
    validate_plan(plan)
    if evidence_ok != "true" or identity != cpu_identity(plan):
        raise ValueError(
            "CPU evidence must match this run, attempt, revision, and plan; use Re-run all jobs"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["validate", "preview", "plan", "check", "matrix", "entry", "cpu"],
    )
    parser.add_argument("--plan", default="selection.json")
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--allow-unauthorized", action="store_true")
    args = parser.parse_args()
    if args.mode == "validate":
        print(f"Validated {len(load_catalog()['entries'])} E2E entries")
        return
    if args.mode == "preview":
        entries, errors = select(load_catalog(), args.labels)
        print(
            json.dumps(
                {"selected": [e["id"] for e in entries], "errors": errors}, indent=2
            )
        )
        raise SystemExit(bool(errors))
    if args.mode == "plan":
        if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
            raise ValueError("E2E plans require a pull_request event")
        event = read_json(os.environ["GITHUB_EVENT_PATH"])
        api = GitHub()
        state = snapshot(api.get(f"pulls/{event['pull_request']['number']}"))
        plan = make_plan(
            load_catalog(),
            state,
            os.environ["GITHUB_SHA"],
            os.environ["GITHUB_RUN_ID"],
            os.environ["GITHUB_RUN_ATTEMPT"],
        )
        # Older events do not get to authorize newer label sets or revisions.
        if snapshot(event["pull_request"]) != state:
            plan["errors"].append(
                "Event payload is stale; trigger a fresh relevant PR event."
            )
        plan["errors"].extend(label_authorization(api, state))
        plan = seal(plan)
        Path(args.plan).write_text(json.dumps(plan, indent=2) + "\n")
        output("requested", str(bool(set(state["labels"]) & LABELS)).lower())
        output("plan_hash", plan["plan_hash"])
        output("identity", f"{plan['run_id']}:{plan['attempt']}:{plan['tested_sha']}")
        summary = [
            "## PPU E2E selection",
            f"Labels: {', '.join(state['labels']) or '(none)'}",
            f"Selected executions: {len(plan['entries'])}",
            "Default: Answer + Accuracy smoke. Format/model selectors use union semantics.",
        ]
        summary += [
            f"- {e['id']} ({e['nodes']} x {e['cards_per_node']} PPUs; job ceiling {e['job_minutes']} min)"
            for e in plan["entries"]
        ]
        summary += ["ERROR: " + error for error in plan["errors"]]
        if "run-ci" not in state["labels"] and plan["entries"]:
            summary.append("Authorization required: no hardware will be allocated.")
        text = "\n\n".join(summary) + "\n"
        print(text)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
                stream.write(text)
        raise SystemExit(bool(plan["errors"]))
    plan = read_json(args.plan)
    validate_plan(plan)
    if os.environ.get("PLAN_HASH") and plan["plan_hash"] != os.environ["PLAN_HASH"]:
        raise ValueError("Caller plan hash mismatch")
    if args.mode == "check":
        check_live(plan, GitHub(), not args.allow_unauthorized)
    elif args.mode == "cpu":
        check_cpu(plan, os.environ["CPU_EVIDENCE_OK"], os.environ["CPU_IDENTITY"])
    elif args.mode == "entry":
        entries = [
            entry for entry in plan["entries"] if entry["id"] == os.environ["ENTRY"]
        ]
        if len(entries) != 1:
            raise ValueError("Matrix entry is not uniquely selected")
        entry = entries[0]
        for key in ("nodes", "job_minutes", "worker_minutes"):
            if str(entry[key]) != os.environ["ENTRY_" + key.upper()]:
                raise ValueError("Matrix resource/budget differs from reviewed plan")
        if phase(entry) != os.environ["ENTRY_PHASE"]:
            raise ValueError("Matrix phase differs from reviewed plan")
    else:
        for name in ("default", "extended", "distributed"):
            entries = [
                {
                    "id": e["id"],
                    "nodes": e["nodes"],
                    "job_minutes": e["job_minutes"],
                    "worker_minutes": e["worker_minutes"],
                }
                for e in plan["entries"]
                if phase(e) == name
            ]
            output(name, {"include": entries})
            output(name + "_selected", str(bool(entries)).lower())


if __name__ == "__main__":
    main()
