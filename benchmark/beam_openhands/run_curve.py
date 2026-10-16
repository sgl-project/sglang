"""Run fresh-server Beam OpenHands points; single-node TP4 is the default.

For optional two-node profiles, both nodes need the same checkout path and local
checkpoints. This driver never installs or downloads weights.
"""

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from curve_config import (
    DEFAULT_CONCURRENCIES,
    DRAFT_REVISION,
    ENGINE_COMMIT,
    MODEL_REVISION,
    inside_repo,
    profile,
)
from guarded_node import save, violations

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def command(settings, rank, argv, **kwargs):
    if rank:
        argv = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            settings["peer_ssh"],
            shlex.join(map(str, argv)),
        ]
    return subprocess.run(list(map(str, argv)), check=True, timeout=60, **kwargs)


def node(settings, rank, action, path):
    result = command(
        settings,
        rank,
        [settings["server_python"], "-B", HERE / "guarded_node.py", action, path],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def write_remote(settings, path, value):
    # Structured bytes on stdin; no shell expansion of JSON or paths.
    script = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())"
    command(
        settings,
        1,
        [settings["server_python"], "-c", script, str(path)],
        input=json.dumps(value).encode(),
        stdout=subprocess.DEVNULL,
    )


def plan(settings, root):
    root = inside_repo(root)
    work = inside_repo(settings["work_dir"])
    if not root.is_relative_to(work):
        raise ValueError("results must be under work_dir for the budget guard")
    harness_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    cases = []
    for layout in settings.get("layouts", ["tp4"]):
        for cc in settings.get("concurrencies_by_layout", {}).get(
            layout, DEFAULT_CONCURRENCIES
        ):
            selected = profile(layout, cc, settings.get("hicache_size", 88))
            selected["resolved_args"]["speculative_draft_model_path"] = settings[
                "draft_path"
            ]
            cases.append(
                dict(
                    layout=layout,
                    cc=cc,
                    profile=selected,
                    discard_first=cc == 128 and settings.get("repeat_cc128", False),
                )
            )
    return dict(
        harness_commit=harness_commit,
        engine_commit=ENGINE_COMMIT,
        model_revision=MODEL_REVISION,
        draft_revision=DRAFT_REVISION,
        primary_y="total input + output tokens/s/GPU, including cached input",
        primary_x="1000 / evalscope TPOT p50 (ms)",
        tail_x="1000 / evalscope TPOT p90 (ms)",
        timing_note="evalscope first choices packet, possibly role-only; not AgentX first-content timing",
        settings=settings,
        results=str(root),
        cases=cases,
    )


def node_job(settings, case, rank, case_dir):
    return dict(
        output=str(case_dir / f"node{rank}"),
        budget_root=settings["work_dir"],
        cwd=str(REPO),
        gpu=True,
        gpu_lock_file=str(HERE / "runtime/.gpu-owner.lock"),
        ports=[settings.get("port", 30080), settings.get("dist_port", 39746)]
        if rank == 0
        else [settings.get("port", 30080)],
        timeout_s=14400,
        argv=[
            settings["server_python"],
            "-B",
            str(HERE / "launch_server.py"),
            str(case_dir / "settings.json"),
            case["layout"],
            str(case["cc"]),
            str(rank),
        ],
    )


def alive(settings, rank, path):
    state = node(settings, rank, "status", path)
    if state["state"] not in ("running", "starting"):
        raise RuntimeError(f"node {rank} stopped: {state}")
    return state


def wait_ready(settings, case_dir, ranks):
    url = f"http://127.0.0.1:{settings.get('port', 30080)}/health"
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        for rank in ranks:
            alive(settings, rank, case_dir / f"node{rank}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError("server startup exceeded 2400s")


def site_env(settings, case_dir, results_name="client-results"):
    values = dict(
        N_GPUS=str(json.loads((case_dir / "profile.json").read_text())["n_gpus"]),
        BASE_URL=f"http://127.0.0.1:{settings.get('port', 30080)}",
        SERVED_MODEL_NAME="beam",
        MODEL_PATH=settings["model_path"],
        MODEL_REVISION=MODEL_REVISION,
        SERVER_SOURCE=ENGINE_COMMIT,
        CLIENT_BASE_PYTHON=settings["client_base_python"],
        HF_HOME=str(Path(settings["work_dir"]) / "hf"),
        WORK_DIR=settings["work_dir"],
        SERVER_PROFILE=str(case_dir / "profile.json"),
        RESULTS_DIR=str(case_dir / results_name),
        CHAT_TEMPLATE_KWARGS='{"allow_missing_thinking":true}',
    )
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in values.items())


def run_case(settings, case, root):
    case_dir = root / f"{case['layout']}-c{case['cc']}"
    # Never overwrite/retry a failed measured point automatically.
    case_dir.mkdir(exist_ok=False)
    for name, value in [("settings.json", settings), ("profile.json", case["profile"])]:
        save(case_dir / name, value)
        if case["profile"]["resolved_args"]["nnodes"] == 2:
            write_remote(settings, case_dir / name, value)
    (case_dir / "site.env").write_text(site_env(settings, case_dir))
    ranks = range(case["profile"]["resolved_args"]["nnodes"])
    started = []
    client_guards = []
    try:
        for rank in reversed(ranks):
            job = node_job(settings, case, rank, case_dir)
            path = case_dir / f"job{rank}.json"
            save(path, job)
            if rank:
                write_remote(settings, path, job)
            started.append(rank)
            save(case_dir / f"launch{rank}.json", node(settings, rank, "start", path))
        wait_ready(settings, case_dir, ranks)
        # Save effective child environments, not only intended launch values.
        from curve_config import SIMULATION

        for rank in ranks:
            state = alive(settings, rank, case_dir / f"node{rank}")
            script = (
                'import pathlib,json,sys; raw=pathlib.Path("/proc/"+sys.argv[1]+"/environ").read_bytes(); '
                'env=dict(x.split(b"=",1) for x in raw.split(b"\\0") if b"=" in x); '
                'keys=json.loads(sys.argv[2]); print(json.dumps({k:env.get(k.encode(),b" ").decode() for k in keys}))'
            )
            actual = command(
                settings,
                rank,
                [
                    settings["server_python"],
                    "-c",
                    script,
                    str(state["pid"]),
                    json.dumps(list(SIMULATION)),
                ],
                capture_output=True,
                text=True,
            )
            actual = json.loads(actual.stdout)
            save(case_dir / f"simulation-node{rank}.json", actual)
            if actual != SIMULATION:
                raise RuntimeError(f"node{rank} simulation mismatch: {actual}")
        phases = ["discarded", "scored"] if case.get("discard_first") else ["scored"]
        for phase in phases:
            results_name = (
                "client-results-discarded" if phase == "discarded" else "client-results"
            )
            guard_name = (
                "client-guard-discarded" if phase == "discarded" else "client-guard"
            )
            site_path = case_dir / f"site-{phase}.env"
            site_path.write_text(site_env(settings, case_dir, results_name))
            job = dict(
                output=str(case_dir / guard_name),
                budget_root=settings["work_dir"],
                cwd=str(HERE),
                gpu=False,
                timeout_s=10800,
                env=dict(
                    SITE_ENV=str(site_path),
                    CONCURRENCIES=str(case["cc"]),
                    MIN_FREE_GIB="1024",
                ),
                argv=["bash", str(HERE / "run_sweep.sh")],
            )
            save(case_dir / f"{guard_name}-job.json", job)
            node(settings, 0, "start", case_dir / f"{guard_name}-job.json")
            client_guards.append(case_dir / guard_name)
            while True:
                for rank in ranks:
                    alive(settings, rank, case_dir / f"node{rank}")
                state = node(settings, 0, "status", case_dir / guard_name)
                if state["state"] not in ("running", "starting"):
                    if state["state"] != "exited" or state["returncode"] != 0:
                        raise RuntimeError(f"client failed: {state}")
                    break
                time.sleep(5)
            row = case_dir / results_name / f"c{case['cc']}" / "rung.json"
            result = json.loads(row.read_text())
            if result["status"] != "OK":
                raise RuntimeError(f"raw request audit failed: {result}")
            if phase == "discarded":
                save(
                    case_dir / "discarded-result.json",
                    {
                        **result,
                        "included_in_curve": False,
                        "reason": "User requested discarding first cc128 measurement and using the rerun",
                    },
                )

    finally:
        errors = []
        for guard in client_guards:
            try:
                node(settings, 0, "stop", guard)
            except Exception as exc:
                errors.append(repr(exc))
        for rank in started:
            try:
                node(settings, rank, "stop", case_dir / f"node{rank}")
                # Wait for cleanup before the next case or resource summary.
                for _ in range(12):
                    r = command(
                        settings,
                        rank,
                        [
                            settings["server_python"],
                            "-c",
                            "import pathlib,sys; print(pathlib.Path(sys.argv[1]).exists())",
                            str(case_dir / f"node{rank}" / "DONE"),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if r.stdout.strip() == "True":
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError(f"node{rank} cleanup did not finish")
            except Exception as exc:
                errors.append(repr(exc))
        # Small control/log evidence only. Raw prompt payloads stay on rank0.
        if 1 in started:
            try:
                for name in (
                    "status.json",
                    "baseline.json",
                    "final-resources.json",
                    "resources.jsonl",
                    "command.log",
                    "supervisor.log",
                ):
                    path = case_dir / "node1" / name
                    r = command(settings, 1, ["cat", str(path)], capture_output=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(r.stdout)
            except Exception as exc:
                errors.append(repr(exc))
        if errors:
            raise RuntimeError(f"teardown/evidence incomplete: {errors}")
    # A later safety stop cannot be rescued by an earlier successful client row.
    for rank in ranks:
        node_dir = case_dir / f"node{rank}"
        baseline = json.loads((node_dir / "baseline.json").read_text())
        final = json.loads((node_dir / "final-resources.json").read_text())
        status = json.loads((node_dir / "status.json").read_text())
        errors = violations(final, baseline, status.get("footprint_bytes", 0))
        if status["state"] not in ("stopped", "exited") or errors:
            raise RuntimeError(f"node{rank} resource audit failed: {status}; {errors}")
    server_after = json.loads(
        (
            case_dir / "client-results" / f"c{case['cc']}" / "server-info-after.json"
        ).read_text()
    )
    observed = [
        state["avg_spec_accept_length"]
        for state in server_after.get("internal_states", [])
        if "avg_spec_accept_length" in state
    ]
    if not observed or any(not 3.75 <= value <= 4.25 for value in observed):
        raise RuntimeError(f"simulated acceptance4 not observed: {observed}")
    result["observed_simulated_accept_length"] = observed
    result["retraction_log_lines"] = {}
    for rank in ranks:
        lines = (
            (case_dir / f"node{rank}" / "command.log")
            .read_text(errors="replace")
            .splitlines()
        )
        result["retraction_log_lines"][f"node{rank}"] = [
            line for line in lines if "retract" in line.lower()
        ]
    save(
        case_dir / "result.json",
        {
            **result,
            "layout": case["layout"],
            "hicache_gb_per_rank": case["profile"]["resolved_args"].get(
                "hicache_size", 0
            ),
            "selection_note": settings.get("selection_note"),
            "profile": case["profile"],
            "selected_measurement": 2 if case.get("discard_first") else 1,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("settings")
    p.add_argument("--results", required=True)
    p.add_argument(
        "--plot-python",
        default=sys.executable,
        help="separate interpreter with matplotlib",
    )
    a = p.parse_args()
    settings = json.loads(Path(a.settings).read_text())
    document = plan(settings, a.results)
    if a.action == "plan":
        print(json.dumps(document, indent=2))
        return
    root = inside_repo(a.results)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / ".campaign.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = root / "plan.json"
    if old.exists() and json.loads(old.read_text()) != document:
        raise ValueError("changed plan: use a fresh result directory")
    save(old, document)
    # Both checkouts must match this committed harness and be clean.
    local_head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    ranks = range(
        max(c["profile"]["resolved_args"]["nnodes"] for c in document["cases"])
    )
    for rank in ranks:
        sha = command(
            settings,
            rank,
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = command(
            settings,
            rank,
            ["git", "-C", REPO, "status", "--porcelain"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if sha != local_head or dirty:
            raise RuntimeError(f"node{rank} checkout differs or is dirty")
    for case in document["cases"]:
        path = root / f"{case['layout']}-c{case['cc']}"
        if (path / "result.json").exists():
            result = json.loads((path / "result.json").read_text())
            if result["status"] == "OK" and all(
                (path / f"node{r}" / "final-resources.json").exists()
                for r in range(case["profile"]["resolved_args"]["nnodes"])
            ):
                continue
        save(root / "status.json", dict(state="running", case=case))
        try:
            run_case(settings, case, root)
        except BaseException as exc:
            save(root / "status.json", dict(state="failed", case=case, error=repr(exc)))
            raise
        subprocess.run(
            [a.plot_python, str(HERE / "plot_curve.py"), str(root)], check=True
        )
    save(root / "status.json", dict(state="complete"))


if __name__ == "__main__":
    main()
