"""Configure, prepare and run the measured four-GB300 Beam OpenHands recipe."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from curve_config import (
    CONCURRENCIES,
    ENGINE_COMMIT,
    MODEL_REVISION,
    inside_repo,
    profile,
)
from guarded_node import GIB, disk_footprint, save, snapshot, violations

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CCS = [1, 4, 8, 16, 32]


def executable(value):
    found = shutil.which(value)
    if not found:
        raise ValueError(f"Interpreter is not executable: {value}")
    # Keep the venv's executable path; resolving its symlink loses the venv.
    return os.path.abspath(found)


def initialize(args):
    work = inside_repo(args.work_dir)
    if (
        not all(1 <= port <= 65535 for port in (args.port, args.dist_port))
        or args.port == args.dist_port
    ):
        raise ValueError("Use two different ports in 1..65535")
    settings = dict(
        work_dir=str(work),
        model_path=str(Path(args.model_path).resolve()),
        draft_path=str(Path(args.draft_path).resolve()),
        server_python=executable(args.server_python),
        client_base_python=executable(args.client_python),
        head_addr="127.0.0.1",
        port=args.port,
        dist_port=args.dist_port,
        layouts=args.layouts,
        concurrencies_by_layout={layout: args.concurrencies for layout in args.layouts},
        hicache_size=88,
        repeat_cc128=False,
        runtime_pythonpaths=[str(Path(p).resolve()) for p in args.runtime_pythonpath],
    )
    for key in ("model_path", "draft_path"):
        if Path(settings[key]).is_relative_to(work):
            raise ValueError("Keep model weights outside work_dir's 80 GiB budget")
    if len(set(args.layouts)) != len(args.layouts) or len(
        set(args.concurrencies)
    ) != len(args.concurrencies):
        raise ValueError("Duplicate layouts or concurrency values")
    bootstrap = profile(args.layouts[0], args.concurrencies[0], hicache_size=88)
    bootstrap["resolved_args"]["speculative_draft_model_path"] = settings["draft_path"]
    env = dict(
        BASE_URL=f"http://127.0.0.1:{args.port}",
        SERVED_MODEL_NAME="beam",
        MODEL_PATH=settings["model_path"],
        MODEL_REVISION=MODEL_REVISION,
        SERVER_SOURCE=ENGINE_COMMIT,
        SERVER_PYTHON=settings["server_python"],
        CLIENT_BASE_PYTHON=settings["client_base_python"],
        WORK_DIR=str(work),
        HF_HOME=str(work / "hf"),
        SERVER_PROFILE=str(work / "bootstrap-profile.json"),
        N_GPUS="4",
        CHAT_TEMPLATE_KWARGS='{"allow_missing_thinking":true}',
    )
    files = {
        "settings.json": json.dumps(settings, indent=2) + "\n",
        "bootstrap-profile.json": json.dumps(bootstrap, indent=2) + "\n",
        "site.env": "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in env.items()
        ),
    }
    for name, content in files.items():
        path = work / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to overwrite {path}; use a new --work-dir")
    work.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if not (work / name).exists():
            (work / name).write_text(content)
    return work / "settings.json"


def preparation_job(settings, label):
    work = inside_repo(settings["work_dir"])
    return dict(
        output=str(work / label),
        budget_root=str(work),
        cwd=str(REPO),
        gpu=False,
        timeout_s=14400,
        argv=["bash", str(HERE / "prepare_repro.sh")],
        # Never serialize the inherited environment: it can contain HF credentials.
        env={
            "SITE_ENV": str(work / "site.env"),
            "BEAM_SERVER_PYTHONPATH": os.pathsep.join(
                [*settings.get("runtime_pythonpaths", []), str(REPO / "python")]
            ),
            "PYTHONNOUSERSITE": "1",
        },
    )


def prepare(settings):
    work = inside_repo(settings["work_dir"])
    label = f"prepare-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    job_path = work / (label + ".json")
    save(job_path, preparation_job(settings, label))
    guard = [settings["server_python"], "-B", str(HERE / "guarded_node.py")]
    subprocess.run([*guard, "start", str(job_path)], check=True)
    output = work / label
    print(f"Preparation log: {output / 'command.log'}", flush=True)
    try:
        while not (output / "DONE").exists():
            time.sleep(1)
    except BaseException:
        subprocess.run([*guard, "stop", str(output)], check=True)
        raise
    status = json.loads((output / "status.json").read_text())
    if status.get("state") != "exited" or status.get("returncode") != 0:
        raise RuntimeError(f"Preparation failed: {status}; inspect {output}")
    print("Client, dataset, serving-runtime checks and plotting environment are ready.")


def check_run_budget(settings):
    """Conservative additional working-memory budget from the measured recipe."""
    work = inside_repo(settings["work_dir"])
    now = snapshot(work)
    errors = violations(now, now, disk_footprint(work))
    high = any(
        cc >= 16 for ccs in settings["concurrencies_by_layout"].values() for cc in ccs
    )
    extra = (600 if high else 64) * GIB
    if now["working_bytes"] + extra > 0.78 * now["limit_bytes"]:
        errors.append(
            "insufficient cgroup headroom for the measured recipe plus margin"
        )
    if now["host_available_bytes"] - extra < 200 * GIB:
        errors.append("insufficient host headroom for the recipe plus 200 GiB reserve")
    document = dict(
        resources=now, additional_working_budget_GiB=extra / GIB, errors=errors
    )
    save(work / f"budget-{time.time_ns()}.json", document)
    if errors:
        raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser(
        "init", help="write ignored settings; no model downloads or GPU work"
    )
    init.add_argument("--model-path", required=True)
    init.add_argument("--draft-path", required=True)
    init.add_argument("--server-python", default=sys.executable)
    init.add_argument("--client-python", default="python3.12")
    init.add_argument("--work-dir", default=str(HERE / "runtime"))
    init.add_argument("--layouts", nargs="+", choices=["tp4", "tep4"], default=["tp4"])
    init.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        choices=CONCURRENCIES,
        default=DEFAULT_CCS,
    )
    init.add_argument("--port", type=int, default=30080)
    init.add_argument("--dist-port", type=int, default=39746)
    init.add_argument("--runtime-pythonpath", action="append", default=[])
    for action in ("prepare", "plan", "run"):
        command = sub.add_parser(action)
        command.add_argument("settings", type=Path)
        if action != "prepare":
            command.add_argument("--results", required=True)
    args = parser.parse_args()
    if args.action == "init":
        print(initialize(args))
        return
    settings = json.loads(args.settings.read_text())
    if any(layout not in ("tp4", "tep4") for layout in settings["layouts"]):
        raise ValueError("This reproduction entry point requires one four-GPU node")
    if args.action == "prepare":
        prepare(settings)
        return
    if args.action == "run":
        work = inside_repo(settings["work_dir"])
        required = (
            "venv/bin/python",
            "plot-env/bin/python",
            "server-runtime.json",
            "tokenizer-view.json",
            "datasets/openhands-padded-1024.json",
        )
        missing = [name for name in required if not (work / name).is_file()]
        if missing:
            raise RuntimeError(f"Run prepare first; missing artifacts: {missing}")
        check_run_budget(settings)
    work = inside_repo(settings["work_dir"])
    subprocess.run(
        [
            settings["server_python"],
            "-B",
            str(HERE / "run_curve.py"),
            args.action,
            str(args.settings.resolve()),
            "--results",
            args.results,
            "--plot-python",
            str(work / "plot-env/bin/python"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
