"""PR-only worker lifecycle and strict suite adapter; never runs on CPU precheck."""

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from select_pr_e2e import ROOT, digest, read_json, validate_plan

PREFIX = {"answer": "ANSWER", "accuracy": "ACCURACY", "perf": "PERF", "pd": "PD_PERF"}


def context():
    plan = read_json(os.environ["PPU_E2E_PLAN"])
    validate_plan(plan)
    entries = [e for e in plan["entries"] if e["id"] == os.environ["PPU_E2E_ENTRY"]]
    if len(entries) != 1:
        raise ValueError("Entry must occur exactly once in the selection plan")
    entry = entries[0]
    rank = int(os.environ.get("NODE_RANK", "0"))
    if not 0 <= rank < entry["nodes"]:
        raise ValueError("Unexpected worker rank")
    directory = Path(os.environ["PPU_EVIDENCE_DIR"])
    return plan, entry, rank, directory


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def suite():
    # Called only after a board adapter has completed installation/network setup.
    plan, entry, rank, directory = context()
    rank_dir = directory / f"rank-{rank}"
    import torch

    import sglang

    if not Path(sglang.__file__).resolve().is_relative_to(ROOT / "python"):
        raise ValueError("Worker is not importing the tested checkout")
    config = read_json(ROOT / entry["config"])
    model_path = Path(config["model"]["path"])
    metadata = read_json(model_path / "config.json")
    weights = sorted(
        str(p.relative_to(model_path)) for p in model_path.rglob("*.safetensors")
    )
    if not weights:
        raise ValueError("Checkpoint contains no safetensors weights")
    if metadata.get("model_type") not in config["model"]["accepted_model_types"]:
        raise ValueError("Checkpoint model type differs from the reviewed config")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != len(config["hardware"]["visible_devices"]):
        raise ValueError("Visible device count differs from the reviewed config")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if commit != plan["tested_sha"]:
        raise ValueError("Worker checkout differs from the selected tested revision")
    save(
        rank_dir / "environment.json",
        {
            "commit": commit,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "checkout_import": str(sglang.__file__),
            "devices": devices,
            "image": os.environ["PPU_BASE_IMAGE_REF"],
            "digest": os.environ["PPU_BASE_IMAGE_DIGEST"],
            "model": {
                "path": str(model_path),
                "metadata_digest": digest(metadata),
                "weights": weights,
            },
            "setup_seconds": time.time() - float(os.environ["PPU_E2E_STARTED"]),
        },
    )
    from validate_pr_e2e_configs import validate_configs

    validate_configs([entry])
    command = [
        sys.executable,
        str(ROOT / "test/run_suite.py"),
        "--hw",
        "ppu",
        "--suite",
        entry["suite"],
        "--nightly",
        "--pytest",
        "--require-nonempty",
        "--junit-dir",
        str(rank_dir / "pytest"),
        "--timeout-per-file",
        str(entry["timeout_seconds"]),
        "--suite-timeout",
        str(entry["timeout_seconds"]),
    ]
    return subprocess.call(command, cwd=ROOT)


def run():
    plan, entry, rank, directory = context()
    rank_dir = directory / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = {
        "id": entry["id"],
        "plan_hash": plan["plan_hash"],
        "rank": rank,
        "tested_sha": plan["tested_sha"],
        "run_id": plan["run_id"],
        "attempt": plan["attempt"],
        "config_digest": entry["config_digest"],
        "exit_code": 1,
    }
    process = None
    try:
        config = read_json(ROOT / entry["config"])
        if entry["blocked"]:
            raise ValueError(entry["blocked"])
        model = Path(config["model"]["path"])
        if not model.is_absolute() or not (model / "config.json").is_file():
            raise ValueError("Checkpoint is not staged at the reviewed absolute path")
        if (
            entry["category"] == "accuracy"
            and not Path(config["evaluation"]["dataset_dir"]).is_dir()
        ):
            raise ValueError("Accuracy dataset is not staged")
        prefix = PREFIX[entry["category"]]
        family = "PERF" if entry["category"] == "pd" else prefix
        env = {
            **os.environ,
            **entry["environment"],
            "PPU_E2E_STARTED": str(started),
            "PPU_INSTALL_PROFILE": "pr",
            "PPU_E2E_REQUIRE_PD": str(entry["category"] == "pd").lower(),
            "PPU_INSTALL_EVIDENCE_DIR": str(rank_dir),
            "GITHUB_WORKSPACE": str(ROOT),
            "GITHUB_STEP_SUMMARY": str(rank_dir / "summary.log"),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(ROOT),
            "PYTHONPATH": str(ROOT / "python") + ":" + str(ROOT / "test"),
            "SGLANG_IS_IN_CI": "true",
            "SGLANG_TEST_MAX_RETRY": "0",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            "SGLANG_SAIL_PLA_CUDA": "0",
            "SGLANG_PPU_SOURCE_REVISION": plan["tested_sha"],
            "SGLANG_PPU_DISTRIBUTED_ACTION_SHA": "06c8ee7cadaae2086db4977e571b19f461653f16",
            "CPLUS_INCLUDE_PATH": "/usr/local/PPU_SDK/targets/x86_64-linux/include",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_CACHE": "/nas_aisw/datasets/hf_cache/hub",
            "PPU_SUPPORTS_FP8": "0",
            "CUDA_VISIBLE_DEVICES": ",".join(
                map(str, config["hardware"]["visible_devices"])
            ),
            "PIP_CACHE_DIR": "/ppu-pr-cache/pip",
            "SGL_KERNEL_WHEEL_CACHE_DIR": "/ppu-pr-cache/aot/ZW-M890P",
            "PPU_JIT_CACHE_LOCAL": f"/tmp/ppu-e2e-{rank}-tvm",
            "SGLANG_JIT_CACHE_DIR": f"/tmp/ppu-e2e-{rank}-jit",
            "DG_JIT_CACHE_DIR": f"/tmp/ppu-e2e-{rank}-dg",
            f"SGLANG_PPU_{prefix}_TEST_CONFIG": str(ROOT / entry["config"]),
            f"SGLANG_PPU_{prefix}_RESULTS_DIR": str(directory / "results"),
            f"{family}_SUITE": entry["suite"],
            f"{family}_TIMEOUT_PER_FILE": str(entry["timeout_seconds"]),
        }
        if entry["category"] == "pd":
            adapter = "run_pd_perf_suite_node.sh"
        elif entry["nodes"] > 1:
            adapter = "run_pr_e2e_multinode.sh"
            env["PPU_E2E_KIND"] = prefix
        else:
            adapter = f"run_{entry['category']}_suite_board.sh"

        def interrupted(signum, frame):
            raise InterruptedError(f"Worker received signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        with (rank_dir / "worker.log").open("w") as log:
            process = subprocess.Popen(
                ["bash", str(ROOT / "scripts/ci/ppu" / adapter)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["exit_code"] = process.wait(
                timeout=entry["worker_minutes"] * 60 - 60
            )
    except Exception as error:
        record["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        if process is not None:
            if record["exit_code"] != 0:
                # Let the strict runner terminate its separate pytest session first.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        record["elapsed"] = time.time() - started
        save(rank_dir / "worker.json", record)
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(suite() if sys.argv[1:] == ["suite"] else run())
