from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

OUTPUT_DIR_NAMES = {
    "benchmarks": Path("outputs/diffusion_benchmarks"),
    "profiles": Path("outputs/diffusion_profiles"),
}


def get_repo_root() -> Path:
    import sglang

    return Path(sglang.__file__).resolve().parents[2]


def get_assets_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "inputs" / "diffusion_benchmark" / "figs"


def get_output_dir(name: str, repo_root: Path | None = None) -> Path:
    if name not in OUTPUT_DIR_NAMES:
        raise KeyError(f"Unknown output dir name: {name}")
    root = repo_root or get_repo_root()
    return root / OUTPUT_DIR_NAMES[name]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_write_access(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    probe_dir = ensure_dir(root / ".cache" / "diffusion_skill_write_test")
    probe_file = probe_dir / "probe.txt"
    probe_file.write_text("ok", encoding="utf-8")
    return probe_file


def _run_nvidia_smi(query: str) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-{query}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[list[str]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append([field.strip() for field in csv.reader([line]).__next__()])
    return rows


def get_gpu_inventory() -> list[dict[str, int | str]]:
    rows = _run_nvidia_smi("gpu=index,uuid,memory.used,memory.total,utilization.gpu")
    inventory = []
    for index, uuid, memory_used, memory_total, utilization_gpu in rows:
        inventory.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(memory_used),
                "memory_total_mib": int(memory_total),
                "utilization_gpu_pct": int(utilization_gpu),
            }
        )
    return inventory


def get_busy_gpu_uuids() -> set[str]:
    rows = _run_nvidia_smi("compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    return {gpu_uuid for gpu_uuid, *_ in rows}


def pick_idle_gpus(
    required_gpus: int,
    max_memory_used_mib: int = 32,
    max_utilization_gpu_pct: int = 5,
) -> list[int]:
    inventory = get_gpu_inventory()
    busy_uuids = get_busy_gpu_uuids()

    idle = [
        int(gpu["index"])
        for gpu in inventory
        if gpu["uuid"] not in busy_uuids
        and int(gpu["memory_used_mib"]) <= max_memory_used_mib
        and int(gpu["utilization_gpu_pct"]) <= max_utilization_gpu_pct
    ]
    if len(idle) < required_gpus:
        raise RuntimeError(
            "Not enough idle GPUs. "
            f"required={required_gpus}, idle={idle}, inventory={inventory}, busy={sorted(busy_uuids)}"
        )
    return idle[:required_gpus]


def configure_runtime_env(required_gpus: int = 1) -> str | None:
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None
    selected = ",".join(str(index) for index in pick_idle_gpus(required_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve SGLang diffusion skill paths and idle GPUs."
    )
    parser.add_argument(
        "command",
        choices=[
            "print-root",
            "print-assets-dir",
            "print-output-dir",
            "print-idle-gpus",
            "check-write-access",
        ],
    )
    parser.add_argument(
        "--kind",
        choices=sorted(OUTPUT_DIR_NAMES),
        help="Output directory kind for print-output-dir.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of idle GPUs to print.",
    )
    parser.add_argument(
        "--mkdir",
        action="store_true",
        help="Create the requested directory before printing it.",
    )
    args = parser.parse_args()

    if args.command == "print-root":
        print(get_repo_root())
        return
    if args.command == "print-assets-dir":
        path = get_assets_dir()
        if args.mkdir:
            ensure_dir(path)
        print(path)
        return
    if args.command == "print-output-dir":
        if not args.kind:
            raise SystemExit("--kind is required for print-output-dir")
        path = get_output_dir(args.kind)
        if args.mkdir:
            ensure_dir(path)
        print(path)
        return
    if args.command == "print-idle-gpus":
        print(",".join(str(index) for index in pick_idle_gpus(args.count)))
        return
    if args.command == "check-write-access":
        print(check_write_access())
        return
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
