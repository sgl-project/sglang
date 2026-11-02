#!/usr/bin/env python3
"""Bind existing experiment checkouts; no clone/install/rebase/model execution."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def git_info(root, expected=None):
    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        top = Path(run("rev-parse", "--show-toplevel")).resolve()
    except subprocess.CalledProcessError:
        top = None
    if top != root:
        if expected:
            raise ValueError(f"{root}: requested ref requires an exact Git root")
        return {"root": str(root), "revision": None, "exported_source": True}
    revision = run("rev-parse", "HEAD")
    if expected and run("rev-parse", "--verify", expected + "^{commit}") != revision:
        raise ValueError(f"{root}: HEAD differs from requested ref {expected}")
    return {
        "root": str(root),
        "revision": revision,
        "dirty": bool(run("status", "--porcelain")),
        "expected_ref": expected,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["simulator", "source", "aic"]:
        p.add_argument("--" + name + "-root", type=Path, required=name != "aic")
        if name != "source":
            p.add_argument("--" + name + "-ref")
    p.add_argument("--systems-path", type=Path)
    p.add_argument("--mode", choices=["afd"], default="afd")
    p.add_argument("--routing", choices=["f-side", "direct"])
    p.add_argument(
        "--afd-config", help="Native AFD JSON with explicit fixed/direct policy"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be new; previous configurations are preserved")
    if (a.aic_ref or a.systems_path) and not a.aic_root:
        p.error("--aic-ref/--systems-path requires --aic-root")
    sim = a.simulator_root.resolve(strict=True)
    source = a.source_root.resolve(strict=True)
    package = sim / "tools/sglang-simulator/src"
    sys.path.insert(0, str(package))
    from sglang_simulator.afd.adapters.sglang.source import verify_source
    from sglang_simulator.afd.native_config import select_native_config

    native, routing, _ = select_native_config(a.afd_config, routing=a.routing)
    files = verify_source(source, mode=a.mode, routing=routing)
    state = {
        "mode": a.mode,
        "routing": routing,
        "native_afd_config": native.raw if native else None,
        "simulator": git_info(sim, a.simulator_ref),
        "source": {"root": str(source)},
        "source_files": files,
        "aic": None,
        "AIC_API_and_data_validated": False,
        "automatic_rebase": False,
        "model_executed": False,
    }
    if a.aic_root:
        ar = a.aic_root.resolve(strict=True)
        core_systems = ar / "aic-core/src/aiconfigurator_core/systems"
        default_systems = (
            core_systems if core_systems.is_dir() else ar / "src/aiconfigurator/systems"
        )
        systems = (a.systems_path or default_systems).resolve(strict=True)
        if not systems.is_dir():
            raise ValueError("--systems-path must be a systems directory")
        state["aic"] = dict(
            git_info(ar, a.aic_ref),
            systems_path=str(systems),
            installation="separately installed editable checkout or pinned wheel",
        )
    state["status"] = "SOURCE_API_CHECKED"
    a.output.mkdir(parents=True)
    (a.output / "environment.json").write_text(json.dumps(state, indent=2) + "\n")
    lines = [
        "# Local AFD environment; no installation performed.",
        "export PYTHONPATH="
        + shlex.quote(str(package))
        + chr(36)
        + "{PYTHONPATH:+:$PYTHONPATH}",
        "export SGLANG_SIMULATOR_SOURCE_ROOT=" + shlex.quote(str(source)),
        "export CUDA_VISIBLE_DEVICES=''",
    ]
    if state["aic"]:
        lines += [
            "export AIC_ROOT=" + shlex.quote(state["aic"]["root"]),
            "export AIC_SYSTEMS_PATH=" + shlex.quote(state["aic"]["systems_path"]),
        ]
    (a.output / "env.sh").write_text("\n".join(lines) + "\n")
    print("source " + shlex.quote(str(a.output.resolve() / "env.sh")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Configuration failed: {exc}", file=sys.stderr)
        sys.exit(2)
