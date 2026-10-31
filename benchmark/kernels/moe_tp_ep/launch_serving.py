"""Record the server's software identity, then launch it in the same process."""

import argparse
import json
import runpy
import sys
from pathlib import Path

from bench_moe_tp_ep import environment_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    launch_args = args.server_args
    if launch_args[:1] == ["--"]:
        launch_args = launch_args[1:]
    if not launch_args:
        parser.error("pass SGLang launch arguments after --")
    report = dict(
        command=[sys.executable, "-m", "sglang.launch_server", *launch_args],
        metadata=environment_metadata(),
    )
    report["metadata"]["execution_mode"] = "serving; configuration in launch command"
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    # Each independent service start needs its own provenance file.
    with args.metadata.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    sys.argv = ["sglang.launch_server", *launch_args]
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
