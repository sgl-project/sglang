"""Native-style role bootstrap for the existing Simulator server entry."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from .native_config import NativeAFDConfig
from .serving import AFDTaskClient, AFDTaskServer, CostCatalog


def prepare_launch(argv):
    """Handle F before SGLang/Torch imports; strip native AFD flags on CPU A.

    The real CPU Scheduler stays non-disaggregated internally. Target A/F ranks
    and native AFD policy are passed to the event service, not CPU collectives.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--afd-execution-mode", choices=("off", "attention", "ffn"), default="off"
    )
    parser.add_argument("--afd-config")
    parser.add_argument("--afd-source-root", type=Path)
    parser.add_argument("--afd-cost-catalog", type=Path)
    parser.add_argument("--afd-output-dir", type=Path)
    parser.add_argument("--afd-allow-fixture", action="store_true")
    parser.add_argument("--sim-config-path")
    args, remaining = parser.parse_known_args(argv)
    if args.afd_execution_mode == "off":
        if any(
            (
                args.afd_config,
                args.afd_source_root,
                args.afd_cost_catalog,
                args.afd_output_dir,
                args.afd_allow_fixture,
            )
        ):
            parser.error("AFD options require --afd-execution-mode attention or ffn")
        return list(argv), None
    if not args.afd_config:
        parser.error("--afd-config is required for an AFD role")
    native = NativeAFDConfig.parse(args.afd_config)
    if args.afd_execution_mode == "ffn":
        if remaining:
            parser.error(
                "The CPU F task service accepts AFD options only; model weights and inference HTTP options belong to A"
            )
        sim_path = args.sim_config_path or os.environ.get(
            "SGLANG_SIMULATOR_CONFIG_PATH"
        )
        settings = (
            json.loads(Path(sim_path).read_text()).get("predictor", {})
            if sim_path
            else {}
        )

        def asset(value):
            if not value:
                return None
            p = Path(value)
            return (
                p
                if p.is_absolute() or not sim_path
                else Path(sim_path).resolve().parent / p
            )

        source_root = args.afd_source_root or asset(settings.get("source_root"))
        catalog_path = args.afd_cost_catalog or asset(settings.get("cost_catalog"))
        output_dir = args.afd_output_dir or asset(settings.get("event_output_dir"))
        if not all((source_root, catalog_path, output_dir)):
            parser.error(
                "F requires source_root, cost_catalog and event_output_dir in simulator JSON, or their --afd-* options"
            )
        catalog = CostCatalog(
            catalog_path,
            source_root,
            native,
            allow_fixture=args.afd_allow_fixture
            or settings.get("allow_fixture") is True,
        )
        with AFDTaskServer(native, catalog, output_dir) as server:
            print(
                "AFD F internal CPU service ready at " + str(server.server_address),
                flush=True,
            )
            server.serve_forever(poll_interval=0.2)
        return None, None
    if any(
        (
            args.afd_source_root,
            args.afd_cost_catalog,
            args.afd_output_dir,
            args.afd_allow_fixture,
        )
    ):
        parser.error(
            "Catalog/source/output/fixture options belong to the F task service"
        )
    source = args.sim_config_path or os.environ.get("SGLANG_SIMULATOR_CONFIG_PATH")
    if not source:
        parser.error(
            "A requires the ordinary --sim-config-path or SGLANG_SIMULATOR_CONFIG_PATH"
        )
    config = json.loads(Path(source).read_text())
    mode = os.environ.get("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE").upper()
    if mode != "OFFLINE":
        parser.error(
            "AFD request integration currently supports OFFLINE logical time only"
        )
    if "--skip-server-warmup" not in remaining:
        remaining.append("--skip-server-warmup")
    peer = AFDTaskClient(native).request("handshake")
    config["predictor"] = dict(name="afd", afd_config=native.raw)
    config["afd_launch"] = dict(
        native_afd_config=native.raw,
        peer=peer,
        cpu_runtime_roles=2,
        capacity_fit="unknown",
        service_accuracy_verified=False,
    )
    temp = tempfile.TemporaryDirectory(prefix="sglang-simulator-afd-")
    path = Path(temp.name) / "simulation.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(path)
    os.environ["SGLANG_SIMULATOR_AFD_REQUESTS"] = "1"
    return remaining + ["--sim-config-path", str(path)], temp
