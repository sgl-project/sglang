import argparse
import dataclasses
import os
import sys
from typing import Optional

from sglang_simulator.compat import (
    apply_simulator_server_args,
    validate_launch_runtime,
)
from sglang_simulator.simulation.sglang.hook_bootstrap import (
    install_simulator_hooks,
    run_simulator_detokenizer_process,
    run_simulator_scheduler_process,
)
from sglang_simulator.utils import get_logger

install_simulator_hooks()


logger = get_logger("sgl_simulator")


@dataclasses.dataclass
class SimulationArgs:
    sim_config_path: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--sim-config-path",
            type=str,
            default=None,
            help="Path to simulation JSON config (same as SGLANG_SIMULATOR_CONFIG_PATH).",
        )

    @classmethod
    def from_cli_args(cls, ns: argparse.Namespace) -> "SimulationArgs":
        return SimulationArgs(sim_config_path=ns.sim_config_path)


def _has_cli_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def apply_simulator_defaults(raw_args: argparse.Namespace, argv: list[str]) -> None:
    """Avoid real model execution while preserving explicit SGLang options."""
    if not _has_cli_option(argv, "--load-format"):
        raw_args.load_format = "dummy"

    if os.getenv("SGLANG_USE_CPU_ENGINE") != "1":
        return

    if not _has_cli_option(argv, "--device"):
        raw_args.device = "cpu"
    if not _has_cli_option(argv, "--attention-backend"):
        raw_args.attention_backend = "torch_native"
    if not _has_cli_option(argv, "--sampling-backend"):
        raw_args.sampling_backend = "pytorch"
    if not (
        _has_cli_option(argv, "--cuda-graph-backend-decode")
        or _has_cli_option(argv, "--cuda-graph-backend-prefill")
        or _has_cli_option(argv, "--disable-cuda-graph")
    ):
        raw_args.disable_cuda_graph = True

    # CPU-only model validation may still query CUDA capability while
    # constructing ServerArgs, before the simulator runner is spawned.
    import torch

    torch.cuda.get_device_capability = lambda *_args, **_kwargs: (10, 0)


if __name__ == "__main__":
    validate_launch_runtime()

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    argv = sys.argv[1:]
    raw_args = parser.parse_args(argv)
    apply_simulator_defaults(raw_args, argv)
    apply_simulator_server_args(raw_args)
    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    config_path = os.getenv("SGLANG_SIMULATOR_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from {config_path}")
    elif simulation_args.sim_config_path:
        os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = simulation_args.sim_config_path

    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_simulator_scheduler_process,
            run_detokenizer_process_func=run_simulator_detokenizer_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
