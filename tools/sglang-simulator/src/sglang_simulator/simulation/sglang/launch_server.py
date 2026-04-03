import argparse
import dataclasses
import os
import sys
from typing import Optional

import sglang_simulator.hook as sgl_simulator_hook
import torch
from sglang_simulator.simulation.sglang import (
    model_runner,
    scheduler,
    sgl_kernel_hook,
)
from sglang_simulator.utils import get_logger

# hook the sglang implementation
if not torch.cuda.is_available():
    # CPU Platform
    sgl_simulator_hook.install_module_hooks(
        [sgl_kernel_hook.M_SGLangKernelLoadUtilHook]
    )
sgl_simulator_hook.install_class_hooks(
    [
        scheduler.C_SchedulerHook,
        model_runner.C_ModelRunnerHook,
    ]
)


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


# Ref: https://github.com/sgl-project/sglang/blob/v0.5.6.post2/python/sglang/launch_server.py
if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    raw_args = parser.parse_args(sys.argv[1:])
    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    config_path = os.getenv("SGLANG_SIMULATOR_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from {config_path}")
    elif simulation_args.sim_config_path:
        os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = simulation_args.sim_config_path

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
